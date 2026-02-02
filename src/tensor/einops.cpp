#include "axiom/einops.hpp"
#include "axiom/operations.hpp"

#include <algorithm>
#include <regex>
#include <set>
#include <sstream>

namespace axiom {
namespace einops {

// ============================================================================
// EinopsExpression Implementation
// ============================================================================

EinopsExpression::EinopsExpression(
    const std::string &pattern, const std::map<std::string, size_t> &axis_sizes)
    : axis_sizes_(axis_sizes) {
    // Split pattern on "->"
    size_t arrow_pos = pattern.find("->");
    if (arrow_pos == std::string::npos) {
        throw EinopsParseError(
            "Pattern must contain '->' to separate input and output: " +
            pattern);
    }

    input_pattern_ = pattern.substr(0, arrow_pos);
    output_pattern_ = pattern.substr(arrow_pos + 2);

    // Trim whitespace
    input_pattern_.erase(0, input_pattern_.find_first_not_of(" \t"));
    input_pattern_.erase(input_pattern_.find_last_not_of(" \t") + 1);
    output_pattern_.erase(0, output_pattern_.find_first_not_of(" \t"));
    output_pattern_.erase(output_pattern_.find_last_not_of(" \t") + 1);

    parse_patterns();
}

void EinopsExpression::parse_patterns() {
    parsed_input_ = parse_single_pattern(input_pattern_);
    parsed_output_ = parse_single_pattern(output_pattern_);

    // Validation: check that all axes in output exist in input (except for
    // repeat operations and special axes like unity/ellipsis)
    auto input_axes = get_pattern_axes(parsed_input_);
    auto output_axes = get_pattern_axes(parsed_output_);

    std::set<std::string> input_set(input_axes.begin(), input_axes.end());
    std::set<std::string> output_set(output_axes.begin(), output_axes.end());

    // Collect anonymous sizes from parsed patterns
    std::set<std::string> known_anonymous;
    for (const auto &element : parsed_input_.elements) {
        if (std::holds_alternative<GroupedAxes>(element)) {
            const auto &group = std::get<GroupedAxes>(element);
            for (const auto &[name, size] : group.anonymous_sizes) {
                known_anonymous.insert(name);
            }
        }
    }
    for (const auto &element : parsed_output_.elements) {
        if (std::holds_alternative<GroupedAxes>(element)) {
            const auto &group = std::get<GroupedAxes>(element);
            for (const auto &[name, size] : group.anonymous_sizes) {
                known_anonymous.insert(name);
            }
        }
    }

    for (const auto &axis : output_set) {
        // Skip special internal markers
        if (axis == "__unity__" || axis == "__ellipsis__" ||
            axis.find("__anon_") == 0 || axis.find("__ellipsis_") == 0) {
            continue;
        }
        // Check if axis is known
        if (input_set.find(axis) == input_set.end() &&
            axis_sizes_.find(axis) == axis_sizes_.end() &&
            known_anonymous.find(axis) == known_anonymous.end()) {
            throw EinopsParseError(
                "Axis '" + axis +
                "' appears in output but not in input and no size specified");
        }
    }
}

ParsedPattern
EinopsExpression::parse_single_pattern(const std::string &pattern) const {
    ParsedPattern result;
    auto tokens = tokenize_pattern(pattern);

    for (const auto &token : tokens) {
        result.elements.push_back(parse_axis_element(token));
    }

    return result;
}

std::vector<std::string>
EinopsExpression::tokenize_pattern(const std::string &pattern) const {
    std::vector<std::string> tokens;
    std::string current_token;
    int paren_depth = 0;

    for (char c : pattern) {
        if (c == ' ' || c == '\t') {
            if (paren_depth == 0 && !current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            } else if (paren_depth > 0) {
                current_token += c;
            }
        } else if (c == '(') {
            current_token += c;
            paren_depth++;
        } else if (c == ')') {
            current_token += c;
            paren_depth--;
            if (paren_depth == 0) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        } else {
            current_token += c;
        }
    }

    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }

    if (paren_depth != 0) {
        throw EinopsParseError("Mismatched parentheses in pattern: " + pattern);
    }

    return tokens;
}

AxisElement
EinopsExpression::parse_axis_element(const std::string &token) const {
    if (token == "...") {
        // Ellipsis - represents remaining dimensions
        return EllipsisAxis{};
    } else if (token == "1") {
        // Unity axis represented by literal '1'
        return UnityAxis{};
    } else if (token.front() == '(' && token.back() == ')') {
        // Check for empty parentheses first
        std::string inner = token.substr(1, token.length() - 2);

        // Trim whitespace from inner content
        inner.erase(0, inner.find_first_not_of(" \t"));
        inner.erase(inner.find_last_not_of(" \t") + 1);

        if (inner.empty()) {
            // Empty composition () - unity axis
            return UnityAxis{};
        }

        // Grouped axes - may contain named axes and anonymous numeric sizes
        std::vector<std::string> axes;
        std::map<std::string, size_t> anonymous_sizes;
        std::istringstream iss(inner);
        std::string axis;
        int anon_counter = 0;

        while (iss >> axis) {
            if (axis.empty()) {
                continue;
            }

            // Check if this is a numeric literal (anonymous axis)
            bool is_numeric = true;
            for (char c : axis) {
                if (!std::isdigit(static_cast<unsigned char>(c))) {
                    is_numeric = false;
                    break;
                }
            }

            if (is_numeric) {
                // Anonymous axis with fixed size
                size_t size = std::stoull(axis);
                std::string anon_name =
                    "__anon_" + std::to_string(anon_counter++);
                axes.push_back(anon_name);
                anonymous_sizes[anon_name] = size;
            } else {
                // Named axis
                axes.push_back(axis);
            }
        }

        if (axes.empty()) {
            throw EinopsParseError("Empty group in pattern: " + token);
        }

        return GroupedAxes{axes, anonymous_sizes};
    } else {
        // Simple axis or numeric literal at top level
        if (token.empty()) {
            throw EinopsParseError("Empty axis name");
        }

        // Check if this is a numeric literal (represents unity-like axis)
        bool is_numeric = true;
        for (char c : token) {
            if (!std::isdigit(static_cast<unsigned char>(c))) {
                is_numeric = false;
                break;
            }
        }

        if (is_numeric) {
            // Numeric literal at top level - treat as unity if 1, error
            // otherwise
            size_t size = std::stoull(token);
            if (size == 1) {
                return UnityAxis{};
            } else {
                throw EinopsParseError(
                    "Numeric literal '" + token +
                    "' not allowed at top level (only '1' or inside groups)");
            }
        }

        // Validate axis name (letters/numbers/underscore)
        if (!std::regex_match(token, std::regex("^[a-zA-Z0-9_]+$"))) {
            throw EinopsParseError("Invalid axis name: " + token);
        }

        return SimpleAxis{token};
    }
}

std::vector<std::string>
EinopsExpression::get_pattern_axes(const ParsedPattern &pattern) const {
    std::vector<std::string> axes;

    for (const auto &element : pattern.elements) {
        if (std::holds_alternative<SimpleAxis>(element)) {
            axes.push_back(std::get<SimpleAxis>(element).name);
        } else if (std::holds_alternative<GroupedAxes>(element)) {
            const auto &group = std::get<GroupedAxes>(element);
            for (const auto &axis : group.axes) {
                // Include all axes including anonymous ones
                axes.push_back(axis);
            }
        } else if (std::holds_alternative<UnityAxis>(element)) {
            // Unity axes don't contribute named axes, but we track them for
            // dimension counting
            axes.push_back("__unity__"); // Special marker for unity axes
        } else if (std::holds_alternative<EllipsisAxis>(element)) {
            // Ellipsis will be handled specially during execution
            axes.push_back("__ellipsis__"); // Special marker for ellipsis
        }
    }

    return axes;
}

void EinopsExpression::validate_input(const Tensor &tensor) const {
    // Count actual dimensions needed (excluding ellipsis)
    size_t input_dims_needed = 0;
    bool has_ellipsis = false;

    for (const auto &element : parsed_input_.elements) {
        if (std::holds_alternative<EllipsisAxis>(element)) {
            if (has_ellipsis) {
                throw EinopsParseError(
                    "Only one ellipsis '...' is allowed per pattern");
            }
            has_ellipsis = true;
        } else {
            input_dims_needed++;
        }
    }

    if (has_ellipsis) {
        // With ellipsis, we need at least input_dims_needed dimensions
        if (tensor.ndim() < input_dims_needed) {
            throw EinopsShapeError("Input pattern needs at least " +
                                   std::to_string(input_dims_needed) +
                                   " dimensions but tensor has " +
                                   std::to_string(tensor.ndim()));
        }
    } else {
        // Without ellipsis, exact match required
        if (tensor.ndim() != input_dims_needed) {
            throw EinopsShapeError(
                "Input pattern has " + std::to_string(input_dims_needed) +
                " dimensions but tensor has " + std::to_string(tensor.ndim()));
        }
    }

    // Additional validation for grouped axes with known sizes
    auto inferred_sizes = infer_axis_sizes(tensor);
    for (const auto &[axis, size] : axis_sizes_) {
        if (inferred_sizes.find(axis) != inferred_sizes.end()) {
            if (inferred_sizes[axis] != size) {
                throw EinopsShapeError(
                    "Axis '" + axis + "' has inferred size " +
                    std::to_string(inferred_sizes[axis]) +
                    " but specified size " + std::to_string(size));
            }
        }
    }
}

std::map<std::string, size_t>
EinopsExpression::infer_axis_sizes(const Tensor &tensor) const {
    std::map<std::string, size_t> sizes =
        axis_sizes_; // Start with provided sizes

    // Add anonymous sizes from parsed patterns
    for (const auto &element : parsed_input_.elements) {
        if (std::holds_alternative<GroupedAxes>(element)) {
            const auto &group = std::get<GroupedAxes>(element);
            for (const auto &[name, size] : group.anonymous_sizes) {
                sizes[name] = size;
            }
        }
    }
    for (const auto &element : parsed_output_.elements) {
        if (std::holds_alternative<GroupedAxes>(element)) {
            const auto &group = std::get<GroupedAxes>(element);
            for (const auto &[name, size] : group.anonymous_sizes) {
                sizes[name] = size;
            }
        }
    }

    const auto &shape = tensor.shape();

    // First pass: find ellipsis and calculate how many dimensions it should
    // consume
    size_t ellipsis_length = 0;
    bool has_ellipsis = false;

    size_t explicit_dims = 0; // Count of non-ellipsis dimensions
    for (const auto &element : parsed_input_.elements) {
        if (std::holds_alternative<EllipsisAxis>(element)) {
            has_ellipsis = true;
        } else {
            explicit_dims++;
        }
    }

    if (has_ellipsis) {
        ellipsis_length = tensor.ndim() - explicit_dims;
    }

    // Second pass: assign sizes to axes
    size_t dim_idx = 0;
    for (const auto &element : parsed_input_.elements) {
        if (std::holds_alternative<SimpleAxis>(element)) {
            const auto &axis = std::get<SimpleAxis>(element);
            sizes[axis.name] = shape[dim_idx];
            dim_idx++;
        } else if (std::holds_alternative<UnityAxis>(element)) {
            // Unity axis always has size 1
            dim_idx++; // Consume one dimension which should be size 1
            if (dim_idx > shape.size() || shape[dim_idx - 1] != 1) {
                throw EinopsShapeError(
                    "Unity axis '()' or '1' requires dimension of size 1");
            }
        } else if (std::holds_alternative<EllipsisAxis>(element)) {
            // Skip ellipsis dimensions for now, they'll be handled in output
            dim_idx += ellipsis_length;
        } else if (std::holds_alternative<GroupedAxes>(element)) {
            // Handle grouped axes
            const auto &group = std::get<GroupedAxes>(element);

            // Calculate the total size for this group
            size_t total_size = shape[dim_idx];

            // Check if all axes in the group have known sizes
            bool all_known = true;
            size_t known_product = 1;
            std::string unknown_axis;

            for (const auto &axis : group.axes) {
                if (sizes.find(axis) != sizes.end()) {
                    known_product *= sizes[axis];
                } else {
                    if (!unknown_axis.empty()) {
                        throw EinopsShapeError(
                            "Cannot infer multiple unknown axes in group");
                    }
                    unknown_axis = axis;
                    all_known = false;
                }
            }

            if (!all_known) {
                if (total_size % known_product != 0) {
                    throw EinopsShapeError("Cannot infer axis size: " +
                                           std::to_string(total_size) +
                                           " is not divisible by " +
                                           std::to_string(known_product));
                }
                sizes[unknown_axis] = total_size / known_product;
            } else {
                // Verify that the product matches
                if (known_product != total_size) {
                    throw EinopsShapeError("Group size mismatch: expected " +
                                           std::to_string(known_product) +
                                           " but got " +
                                           std::to_string(total_size));
                }
            }

            dim_idx++;
        }
    }

    return sizes;
}

Shape EinopsExpression::get_output_shape(const Tensor &input) const {
    auto sizes = infer_axis_sizes(input);
    return calculate_reshape_dims(sizes, true);
}

Shape EinopsExpression::calculate_reshape_dims(
    const std::map<std::string, size_t> &sizes, bool is_output) const {
    const auto &pattern = is_output ? parsed_output_ : parsed_input_;
    Shape dims;

    for (const auto &element : pattern.elements) {
        if (std::holds_alternative<SimpleAxis>(element)) {
            const auto &axis = std::get<SimpleAxis>(element);
            auto it = sizes.find(axis.name);
            if (it == sizes.end()) {
                throw EinopsShapeError("Unknown axis size: " + axis.name);
            }
            dims.push_back(it->second);
        } else if (std::holds_alternative<UnityAxis>(element)) {
            // Unity axis always contributes size 1
            dims.push_back(1);
        } else if (std::holds_alternative<EllipsisAxis>(element)) {
            // For ellipsis, we need to copy the corresponding dimensions from
            // the input tensor This is tricky - for now, we'll handle it in the
            // apply method Here we just mark it as a special case
            dims.push_back(
                0); // Placeholder - will be resolved during execution
        } else if (std::holds_alternative<GroupedAxes>(element)) {
            const auto &group = std::get<GroupedAxes>(element);
            size_t total_size = calculate_grouped_size(group, sizes);
            dims.push_back(total_size);
        }
    }

    return dims;
}

size_t EinopsExpression::calculate_grouped_size(
    const GroupedAxes &group,
    const std::map<std::string, size_t> &sizes) const {
    size_t total = 1;
    for (const auto &axis : group.axes) {
        auto it = sizes.find(axis);
        if (it == sizes.end()) {
            throw EinopsShapeError("Unknown axis size: " + axis);
        }
        total *= it->second;
    }
    return total;
}

std::vector<int> EinopsExpression::calculate_transpose_axes(
    [[maybe_unused]] const std::map<std::string, size_t> &sizes) const {
    // Build mapping from axis name to position in input
    std::map<std::string, int> axis_to_input_pos;
    auto input_axes = get_pattern_axes(parsed_input_);
    for (size_t i = 0; i < input_axes.size(); ++i) {
        axis_to_input_pos[input_axes[i]] = static_cast<int>(i);
    }

    // Build transpose order based on output pattern
    std::vector<int> transpose_axes;
    auto output_axes = get_pattern_axes(parsed_output_);

    for (const auto &axis : output_axes) {
        auto it = axis_to_input_pos.find(axis);
        if (it != axis_to_input_pos.end()) {
            transpose_axes.push_back(it->second);
        }
    }

    return transpose_axes;
}

Tensor EinopsExpression::apply(const Tensor &tensor) const {
    validate_input(tensor);

    auto sizes = infer_axis_sizes(tensor);

    // Step 1: If we need to change the memory layout, do transpose first
    auto input_axes = get_pattern_axes(parsed_input_);
    auto output_axes = get_pattern_axes(parsed_output_);

    Tensor result = tensor;

    // Check if we need transpose (axis reordering)
    bool needs_transpose = false;
    if (input_axes.size() == output_axes.size()) {
        for (size_t i = 0; i < input_axes.size(); ++i) {
            if (input_axes[i] != output_axes[i]) {
                needs_transpose = true;
                break;
            }
        }
    }

    // For complex transformations, we might need intermediate steps
    if (parsed_input_.elements.size() != parsed_output_.elements.size() ||
        needs_transpose) {
        // Step 1: Flatten to 1D if we have groupings in input
        bool has_input_groups = false;
        for (const auto &element : parsed_input_.elements) {
            if (std::holds_alternative<GroupedAxes>(element)) {
                has_input_groups = true;
                break;
            }
        }

        if (has_input_groups) {
            // Reshape to separate all axes
            auto flat_shape = calculate_reshape_dims(sizes, false);
            result = result.reshape(flat_shape);
        }

        // Step 2: Transpose if needed
        if (needs_transpose && input_axes.size() == output_axes.size()) {
            auto transpose_order = calculate_transpose_axes(sizes);
            if (transpose_order.size() == result.ndim()) {
                result = result.transpose(transpose_order);
            }
        }

        // Step 3: Reshape to final output shape
        auto output_shape = get_output_shape(tensor);
        if (result.shape() != output_shape) {
            result = result.reshape(output_shape);
        }
    }

    return result;
}

// ============================================================================
// High-level API functions
// ============================================================================

Tensor rearrange(const Tensor &tensor, const std::string &pattern,
                 const std::map<std::string, size_t> &axis_sizes) {
    EinopsExpression expr(pattern, axis_sizes);
    return expr.apply(tensor);
}

Tensor reduce(const Tensor &tensor, const std::string &pattern,
              const std::string &reduction,
              const std::map<std::string, size_t> &axis_sizes) {
    // Parse the reduction type
    enum class ReductionOp { Sum, Mean, Max, Min, Prod, Any, All };
    ReductionOp red_op;

    if (reduction == "sum") {
        red_op = ReductionOp::Sum;
    } else if (reduction == "mean") {
        red_op = ReductionOp::Mean;
    } else if (reduction == "max") {
        red_op = ReductionOp::Max;
    } else if (reduction == "min") {
        red_op = ReductionOp::Min;
    } else if (reduction == "prod") {
        red_op = ReductionOp::Prod;
    } else if (reduction == "any") {
        red_op = ReductionOp::Any;
    } else if (reduction == "all") {
        red_op = ReductionOp::All;
    } else {
        throw EinopsError(
            "Unknown reduction: " + reduction +
            ". Use 'sum', 'mean', 'max', 'min', 'prod', 'any', or 'all'");
    }

    // Parse the einops expression
    EinopsExpression expr(pattern, axis_sizes);

    // Split pattern for parsing
    size_t arrow_pos = pattern.find("->");
    if (arrow_pos == std::string::npos) {
        throw EinopsParseError(
            "Pattern must contain '->' to separate input and output: " +
            pattern);
    }

    std::string input_pattern_str = pattern.substr(0, arrow_pos);
    std::string output_pattern_str = pattern.substr(arrow_pos + 2);

    // Trim whitespace
    input_pattern_str.erase(0, input_pattern_str.find_first_not_of(" \t"));
    input_pattern_str.erase(input_pattern_str.find_last_not_of(" \t") + 1);
    output_pattern_str.erase(0, output_pattern_str.find_first_not_of(" \t"));
    output_pattern_str.erase(output_pattern_str.find_last_not_of(" \t") + 1);

    // Parse patterns
    auto parsed_input = expr.parse_single_pattern(input_pattern_str);
    auto parsed_output = expr.parse_single_pattern(output_pattern_str);

    // Compute axis sizes from tensor
    auto sizes = expr.infer_axis_sizes(tensor);

    // Step 1: Expand grouped axes in input to individual dimensions
    // Build the expanded shape and track axis names
    Tensor working = tensor;
    std::vector<std::string> expanded_axes;
    Shape expanded_shape;

    size_t dim_idx = 0;
    for (const auto &element : parsed_input.elements) {
        if (std::holds_alternative<SimpleAxis>(element)) {
            const auto &axis = std::get<SimpleAxis>(element);
            expanded_axes.push_back(axis.name);
            expanded_shape.push_back(sizes.at(axis.name));
            dim_idx++;
        } else if (std::holds_alternative<UnityAxis>(element)) {
            expanded_axes.push_back("__unity__");
            expanded_shape.push_back(1);
            dim_idx++;
        } else if (std::holds_alternative<GroupedAxes>(element)) {
            const auto &group = std::get<GroupedAxes>(element);
            for (const auto &axis : group.axes) {
                expanded_axes.push_back(axis);
                expanded_shape.push_back(sizes.at(axis));
            }
            dim_idx++;
        } else if (std::holds_alternative<EllipsisAxis>(element)) {
            // Handle ellipsis - copy remaining dimensions
            size_t explicit_dims = 0;
            for (const auto &e : parsed_input.elements) {
                if (!std::holds_alternative<EllipsisAxis>(e)) {
                    explicit_dims++;
                }
            }
            size_t ellipsis_dims = tensor.ndim() - explicit_dims;
            for (size_t i = 0; i < ellipsis_dims; ++i) {
                std::string axis_name =
                    "__ellipsis_" + std::to_string(i) + "__";
                expanded_axes.push_back(axis_name);
                expanded_shape.push_back(tensor.shape()[dim_idx + i]);
            }
            dim_idx += ellipsis_dims;
        }
    }

    // Reshape to expanded form if needed
    if (expanded_shape != working.shape()) {
        working = working.reshape(expanded_shape);
    }

    // Step 2: Build output axis list and identify which axes to reduce
    std::vector<std::string> output_axes;
    std::vector<bool> output_is_unity; // Track which output dims are unity

    for (const auto &element : parsed_output.elements) {
        if (std::holds_alternative<SimpleAxis>(element)) {
            const auto &axis = std::get<SimpleAxis>(element);
            output_axes.push_back(axis.name);
            output_is_unity.push_back(false);
        } else if (std::holds_alternative<UnityAxis>(element)) {
            output_axes.push_back("__unity__");
            output_is_unity.push_back(true);
        } else if (std::holds_alternative<GroupedAxes>(element)) {
            const auto &group = std::get<GroupedAxes>(element);
            for (const auto &axis : group.axes) {
                output_axes.push_back(axis);
                output_is_unity.push_back(false);
            }
        } else if (std::holds_alternative<EllipsisAxis>(element)) {
            // Copy ellipsis axes from input
            for (const auto &axis : expanded_axes) {
                if (axis.find("__ellipsis_") == 0) {
                    output_axes.push_back(axis);
                    output_is_unity.push_back(false);
                }
            }
        }
    }

    // Step 3: Find axes to reduce (in input but not in output, excluding unity)
    std::set<std::string> output_set;
    for (const auto &axis : output_axes) {
        if (axis != "__unity__") {
            output_set.insert(axis);
        }
    }

    std::vector<int> reduce_dims;
    std::vector<std::string> axes_after_reduce;

    for (size_t i = 0; i < expanded_axes.size(); ++i) {
        const auto &axis = expanded_axes[i];
        bool is_anonymous = axis.find("__anon_") == 0;
        bool is_in_output = output_set.find(axis) != output_set.end();

        if (!is_in_output && axis != "__unity__") {
            // This axis should be reduced
            reduce_dims.push_back(static_cast<int>(i));
        } else if (!is_anonymous || is_in_output) {
            axes_after_reduce.push_back(axis);
        }
    }

    // Step 4: Apply reduction (reduce all axes at once for efficiency)
    if (!reduce_dims.empty()) {
        // Sort axes in ascending order for multi-axis reduction
        std::sort(reduce_dims.begin(), reduce_dims.end());

        switch (red_op) {
        case ReductionOp::Sum:
            working = ops::sum(working, reduce_dims, false);
            break;
        case ReductionOp::Mean:
            working = ops::mean(working, reduce_dims, false);
            break;
        case ReductionOp::Max:
            working = ops::max(working, reduce_dims, false);
            break;
        case ReductionOp::Min:
            working = ops::min(working, reduce_dims, false);
            break;
        case ReductionOp::Prod:
            working = ops::prod(working, reduce_dims, false);
            break;
        case ReductionOp::Any:
            working = ops::any(working, reduce_dims, false);
            break;
        case ReductionOp::All:
            working = ops::all(working, reduce_dims, false);
            break;
        }
    }

    // Step 5: Transpose if axes are reordered in output
    // Build mapping from axis name to current position
    std::map<std::string, int> axis_to_pos;
    for (size_t i = 0; i < axes_after_reduce.size(); ++i) {
        axis_to_pos[axes_after_reduce[i]] = static_cast<int>(i);
    }

    // Build transpose order based on output pattern (excluding unity axes)
    std::vector<int> transpose_order;
    std::vector<std::string> non_unity_output;
    for (size_t i = 0; i < output_axes.size(); ++i) {
        if (!output_is_unity[i]) {
            non_unity_output.push_back(output_axes[i]);
        }
    }

    bool needs_transpose = false;
    for (const auto &axis : non_unity_output) {
        auto it = axis_to_pos.find(axis);
        if (it != axis_to_pos.end()) {
            transpose_order.push_back(it->second);
        }
    }

    if (transpose_order.size() == working.ndim() && working.ndim() > 1) {
        // Check if transpose is actually needed
        for (size_t i = 0; i < transpose_order.size(); ++i) {
            if (transpose_order[i] != static_cast<int>(i)) {
                needs_transpose = true;
                break;
            }
        }
        if (needs_transpose) {
            working = working.transpose(transpose_order);
        }
    }

    // Step 6: Add unity dimensions where needed in output
    // Count unity axes and their positions
    std::vector<size_t> unity_positions;
    for (size_t i = 0; i < output_is_unity.size(); ++i) {
        if (output_is_unity[i]) {
            unity_positions.push_back(i);
        }
    }

    if (!unity_positions.empty()) {
        // Build final shape with unity dimensions inserted
        Shape final_shape;
        size_t working_dim = 0;
        for (size_t i = 0; i < output_axes.size(); ++i) {
            if (output_is_unity[i]) {
                final_shape.push_back(1);
            } else {
                if (working_dim < working.ndim()) {
                    final_shape.push_back(working.shape()[working_dim]);
                    working_dim++;
                }
            }
        }
        working = working.reshape(final_shape);
    }

    // Step 7: Handle grouped axes in output (merge dimensions)
    // Build a map of ellipsis axis names to their sizes from expanded_shape
    std::map<std::string, size_t> ellipsis_sizes;
    for (size_t i = 0; i < expanded_axes.size(); ++i) {
        if (expanded_axes[i].find("__ellipsis_") == 0) {
            ellipsis_sizes[expanded_axes[i]] = expanded_shape[i];
        }
    }

    Shape final_output_shape;
    for (const auto &element : parsed_output.elements) {
        if (std::holds_alternative<SimpleAxis>(element)) {
            const auto &axis = std::get<SimpleAxis>(element);
            final_output_shape.push_back(sizes.at(axis.name));
        } else if (std::holds_alternative<UnityAxis>(element)) {
            final_output_shape.push_back(1);
        } else if (std::holds_alternative<GroupedAxes>(element)) {
            const auto &group = std::get<GroupedAxes>(element);
            size_t group_size = 1;
            for (const auto &axis : group.axes) {
                group_size *= sizes.at(axis);
            }
            final_output_shape.push_back(group_size);
        } else if (std::holds_alternative<EllipsisAxis>(element)) {
            // Add ellipsis dimensions
            for (const auto &axis : expanded_axes) {
                if (axis.find("__ellipsis_") == 0) {
                    final_output_shape.push_back(ellipsis_sizes.at(axis));
                }
            }
        }
    }

    if (working.shape() != final_output_shape) {
        working = working.reshape(final_output_shape);
    }

    return working;
}

} // namespace einops
} // namespace axiom