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

// ============================================================================
// Einsum implementation
// ============================================================================

namespace {

struct EinsumParsed {
    std::vector<std::string> input_subscripts; // e.g., ["ij", "jk"]
    std::string output_subscript;              // e.g., "ik"
    bool has_explicit_output;
};

EinsumParsed parse_einsum_equation(const std::string &equation) {
    EinsumParsed result;

    // Find the arrow separator
    size_t arrow_pos = equation.find("->");
    std::string inputs_part;
    if (arrow_pos != std::string::npos) {
        inputs_part = equation.substr(0, arrow_pos);
        result.output_subscript = equation.substr(arrow_pos + 2);
        result.has_explicit_output = true;

        // Trim whitespace from output
        result.output_subscript.erase(
            0, result.output_subscript.find_first_not_of(" \t"));
        result.output_subscript.erase(
            result.output_subscript.find_last_not_of(" \t") + 1);
    } else {
        inputs_part = equation;
        result.has_explicit_output = false;
    }

    // Split inputs by comma
    size_t start = 0;
    size_t comma_pos;
    while ((comma_pos = inputs_part.find(',', start)) != std::string::npos) {
        std::string sub = inputs_part.substr(start, comma_pos - start);
        // Trim whitespace
        sub.erase(0, sub.find_first_not_of(" \t"));
        sub.erase(sub.find_last_not_of(" \t") + 1);
        result.input_subscripts.push_back(sub);
        start = comma_pos + 1;
    }
    // Last input
    std::string sub = inputs_part.substr(start);
    sub.erase(0, sub.find_first_not_of(" \t"));
    sub.erase(sub.find_last_not_of(" \t") + 1);
    result.input_subscripts.push_back(sub);

    return result;
}

// Build index-to-size map from inputs
std::map<char, size_t>
build_index_sizes(const std::vector<std::string> &subscripts,
                  const std::vector<Tensor> &operands) {
    std::map<char, size_t> sizes;

    for (size_t i = 0; i < subscripts.size(); ++i) {
        const auto &sub = subscripts[i];
        const auto &tensor = operands[i];

        if (sub.length() != tensor.ndim()) {
            throw EinopsError("einsum: subscript '" + sub + "' has " +
                              std::to_string(sub.length()) +
                              " indices but tensor has " +
                              std::to_string(tensor.ndim()) + " dimensions");
        }

        for (size_t j = 0; j < sub.length(); ++j) {
            char idx = sub[j];
            size_t dim_size = tensor.shape()[j];

            auto it = sizes.find(idx);
            if (it != sizes.end()) {
                if (it->second != dim_size) {
                    throw EinopsError("einsum: index '" + std::string(1, idx) +
                                      "' has inconsistent sizes: " +
                                      std::to_string(it->second) + " vs " +
                                      std::to_string(dim_size));
                }
            } else {
                sizes[idx] = dim_size;
            }
        }
    }

    return sizes;
}

// Determine output indices (implicit mode: sorted unique indices that appear
// exactly once)
std::string
compute_implicit_output(const std::vector<std::string> &subscripts) {
    std::map<char, int> counts;
    for (const auto &sub : subscripts) {
        for (char c : sub) {
            counts[c]++;
        }
    }

    std::string output;
    // Collect indices that appear exactly once, in sorted order
    for (const auto &[idx, count] : counts) {
        if (count == 1) {
            output += idx;
        }
    }
    std::sort(output.begin(), output.end());
    return output;
}

} // namespace

Tensor einsum(const std::string &equation,
              const std::vector<Tensor> &operands) {
    if (operands.empty()) {
        throw EinopsError("einsum: at least one operand required");
    }

    auto parsed = parse_einsum_equation(equation);

    if (parsed.input_subscripts.size() != operands.size()) {
        throw EinopsError("einsum: number of subscripts (" +
                          std::to_string(parsed.input_subscripts.size()) +
                          ") doesn't match number of operands (" +
                          std::to_string(operands.size()) + ")");
    }

    auto sizes = build_index_sizes(parsed.input_subscripts, operands);

    std::string output_sub =
        parsed.has_explicit_output
            ? parsed.output_subscript
            : compute_implicit_output(parsed.input_subscripts);

    // Collect all unique indices
    std::set<char> all_indices;
    for (const auto &sub : parsed.input_subscripts) {
        for (char c : sub) {
            all_indices.insert(c);
        }
    }

    // Identify contracted indices (in inputs but not in output)
    std::set<char> output_indices(output_sub.begin(), output_sub.end());
    std::vector<char> contracted_indices;
    for (char idx : all_indices) {
        if (output_indices.find(idx) == output_indices.end()) {
            contracted_indices.push_back(idx);
        }
    }

    // Special case: single operand
    if (operands.size() == 1) {
        const Tensor &input = operands[0];
        const std::string &in_sub = parsed.input_subscripts[0];

        // Handle trace: "ii->" or "ii->i"
        bool is_trace = in_sub.length() >= 2 && contracted_indices.size() > 0;
        for (char c : contracted_indices) {
            if (std::count(in_sub.begin(), in_sub.end(), c) >= 2) {
                is_trace = true;
                break;
            }
        }

        if (is_trace && contracted_indices.size() > 0) {
            // Check for diagonal trace pattern
            char trace_idx = 0;
            for (char c : contracted_indices) {
                if (std::count(in_sub.begin(), in_sub.end(), c) >= 2) {
                    trace_idx = c;
                    break;
                }
            }

            if (trace_idx != 0) {
                // Simple 2D trace: "ii->"
                if (in_sub == "ii" && output_sub.empty()) {
                    return input.trace();
                }
            }
        }

        // Handle transpose: "ij->ji"
        if (contracted_indices.empty() &&
            in_sub.length() == output_sub.length()) {
            std::vector<int> perm;
            for (char c : output_sub) {
                size_t pos = in_sub.find(c);
                if (pos == std::string::npos) {
                    throw EinopsError("einsum: output index '" +
                                      std::string(1, c) +
                                      "' not found in input");
                }
                perm.push_back(static_cast<int>(pos));
            }
            return input.transpose(perm);
        }

        // Handle sum reduction: "ijk->j"
        if (!contracted_indices.empty()) {
            std::vector<int> reduce_axes;
            for (size_t i = 0; i < in_sub.length(); ++i) {
                if (output_indices.find(in_sub[i]) == output_indices.end()) {
                    reduce_axes.push_back(static_cast<int>(i));
                }
            }

            Tensor result = ops::sum(input, reduce_axes, false);

            // Transpose if output order differs
            if (output_sub.length() > 1) {
                std::string remaining;
                for (char c : in_sub) {
                    if (output_indices.find(c) != output_indices.end()) {
                        remaining += c;
                    }
                }
                if (remaining != output_sub) {
                    std::vector<int> perm;
                    for (char c : output_sub) {
                        size_t pos = remaining.find(c);
                        perm.push_back(static_cast<int>(pos));
                    }
                    result = result.transpose(perm);
                }
            }

            return result;
        }
    }

    // Two operands case
    if (operands.size() == 2) {
        const Tensor &A = operands[0];
        const Tensor &B = operands[1];
        const std::string &sub_a = parsed.input_subscripts[0];
        const std::string &sub_b = parsed.input_subscripts[1];

        // Element-wise multiply: "ij,ij->ij"
        if (sub_a == sub_b && sub_a == output_sub) {
            return ops::multiply(A, B);
        }

        // Matrix multiply: "ij,jk->ik"
        // Batched matmul: "bij,bjk->bik"
        // General contraction with one or more contracted indices

        // Find contracted (common) indices
        std::set<char> a_indices(sub_a.begin(), sub_a.end());
        std::set<char> b_indices(sub_b.begin(), sub_b.end());

        std::vector<char> contract_idx;
        for (char c : contracted_indices) {
            if (a_indices.count(c) && b_indices.count(c)) {
                contract_idx.push_back(c);
            }
        }

        // Standard matmul pattern detection
        // "ij,jk->ik" or "...ij,...jk->...ik"
        if (contract_idx.size() == 1) {
            char k = contract_idx[0];

            // Find positions of contracted index
            size_t a_k_pos = sub_a.find(k);
            size_t b_k_pos = sub_b.find(k);

            // Check if this is a standard matmul pattern
            bool a_contract_last = (a_k_pos == sub_a.length() - 1);
            bool b_contract_first_nonbatch = true;

            // Identify batch dimensions
            std::vector<char> batch_dims;
            for (char c : output_sub) {
                if (a_indices.count(c) && b_indices.count(c)) {
                    batch_dims.push_back(c);
                }
            }

            // Simple 2D matmul: "ij,jk->ik"
            if (sub_a.length() == 2 && sub_b.length() == 2 &&
                output_sub.length() == 2 && batch_dims.empty()) {
                bool trans_a = (a_k_pos == 0);
                bool trans_b = (b_k_pos == 1);
                return ops::matmul(A, B, trans_a, trans_b);
            }

            // Batched matmul: "bij,bjk->bik"
            if (batch_dims.size() > 0 && sub_a.length() >= 3 &&
                sub_b.length() >= 3) {
                // Reshape for batch matmul and use matmul
                // This is a simplified implementation
                bool trans_a = (sub_a[sub_a.length() - 1] != k);
                bool trans_b = (sub_b[sub_b.length() - 1] == k);
                return ops::matmul(A, B, trans_a, trans_b);
            }
        }

        // General case: expand, multiply, and reduce
        // This handles arbitrary contractions but is less efficient

        // Build unified shape for broadcasting
        std::vector<char> all_idx_ordered;
        for (char c : sub_a) {
            if (std::find(all_idx_ordered.begin(), all_idx_ordered.end(), c) ==
                all_idx_ordered.end()) {
                all_idx_ordered.push_back(c);
            }
        }
        for (char c : sub_b) {
            if (std::find(all_idx_ordered.begin(), all_idx_ordered.end(), c) ==
                all_idx_ordered.end()) {
                all_idx_ordered.push_back(c);
            }
        }

        // Build shapes for expansion
        Shape expand_shape_a, expand_shape_b;
        std::vector<int> perm_a, perm_b;

        for (char c : all_idx_ordered) {
            size_t pos_a = sub_a.find(c);
            size_t pos_b = sub_b.find(c);

            if (pos_a != std::string::npos) {
                perm_a.push_back(static_cast<int>(pos_a));
                expand_shape_a.push_back(sizes[c]);
            } else {
                expand_shape_a.push_back(1);
            }

            if (pos_b != std::string::npos) {
                perm_b.push_back(static_cast<int>(pos_b));
                expand_shape_b.push_back(sizes[c]);
            } else {
                expand_shape_b.push_back(1);
            }
        }

        // Transpose and expand
        Tensor a_work = (perm_a.size() == A.ndim()) ? A.transpose(perm_a) : A;
        Tensor b_work = (perm_b.size() == B.ndim()) ? B.transpose(perm_b) : B;

        // Reshape with 1s for broadcasting
        Shape full_shape;
        for (char c : all_idx_ordered) {
            full_shape.push_back(sizes[c]);
        }

        Shape a_broadcast_shape = a_work.shape();
        Shape b_broadcast_shape = b_work.shape();

        // Pad shapes to full_shape.size()
        while (a_broadcast_shape.size() < full_shape.size()) {
            a_broadcast_shape.insert(a_broadcast_shape.begin(), 1);
        }
        while (b_broadcast_shape.size() < full_shape.size()) {
            b_broadcast_shape.insert(b_broadcast_shape.begin(), 1);
        }

        a_work = a_work.reshape(a_broadcast_shape).expand(full_shape);
        b_work = b_work.reshape(b_broadcast_shape).expand(full_shape);

        // Multiply
        Tensor product = ops::multiply(a_work, b_work);

        // Sum over contracted indices
        std::vector<int> reduce_axes;
        for (size_t i = 0; i < all_idx_ordered.size(); ++i) {
            if (output_indices.find(all_idx_ordered[i]) ==
                output_indices.end()) {
                reduce_axes.push_back(static_cast<int>(i));
            }
        }

        Tensor result = product;
        if (!reduce_axes.empty()) {
            result = ops::sum(result, reduce_axes, false);
        }

        // Transpose to output order if needed
        std::string remaining;
        for (char c : all_idx_ordered) {
            if (output_indices.find(c) != output_indices.end()) {
                remaining += c;
            }
        }

        if (remaining != output_sub && output_sub.length() > 1) {
            std::vector<int> perm;
            for (char c : output_sub) {
                size_t pos = remaining.find(c);
                if (pos != std::string::npos) {
                    perm.push_back(static_cast<int>(pos));
                }
            }
            if (perm.size() == result.ndim()) {
                result = result.transpose(perm);
            }
        }

        return result;
    }

    // More than 2 operands: reduce pairwise
    if (operands.size() > 2) {
        // Greedy pairwise contraction
        // This is a simplified approach - a full implementation would optimize
        // contraction order

        // For now, just contract left to right
        std::vector<Tensor> remaining_ops = operands;
        std::vector<std::string> remaining_subs = parsed.input_subscripts;

        while (remaining_ops.size() > 1) {
            // Contract first two operands
            Tensor a = remaining_ops[0];
            Tensor b = remaining_ops[1];
            std::string sub_a = remaining_subs[0];
            std::string sub_b = remaining_subs[1];

            // Determine intermediate output (all unique indices)
            std::set<char> intermediate_set;
            for (char c : sub_a)
                intermediate_set.insert(c);
            for (char c : sub_b)
                intermediate_set.insert(c);

            // Remove indices that appear in both and will be contracted
            // (unless they appear in final output)
            std::string intermediate;
            for (char c : intermediate_set) {
                bool in_a = sub_a.find(c) != std::string::npos;
                bool in_b = sub_b.find(c) != std::string::npos;
                bool in_output = output_sub.find(c) != std::string::npos;
                bool in_later = false;

                for (size_t i = 2; i < remaining_subs.size(); ++i) {
                    if (remaining_subs[i].find(c) != std::string::npos) {
                        in_later = true;
                        break;
                    }
                }

                if (!(in_a && in_b && !in_output && !in_later)) {
                    intermediate += c;
                }
            }

            std::sort(intermediate.begin(), intermediate.end());

            std::string pair_eq = sub_a + "," + sub_b + "->" + intermediate;
            Tensor contracted = einsum(pair_eq, {a, b});

            remaining_ops.erase(remaining_ops.begin(),
                                remaining_ops.begin() + 2);
            remaining_ops.insert(remaining_ops.begin(), contracted);

            remaining_subs.erase(remaining_subs.begin(),
                                 remaining_subs.begin() + 2);
            remaining_subs.insert(remaining_subs.begin(), intermediate);
        }

        // Final adjustment to output
        if (remaining_subs[0] != output_sub) {
            std::string final_eq = remaining_subs[0] + "->" + output_sub;
            return einsum(final_eq, {remaining_ops[0]});
        }

        return remaining_ops[0];
    }

    throw EinopsError("einsum: unsupported equation pattern: " + equation);
}

} // namespace einops
} // namespace axiom