#include "axiom/einops.hpp"

#include <regex>
#include <set>
#include <sstream>

namespace axiom {
namespace einops {

// ============================================================================
// EinopsExpression Implementation
// ============================================================================

EinopsExpression::EinopsExpression(
    const std::string& pattern, const std::map<std::string, size_t>& axis_sizes)
    : axis_sizes_(axis_sizes) {
  // Split pattern on "->"
  size_t arrow_pos = pattern.find("->");
  if (arrow_pos == std::string::npos) {
    throw EinopsParseError(
        "Pattern must contain '->' to separate input and output: " + pattern);
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

  // Validation: check that all axes in output exist in input (except for repeat
  // operations)
  auto input_axes = get_pattern_axes(parsed_input_);
  auto output_axes = get_pattern_axes(parsed_output_);

  std::set<std::string> input_set(input_axes.begin(), input_axes.end());
  std::set<std::string> output_set(output_axes.begin(), output_axes.end());

  for (const auto& axis : output_set) {
    if (input_set.find(axis) == input_set.end() &&
        axis_sizes_.find(axis) == axis_sizes_.end()) {
      throw EinopsParseError(
          "Axis '" + axis +
          "' appears in output but not in input and no size specified");
    }
  }
}

ParsedPattern EinopsExpression::parse_single_pattern(
    const std::string& pattern) const {
  ParsedPattern result;
  auto tokens = tokenize_pattern(pattern);

  for (const auto& token : tokens) {
    result.elements.push_back(parse_axis_element(token));
  }

  return result;
}

std::vector<std::string> EinopsExpression::tokenize_pattern(
    const std::string& pattern) const {
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

AxisElement EinopsExpression::parse_axis_element(
    const std::string& token) const {
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

    // Grouped axes
    std::vector<std::string> axes;
    std::istringstream iss(inner);
    std::string axis;
    while (iss >> axis) {
      if (!axis.empty()) {
        axes.push_back(axis);
      }
    }

    if (axes.empty()) {
      throw EinopsParseError("Empty group in pattern: " + token);
    }

    return GroupedAxes{axes};
  } else {
    // Simple axis
    if (token.empty()) {
      throw EinopsParseError("Empty axis name");
    }

    // Validate axis name (letters/numbers/underscore)
    if (!std::regex_match(token, std::regex("^[a-zA-Z0-9_]+$"))) {
      throw EinopsParseError("Invalid axis name: " + token);
    }

    return SimpleAxis{token};
  }
}

std::vector<std::string> EinopsExpression::get_pattern_axes(
    const ParsedPattern& pattern) const {
  std::vector<std::string> axes;

  for (const auto& element : pattern.elements) {
    if (std::holds_alternative<SimpleAxis>(element)) {
      axes.push_back(std::get<SimpleAxis>(element).name);
    } else if (std::holds_alternative<GroupedAxes>(element)) {
      const auto& group = std::get<GroupedAxes>(element);
      for (const auto& axis : group.axes) {
        axes.push_back(axis);
      }
    } else if (std::holds_alternative<UnityAxis>(element)) {
      // Unity axes don't contribute named axes, but we track them for dimension
      // counting
      axes.push_back("__unity__");  // Special marker for unity axes
    } else if (std::holds_alternative<EllipsisAxis>(element)) {
      // Ellipsis will be handled specially during execution
      axes.push_back("__ellipsis__");  // Special marker for ellipsis
    }
  }

  return axes;
}

void EinopsExpression::validate_input(const Tensor& tensor) const {
  // Count actual dimensions needed (excluding ellipsis)
  size_t input_dims_needed = 0;
  bool has_ellipsis = false;

  for (const auto& element : parsed_input_.elements) {
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
      throw EinopsShapeError(
          "Input pattern needs at least " + std::to_string(input_dims_needed) +
          " dimensions but tensor has " + std::to_string(tensor.ndim()));
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
  for (const auto& [axis, size] : axis_sizes_) {
    if (inferred_sizes.find(axis) != inferred_sizes.end()) {
      if (inferred_sizes[axis] != size) {
        throw EinopsShapeError("Axis '" + axis + "' has inferred size " +
                               std::to_string(inferred_sizes[axis]) +
                               " but specified size " + std::to_string(size));
      }
    }
  }
}

std::map<std::string, size_t> EinopsExpression::infer_axis_sizes(
    const Tensor& tensor) const {
  std::map<std::string, size_t> sizes =
      axis_sizes_;  // Start with provided sizes
  const auto& shape = tensor.shape();

  // First pass: find ellipsis and calculate how many dimensions it should
  // consume
  size_t ellipsis_start = 0;
  size_t ellipsis_length = 0;
  bool has_ellipsis = false;

  size_t explicit_dims = 0;  // Count of non-ellipsis dimensions
  for (const auto& element : parsed_input_.elements) {
    if (std::holds_alternative<EllipsisAxis>(element)) {
      has_ellipsis = true;
      ellipsis_start = explicit_dims;
    } else {
      explicit_dims++;
    }
  }

  if (has_ellipsis) {
    ellipsis_length = tensor.ndim() - explicit_dims;
  }

  // Second pass: assign sizes to axes
  size_t dim_idx = 0;
  for (const auto& element : parsed_input_.elements) {
    if (std::holds_alternative<SimpleAxis>(element)) {
      const auto& axis = std::get<SimpleAxis>(element);
      sizes[axis.name] = shape[dim_idx];
      dim_idx++;

    } else if (std::holds_alternative<UnityAxis>(element)) {
      // Unity axis always has size 1
      dim_idx++;  // Consume one dimension which should be size 1
      if (dim_idx > shape.size() || shape[dim_idx - 1] != 1) {
        throw EinopsShapeError(
            "Unity axis '()' or '1' requires dimension of size 1");
      }

    } else if (std::holds_alternative<EllipsisAxis>(element)) {
      // Skip ellipsis dimensions for now, they'll be handled in output
      dim_idx += ellipsis_length;

    } else if (std::holds_alternative<GroupedAxes>(element)) {
      // Handle grouped axes
      const auto& group = std::get<GroupedAxes>(element);

      // Calculate the total size for this group
      size_t total_size = shape[dim_idx];

      // Check if all axes in the group have known sizes
      bool all_known = true;
      size_t known_product = 1;
      std::string unknown_axis;

      for (const auto& axis : group.axes) {
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
          throw EinopsShapeError(
              "Cannot infer axis size: " + std::to_string(total_size) +
              " is not divisible by " + std::to_string(known_product));
        }
        sizes[unknown_axis] = total_size / known_product;
      } else {
        // Verify that the product matches
        if (known_product != total_size) {
          throw EinopsShapeError("Group size mismatch: expected " +
                                 std::to_string(known_product) + " but got " +
                                 std::to_string(total_size));
        }
      }

      dim_idx++;
    }
  }

  return sizes;
}

Shape EinopsExpression::get_output_shape(const Tensor& input) const {
  auto sizes = infer_axis_sizes(input);
  return calculate_reshape_dims(sizes, true);
}

Shape EinopsExpression::calculate_reshape_dims(
    const std::map<std::string, size_t>& sizes, bool is_output) const {
  const auto& pattern = is_output ? parsed_output_ : parsed_input_;
  Shape dims;

  for (const auto& element : pattern.elements) {
    if (std::holds_alternative<SimpleAxis>(element)) {
      const auto& axis = std::get<SimpleAxis>(element);
      auto it = sizes.find(axis.name);
      if (it == sizes.end()) {
        throw EinopsShapeError("Unknown axis size: " + axis.name);
      }
      dims.push_back(it->second);

    } else if (std::holds_alternative<UnityAxis>(element)) {
      // Unity axis always contributes size 1
      dims.push_back(1);

    } else if (std::holds_alternative<EllipsisAxis>(element)) {
      // For ellipsis, we need to copy the corresponding dimensions from the
      // input tensor This is tricky - for now, we'll handle it in the apply
      // method Here we just mark it as a special case
      dims.push_back(0);  // Placeholder - will be resolved during execution

    } else if (std::holds_alternative<GroupedAxes>(element)) {
      const auto& group = std::get<GroupedAxes>(element);
      size_t total_size = calculate_grouped_size(group, sizes);
      dims.push_back(total_size);
    }
  }

  return dims;
}

size_t EinopsExpression::calculate_grouped_size(
    const GroupedAxes& group,
    const std::map<std::string, size_t>& sizes) const {
  size_t total = 1;
  for (const auto& axis : group.axes) {
    auto it = sizes.find(axis);
    if (it == sizes.end()) {
      throw EinopsShapeError("Unknown axis size: " + axis);
    }
    total *= it->second;
  }
  return total;
}

std::vector<int> EinopsExpression::calculate_transpose_axes(
    const std::map<std::string, size_t>& sizes) const {
  // Build mapping from axis name to position in input
  std::map<std::string, int> axis_to_input_pos;
  auto input_axes = get_pattern_axes(parsed_input_);
  for (size_t i = 0; i < input_axes.size(); ++i) {
    axis_to_input_pos[input_axes[i]] = static_cast<int>(i);
  }

  // Build transpose order based on output pattern
  std::vector<int> transpose_axes;
  auto output_axes = get_pattern_axes(parsed_output_);

  for (const auto& axis : output_axes) {
    auto it = axis_to_input_pos.find(axis);
    if (it != axis_to_input_pos.end()) {
      transpose_axes.push_back(it->second);
    }
  }

  return transpose_axes;
}

Tensor EinopsExpression::apply(const Tensor& tensor) const {
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
    for (const auto& element : parsed_input_.elements) {
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

Tensor rearrange(const Tensor& tensor, const std::string& pattern,
                 const std::map<std::string, size_t>& axis_sizes) {
  EinopsExpression expr(pattern, axis_sizes);
  return expr.apply(tensor);
}

}  // namespace einops
}  // namespace axiom