#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "tensor.hpp"

namespace axiom {
namespace einops {

class EinopsPattern;
class EinopsExpression;

// ============================================================================
// Types for parsing and representing einops patterns
// ============================================================================

enum class AxisElementType {
    Simple,  // Single axis name like 'h', 'w', 'c'
    Grouped, // Grouped axes like '(h w)' or '(b h1 w1)'
    Unity,   // Unity axis '()' or '1' representing dimension of size 1
    Ellipsis // Ellipsis '...' representing remaining dimensions
};

struct SimpleAxis {
    std::string name;
};

struct GroupedAxes {
    std::vector<std::string> axes;
    // Map from axis name to anonymous size (for numeric literals in groups)
    // e.g., "(h1 2)" has anonymous_sizes["__anon_0"] = 2
    std::map<std::string, size_t> anonymous_sizes;
};

struct UnityAxis {
    // Represents a dimension of size 1, created by '()' or '1'
};

struct EllipsisAxis {
    // Represents remaining dimensions, created by '...'
};

using AxisElement =
    std::variant<SimpleAxis, GroupedAxes, UnityAxis, EllipsisAxis>;

struct ParsedPattern {
    std::vector<AxisElement> elements;
};

// ============================================================================
// Einops expression parser and executor
// ============================================================================

class EinopsExpression {
  private:
    std::string input_pattern_;
    std::string output_pattern_;
    std::map<std::string, size_t> axis_sizes_;
    mutable int anonymous_axis_counter_ = 0;

    ParsedPattern parsed_input_;
    ParsedPattern parsed_output_;

  public:
    EinopsExpression(const std::string &pattern,
                     const std::map<std::string, size_t> &axis_sizes = {});

    // Apply the transformation to a tensor
    Tensor apply(const Tensor &tensor) const;

    // Validate that the pattern is compatible with the tensor shape
    void validate_input(const Tensor &tensor) const;

    // Get the expected output shape
    Shape get_output_shape(const Tensor &input) const;

    // Pattern parsing (public for reduce() to use)
    ParsedPattern parse_single_pattern(const std::string &pattern) const;
    std::vector<std::string>
    get_pattern_axes(const ParsedPattern &pattern) const;
    std::map<std::string, size_t> infer_axis_sizes(const Tensor &tensor) const;

    // Access already-parsed patterns (avoids re-parsing with different
    // counters)
    const ParsedPattern &parsed_input() const { return parsed_input_; }
    const ParsedPattern &parsed_output() const { return parsed_output_; }

  private:
    void parse_patterns();

    std::vector<std::string> tokenize_pattern(const std::string &pattern) const;
    AxisElement parse_axis_element(const std::string &token) const;

    // Execution methods
    std::vector<int>
    calculate_transpose_axes(const std::map<std::string, size_t> &sizes) const;
    Shape calculate_reshape_dims(const std::map<std::string, size_t> &sizes,
                                 bool is_output) const;

    // Helper methods
    std::vector<std::string> get_all_axes() const;
    size_t
    calculate_grouped_size(const GroupedAxes &group,
                           const std::map<std::string, size_t> &sizes) const;
};

// ============================================================================
// High-level einops functions (NumPy/PyTorch style API)
// ============================================================================

/**
 * Rearrange tensor according to einops pattern
 * @param tensor Input tensor
 * @param pattern Einops pattern string like "b h w c -> b c h w"
 * @param axis_sizes Optional axis size specifications for splitting
 * @return Rearranged tensor
 */
Tensor rearrange(const Tensor &tensor, const std::string &pattern,
                 const std::map<std::string, size_t> &axis_sizes = {});

/**
 * Reduce tensor according to einops pattern
 * @param tensor Input tensor
 * @param pattern Einops pattern string like "b h w c -> b c" (axes not in
 * output are reduced)
 * @param reduction Reduction operation: "sum", "mean", "max", "min", "prod"
 * @param axis_sizes Optional axis size specifications
 * @return Reduced tensor
 *
 * Example:
 *   reduce(x, "b h w c -> b c", "mean")  // Pool over h and w
 *   reduce(x, "b (h h2) (w w2) c -> b h w c", "mean", {{"h2", 2}, {"w2", 2}})
 */
Tensor reduce(const Tensor &tensor, const std::string &pattern,
              const std::string &reduction,
              const std::map<std::string, size_t> &axis_sizes = {});

/**
 * Einstein summation convention
 * @param equation Einsum equation like "ij,jk->ik" for matrix multiply
 * @param operands Vector of input tensors
 * @return Result tensor
 *
 * Supported patterns:
 *   einsum("ij,jk->ik", {A, B})     // matrix multiply
 *   einsum("ii->", {A})             // trace
 *   einsum("ij->ji", {A})           // transpose
 *   einsum("ij,ij->ij", {A, B})     // element-wise multiply
 *   einsum("bij,bjk->bik", {A, B})  // batched matmul
 *   einsum("ijk->", {A})            // sum all elements
 *   einsum("ij->j", {A})            // sum over rows
 */
Tensor einsum(const std::string &equation, const std::vector<Tensor> &operands);

/**
 * Repeat tensor according to einops pattern (add/tile dimensions)
 * @param tensor Input tensor
 * @param pattern Einops pattern string like "h w -> h w c" (new axes in output)
 * @param axis_sizes Sizes for new axes
 * @return Repeated tensor
 *
 * Example:
 *   repeat(x, "h w -> h w c", {{"c", 3}})       // Add channel dim
 *   repeat(x, "h w -> h repeat w", {{"repeat", 3}})  // Tile rows
 */
Tensor repeat(const Tensor &tensor, const std::string &pattern,
              const std::map<std::string, size_t> &axis_sizes = {});

/**
 * Pack multiple tensors into a single tensor with a wildcard dimension
 * @param tensors Input tensors
 * @param pattern Pattern string with exactly one '*' wildcard
 * @return Pair of (packed tensor, packed_shapes for unpack)
 *
 * Example:
 *   auto [packed, ps] = pack({img1, img2, img3}, "* h w")
 */
std::pair<Tensor, std::vector<Shape>> pack(const std::vector<Tensor> &tensors,
                                           const std::string &pattern);

/**
 * Unpack a packed tensor back into individual tensors
 * @param tensor Packed tensor
 * @param packed_shapes Shapes from pack() for each tensor's wildcard dims
 * @param pattern Pattern string with exactly one '*' wildcard
 * @return Vector of unpacked tensors
 *
 * Example:
 *   auto tensors = unpack(packed, ps, "* h w")
 */
std::vector<Tensor> unpack(const Tensor &tensor,
                           const std::vector<Shape> &packed_shapes,
                           const std::string &pattern);

// ============================================================================
// Exception types for einops operations
// ============================================================================

class EinopsError : public std::runtime_error {
  public:
    explicit EinopsError(const std::string &message)
        : std::runtime_error("Einops error: " + message) {}
};

class EinopsParseError : public EinopsError {
  public:
    explicit EinopsParseError(const std::string &message)
        : EinopsError("Parse error: " + message) {}
};

class EinopsShapeError : public EinopsError {
  public:
    explicit EinopsShapeError(const std::string &message)
        : EinopsError("Shape error: " + message) {}
};

} // namespace einops
} // namespace axiom