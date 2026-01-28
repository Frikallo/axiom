#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "dtype.hpp"
#include "shape.hpp"
#include "storage.hpp"
#include "tensor.hpp"

namespace axiom {

namespace ops {

// ============================================================================
// Broadcasting utilities
// ============================================================================

struct BroadcastInfo {
    Shape result_shape;
    std::vector<int> lhs_strides_adjustment;
    std::vector<int> rhs_strides_adjustment;
    bool needs_broadcast;
};

BroadcastInfo compute_broadcast_info(const Shape &lhs_shape,
                                     const Shape &rhs_shape);
bool are_broadcastable(const Shape &lhs_shape, const Shape &rhs_shape);

// ============================================================================
// Type promotion utilities
// ============================================================================

DType promote_types(DType lhs_dtype, DType rhs_dtype);
DType result_type(const Tensor &lhs, const Tensor &rhs);

// ============================================================================
// Operation Interface
// ============================================================================

enum class OpType {
    // Binary operations
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Modulo,

    // Comparison operations
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,

    // Logical operations
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    LogicalNot,

    // Bitwise operations
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,

    // Math operations
    Maximum,
    Minimum,
    Atan2,
    Hypot,

    // Unary operations
    Negate,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Erf,

    // Complex operations
    Conj,
    Real,
    Imag,

    // Activation operations
    ReLU,
    LeakyReLU,
    SiLU,
    Sigmoid,
    Tanh,
    GELU,
    Softmax,
    LogSoftmax,

    // Reduction operations
    Sum,
    Mean,
    Max,
    Min,
    ArgMax,
    ArgMin,
    Any,
    All,

    // Matrix operations
    MatMul,
    BatchMatMul,

    // Conditional operations
    Where,

    // Masking operations
    MaskedFill,
    MaskedSelect,

    // Indexing operations
    Gather,
    Scatter,
    IndexSelect,

    // Normalization operations
    LayerNorm,
    RMSNorm,

    // Dropout
    Dropout
};

class Operation {
  public:
    virtual ~Operation() = default;

    virtual OpType type() const = 0;
    virtual std::string name() const = 0;
    virtual Device device() const = 0;

    // A way to check for feature support like broadcasting
    virtual bool supports_binary(const Tensor &lhs, const Tensor &rhs) const {
        // By default, assume basic support (same shapes, no broadcasting)
        return lhs.shape() == rhs.shape();
    }

    // For binary operations
    virtual Tensor execute_binary(const Tensor &lhs,
                                  const Tensor &rhs) const = 0;

    // For unary operations
    virtual Tensor execute_unary(const Tensor &input) const;

    // For reduction operations
    virtual Tensor execute_reduction(const Tensor &input,
                                     const std::vector<int> &axis,
                                     bool keep_dims) const;

    // For matrix multiplication operations
    // transpose_a/b: if true, the last two dimensions are treated as transposed
    // This allows zero-copy transposed views without materializing
    virtual Tensor execute_matmul(const Tensor &a, const Tensor &b,
                                  bool transpose_a, bool transpose_b) const;

    // For conditional selection (where)
    // Returns elements from 'a' where condition is true, 'b' otherwise
    virtual Tensor execute_where(const Tensor &condition, const Tensor &a,
                                 const Tensor &b) const;

    // For masked fill operation
    // Returns tensor with masked positions filled with value
    virtual Tensor execute_masked_fill(const Tensor &input, const Tensor &mask,
                                       const Tensor &value) const;

    // For masked select operation
    // Returns 1D tensor of elements where mask is true
    virtual Tensor execute_masked_select(const Tensor &input,
                                         const Tensor &mask) const;

    // For gather operation
    // Gathers values along an axis according to indices
    virtual Tensor execute_gather(const Tensor &input, int dim,
                                  const Tensor &indices) const;

    // For scatter operation
    // Scatters values along an axis according to indices
    virtual Tensor execute_scatter(const Tensor &input, int dim,
                                   const Tensor &indices,
                                   const Tensor &src) const;

    // For index_select operation
    // Selects elements along a dimension using indices
    virtual Tensor execute_index_select(const Tensor &input, int dim,
                                        const Tensor &indices) const;

    // For in-place operations (future extension)
    virtual void execute_binary_inplace(Tensor &lhs, const Tensor &rhs) const;
};

// ============================================================================
// Operation Registry
// ============================================================================

class OperationRegistry {
  public:
    static void register_operation(OpType op_type, Device device,
                                   std::unique_ptr<Operation> operation);

    static const Operation *get_operation(OpType op_type, Device device);

    static std::vector<Device> available_devices_for_operation(OpType op_type);

    static bool is_operation_available(OpType op_type, Device device);

    // Initialize built-in operations
    static void initialize_builtin_operations();

  private:
    static std::map<std::pair<OpType, Device>, std::unique_ptr<Operation>> &
    get_registry();
};

// ============================================================================
// High-level operation functions
// ============================================================================

// Binary operations
Tensor add(const Tensor &lhs, const Tensor &rhs);
Tensor subtract(const Tensor &lhs, const Tensor &rhs);
Tensor multiply(const Tensor &lhs, const Tensor &rhs);
Tensor divide(const Tensor &lhs, const Tensor &rhs);
Tensor power(const Tensor &lhs, const Tensor &rhs);
Tensor modulo(const Tensor &lhs, const Tensor &rhs);

// Comparison operations
Tensor equal(const Tensor &lhs, const Tensor &rhs);
Tensor not_equal(const Tensor &lhs, const Tensor &rhs);
Tensor less(const Tensor &lhs, const Tensor &rhs);
Tensor less_equal(const Tensor &lhs, const Tensor &rhs);
Tensor greater(const Tensor &lhs, const Tensor &rhs);
Tensor greater_equal(const Tensor &lhs, const Tensor &rhs);

// Logical operations
Tensor logical_and(const Tensor &lhs, const Tensor &rhs);
Tensor logical_or(const Tensor &lhs, const Tensor &rhs);
Tensor logical_xor(const Tensor &lhs, const Tensor &rhs);
Tensor logical_not(const Tensor &input);

// Bitwise operations
Tensor bitwise_and(const Tensor &lhs, const Tensor &rhs);
Tensor bitwise_or(const Tensor &lhs, const Tensor &rhs);
Tensor bitwise_xor(const Tensor &lhs, const Tensor &rhs);
Tensor left_shift(const Tensor &lhs, const Tensor &rhs);
Tensor right_shift(const Tensor &lhs, const Tensor &rhs);

// Math operations
Tensor maximum(const Tensor &lhs, const Tensor &rhs);
Tensor minimum(const Tensor &lhs, const Tensor &rhs);
Tensor atan2(const Tensor &lhs, const Tensor &rhs);
Tensor hypot(const Tensor &lhs, const Tensor &rhs);

// Unary operations
Tensor negate(const Tensor &input);
Tensor abs(const Tensor &input);
Tensor sqrt(const Tensor &input);
Tensor exp(const Tensor &input);
Tensor log(const Tensor &input);
Tensor sin(const Tensor &input);
Tensor cos(const Tensor &input);
Tensor tan(const Tensor &input);
Tensor erf(const Tensor &input);

// Complex operations
Tensor conj(const Tensor &input);
Tensor real(const Tensor &input);
Tensor imag(const Tensor &input);

// Activation operations
Tensor relu(const Tensor &input);
Tensor leaky_relu(const Tensor &input, float negative_slope = 0.01f);
Tensor silu(const Tensor &input); // SiLU/Swish: x * sigmoid(x)
Tensor sigmoid(const Tensor &input);
Tensor tanh(const Tensor &input);
Tensor gelu(const Tensor &input);
Tensor softmax(const Tensor &input, int axis = -1);
Tensor log_softmax(const Tensor &input, int axis = -1);

// Reduction operations
Tensor sum(const Tensor &input, const std::vector<int> &axis = {},
           bool keep_dims = false);
Tensor mean(const Tensor &input, const std::vector<int> &axis = {},
            bool keep_dims = false);
Tensor max(const Tensor &input, const std::vector<int> &axis = {},
           bool keep_dims = false);
Tensor min(const Tensor &input, const std::vector<int> &axis = {},
           bool keep_dims = false);

// Argmax/Argmin - returns indices of max/min values along an axis
// Returns Int64 tensor with indices
Tensor argmax(const Tensor &input, int axis = -1, bool keep_dims = false);
Tensor argmin(const Tensor &input, int axis = -1, bool keep_dims = false);

// Boolean reductions - returns Bool tensor
// any: True if any element is non-zero along axis
// all: True if all elements are non-zero along axis
Tensor any(const Tensor &input, const std::vector<int> &axis = {},
           bool keep_dims = false);
Tensor all(const Tensor &input, const std::vector<int> &axis = {},
           bool keep_dims = false);

// Matrix multiplication operations
// matmul: General matrix multiplication with broadcasting of batch dimensions
// Supports:
//   - 2D x 2D: standard matrix multiply (M,K) @ (K,N) -> (M,N)
//   - ND x 2D: batch matmul with broadcasting
//   - 2D x ND: batch matmul with broadcasting
//   - ND x ND: batch matmul with auto-broadcasted batch dims
// transpose_a/transpose_b: treat input matrices as transposed without
// materializing This enables zero-copy transposed matrix multiplication
Tensor matmul(const Tensor &a, const Tensor &b, bool transpose_a = false,
              bool transpose_b = false);

// Conditional selection
// where: Returns elements from 'a' where condition is true, 'b' otherwise
// Equivalent to numpy.where(condition, a, b)
// All inputs are broadcast together
Tensor where(const Tensor &condition, const Tensor &a, const Tensor &b);

// ============================================================================
// Masking operations
// ============================================================================

// masked_fill: Fill elements where mask is true with the given value
// Returns a new tensor with masked positions filled
Tensor masked_fill(const Tensor &input, const Tensor &mask, float value);
Tensor masked_fill(const Tensor &input, const Tensor &mask, double value);
Tensor masked_fill(const Tensor &input, const Tensor &mask,
                   const Tensor &value);

// masked_select: Select elements where mask is true
// Returns a 1D tensor containing the selected elements
Tensor masked_select(const Tensor &input, const Tensor &mask);

// ============================================================================
// Indexing operations
// ============================================================================

// gather: Gather values along an axis according to indices
// Like PyTorch's torch.gather(input, dim, index)
// output[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
// output[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
Tensor gather(const Tensor &input, int dim, const Tensor &indices);

// scatter: Scatter values into tensor at indices
// Like PyTorch's tensor.scatter_(dim, index, src)
// self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
Tensor scatter(const Tensor &input, int dim, const Tensor &indices,
               const Tensor &src);

// index_select: Select elements along a dimension using 1D indices
// Like PyTorch's torch.index_select(input, dim, index)
// More efficient than gather when selecting whole slices
Tensor index_select(const Tensor &input, int dim, const Tensor &indices);

// Normalization operations
// layer_norm: (x - mean) / sqrt(var + eps) * weight + bias
Tensor layer_norm(const Tensor &input, const Tensor &weight, const Tensor &bias,
                  int axis = -1, float eps = 1e-5f);
// rms_norm: x / sqrt(mean(x²) + eps) * weight
Tensor rms_norm(const Tensor &input, const Tensor &weight, int axis = -1,
                float eps = 1e-5f);

// Dropout operation
// Returns pair of (output, mask) where mask is Bool tensor of kept values
// Scale factor 1/(1-p) is applied when training=true
std::pair<Tensor, Tensor> dropout(const Tensor &input, float p = 0.5f,
                                  bool training = true);

// In-place operations
void add_inplace(Tensor &lhs, const Tensor &rhs);
void subtract_inplace(Tensor &lhs, const Tensor &rhs);
void multiply_inplace(Tensor &lhs, const Tensor &rhs);
void divide_inplace(Tensor &lhs, const Tensor &rhs);

void execute_binary_inplace(OpType op_type, Tensor &lhs, const Tensor &rhs);

} // namespace ops
} // namespace axiom