#include "axiom/operations.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>

#include "axiom/error.hpp"
#include "axiom/system.hpp"
#include "axiom/tensor.hpp"
#include "backends/cpu/cpu_operations.hpp"
#include "backends/metal/metal_operations.hpp"

namespace axiom {
namespace ops {

// ============================================================================
// Complex type legality enforcement
// ============================================================================

// Operations allowed for complex types
static const std::set<OpType> complex_allowed_ops = {
    OpType::Add,    OpType::Subtract, OpType::Multiply, OpType::Divide,
    OpType::Negate, OpType::Exp,      OpType::Log,      OpType::Sqrt,
    OpType::Sin,    OpType::Cos,      OpType::Tan,      OpType::Sum,
    OpType::Mean,   OpType::MatMul,   OpType::Conj,     OpType::Real,
    OpType::Imag};

static std::string op_type_name(OpType op) {
    switch (op) {
    case OpType::Add:
        return "add";
    case OpType::Subtract:
        return "subtract";
    case OpType::Multiply:
        return "multiply";
    case OpType::Divide:
        return "divide";
    case OpType::Power:
        return "power";
    case OpType::Modulo:
        return "modulo";
    case OpType::Equal:
        return "equal";
    case OpType::NotEqual:
        return "not_equal";
    case OpType::Less:
        return "less";
    case OpType::LessEqual:
        return "less_equal";
    case OpType::Greater:
        return "greater";
    case OpType::GreaterEqual:
        return "greater_equal";
    case OpType::LogicalAnd:
        return "logical_and";
    case OpType::LogicalOr:
        return "logical_or";
    case OpType::LogicalXor:
        return "logical_xor";
    case OpType::LogicalNot:
        return "logical_not";
    case OpType::Maximum:
        return "maximum";
    case OpType::Minimum:
        return "minimum";
    case OpType::Atan2:
        return "atan2";
    case OpType::Hypot:
        return "hypot";
    case OpType::Negate:
        return "negate";
    case OpType::Abs:
        return "abs";
    case OpType::Sqrt:
        return "sqrt";
    case OpType::Exp:
        return "exp";
    case OpType::Log:
        return "log";
    case OpType::Sin:
        return "sin";
    case OpType::Cos:
        return "cos";
    case OpType::Tan:
        return "tan";
    case OpType::Erf:
        return "erf";
    case OpType::Conj:
        return "conj";
    case OpType::Real:
        return "real";
    case OpType::Imag:
        return "imag";
    case OpType::GELU:
        return "gelu";
    case OpType::Softmax:
        return "softmax";
    case OpType::LogSoftmax:
        return "log_softmax";
    case OpType::Sum:
        return "sum";
    case OpType::Mean:
        return "mean";
    case OpType::Max:
        return "max";
    case OpType::Min:
        return "min";
    case OpType::ArgMax:
        return "argmax";
    case OpType::ArgMin:
        return "argmin";
    case OpType::Any:
        return "any";
    case OpType::All:
        return "all";
    case OpType::MatMul:
        return "matmul";
    case OpType::BatchMatMul:
        return "batch_matmul";
    case OpType::Where:
        return "where";
    case OpType::LayerNorm:
        return "layer_norm";
    case OpType::RMSNorm:
        return "rms_norm";
    case OpType::Dropout:
        return "dropout";
    case OpType::MaskedFill:
        return "masked_fill";
    case OpType::MaskedSelect:
        return "masked_select";
    case OpType::Gather:
        return "gather";
    case OpType::Scatter:
        return "scatter";
    case OpType::IndexSelect:
        return "index_select";
    default:
        return "unknown";
    }
}

static void assert_complex_legal(OpType op, DType dtype) {
    if (is_complex_dtype(dtype) &&
        complex_allowed_ops.find(op) == complex_allowed_ops.end()) {
        throw TypeError("Operation '" + op_type_name(op) +
                        "' not supported for complex types");
    }
}

// Helper function to convert vector to string
template <typename T>
static std::string vec_to_string(const std::vector<T> &vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << vec[i];
    }
    oss << "]";
    return oss.str();
}

// ============================================================================
// Broadcasting utilities
// ============================================================================

BroadcastInfo compute_broadcast_info(const Shape &lhs_shape,
                                     const Shape &rhs_shape) {
    if (lhs_shape.empty() && rhs_shape.empty()) {
        return {{}, {}, {}, false};
    }
    BroadcastInfo info;
    info.needs_broadcast = false;

    // Get the maximum number of dimensions
    size_t max_dims = std::max(lhs_shape.size(), rhs_shape.size());

    // Start with the result shape
    info.result_shape.resize(max_dims);
    info.lhs_strides_adjustment.resize(max_dims, 0);
    info.rhs_strides_adjustment.resize(max_dims, 0);

    // Work backwards from the last dimension
    for (int i = max_dims - 1; i >= 0; --i) {
        int lhs_idx = static_cast<int>(lhs_shape.size()) - (max_dims - i);
        int rhs_idx = static_cast<int>(rhs_shape.size()) - (max_dims - i);

        size_t lhs_dim = (lhs_idx >= 0) ? lhs_shape[lhs_idx] : 1;
        size_t rhs_dim = (rhs_idx >= 0) ? rhs_shape[rhs_idx] : 1;

        if (lhs_dim == rhs_dim) {
            info.result_shape[i] = lhs_dim;
        } else if (lhs_dim == 1) {
            info.result_shape[i] = rhs_dim;
            info.lhs_strides_adjustment[i] =
                0; // Stride becomes 0 for broadcasting
            info.needs_broadcast = true;
        } else if (rhs_dim == 1) {
            info.result_shape[i] = lhs_dim;
            info.rhs_strides_adjustment[i] =
                0; // Stride becomes 0 for broadcasting
            info.needs_broadcast = true;
        } else {
            throw ShapeError::broadcast_incompatible(
                "dimension mismatch at axis " + std::to_string(i) + ": " +
                std::to_string(lhs_dim) + " vs " + std::to_string(rhs_dim));
        }
    }

    return info;
}

bool are_broadcastable(const Shape &lhs_shape, const Shape &rhs_shape) {
    try {
        compute_broadcast_info(lhs_shape, rhs_shape);
        return true;
    } catch (const std::runtime_error &) {
        return false;
    }
}

// ============================================================================
// Type promotion utilities
// ============================================================================

DType promote_types(DType lhs, DType rhs) {
    if (lhs == rhs)
        return lhs;

    // Highest rank wins
    static const std::map<DType, int> rank = {
        {DType::Bool, 0},       {DType::Int8, 1},       {DType::UInt8, 2},
        {DType::Int16, 3},      {DType::UInt16, 4},     {DType::Int32, 5},
        {DType::UInt32, 6},     {DType::Int64, 7},      {DType::UInt64, 8},
        {DType::Float16, 9},    {DType::Float32, 10},   {DType::Float64, 11},
        {DType::Complex64, 12}, {DType::Complex128, 13}};

    DType t1 = rank.at(lhs) > rank.at(rhs) ? lhs : rhs;
    DType t2 = rank.at(lhs) > rank.at(rhs) ? rhs : lhs;

    if (is_complex_dtype(t1))
        return t1;
    if (is_floating_dtype(t1))
        return t1;

    // Both are integers
    bool signed1 = is_signed_integer_dtype(t1);
    bool signed2 = is_signed_integer_dtype(t2);

    size_t size1 = dtype_size(t1);
    size_t size2 = dtype_size(t2);

    if (signed1 == signed2)
        return t1;

    // t1 is higher rank. if signed, it can hold t2.
    if (signed1 && size1 > size2)
        return t1;

    // t1 is unsigned. if bigger, it can hold t2.
    if (!signed1 && size1 > size2)
        return t1;

    // if sizes are equal, promote to next size signed
    // if t2 is signed and bigger, t1 wins (e.g. uint8+int16 -> int16)
    // if we are here, we need to promote.
    switch (std::max(size1, size2)) {
    case 1:
        return DType::Int16;
    case 2:
        return DType::Int32;
    case 4:
        return DType::Int64;
    default:
        return DType::Float64;
    }
}

DType result_type(const Tensor &lhs, const Tensor &rhs) {
    return promote_types(lhs.dtype(), rhs.dtype());
}

// ============================================================================
// Operation Registry
// ============================================================================

std::map<std::pair<OpType, Device>, std::unique_ptr<Operation>> &
OperationRegistry::get_registry() {
    static std::map<std::pair<OpType, Device>, std::unique_ptr<Operation>>
        registry;
    return registry;
}

void OperationRegistry::register_operation(
    OpType op_type, Device device, std::unique_ptr<Operation> operation) {
    auto &registry = get_registry();
    registry[{op_type, device}] = std::move(operation);
}

const Operation *OperationRegistry::get_operation(OpType op_type,
                                                  Device device) {
    auto &registry = get_registry();
    auto it = registry.find({op_type, device});
    return (it != registry.end()) ? it->second.get() : nullptr;
}

std::vector<Device>
OperationRegistry::available_devices_for_operation(OpType op_type) {
    std::vector<Device> devices;
    auto &registry = get_registry();

    for (const auto &[key, operation] : registry) {
        if (key.first == op_type) {
            devices.push_back(key.second);
        }
    }

    return devices;
}

bool OperationRegistry::is_operation_available(OpType op_type, Device device) {
    return get_operation(op_type, device) != nullptr;
}

void OperationRegistry::initialize_builtin_operations() {
    // Register CPU operations
    backends::cpu::register_cpu_operations();

#ifdef __APPLE__
    backends::metal::register_metal_operations();
#endif
}

// ============================================================================
// Default Operation implementations
// ============================================================================

Tensor Operation::execute_unary(const Tensor &input) const {
    (void)input; // Suppress unused parameter warning
    throw RuntimeError::not_implemented("Unary operations for " + name());
}

Tensor Operation::execute_reduction(const Tensor &input,
                                    const std::vector<int> &axis,
                                    bool keep_dims) const {
    (void)input;
    (void)axis;
    (void)keep_dims;
    throw RuntimeError::not_implemented("Reduction operations for " + name());
}

Tensor Operation::execute_matmul(const Tensor &a, const Tensor &b,
                                 bool transpose_a, bool transpose_b) const {
    (void)a;
    (void)b;
    (void)transpose_a;
    (void)transpose_b;
    throw RuntimeError::not_implemented("MatMul operations for " + name());
}

Tensor Operation::execute_where(const Tensor &condition, const Tensor &a,
                                const Tensor &b) const {
    (void)condition;
    (void)a;
    (void)b;
    throw RuntimeError::not_implemented("Where operations for " + name());
}

Tensor Operation::execute_masked_fill(const Tensor &input, const Tensor &mask,
                                      const Tensor &value) const {
    (void)input;
    (void)mask;
    (void)value;
    throw RuntimeError::not_implemented("MaskedFill operations for " + name());
}

Tensor Operation::execute_masked_select(const Tensor &input,
                                        const Tensor &mask) const {
    (void)input;
    (void)mask;
    throw RuntimeError::not_implemented("MaskedSelect operations for " + name());
}

Tensor Operation::execute_gather(const Tensor &input, int dim,
                                 const Tensor &indices) const {
    (void)input;
    (void)dim;
    (void)indices;
    throw RuntimeError::not_implemented("Gather operations for " + name());
}

Tensor Operation::execute_scatter(const Tensor &input, int dim,
                                  const Tensor &indices,
                                  const Tensor &src) const {
    (void)input;
    (void)dim;
    (void)indices;
    (void)src;
    throw RuntimeError::not_implemented("Scatter operations for " + name());
}

Tensor Operation::execute_index_select(const Tensor &input, int dim,
                                       const Tensor &indices) const {
    (void)input;
    (void)dim;
    (void)indices;
    throw RuntimeError::not_implemented("IndexSelect operations for " + name());
}

void Operation::execute_binary_inplace(Tensor &lhs, const Tensor &rhs) const {
    (void)lhs; // Suppress unused parameter warning
    (void)rhs; // Suppress unused parameter warning
    throw RuntimeError::not_implemented("In-place operations for " + name());
}

// ============================================================================
// Helper function for executing unary operations
// ============================================================================

static Tensor execute_unary_operation(OpType op_type, const Tensor &input) {
    Device target_device = input.device();

    const Operation *op =
        OperationRegistry::get_operation(op_type, target_device);

    if (!op) {
        throw DeviceError("Operation not available for device: " +
                          axiom::system::device_to_string(target_device));
    }

    return op->execute_unary(input);
}

// ============================================================================
// Helper function for executing reduction operations
// ============================================================================
static Tensor execute_reduction_operation(OpType op_type, const Tensor &input,
                                          const std::vector<int> &axis,
                                          bool keep_dims) {
    Device target_device = input.device();

    const auto *op = OperationRegistry::get_operation(op_type, target_device);
    if (!op) {
        throw DeviceError("Reduction operation not available for device: " +
                          axiom::system::device_to_string(target_device));
    }

    return op->execute_reduction(input, axis, keep_dims);
}

// ============================================================================
// Helper function for executing matmul operations
// ============================================================================
static Tensor execute_matmul_operation(const Tensor &a, const Tensor &b,
                                       bool transpose_a, bool transpose_b) {
    // Determine target device (prefer GPU if available)
    Device target_device =
        (a.device() == Device::GPU || b.device() == Device::GPU) ? Device::GPU
                                                                 : Device::CPU;

    const Operation *op =
        OperationRegistry::get_operation(OpType::MatMul, target_device);

    // Fallback to CPU if GPU op not available
    if (target_device == Device::GPU && !op) {
        target_device = Device::CPU;
        op = OperationRegistry::get_operation(OpType::MatMul, target_device);
    }

    if (!op) {
        throw DeviceError("MatMul operation not available for any device");
    }

    // Move tensors to target device if needed
    Tensor a_target = (a.device() == target_device) ? a : a.to(target_device);
    Tensor b_target = (b.device() == target_device) ? b : b.to(target_device);

    return op->execute_matmul(a_target, b_target, transpose_a, transpose_b);
}

// ============================================================================
// Helper function for executing binary operations
// ============================================================================

static Tensor execute_binary_operation(OpType op_type, const Tensor &lhs,
                                       const Tensor &rhs) {
    // Check if tensors are broadcastable
    if (!are_broadcastable(lhs.shape(), rhs.shape())) {
        throw ShapeError::broadcast_incompatible(
            "shapes " + vec_to_string(lhs.shape()) + " and " +
            vec_to_string(rhs.shape()));
    }

    // Determine the target device (prefer GPU if available)
    Device target_device =
        (lhs.device() == Device::GPU || rhs.device() == Device::GPU)
            ? Device::GPU
            : Device::CPU;

    // Get the operation implementation
    const Operation *op =
        OperationRegistry::get_operation(op_type, target_device);

    // Fallback to CPU if GPU op is not available or doesn't support the inputs
    if (target_device == Device::GPU &&
        (!op || !op->supports_binary(lhs, rhs))) {
        target_device = Device::CPU;
        op = OperationRegistry::get_operation(op_type, target_device);
    }

    if (!op) {
        throw DeviceError("Operation not available for any device");
    }

    // Move tensors to target device if needed
    Tensor lhs_target =
        (lhs.device() == target_device) ? lhs : lhs.to(target_device);
    Tensor rhs_target =
        (rhs.device() == target_device) ? rhs : rhs.to(target_device);

    // Execute the operation
    return op->execute_binary(lhs_target, rhs_target);
}

// ============================================================================
// High-level operation functions
// ============================================================================

// Binary operations
Tensor add(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Add, lhs, rhs);
}

Tensor subtract(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Subtract, lhs, rhs);
}

Tensor multiply(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Multiply, lhs, rhs);
}

Tensor divide(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Divide, lhs, rhs);
}

Tensor power(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Power, lhs, rhs);
}

Tensor modulo(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Modulo, lhs, rhs);
}

// Comparison operations
Tensor equal(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Equal, lhs, rhs);
}

Tensor not_equal(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::NotEqual, lhs, rhs);
}

Tensor less(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Less, lhs, rhs);
}

Tensor less_equal(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::LessEqual, lhs, rhs);
}

Tensor greater(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Greater, lhs, rhs);
}

Tensor greater_equal(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::GreaterEqual, lhs, rhs);
}

// Logical operations
Tensor logical_and(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::LogicalAnd, lhs, rhs);
}

Tensor logical_or(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::LogicalOr, lhs, rhs);
}

Tensor logical_xor(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::LogicalXor, lhs, rhs);
}

// Bitwise operations
Tensor bitwise_and(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::BitwiseAnd, lhs, rhs);
}

Tensor bitwise_or(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::BitwiseOr, lhs, rhs);
}

Tensor bitwise_xor(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::BitwiseXor, lhs, rhs);
}

Tensor left_shift(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::LeftShift, lhs, rhs);
}

Tensor right_shift(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::RightShift, lhs, rhs);
}

// Math operations
Tensor maximum(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Maximum, lhs, rhs);
}

Tensor minimum(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Minimum, lhs, rhs);
}

Tensor atan2(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Atan2, lhs, rhs);
}

Tensor hypot(const Tensor &lhs, const Tensor &rhs) {
    return execute_binary_operation(OpType::Hypot, lhs, rhs);
}

// Unary operations
Tensor negate(const Tensor &input) {
    return execute_unary_operation(OpType::Negate, input);
}

Tensor abs(const Tensor &input) {
    return execute_unary_operation(OpType::Abs, input);
}

Tensor sqrt(const Tensor &input) {
    return execute_unary_operation(OpType::Sqrt, input);
}

Tensor exp(const Tensor &input) {
    return execute_unary_operation(OpType::Exp, input);
}

Tensor log(const Tensor &input) {
    return execute_unary_operation(OpType::Log, input);
}

Tensor sin(const Tensor &input) {
    return execute_unary_operation(OpType::Sin, input);
}

Tensor cos(const Tensor &input) {
    return execute_unary_operation(OpType::Cos, input);
}

Tensor tan(const Tensor &input) {
    return execute_unary_operation(OpType::Tan, input);
}

Tensor erf(const Tensor &input) {
    return execute_unary_operation(OpType::Erf, input);
}

// Complex operations
Tensor conj(const Tensor &input) {
    if (!is_complex_dtype(input.dtype())) {
        throw TypeError("conj() requires complex tensor, got " +
                        input.dtype_name());
    }
    return execute_unary_operation(OpType::Conj, input);
}

Tensor real(const Tensor &input) { return input.real(); }

Tensor imag(const Tensor &input) { return input.imag(); }

// Activation operations
Tensor gelu(const Tensor &input) {
    return execute_unary_operation(OpType::GELU, input);
}

Tensor softmax(const Tensor &input, int axis) {
    Device target_device = input.device();

    const Operation *op =
        OperationRegistry::get_operation(OpType::Softmax, target_device);

    if (!op) {
        throw DeviceError("Softmax operation not available for device: " +
                          axiom::system::device_to_string(target_device));
    }

    // Softmax uses execute_reduction with a single axis
    return op->execute_reduction(input, {axis}, false);
}

Tensor log_softmax(const Tensor &input, int axis) {
    Device target_device = input.device();

    const Operation *op =
        OperationRegistry::get_operation(OpType::LogSoftmax, target_device);

    if (!op) {
        throw DeviceError("LogSoftmax operation not available for device: " +
                          axiom::system::device_to_string(target_device));
    }

    return op->execute_reduction(input, {axis}, false);
}

// Reduction operations
Tensor sum(const Tensor &input, const std::vector<int> &axis, bool keep_dims) {
    return execute_reduction_operation(OpType::Sum, input, axis, keep_dims);
}

Tensor mean(const Tensor &input, const std::vector<int> &axis, bool keep_dims) {
    return execute_reduction_operation(OpType::Mean, input, axis, keep_dims);
}

Tensor max(const Tensor &input, const std::vector<int> &axis, bool keep_dims) {
    return execute_reduction_operation(OpType::Max, input, axis, keep_dims);
}

Tensor min(const Tensor &input, const std::vector<int> &axis, bool keep_dims) {
    return execute_reduction_operation(OpType::Min, input, axis, keep_dims);
}

Tensor argmax(const Tensor &input, int axis, bool keep_dims) {
    return execute_reduction_operation(OpType::ArgMax, input, {axis},
                                       keep_dims);
}

Tensor argmin(const Tensor &input, int axis, bool keep_dims) {
    return execute_reduction_operation(OpType::ArgMin, input, {axis},
                                       keep_dims);
}

Tensor any(const Tensor &input, const std::vector<int> &axis, bool keep_dims) {
    return execute_reduction_operation(OpType::Any, input, axis, keep_dims);
}

Tensor all(const Tensor &input, const std::vector<int> &axis, bool keep_dims) {
    return execute_reduction_operation(OpType::All, input, axis, keep_dims);
}

void execute_binary_inplace(OpType op_type, Tensor &lhs, const Tensor &rhs) {
    auto device = lhs.device(); // In-place ops run on the device of the lhs
    auto op = OperationRegistry::get_operation(op_type, device);
    if (!op) {
        throw DeviceError("Operation not available for device: " +
                          axiom::system::device_to_string(device));
    }
    op->execute_binary_inplace(lhs, rhs);
}

// In-place operations
void add_inplace(Tensor &lhs, const Tensor &rhs) {
    execute_binary_inplace(OpType::Add, lhs, rhs);
}
void subtract_inplace(Tensor &lhs, const Tensor &rhs) {
    execute_binary_inplace(OpType::Subtract, lhs, rhs);
}
void multiply_inplace(Tensor &lhs, const Tensor &rhs) {
    execute_binary_inplace(OpType::Multiply, lhs, rhs);
}
void divide_inplace(Tensor &lhs, const Tensor &rhs) {
    execute_binary_inplace(OpType::Divide, lhs, rhs);
}

// Matrix multiplication
Tensor matmul(const Tensor &a, const Tensor &b, bool transpose_a,
              bool transpose_b) {
    return execute_matmul_operation(a, b, transpose_a, transpose_b);
}

// Conditional selection (where)
Tensor where(const Tensor &condition, const Tensor &a, const Tensor &b) {
    // Determine device - prefer GPU if any input is on GPU
    Device device = Device::CPU;
    if (condition.device() == Device::GPU || a.device() == Device::GPU ||
        b.device() == Device::GPU) {
        device = Device::GPU;
    }

    // Get operation for device
    auto op = OperationRegistry::get_operation(OpType::Where, device);
    if (!op) {
        throw DeviceError("Where operation not available for device: " +
                          axiom::system::device_to_string(device));
    }

    // Move tensors to target device if needed
    Tensor cond_on_device =
        (condition.device() == device) ? condition : condition.to(device);
    Tensor a_on_device = (a.device() == device) ? a : a.to(device);
    Tensor b_on_device = (b.device() == device) ? b : b.to(device);

    return op->execute_where(cond_on_device, a_on_device, b_on_device);
}

// ============================================================================
// Masking operations
// ============================================================================

Tensor masked_fill(const Tensor &input, const Tensor &mask, float value) {
    auto value_tensor = Tensor::full({1}, value, input.device());
    return masked_fill(input, mask, value_tensor);
}

Tensor masked_fill(const Tensor &input, const Tensor &mask, double value) {
    auto value_tensor = Tensor::full({1}, static_cast<float>(value), input.device());
    return masked_fill(input, mask, value_tensor);
}

Tensor masked_fill(const Tensor &input, const Tensor &mask,
                   const Tensor &value) {
    // Determine device - prefer GPU if any input is on GPU
    Device device = Device::CPU;
    if (input.device() == Device::GPU || mask.device() == Device::GPU ||
        value.device() == Device::GPU) {
        device = Device::GPU;
    }

    // Get operation for device
    auto op = OperationRegistry::get_operation(OpType::MaskedFill, device);
    if (!op) {
        // Fallback to CPU
        device = Device::CPU;
        op = OperationRegistry::get_operation(OpType::MaskedFill, device);
    }
    if (!op) {
        throw DeviceError("MaskedFill operation not available for device: " +
                          axiom::system::device_to_string(device));
    }

    // Move tensors to target device if needed
    Tensor input_on_device =
        (input.device() == device) ? input : input.to(device);
    Tensor mask_on_device =
        (mask.device() == device) ? mask : mask.to(device);
    Tensor value_on_device =
        (value.device() == device) ? value : value.to(device);

    return op->execute_masked_fill(input_on_device, mask_on_device,
                                   value_on_device);
}

Tensor masked_select(const Tensor &input, const Tensor &mask) {
    // Determine device
    Device device = Device::CPU;
    if (input.device() == Device::GPU || mask.device() == Device::GPU) {
        device = Device::GPU;
    }

    // Get operation for device
    auto op = OperationRegistry::get_operation(OpType::MaskedSelect, device);
    if (!op) {
        // Fallback to CPU
        device = Device::CPU;
        op = OperationRegistry::get_operation(OpType::MaskedSelect, device);
    }
    if (!op) {
        throw DeviceError("MaskedSelect operation not available for device: " +
                          axiom::system::device_to_string(device));
    }

    // Move tensors to target device if needed
    Tensor input_on_device =
        (input.device() == device) ? input : input.to(device);
    Tensor mask_on_device =
        (mask.device() == device) ? mask : mask.to(device);

    return op->execute_masked_select(input_on_device, mask_on_device);
}

// ============================================================================
// Indexing operations
// ============================================================================

Tensor gather(const Tensor &input, int dim, const Tensor &indices) {
    // Determine device
    Device device = Device::CPU;
    if (input.device() == Device::GPU || indices.device() == Device::GPU) {
        device = Device::GPU;
    }

    // Get operation for device
    auto op = OperationRegistry::get_operation(OpType::Gather, device);
    if (!op) {
        // Fallback to CPU
        device = Device::CPU;
        op = OperationRegistry::get_operation(OpType::Gather, device);
    }
    if (!op) {
        throw DeviceError("Gather operation not available for device: " +
                          axiom::system::device_to_string(device));
    }

    // Move tensors to target device if needed
    Tensor input_on_device =
        (input.device() == device) ? input : input.to(device);
    Tensor indices_on_device =
        (indices.device() == device) ? indices : indices.to(device);

    return op->execute_gather(input_on_device, dim, indices_on_device);
}

Tensor scatter(const Tensor &input, int dim, const Tensor &indices,
               const Tensor &src) {
    // Determine device
    Device device = Device::CPU;
    if (input.device() == Device::GPU || indices.device() == Device::GPU ||
        src.device() == Device::GPU) {
        device = Device::GPU;
    }

    // Get operation for device
    auto op = OperationRegistry::get_operation(OpType::Scatter, device);
    if (!op) {
        // Fallback to CPU
        device = Device::CPU;
        op = OperationRegistry::get_operation(OpType::Scatter, device);
    }
    if (!op) {
        throw DeviceError("Scatter operation not available for device: " +
                          axiom::system::device_to_string(device));
    }

    // Move tensors to target device if needed
    Tensor input_on_device =
        (input.device() == device) ? input : input.to(device);
    Tensor indices_on_device =
        (indices.device() == device) ? indices : indices.to(device);
    Tensor src_on_device = (src.device() == device) ? src : src.to(device);

    return op->execute_scatter(input_on_device, dim, indices_on_device,
                               src_on_device);
}

Tensor index_select(const Tensor &input, int dim, const Tensor &indices) {
    // Determine device
    Device device = Device::CPU;
    if (input.device() == Device::GPU || indices.device() == Device::GPU) {
        device = Device::GPU;
    }

    // Get operation for device
    auto op = OperationRegistry::get_operation(OpType::IndexSelect, device);
    if (!op) {
        // Fallback to CPU
        device = Device::CPU;
        op = OperationRegistry::get_operation(OpType::IndexSelect, device);
    }
    if (!op) {
        throw DeviceError("IndexSelect operation not available for device: " +
                          axiom::system::device_to_string(device));
    }

    // Move tensors to target device if needed
    Tensor input_on_device =
        (input.device() == device) ? input : input.to(device);
    Tensor indices_on_device =
        (indices.device() == device) ? indices : indices.to(device);

    return op->execute_index_select(input_on_device, dim, indices_on_device);
}

// Normalization operations
Tensor layer_norm(const Tensor &input, const Tensor &weight, const Tensor &bias,
                  int axis, float eps) {
    // Determine device
    Device device = Device::CPU;
    if (input.device() == Device::GPU || weight.device() == Device::GPU ||
        bias.device() == Device::GPU) {
        device = Device::GPU;
    }

    // Move tensors to target device if needed
    Tensor input_on_device =
        (input.device() == device) ? input : input.to(device);
    Tensor weight_on_device =
        (weight.device() == device) ? weight : weight.to(device);
    Tensor bias_on_device = (bias.device() == device) ? bias : bias.to(device);

    // Normalize axis
    int norm_axis = axis;
    if (norm_axis < 0) {
        norm_axis += static_cast<int>(input.ndim());
    }

    // Compute mean and variance along axis
    auto input_mean = mean(input_on_device, {norm_axis}, true);
    auto x_centered = subtract(input_on_device, input_mean);
    auto x_sq = multiply(x_centered, x_centered);
    auto variance = mean(x_sq, {norm_axis}, true);

    // Normalize: (x - mean) / sqrt(var + eps)
    auto eps_tensor = Tensor::full({1}, eps, device);
    auto var_eps = add(variance, eps_tensor);
    auto std_dev = ops::sqrt(var_eps);
    auto normalized = divide(x_centered, std_dev);

    // Apply weight and bias
    auto scaled = multiply(normalized, weight_on_device);
    return add(scaled, bias_on_device);
}

Tensor rms_norm(const Tensor &input, const Tensor &weight, int axis,
                float eps) {
    // Determine device
    Device device = Device::CPU;
    if (input.device() == Device::GPU || weight.device() == Device::GPU) {
        device = Device::GPU;
    }

    // Move tensors to target device if needed
    Tensor input_on_device =
        (input.device() == device) ? input : input.to(device);
    Tensor weight_on_device =
        (weight.device() == device) ? weight : weight.to(device);

    // Normalize axis
    int norm_axis = axis;
    if (norm_axis < 0) {
        norm_axis += static_cast<int>(input.ndim());
    }

    // Compute RMS: sqrt(mean(x²) + eps)
    auto x_sq = multiply(input_on_device, input_on_device);
    auto mean_x_sq = mean(x_sq, {norm_axis}, true);
    auto eps_tensor = Tensor::full({1}, eps, device);
    auto mean_eps = add(mean_x_sq, eps_tensor);
    auto rms = ops::sqrt(mean_eps);

    // Normalize: x / rms * weight
    auto normalized = divide(input_on_device, rms);
    return multiply(normalized, weight_on_device);
}

// Dropout operation
std::pair<Tensor, Tensor> dropout(const Tensor &input, float p, bool training) {
    if (!training || p == 0.0f) {
        // No dropout during inference or when p=0
        return {input,
                Tensor::ones(input.shape(), DType::Bool, input.device())};
    }

    if (p < 0.0f || p >= 1.0f) {
        throw ValueError("Dropout probability must be in [0, 1), got " +
                         std::to_string(p));
    }

    Device device = input.device();

    // Generate random mask on CPU, then move to device
    auto rand_tensor =
        Tensor::randn(input.shape(), DType::Float32, Device::CPU);

    // Create mask: keep with probability (1-p)
    // We use comparison: random > p to get True for values to keep
    auto threshold = Tensor::full(input.shape(), p, Device::CPU);
    auto rand_uniform = Tensor(input.shape(), DType::Float32, Device::CPU);

    // Generate uniform random values using Box-Muller transform result
    // For simplicity, use abs of randn and scale to [0,1)
    float *rand_data = rand_tensor.typed_data<float>();
    float *uniform_data = rand_uniform.typed_data<float>();
    for (size_t i = 0; i < input.size(); ++i) {
        // Simple hash to uniform - this is a quick approximation
        uniform_data[i] = std::fmod(
            std::abs(rand_data[i] * 0.3989422804014327f + 0.5f), 1.0f);
    }

    auto mask = greater(rand_uniform, threshold);

    // Move to device if needed
    if (device != Device::CPU) {
        mask = mask.to(device);
    }

    // Scale factor: 1 / (1 - p)
    float scale = 1.0f / (1.0f - p);
    auto scale_tensor = Tensor::full(input.shape(), scale, device);

    // Apply mask and scale
    auto mask_float = mask.astype(input.dtype());
    auto masked = multiply(input, mask_float);
    auto scaled = multiply(masked, scale_tensor);

    return {scaled, mask};
}

} // namespace ops
} // namespace axiom