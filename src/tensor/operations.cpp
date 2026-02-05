#include "axiom/operations.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>

#include "axiom/error.hpp"
#include "axiom/graph/graph_registry.hpp"
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
    OpType::Negate, OpType::Abs,      OpType::Exp,      OpType::Log,
    OpType::Sqrt,   OpType::Sin,      OpType::Cos,      OpType::Tan,
    OpType::Sum,    OpType::Mean,     OpType::MatMul,   OpType::Conj,
    OpType::Real,   OpType::Imag,     OpType::Where};

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
    case OpType::Sign:
        return "sign";
    case OpType::Floor:
        return "floor";
    case OpType::Ceil:
        return "ceil";
    case OpType::Trunc:
        return "trunc";
    case OpType::Round:
        return "round";
    case OpType::Reciprocal:
        return "reciprocal";
    case OpType::Square:
        return "square";
    case OpType::Cbrt:
        return "cbrt";
    case OpType::IsNaN:
        return "isnan";
    case OpType::IsInf:
        return "isinf";
    case OpType::IsFinite:
        return "isfinite";
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
    case OpType::Prod:
        return "prod";
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
    case OpType::Take:
        return "take";
    case OpType::TakeAlongAxis:
        return "take_along_axis";
    case OpType::MaxPool1D:
        return "max_pool1d";
    case OpType::MaxPool2D:
        return "max_pool2d";
    case OpType::MaxPool3D:
        return "max_pool3d";
    case OpType::AvgPool1D:
        return "avg_pool1d";
    case OpType::AvgPool2D:
        return "avg_pool2d";
    case OpType::AvgPool3D:
        return "avg_pool3d";
    case OpType::AdaptiveMaxPool2D:
        return "adaptive_max_pool2d";
    case OpType::AdaptiveAvgPool2D:
        return "adaptive_avg_pool2d";
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

Shape broadcast_shapes(const std::vector<Shape> &shapes) {
    if (shapes.empty()) {
        return {};
    }
    if (shapes.size() == 1) {
        return shapes[0];
    }

    // Pairwise broadcast: result = broadcast(broadcast(s1, s2), s3)...
    Shape result = shapes[0];
    for (size_t i = 1; i < shapes.size(); ++i) {
        result = ShapeUtils::broadcast_shape(result, shapes[i]);
    }
    return result;
}

std::vector<Tensor> broadcast_tensors(const std::vector<Tensor> &tensors) {
    if (tensors.empty()) {
        return {};
    }
    if (tensors.size() == 1) {
        return tensors;
    }

    // Collect all shapes
    std::vector<Shape> shapes;
    shapes.reserve(tensors.size());
    for (const auto &t : tensors) {
        shapes.push_back(t.shape());
    }

    // Compute broadcast shape
    Shape target_shape = broadcast_shapes(shapes);

    // Expand each tensor to target shape (zero-copy via strides)
    std::vector<Tensor> result;
    result.reserve(tensors.size());
    for (const auto &t : tensors) {
        result.push_back(t.expand(target_shape));
    }

    return result;
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
    static bool initialized = false;
    if (!initialized) {
        initialized = true; // Set first to prevent recursion
        backends::cpu::register_cpu_operations();
#ifdef __APPLE__
        backends::metal::register_metal_operations();
#endif
    }
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
    throw RuntimeError::not_implemented("MaskedSelect operations for " +
                                        name());
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

Tensor Operation::execute_cast(const Tensor &input, DType target_dtype) const {
    (void)input;
    (void)target_dtype;
    throw RuntimeError::not_implemented("Cast operations for " + name());
}

void Operation::execute_binary_inplace(Tensor &lhs, const Tensor &rhs) const {
    (void)lhs; // Suppress unused parameter warning
    (void)rhs; // Suppress unused parameter warning
    throw RuntimeError::not_implemented("In-place operations for " + name());
}

// ============================================================================
// Helper function for complex unary operations
// ============================================================================
static Tensor execute_complex_unary(OpType op_type, const Tensor &input) {
    // Complex unary ops: Negate, Abs, Sqrt, Exp, Log, Sin, Cos, Tan, Conj
    Tensor input_cpu = input.cpu();
    DType dtype = input.dtype();

    // Abs returns real type
    if (op_type == OpType::Abs) {
        DType result_dtype =
            (dtype == DType::Complex64) ? DType::Float32 : DType::Float64;
        Tensor result(input.shape(), result_dtype, Device::CPU);

        if (dtype == DType::Complex64) {
            const complex64_t *in = input_cpu.typed_data<complex64_t>();
            float *out = result.typed_data<float>();
            for (size_t i = 0; i < input.size(); ++i)
                out[i] = std::abs(in[i]);
        } else {
            const complex128_t *in = input_cpu.typed_data<complex128_t>();
            double *out = result.typed_data<double>();
            for (size_t i = 0; i < input.size(); ++i)
                out[i] = std::abs(in[i]);
        }
        return result;
    }

    // All other ops return complex
    Tensor result(input.shape(), dtype, Device::CPU);

    if (dtype == DType::Complex64) {
        const complex64_t *in = input_cpu.typed_data<complex64_t>();
        complex64_t *out = result.typed_data<complex64_t>();

        for (size_t i = 0; i < input.size(); ++i) {
            switch (op_type) {
            case OpType::Negate:
                out[i] = -in[i];
                break;
            case OpType::Sqrt:
                out[i] = std::sqrt(in[i]);
                break;
            case OpType::Exp:
                out[i] = std::exp(in[i]);
                break;
            case OpType::Log:
                out[i] = std::log(in[i]);
                break;
            case OpType::Sin:
                out[i] = std::sin(in[i]);
                break;
            case OpType::Cos:
                out[i] = std::cos(in[i]);
                break;
            case OpType::Tan:
                out[i] = std::tan(in[i]);
                break;
            case OpType::Conj:
                out[i] = std::conj(in[i]);
                break;
            default:
                throw TypeError("Unary operation " + op_type_name(op_type) +
                                " not supported for complex types");
            }
        }
    } else { // Complex128
        const complex128_t *in = input_cpu.typed_data<complex128_t>();
        complex128_t *out = result.typed_data<complex128_t>();

        for (size_t i = 0; i < input.size(); ++i) {
            switch (op_type) {
            case OpType::Negate:
                out[i] = -in[i];
                break;
            case OpType::Sqrt:
                out[i] = std::sqrt(in[i]);
                break;
            case OpType::Exp:
                out[i] = std::exp(in[i]);
                break;
            case OpType::Log:
                out[i] = std::log(in[i]);
                break;
            case OpType::Sin:
                out[i] = std::sin(in[i]);
                break;
            case OpType::Cos:
                out[i] = std::cos(in[i]);
                break;
            case OpType::Tan:
                out[i] = std::tan(in[i]);
                break;
            case OpType::Conj:
                out[i] = std::conj(in[i]);
                break;
            default:
                throw TypeError("Unary operation " + op_type_name(op_type) +
                                " not supported for complex types");
            }
        }
    }

    return result;
}

// Helper function for executing unary operations
// ============================================================================

static Tensor execute_unary_operation(OpType op_type, const Tensor &input) {
    // Handle complex types directly (no lazy evaluation for complex)
    if (is_complex_dtype(input.dtype())) {
        assert_complex_legal(op_type, input.dtype());
        return execute_complex_unary(op_type, input);
    }

    // Use lazy evaluation for non-complex types
    return graph::GraphRegistry::create_lazy_unary(op_type, input);
}

// Eager execution path for unary operations (used by lazy evaluation)
static Tensor execute_unary_operation_eager(OpType op_type,
                                            const Tensor &input) {
    Device target_device = input.device();

    const Operation *op =
        OperationRegistry::get_operation(op_type, target_device);

    if (!op) {
        throw DeviceError("Operation not available for device: " +
                          axiom::system::device_to_string(target_device));
    }

    // Move tensor to target device if needed
    Tensor input_target =
        (input.device() == target_device) ? input : input.to(target_device);

    return op->execute_unary(input_target);
}

// ============================================================================
// Helper for complex reductions (Sum and Mean only)
// ============================================================================
static Tensor execute_complex_reduction(OpType op_type, const Tensor &input,
                                        const std::vector<int> &axes,
                                        bool keep_dims) {
    // Only Sum and Mean are valid for complex types
    if (op_type != OpType::Sum && op_type != OpType::Mean) {
        throw TypeError(
            "Reduction '" + op_type_name(op_type) +
            "' not supported for complex types (no total ordering)");
    }

    Tensor input_cpu = input.cpu();
    std::vector<int> norm_axes = axes;
    if (norm_axes.empty()) {
        for (size_t i = 0; i < input.ndim(); ++i)
            norm_axes.push_back(static_cast<int>(i));
    }

    // Normalize negative axes
    for (int &ax : norm_axes) {
        if (ax < 0)
            ax += static_cast<int>(input.ndim());
    }

    // Calculate result shape
    Shape result_shape;
    for (size_t i = 0; i < input.shape().size(); ++i) {
        bool is_reduced = false;
        for (int ax : norm_axes) {
            if (i == static_cast<size_t>(ax)) {
                is_reduced = true;
                break;
            }
        }
        if (is_reduced) {
            if (keep_dims)
                result_shape.push_back(1);
        } else {
            result_shape.push_back(input.shape()[i]);
        }
    }
    if (result_shape.empty())
        result_shape.push_back(1);

    DType dtype = input.dtype();
    Tensor result(result_shape, dtype, Device::CPU);

    size_t reduction_size = 1;
    for (int ax : norm_axes) {
        reduction_size *= input.shape()[ax];
    }

    if (dtype == DType::Complex64) {
        complex64_t *out = result.typed_data<complex64_t>();
        // Initialize to zero
        for (size_t i = 0; i < result.size(); ++i)
            out[i] = complex64_t(0, 0);

        // Simple full reduction case
        if (norm_axes.size() == input.ndim()) {
            complex64_t sum(0, 0);
            std::vector<size_t> coords(input.ndim(), 0);
            for (size_t i = 0; i < input.size(); ++i) {
                sum += input_cpu.item<complex64_t>(coords);
                ShapeUtils::increment_coords(coords, input.shape());
            }
            if (op_type == OpType::Mean)
                sum /= static_cast<float>(input.size());
            out[0] = sum;
        } else {
            // Partial reduction - iterate over output positions
            std::vector<size_t> out_coords(result_shape.size(), 0);
            for (size_t out_i = 0; out_i < result.size(); ++out_i) {
                complex64_t sum(0, 0);
                // For each output position, sum over reduction dims
                std::vector<size_t> in_coords(input.ndim(), 0);
                size_t out_dim_idx = 0;
                for (size_t d = 0; d < input.ndim(); ++d) {
                    bool is_reduced = false;
                    for (int ax : norm_axes)
                        if (d == static_cast<size_t>(ax))
                            is_reduced = true;
                    if (!is_reduced) {
                        in_coords[d] = out_coords[out_dim_idx++];
                    }
                }
                // Sum over all combinations of reduction dimensions
                for (size_t r = 0; r < reduction_size; ++r) {
                    sum += input_cpu.item<complex64_t>(in_coords);
                    // Increment reduction coordinates
                    for (int ax_i = norm_axes.size() - 1; ax_i >= 0; --ax_i) {
                        size_t d = norm_axes[ax_i];
                        if (++in_coords[d] < input.shape()[d])
                            break;
                        in_coords[d] = 0;
                    }
                }
                if (op_type == OpType::Mean)
                    sum /= static_cast<float>(reduction_size);
                out[out_i] = sum;
                // Increment output coordinates
                for (int j = result_shape.size() - 1; j >= 0; --j) {
                    if (++out_coords[j] < result_shape[j])
                        break;
                    out_coords[j] = 0;
                }
            }
        }
    } else { // Complex128
        complex128_t *out = result.typed_data<complex128_t>();
        for (size_t i = 0; i < result.size(); ++i)
            out[i] = complex128_t(0, 0);

        if (norm_axes.size() == input.ndim()) {
            complex128_t sum(0, 0);
            std::vector<size_t> coords(input.ndim(), 0);
            for (size_t i = 0; i < input.size(); ++i) {
                sum += input_cpu.item<complex128_t>(coords);
                ShapeUtils::increment_coords(coords, input.shape());
            }
            if (op_type == OpType::Mean)
                sum /= static_cast<double>(input.size());
            out[0] = sum;
        } else {
            std::vector<size_t> out_coords(result_shape.size(), 0);
            for (size_t out_i = 0; out_i < result.size(); ++out_i) {
                complex128_t sum(0, 0);
                std::vector<size_t> in_coords(input.ndim(), 0);
                size_t out_dim_idx = 0;
                for (size_t d = 0; d < input.ndim(); ++d) {
                    bool is_reduced = false;
                    for (int ax : norm_axes)
                        if (d == static_cast<size_t>(ax))
                            is_reduced = true;
                    if (!is_reduced)
                        in_coords[d] = out_coords[out_dim_idx++];
                }
                for (size_t r = 0; r < reduction_size; ++r) {
                    sum += input_cpu.item<complex128_t>(in_coords);
                    for (int ax_i = norm_axes.size() - 1; ax_i >= 0; --ax_i) {
                        size_t d = norm_axes[ax_i];
                        if (++in_coords[d] < input.shape()[d])
                            break;
                        in_coords[d] = 0;
                    }
                }
                if (op_type == OpType::Mean)
                    sum /= static_cast<double>(reduction_size);
                out[out_i] = sum;
                for (int j = result_shape.size() - 1; j >= 0; --j) {
                    if (++out_coords[j] < result_shape[j])
                        break;
                    out_coords[j] = 0;
                }
            }
        }
    }

    return result;
}

// ============================================================================
// Helper function for executing reduction operations
// ============================================================================
static Tensor execute_reduction_operation(OpType op_type, const Tensor &input,
                                          const std::vector<int> &axis,
                                          bool keep_dims) {
    // Handle complex types directly (no lazy evaluation for complex)
    if (is_complex_dtype(input.dtype())) {
        assert_complex_legal(op_type, input.dtype());
        return execute_complex_reduction(op_type, input, axis, keep_dims);
    }

    // Use lazy evaluation for non-complex types
    return graph::GraphRegistry::create_lazy_reduction(op_type, input, axis,
                                                       keep_dims);
}

// Eager execution path for reduction operations (used by lazy evaluation)
static Tensor execute_reduction_operation_eager(OpType op_type,
                                                const Tensor &input,
                                                const std::vector<int> &axis,
                                                bool keep_dims) {
    Device target_device = input.device();

    const auto *op = OperationRegistry::get_operation(op_type, target_device);
    if (!op) {
        throw DeviceError("Reduction operation not available for device: " +
                          axiom::system::device_to_string(target_device));
    }

    // Move tensor to target device if needed
    Tensor input_target =
        (input.device() == target_device) ? input : input.to(target_device);

    return op->execute_reduction(input_target, axis, keep_dims);
}

// ============================================================================
// Helper function for executing matmul operations
// ============================================================================

// Eager execution path for matmul operations (used by lazy evaluation)
static Tensor execute_matmul_operation_eager(const Tensor &a, const Tensor &b,
                                             bool transpose_a,
                                             bool transpose_b) {
    // Determine target device (prefer GPU if available)
    Device target_device =
        (a.device() == Device::GPU || b.device() == Device::GPU) ? Device::GPU
                                                                 : Device::CPU;

    // Force CPU for complex types (GPU complex support is limited/unstable)
    DType promoted_dtype = promote_types(a.dtype(), b.dtype());
    if (is_complex_dtype(promoted_dtype)) {
        target_device = Device::CPU;
    }

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

static Tensor execute_matmul_operation(const Tensor &a, const Tensor &b,
                                       bool transpose_a, bool transpose_b) {
    // Force CPU for complex types (GPU complex support is limited/unstable)
    DType promoted_dtype = promote_types(a.dtype(), b.dtype());
    if (is_complex_dtype(promoted_dtype)) {
        // Execute eagerly for complex types
        return execute_matmul_operation_eager(a, b, transpose_a, transpose_b);
    }

    // Use lazy evaluation for non-complex types
    return graph::GraphRegistry::create_lazy_matmul(a, b, transpose_a,
                                                    transpose_b);
}

// ============================================================================
// Helper function for executing binary operations
// ============================================================================

// Helper for complex binary arithmetic
static Tensor execute_complex_binary(OpType op_type, const Tensor &lhs,
                                     const Tensor &rhs) {
    auto broadcast_info = compute_broadcast_info(lhs.shape(), rhs.shape());
    DType result_dtype = result_type(lhs, rhs);

    Tensor lhs_cpu = lhs.cpu().astype(result_dtype);
    Tensor rhs_cpu = rhs.cpu().astype(result_dtype);
    Tensor result(broadcast_info.result_shape, result_dtype, Device::CPU);

    size_t total = ShapeUtils::size(broadcast_info.result_shape);

    if (result_dtype == DType::Complex64) {
        const complex64_t *l = lhs_cpu.typed_data<complex64_t>();
        const complex64_t *r = rhs_cpu.typed_data<complex64_t>();
        complex64_t *out = result.typed_data<complex64_t>();

        // Simple case: same shape, contiguous
        if (lhs_cpu.shape() == rhs_cpu.shape() && lhs_cpu.is_contiguous() &&
            rhs_cpu.is_contiguous()) {
            for (size_t i = 0; i < total; ++i) {
                switch (op_type) {
                case OpType::Add:
                    out[i] = l[i] + r[i];
                    break;
                case OpType::Subtract:
                    out[i] = l[i] - r[i];
                    break;
                case OpType::Multiply:
                    out[i] = l[i] * r[i];
                    break;
                case OpType::Divide:
                    out[i] = l[i] / r[i];
                    break;
                default:
                    throw TypeError("Operation " + op_type_name(op_type) +
                                    " not supported for complex types");
                }
            }
        } else {
            // Broadcasting case - use item() for proper indexing
            std::vector<size_t> coords(broadcast_info.result_shape.size(), 0);
            for (size_t i = 0; i < total; ++i) {
                complex64_t lv = lhs_cpu.item<complex64_t>(coords);
                complex64_t rv = rhs_cpu.item<complex64_t>(coords);
                switch (op_type) {
                case OpType::Add:
                    out[i] = lv + rv;
                    break;
                case OpType::Subtract:
                    out[i] = lv - rv;
                    break;
                case OpType::Multiply:
                    out[i] = lv * rv;
                    break;
                case OpType::Divide:
                    out[i] = lv / rv;
                    break;
                default:
                    throw TypeError("Operation " + op_type_name(op_type) +
                                    " not supported for complex types");
                }
                ShapeUtils::increment_coords(coords,
                                             broadcast_info.result_shape);
            }
        }
    } else { // Complex128
        const complex128_t *l = lhs_cpu.typed_data<complex128_t>();
        const complex128_t *r = rhs_cpu.typed_data<complex128_t>();
        complex128_t *out = result.typed_data<complex128_t>();

        if (lhs_cpu.shape() == rhs_cpu.shape() && lhs_cpu.is_contiguous() &&
            rhs_cpu.is_contiguous()) {
            for (size_t i = 0; i < total; ++i) {
                switch (op_type) {
                case OpType::Add:
                    out[i] = l[i] + r[i];
                    break;
                case OpType::Subtract:
                    out[i] = l[i] - r[i];
                    break;
                case OpType::Multiply:
                    out[i] = l[i] * r[i];
                    break;
                case OpType::Divide:
                    out[i] = l[i] / r[i];
                    break;
                default:
                    throw TypeError("Operation " + op_type_name(op_type) +
                                    " not supported for complex types");
                }
            }
        } else {
            std::vector<size_t> coords(broadcast_info.result_shape.size(), 0);
            for (size_t i = 0; i < total; ++i) {
                complex128_t lv = lhs_cpu.item<complex128_t>(coords);
                complex128_t rv = rhs_cpu.item<complex128_t>(coords);
                switch (op_type) {
                case OpType::Add:
                    out[i] = lv + rv;
                    break;
                case OpType::Subtract:
                    out[i] = lv - rv;
                    break;
                case OpType::Multiply:
                    out[i] = lv * rv;
                    break;
                case OpType::Divide:
                    out[i] = lv / rv;
                    break;
                default:
                    throw TypeError("Operation " + op_type_name(op_type) +
                                    " not supported for complex types");
                }
                ShapeUtils::increment_coords(coords,
                                             broadcast_info.result_shape);
            }
        }
    }

    return result;
}

static Tensor execute_binary_operation(OpType op_type, const Tensor &lhs,
                                       const Tensor &rhs) {
    // Check if tensors are broadcastable
    if (!are_broadcastable(lhs.shape(), rhs.shape())) {
        throw ShapeError::broadcast_incompatible(
            "shapes " + vec_to_string(lhs.shape()) + " and " +
            vec_to_string(rhs.shape()));
    }

    // Check for complex types - handle directly (no lazy evaluation for
    // complex)
    DType promoted_dtype = promote_types(lhs.dtype(), rhs.dtype());
    if (is_complex_dtype(promoted_dtype)) {
        assert_complex_legal(op_type, promoted_dtype);
        return execute_complex_binary(op_type, lhs, rhs);
    }

    // Use lazy evaluation for non-complex types
    return graph::GraphRegistry::create_lazy_binary(op_type, lhs, rhs);
}

// Eager execution path for binary operations (used by lazy evaluation)
static Tensor execute_binary_operation_eager(OpType op_type, const Tensor &lhs,
                                             const Tensor &rhs) {
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

Tensor logical_not(const Tensor &input) {
    return execute_unary_operation(OpType::LogicalNot, input);
}

// Bitwise operations - require integer types
static void assert_bitwise_types(const Tensor &lhs, const Tensor &rhs,
                                 const std::string &op_name) {
    auto check_integral = [&](DType dtype) {
        return dtype == DType::Int8 || dtype == DType::Int16 ||
               dtype == DType::Int32 || dtype == DType::Int64 ||
               dtype == DType::UInt8 || dtype == DType::UInt16 ||
               dtype == DType::UInt32 || dtype == DType::UInt64 ||
               dtype == DType::Bool;
    };
    if (!check_integral(lhs.dtype())) {
        throw TypeError(op_name + " requires integer types, got " +
                        dtype_name(lhs.dtype()));
    }
    if (!check_integral(rhs.dtype())) {
        throw TypeError(op_name + " requires integer types, got " +
                        dtype_name(rhs.dtype()));
    }
}

Tensor bitwise_and(const Tensor &lhs, const Tensor &rhs) {
    assert_bitwise_types(lhs, rhs, "bitwise_and");
    return execute_binary_operation(OpType::BitwiseAnd, lhs, rhs);
}

Tensor bitwise_or(const Tensor &lhs, const Tensor &rhs) {
    assert_bitwise_types(lhs, rhs, "bitwise_or");
    return execute_binary_operation(OpType::BitwiseOr, lhs, rhs);
}

Tensor bitwise_xor(const Tensor &lhs, const Tensor &rhs) {
    assert_bitwise_types(lhs, rhs, "bitwise_xor");
    return execute_binary_operation(OpType::BitwiseXor, lhs, rhs);
}

Tensor left_shift(const Tensor &lhs, const Tensor &rhs) {
    assert_bitwise_types(lhs, rhs, "left_shift");
    return execute_binary_operation(OpType::LeftShift, lhs, rhs);
}

Tensor right_shift(const Tensor &lhs, const Tensor &rhs) {
    assert_bitwise_types(lhs, rhs, "right_shift");
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

// NumPy-like math operations
Tensor sign(const Tensor &input) {
    return execute_unary_operation(OpType::Sign, input);
}

Tensor floor(const Tensor &input) {
    return execute_unary_operation(OpType::Floor, input);
}

Tensor ceil(const Tensor &input) {
    return execute_unary_operation(OpType::Ceil, input);
}

Tensor trunc(const Tensor &input) {
    return execute_unary_operation(OpType::Trunc, input);
}

Tensor round(const Tensor &input, int decimals) {
    if (decimals == 0) {
        return execute_unary_operation(OpType::Round, input);
    }
    // For non-zero decimals: round(x * 10^d) / 10^d
    double scale = std::pow(10.0, decimals);
    auto scale_tensor = Tensor::full({1}, static_cast<float>(scale));
    auto scaled = multiply(input, scale_tensor);
    auto rounded = execute_unary_operation(OpType::Round, scaled);
    auto inv_scale_tensor = Tensor::full({1}, static_cast<float>(1.0 / scale));
    return multiply(rounded, inv_scale_tensor);
}

Tensor reciprocal(const Tensor &input) {
    return execute_unary_operation(OpType::Reciprocal, input);
}

Tensor square(const Tensor &input) {
    return execute_unary_operation(OpType::Square, input);
}

Tensor cbrt(const Tensor &input) {
    return execute_unary_operation(OpType::Cbrt, input);
}

// Element-wise testing operations
Tensor isnan(const Tensor &input) {
    return execute_unary_operation(OpType::IsNaN, input);
}

Tensor isinf(const Tensor &input) {
    return execute_unary_operation(OpType::IsInf, input);
}

Tensor isfinite(const Tensor &input) {
    return execute_unary_operation(OpType::IsFinite, input);
}

// Clipping operation (composition-based, no new backend op)
Tensor clip(const Tensor &input, const Tensor &min_val, const Tensor &max_val) {
    return maximum(minimum(input, max_val), min_val);
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
Tensor relu(const Tensor &input) {
    return execute_unary_operation(OpType::ReLU, input);
}

Tensor leaky_relu(const Tensor &input, float negative_slope) {
    // For non-default slopes, we need a custom implementation
    // For now, use the registered default (0.01)
    (void)negative_slope; // TODO: support custom slopes
    return execute_unary_operation(OpType::LeakyReLU, input);
}

Tensor silu(const Tensor &input) {
    return execute_unary_operation(OpType::SiLU, input);
}

Tensor sigmoid(const Tensor &input) {
    return execute_unary_operation(OpType::Sigmoid, input);
}

Tensor tanh(const Tensor &input) {
    return execute_unary_operation(OpType::Tanh, input);
}

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

Tensor prod(const Tensor &input, const std::vector<int> &axis, bool keep_dims) {
    return execute_reduction_operation(OpType::Prod, input, axis, keep_dims);
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
    auto value_tensor =
        Tensor::full({1}, static_cast<float>(value), input.device());
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
    Tensor mask_on_device = (mask.device() == device) ? mask : mask.to(device);
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
    Tensor mask_on_device = (mask.device() == device) ? mask : mask.to(device);

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

// ============================================================================
// Advanced indexing operations
// ============================================================================

Tensor take(const Tensor &input, const Tensor &indices, int axis) {
    // Validate indices dtype
    if (!is_integer_dtype(indices.dtype())) {
        throw TypeError("take: indices must be integer type, got " +
                        dtype_name(indices.dtype()));
    }

    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();
    Tensor indices_cpu =
        indices.device() == Device::CPU ? indices : indices.cpu();

    // Convert indices to Int64 for uniform handling
    if (indices_cpu.dtype() != DType::Int64) {
        indices_cpu = indices_cpu.astype(DType::Int64);
    }

    if (axis == -1) {
        // Take from flattened tensor
        Tensor flat = input_cpu.flatten();
        size_t n = flat.size();

        // Output shape is same as indices shape
        Tensor result(indices.shape(), input.dtype(), Device::CPU);

        const int64_t *idx_data = indices_cpu.typed_data<int64_t>();
        size_t out_size = indices_cpu.size();

        switch (input.dtype()) {
#define TAKE_CASE(DTYPE, CTYPE)                                                \
    case DTYPE: {                                                              \
        const CTYPE *src = flat.typed_data<CTYPE>();                           \
        CTYPE *dst = result.typed_data<CTYPE>();                               \
        for (size_t i = 0; i < out_size; ++i) {                                \
            int64_t idx = idx_data[i];                                         \
            if (idx < 0)                                                       \
                idx += static_cast<int64_t>(n);                                \
            if (idx < 0 || idx >= static_cast<int64_t>(n)) {                   \
                throw IndexError(                                              \
                    "take: index " + std::to_string(idx_data[i]) +             \
                    " out of bounds for size " + std::to_string(n));           \
            }                                                                  \
            dst[i] = src[idx];                                                 \
        }                                                                      \
        break;                                                                 \
    }
            TAKE_CASE(DType::Float32, float)
            TAKE_CASE(DType::Float64, double)
            TAKE_CASE(DType::Int32, int32_t)
            TAKE_CASE(DType::Int64, int64_t)
            TAKE_CASE(DType::Int8, int8_t)
            TAKE_CASE(DType::Int16, int16_t)
#undef TAKE_CASE
        default:
            throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                               "take");
        }

        return input.device() == Device::GPU ? result.gpu() : result;
    }

    // Take along specific axis
    int ndim = static_cast<int>(input.ndim());
    if (axis < 0)
        axis += ndim;
    if (axis < 0 || axis >= ndim) {
        throw ShapeError::invalid_axis(axis, ndim);
    }

    // Use index_select for axis-based take
    return index_select(input, axis, indices.flatten());
}

Tensor take_along_axis(const Tensor &input, const Tensor &indices, int axis) {
    // Validate indices dtype
    if (!is_integer_dtype(indices.dtype())) {
        throw TypeError("take_along_axis: indices must be integer type, got " +
                        dtype_name(indices.dtype()));
    }

    int ndim = static_cast<int>(input.ndim());
    if (axis < 0)
        axis += ndim;
    if (axis < 0 || axis >= ndim) {
        throw ShapeError::invalid_axis(axis, ndim);
    }

    // indices and input must have same number of dimensions
    if (indices.ndim() != input.ndim()) {
        throw ShapeError(
            "take_along_axis: indices must have same ndim as input, got " +
            std::to_string(indices.ndim()) + " vs " + std::to_string(ndim));
    }

    // Check shapes are compatible (same except possibly along axis)
    for (int d = 0; d < ndim; ++d) {
        if (d != axis && indices.shape()[d] != input.shape()[d]) {
            throw ShapeError(
                "take_along_axis: indices shape must match input shape except "
                "along axis");
        }
    }

    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();
    Tensor indices_cpu =
        indices.device() == Device::CPU ? indices : indices.cpu();

    if (indices_cpu.dtype() != DType::Int64) {
        indices_cpu = indices_cpu.astype(DType::Int64);
    }

    // Output has same shape as indices
    Tensor result(indices.shape(), input.dtype(), Device::CPU);

    const int64_t *idx_data = indices_cpu.typed_data<int64_t>();
    size_t axis_size = input.shape()[axis];

    // Iterate over all positions in indices
    size_t total = indices_cpu.size();
    std::vector<size_t> coords(ndim, 0);

    switch (input.dtype()) {
#define TAKE_ALONG_CASE(DTYPE, CTYPE)                                          \
    case DTYPE: {                                                              \
        for (size_t i = 0; i < total; ++i) {                                   \
            int64_t idx = idx_data[i];                                         \
            if (idx < 0)                                                       \
                idx += static_cast<int64_t>(axis_size);                        \
            if (idx < 0 || idx >= static_cast<int64_t>(axis_size)) {           \
                throw IndexError("take_along_axis: index out of bounds");      \
            }                                                                  \
            std::vector<size_t> src_coords = coords;                           \
            src_coords[axis] = static_cast<size_t>(idx);                       \
            result.set_item<CTYPE>(coords, input_cpu.item<CTYPE>(src_coords)); \
            ShapeUtils::increment_coords(coords, indices.shape());             \
        }                                                                      \
        break;                                                                 \
    }
        TAKE_ALONG_CASE(DType::Float32, float)
        TAKE_ALONG_CASE(DType::Float64, double)
        TAKE_ALONG_CASE(DType::Int32, int32_t)
        TAKE_ALONG_CASE(DType::Int64, int64_t)
#undef TAKE_ALONG_CASE
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "take_along_axis");
    }

    return input.device() == Device::GPU ? result.gpu() : result;
}

Tensor put_along_axis(const Tensor &input, const Tensor &indices,
                      const Tensor &values, int axis) {
    // Validate indices dtype
    if (!is_integer_dtype(indices.dtype())) {
        throw TypeError("put_along_axis: indices must be integer type, got " +
                        dtype_name(indices.dtype()));
    }

    int ndim = static_cast<int>(input.ndim());
    if (axis < 0)
        axis += ndim;
    if (axis < 0 || axis >= ndim) {
        throw ShapeError::invalid_axis(axis, ndim);
    }

    // indices and values must have same shape
    if (indices.shape() != values.shape()) {
        throw ShapeError(
            "put_along_axis: indices and values must have same shape");
    }

    // indices must have same ndim as input
    if (indices.ndim() != input.ndim()) {
        throw ShapeError(
            "put_along_axis: indices must have same ndim as input");
    }

    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();
    Tensor indices_cpu =
        indices.device() == Device::CPU ? indices : indices.cpu();
    Tensor values_cpu = values.device() == Device::CPU ? values : values.cpu();

    if (indices_cpu.dtype() != DType::Int64) {
        indices_cpu = indices_cpu.astype(DType::Int64);
    }

    // Create copy of input to modify
    Tensor result = input_cpu.clone();

    const int64_t *idx_data = indices_cpu.typed_data<int64_t>();
    size_t axis_size = input.shape()[axis];

    size_t total = indices_cpu.size();
    std::vector<size_t> coords(ndim, 0);

    switch (input.dtype()) {
#define PUT_ALONG_CASE(DTYPE, CTYPE)                                           \
    case DTYPE: {                                                              \
        for (size_t i = 0; i < total; ++i) {                                   \
            int64_t idx = idx_data[i];                                         \
            if (idx < 0)                                                       \
                idx += static_cast<int64_t>(axis_size);                        \
            if (idx < 0 || idx >= static_cast<int64_t>(axis_size)) {           \
                throw IndexError("put_along_axis: index out of bounds");       \
            }                                                                  \
            std::vector<size_t> dst_coords = coords;                           \
            dst_coords[axis] = static_cast<size_t>(idx);                       \
            result.set_item<CTYPE>(dst_coords,                                 \
                                   values_cpu.item<CTYPE>(coords));            \
            ShapeUtils::increment_coords(coords, indices.shape());             \
        }                                                                      \
        break;                                                                 \
    }
        PUT_ALONG_CASE(DType::Float32, float)
        PUT_ALONG_CASE(DType::Float64, double)
        PUT_ALONG_CASE(DType::Int32, int32_t)
        PUT_ALONG_CASE(DType::Int64, int64_t)
#undef PUT_ALONG_CASE
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "put_along_axis");
    }

    return input.device() == Device::GPU ? result.gpu() : result;
}

// ============================================================================
// Shape manipulation operations
// ============================================================================

std::vector<Tensor> meshgrid(const std::vector<Tensor> &tensors,
                             const std::string &indexing) {
    if (tensors.empty()) {
        return {};
    }

    // Validate all tensors are 1D
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].ndim() != 1) {
            throw ShapeError("meshgrid: all input tensors must be 1D, got " +
                             std::to_string(tensors[i].ndim()) + "D at index " +
                             std::to_string(i));
        }
    }

    if (indexing != "xy" && indexing != "ij") {
        throw ValueError("meshgrid: indexing must be 'xy' or 'ij', got '" +
                         indexing + "'");
    }

    size_t ndim = tensors.size();

    // Build output shape: [len(t0), len(t1), ..., len(tn)]
    Shape output_shape;
    for (const auto &t : tensors) {
        output_shape.push_back(t.shape()[0]);
    }

    // For "xy" indexing (Cartesian), swap first two dimensions
    if (indexing == "xy" && ndim >= 2) {
        std::swap(output_shape[0], output_shape[1]);
    }

    std::vector<Tensor> result;
    result.reserve(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        // Determine which dimension this tensor expands along
        size_t expand_dim = i;
        if (indexing == "xy" && ndim >= 2) {
            if (i == 0)
                expand_dim = 1;
            else if (i == 1)
                expand_dim = 0;
        }

        // Reshape tensor to have 1s in all dimensions except expand_dim
        Shape reshape_shape(ndim, 1);
        reshape_shape[expand_dim] = tensors[i].shape()[0];

        Tensor reshaped = tensors[i].reshape(reshape_shape);
        Tensor expanded = reshaped.expand(output_shape);

        result.push_back(expanded);
    }

    return result;
}

Tensor pad(const Tensor &input,
           const std::vector<std::pair<size_t, size_t>> &pad_width,
           const std::string &mode, double value) {
    if (pad_width.size() != input.ndim()) {
        throw ShapeError("pad: pad_width size (" +
                         std::to_string(pad_width.size()) +
                         ") must match tensor dimensions (" +
                         std::to_string(input.ndim()) + ")");
    }

    if (mode != "constant" && mode != "reflect" && mode != "replicate" &&
        mode != "circular") {
        throw ValueError(
            "pad: mode must be 'constant', 'reflect', 'replicate', or "
            "'circular', got '" +
            mode + "'");
    }

    // Calculate output shape
    Shape output_shape;
    for (size_t i = 0; i < input.ndim(); ++i) {
        output_shape.push_back(pad_width[i].first + input.shape()[i] +
                               pad_width[i].second);
    }

    // Create output tensor
    Tensor result = Tensor::zeros(output_shape, input.dtype(), input.device(),
                                  input.memory_order());

    // For GPU tensors, move to CPU for padding, then back
    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();
    Tensor result_cpu = result.device() == Device::CPU ? result : result.cpu();

    // Fill with constant value if mode is constant
    if (mode == "constant" && value != 0.0) {
        switch (result_cpu.dtype()) {
        case DType::Float32:
            result_cpu.fill<float>(static_cast<float>(value));
            break;
        case DType::Float64:
            result_cpu.fill<double>(value);
            break;
        case DType::Int32:
            result_cpu.fill<int32_t>(static_cast<int32_t>(value));
            break;
        case DType::Int64:
            result_cpu.fill<int64_t>(static_cast<int64_t>(value));
            break;
        default:
            break;
        }
    }

    // Helper to compute source index based on padding mode
    auto get_source_idx = [&](int64_t idx, size_t dim_size,
                              const std::string &pad_mode) -> int64_t {
        if (idx >= 0 && idx < static_cast<int64_t>(dim_size)) {
            return idx;
        }

        if (pad_mode == "constant") {
            return -1; // Signal to use constant value
        } else if (pad_mode == "reflect") {
            // Reflect at boundaries (excluding boundary)
            while (idx < 0 || idx >= static_cast<int64_t>(dim_size)) {
                if (idx < 0) {
                    idx = -idx;
                }
                if (idx >= static_cast<int64_t>(dim_size)) {
                    idx = 2 * static_cast<int64_t>(dim_size) - idx - 2;
                }
            }
            return idx;
        } else if (pad_mode == "replicate") {
            // Clamp to boundary
            return std::max<int64_t>(
                0, std::min<int64_t>(idx, static_cast<int64_t>(dim_size) - 1));
        } else { // circular
            // Wrap around
            idx = idx % static_cast<int64_t>(dim_size);
            if (idx < 0)
                idx += static_cast<int64_t>(dim_size);
            return idx;
        }
    };

    // Copy data with padding
    size_t total_size = ShapeUtils::size(output_shape);
    std::vector<size_t> out_coords(input.ndim(), 0);

    for (size_t i = 0; i < total_size; ++i) {
        // Compute source coordinates
        std::vector<size_t> src_coords(input.ndim());
        bool use_constant = false;

        for (size_t d = 0; d < input.ndim(); ++d) {
            int64_t src_idx = static_cast<int64_t>(out_coords[d]) -
                              static_cast<int64_t>(pad_width[d].first);
            int64_t mapped_idx =
                get_source_idx(src_idx, input.shape()[d], mode);
            if (mapped_idx < 0) {
                use_constant = true;
                break;
            }
            src_coords[d] = static_cast<size_t>(mapped_idx);
        }

        if (!use_constant) {
            // Copy value from source
            switch (result_cpu.dtype()) {
            case DType::Float32:
                result_cpu.set_item<float>(out_coords,
                                           input_cpu.item<float>(src_coords));
                break;
            case DType::Float64:
                result_cpu.set_item<double>(out_coords,
                                            input_cpu.item<double>(src_coords));
                break;
            case DType::Int32:
                result_cpu.set_item<int32_t>(
                    out_coords, input_cpu.item<int32_t>(src_coords));
                break;
            case DType::Int64:
                result_cpu.set_item<int64_t>(
                    out_coords, input_cpu.item<int64_t>(src_coords));
                break;
            default:
                break;
            }
        }

        // Increment output coordinates
        for (int d = static_cast<int>(input.ndim()) - 1; d >= 0; --d) {
            if (++out_coords[d] < output_shape[d])
                break;
            out_coords[d] = 0;
        }
    }

    if (input.device() == Device::GPU) {
        return result_cpu.gpu();
    }
    return result_cpu;
}

Tensor atleast_1d(const Tensor &tensor) {
    if (tensor.ndim() >= 1) {
        return tensor;
    }
    // 0D tensor -> 1D with shape [1]
    return tensor.reshape({1});
}

Tensor atleast_2d(const Tensor &tensor) {
    if (tensor.ndim() >= 2) {
        return tensor;
    }
    if (tensor.ndim() == 1) {
        // [n] -> [1, n]
        return tensor.unsqueeze(0);
    }
    // 0D -> [1, 1]
    return tensor.reshape({1, 1});
}

Tensor atleast_3d(const Tensor &tensor) {
    if (tensor.ndim() >= 3) {
        return tensor;
    }
    if (tensor.ndim() == 2) {
        // [m, n] -> [m, n, 1]
        return tensor.unsqueeze(2);
    }
    if (tensor.ndim() == 1) {
        // [n] -> [1, n, 1]
        return tensor.unsqueeze(0).unsqueeze(2);
    }
    // 0D -> [1, 1, 1]
    return tensor.reshape({1, 1, 1});
}

std::vector<Tensor> atleast_1d(const std::vector<Tensor> &tensors) {
    std::vector<Tensor> result;
    result.reserve(tensors.size());
    for (const auto &t : tensors) {
        result.push_back(atleast_1d(t));
    }
    return result;
}

std::vector<Tensor> atleast_2d(const std::vector<Tensor> &tensors) {
    std::vector<Tensor> result;
    result.reserve(tensors.size());
    for (const auto &t : tensors) {
        result.push_back(atleast_2d(t));
    }
    return result;
}

std::vector<Tensor> atleast_3d(const std::vector<Tensor> &tensors) {
    std::vector<Tensor> result;
    result.reserve(tensors.size());
    for (const auto &t : tensors) {
        result.push_back(atleast_3d(t));
    }
    return result;
}

// ============================================================================
// Pooling operations
// ============================================================================

namespace {

// Helper to compute output size for pooling
size_t pool_output_size(size_t input_size, int kernel_size, int stride,
                        int padding) {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

} // namespace

Tensor max_pool1d(const Tensor &input, int kernel_size, int stride,
                  int padding) {
    if (stride <= 0)
        stride = kernel_size;
    if (input.ndim() < 2 || input.ndim() > 3) {
        throw ShapeError(
            "max_pool1d: expected 2D or 3D input (C,L) or (N,C,L), got " +
            std::to_string(input.ndim()) + "D");
    }

    bool has_batch = (input.ndim() == 3);
    size_t batch_size = has_batch ? input.shape()[0] : 1;
    size_t channels = has_batch ? input.shape()[1] : input.shape()[0];
    size_t length = has_batch ? input.shape()[2] : input.shape()[1];

    size_t out_length = pool_output_size(length, kernel_size, stride, padding);

    Shape out_shape = has_batch ? Shape{batch_size, channels, out_length}
                                : Shape{channels, out_length};

    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();
    Tensor result(out_shape, input.dtype(), Device::CPU);

    // Process each channel
    switch (input.dtype()) {
#define MAX_POOL1D_CASE(DTYPE, CTYPE, MINVAL)                                  \
    case DTYPE: {                                                              \
        for (size_t b = 0; b < batch_size; ++b) {                              \
            for (size_t c = 0; c < channels; ++c) {                            \
                for (size_t o = 0; o < out_length; ++o) {                      \
                    CTYPE max_val = MINVAL;                                    \
                    int start = static_cast<int>(o) * stride - padding;        \
                    for (int k = 0; k < kernel_size; ++k) {                    \
                        int idx = start + k;                                   \
                        if (idx >= 0 && idx < static_cast<int>(length)) {      \
                            std::vector<size_t> coords =                       \
                                has_batch                                      \
                                    ? std::vector<size_t>{b, c,                \
                                                          static_cast<size_t>( \
                                                              idx)}            \
                                    : std::vector<size_t>{                     \
                                          c, static_cast<size_t>(idx)};        \
                            CTYPE val = input_cpu.item<CTYPE>(coords);         \
                            if (val > max_val)                                 \
                                max_val = val;                                 \
                        }                                                      \
                    }                                                          \
                    std::vector<size_t> out_coords =                           \
                        has_batch ? std::vector<size_t>{b, c, o}               \
                                  : std::vector<size_t>{c, o};                 \
                    result.set_item<CTYPE>(out_coords, max_val);               \
                }                                                              \
            }                                                                  \
        }                                                                      \
        break;                                                                 \
    }
        MAX_POOL1D_CASE(DType::Float32, float,
                        -std::numeric_limits<float>::infinity())
        MAX_POOL1D_CASE(DType::Float64, double,
                        -std::numeric_limits<double>::infinity())
#undef MAX_POOL1D_CASE
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "max_pool1d");
    }

    return input.device() == Device::GPU ? result.gpu() : result;
}

Tensor avg_pool1d(const Tensor &input, int kernel_size, int stride, int padding,
                  bool count_include_pad) {
    if (stride <= 0)
        stride = kernel_size;
    if (input.ndim() < 2 || input.ndim() > 3) {
        throw ShapeError("avg_pool1d: expected 2D or 3D input, got " +
                         std::to_string(input.ndim()) + "D");
    }

    bool has_batch = (input.ndim() == 3);
    size_t batch_size = has_batch ? input.shape()[0] : 1;
    size_t channels = has_batch ? input.shape()[1] : input.shape()[0];
    size_t length = has_batch ? input.shape()[2] : input.shape()[1];

    size_t out_length = pool_output_size(length, kernel_size, stride, padding);

    Shape out_shape = has_batch ? Shape{batch_size, channels, out_length}
                                : Shape{channels, out_length};

    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();
    Tensor result(out_shape, input.dtype(), Device::CPU);

    switch (input.dtype()) {
#define AVG_POOL1D_CASE(DTYPE, CTYPE)                                          \
    case DTYPE: {                                                              \
        for (size_t b = 0; b < batch_size; ++b) {                              \
            for (size_t c = 0; c < channels; ++c) {                            \
                for (size_t o = 0; o < out_length; ++o) {                      \
                    CTYPE sum = 0;                                             \
                    int count = 0;                                             \
                    int start = static_cast<int>(o) * stride - padding;        \
                    for (int k = 0; k < kernel_size; ++k) {                    \
                        int idx = start + k;                                   \
                        if (idx >= 0 && idx < static_cast<int>(length)) {      \
                            std::vector<size_t> coords =                       \
                                has_batch                                      \
                                    ? std::vector<size_t>{b, c,                \
                                                          static_cast<size_t>( \
                                                              idx)}            \
                                    : std::vector<size_t>{                     \
                                          c, static_cast<size_t>(idx)};        \
                            sum += input_cpu.item<CTYPE>(coords);              \
                            count++;                                           \
                        } else if (count_include_pad) {                        \
                            count++;                                           \
                        }                                                      \
                    }                                                          \
                    CTYPE avg = count > 0 ? sum / count : CTYPE(0);            \
                    std::vector<size_t> out_coords =                           \
                        has_batch ? std::vector<size_t>{b, c, o}               \
                                  : std::vector<size_t>{c, o};                 \
                    result.set_item<CTYPE>(out_coords, avg);                   \
                }                                                              \
            }                                                                  \
        }                                                                      \
        break;                                                                 \
    }
        AVG_POOL1D_CASE(DType::Float32, float)
        AVG_POOL1D_CASE(DType::Float64, double)
#undef AVG_POOL1D_CASE
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "avg_pool1d");
    }

    return input.device() == Device::GPU ? result.gpu() : result;
}

Tensor max_pool2d(const Tensor &input, const std::vector<int> &kernel_size,
                  const std::vector<int> &stride,
                  const std::vector<int> &padding) {
    if (kernel_size.size() != 2) {
        throw ValueError("max_pool2d: kernel_size must have 2 elements");
    }

    std::vector<int> actual_stride = stride.empty() ? kernel_size : stride;
    std::vector<int> actual_padding =
        padding.empty() ? std::vector<int>{0, 0} : padding;

    if (actual_stride.size() != 2 || actual_padding.size() != 2) {
        throw ValueError("max_pool2d: stride and padding must have 2 elements");
    }

    if (input.ndim() < 3 || input.ndim() > 4) {
        throw ShapeError(
            "max_pool2d: expected 3D or 4D input (C,H,W) or (N,C,H,W), got " +
            std::to_string(input.ndim()) + "D");
    }

    bool has_batch = (input.ndim() == 4);
    size_t batch_size = has_batch ? input.shape()[0] : 1;
    size_t channels = has_batch ? input.shape()[1] : input.shape()[0];
    size_t height = has_batch ? input.shape()[2] : input.shape()[1];
    size_t width = has_batch ? input.shape()[3] : input.shape()[2];

    size_t out_h = pool_output_size(height, kernel_size[0], actual_stride[0],
                                    actual_padding[0]);
    size_t out_w = pool_output_size(width, kernel_size[1], actual_stride[1],
                                    actual_padding[1]);

    Shape out_shape = has_batch ? Shape{batch_size, channels, out_h, out_w}
                                : Shape{channels, out_h, out_w};

    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();
    Tensor result(out_shape, input.dtype(), Device::CPU);

    switch (input.dtype()) {
#define MAX_POOL2D_CASE(DTYPE, CTYPE, MINVAL)                                  \
    case DTYPE: {                                                              \
        for (size_t b = 0; b < batch_size; ++b) {                              \
            for (size_t c = 0; c < channels; ++c) {                            \
                for (size_t oh = 0; oh < out_h; ++oh) {                        \
                    for (size_t ow = 0; ow < out_w; ++ow) {                    \
                        CTYPE max_val = MINVAL;                                \
                        int h_start =                                          \
                            static_cast<int>(oh) * actual_stride[0] -          \
                            actual_padding[0];                                 \
                        int w_start =                                          \
                            static_cast<int>(ow) * actual_stride[1] -          \
                            actual_padding[1];                                 \
                        for (int kh = 0; kh < kernel_size[0]; ++kh) {          \
                            for (int kw = 0; kw < kernel_size[1]; ++kw) {      \
                                int h = h_start + kh;                          \
                                int w = w_start + kw;                          \
                                if (h >= 0 && h < static_cast<int>(height) &&  \
                                    w >= 0 && w < static_cast<int>(width)) {   \
                                    std::vector<size_t> coords =               \
                                        has_batch                              \
                                            ? std::vector<                     \
                                                  size_t>{b, c,                \
                                                          static_cast<size_t>( \
                                                              h),              \
                                                          static_cast<size_t>( \
                                                              w)}              \
                                            : std::vector<size_t>{             \
                                                  c, static_cast<size_t>(h),   \
                                                  static_cast<size_t>(w)};     \
                                    CTYPE val = input_cpu.item<CTYPE>(coords); \
                                    if (val > max_val)                         \
                                        max_val = val;                         \
                                }                                              \
                            }                                                  \
                        }                                                      \
                        std::vector<size_t> out_coords =                       \
                            has_batch ? std::vector<size_t>{b, c, oh, ow}      \
                                      : std::vector<size_t>{c, oh, ow};        \
                        result.set_item<CTYPE>(out_coords, max_val);           \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        break;                                                                 \
    }
        MAX_POOL2D_CASE(DType::Float32, float,
                        -std::numeric_limits<float>::infinity())
        MAX_POOL2D_CASE(DType::Float64, double,
                        -std::numeric_limits<double>::infinity())
#undef MAX_POOL2D_CASE
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "max_pool2d");
    }

    return input.device() == Device::GPU ? result.gpu() : result;
}

Tensor avg_pool2d(const Tensor &input, const std::vector<int> &kernel_size,
                  const std::vector<int> &stride,
                  const std::vector<int> &padding, bool count_include_pad) {
    if (kernel_size.size() != 2) {
        throw ValueError("avg_pool2d: kernel_size must have 2 elements");
    }

    std::vector<int> actual_stride = stride.empty() ? kernel_size : stride;
    std::vector<int> actual_padding =
        padding.empty() ? std::vector<int>{0, 0} : padding;

    if (input.ndim() < 3 || input.ndim() > 4) {
        throw ShapeError("avg_pool2d: expected 3D or 4D input, got " +
                         std::to_string(input.ndim()) + "D");
    }

    bool has_batch = (input.ndim() == 4);
    size_t batch_size = has_batch ? input.shape()[0] : 1;
    size_t channels = has_batch ? input.shape()[1] : input.shape()[0];
    size_t height = has_batch ? input.shape()[2] : input.shape()[1];
    size_t width = has_batch ? input.shape()[3] : input.shape()[2];

    size_t out_h = pool_output_size(height, kernel_size[0], actual_stride[0],
                                    actual_padding[0]);
    size_t out_w = pool_output_size(width, kernel_size[1], actual_stride[1],
                                    actual_padding[1]);

    Shape out_shape = has_batch ? Shape{batch_size, channels, out_h, out_w}
                                : Shape{channels, out_h, out_w};

    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();
    Tensor result(out_shape, input.dtype(), Device::CPU);

    switch (input.dtype()) {
#define AVG_POOL2D_CASE(DTYPE, CTYPE)                                          \
    case DTYPE: {                                                              \
        for (size_t b = 0; b < batch_size; ++b) {                              \
            for (size_t c = 0; c < channels; ++c) {                            \
                for (size_t oh = 0; oh < out_h; ++oh) {                        \
                    for (size_t ow = 0; ow < out_w; ++ow) {                    \
                        CTYPE sum = 0;                                         \
                        int count = 0;                                         \
                        int h_start =                                          \
                            static_cast<int>(oh) * actual_stride[0] -          \
                            actual_padding[0];                                 \
                        int w_start =                                          \
                            static_cast<int>(ow) * actual_stride[1] -          \
                            actual_padding[1];                                 \
                        for (int kh = 0; kh < kernel_size[0]; ++kh) {          \
                            for (int kw = 0; kw < kernel_size[1]; ++kw) {      \
                                int h = h_start + kh;                          \
                                int w = w_start + kw;                          \
                                if (h >= 0 && h < static_cast<int>(height) &&  \
                                    w >= 0 && w < static_cast<int>(width)) {   \
                                    std::vector<size_t> coords =               \
                                        has_batch                              \
                                            ? std::vector<                     \
                                                  size_t>{b, c,                \
                                                          static_cast<size_t>( \
                                                              h),              \
                                                          static_cast<size_t>( \
                                                              w)}              \
                                            : std::vector<size_t>{             \
                                                  c, static_cast<size_t>(h),   \
                                                  static_cast<size_t>(w)};     \
                                    sum += input_cpu.item<CTYPE>(coords);      \
                                    count++;                                   \
                                } else if (count_include_pad) {                \
                                    count++;                                   \
                                }                                              \
                            }                                                  \
                        }                                                      \
                        CTYPE avg = count > 0 ? sum / count : CTYPE(0);        \
                        std::vector<size_t> out_coords =                       \
                            has_batch ? std::vector<size_t>{b, c, oh, ow}      \
                                      : std::vector<size_t>{c, oh, ow};        \
                        result.set_item<CTYPE>(out_coords, avg);               \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        break;                                                                 \
    }
        AVG_POOL2D_CASE(DType::Float32, float)
        AVG_POOL2D_CASE(DType::Float64, double)
#undef AVG_POOL2D_CASE
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "avg_pool2d");
    }

    return input.device() == Device::GPU ? result.gpu() : result;
}

Tensor max_pool3d(const Tensor &input, const std::vector<int> &kernel_size,
                  const std::vector<int> &stride,
                  const std::vector<int> &padding) {
    if (kernel_size.size() != 3) {
        throw ValueError("max_pool3d: kernel_size must have 3 elements");
    }

    std::vector<int> actual_stride = stride.empty() ? kernel_size : stride;
    std::vector<int> actual_padding =
        padding.empty() ? std::vector<int>{0, 0, 0} : padding;

    if (input.ndim() < 4 || input.ndim() > 5) {
        throw ShapeError("max_pool3d: expected 4D or 5D input, got " +
                         std::to_string(input.ndim()) + "D");
    }

    bool has_batch = (input.ndim() == 5);
    size_t batch_size = has_batch ? input.shape()[0] : 1;
    size_t channels = has_batch ? input.shape()[1] : input.shape()[0];
    size_t depth = has_batch ? input.shape()[2] : input.shape()[1];
    size_t height = has_batch ? input.shape()[3] : input.shape()[2];
    size_t width = has_batch ? input.shape()[4] : input.shape()[3];

    size_t out_d = pool_output_size(depth, kernel_size[0], actual_stride[0],
                                    actual_padding[0]);
    size_t out_h = pool_output_size(height, kernel_size[1], actual_stride[1],
                                    actual_padding[1]);
    size_t out_w = pool_output_size(width, kernel_size[2], actual_stride[2],
                                    actual_padding[2]);

    Shape out_shape = has_batch
                          ? Shape{batch_size, channels, out_d, out_h, out_w}
                          : Shape{channels, out_d, out_h, out_w};

    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();
    Tensor result(out_shape, input.dtype(), Device::CPU);

    switch (input.dtype()) {
#define MAX_POOL3D_CASE(DTYPE, CTYPE, MINVAL)                                  \
    case DTYPE: {                                                              \
        for (size_t b = 0; b < batch_size; ++b) {                              \
            for (size_t c = 0; c < channels; ++c) {                            \
                for (size_t od = 0; od < out_d; ++od) {                        \
                    for (size_t oh = 0; oh < out_h; ++oh) {                    \
                        for (size_t ow = 0; ow < out_w; ++ow) {                \
                            CTYPE max_val = MINVAL;                            \
                            int d_start =                                      \
                                static_cast<int>(od) * actual_stride[0] -      \
                                actual_padding[0];                             \
                            int h_start =                                      \
                                static_cast<int>(oh) * actual_stride[1] -      \
                                actual_padding[1];                             \
                            int w_start =                                      \
                                static_cast<int>(ow) * actual_stride[2] -      \
                                actual_padding[2];                             \
                            for (int kd = 0; kd < kernel_size[0]; ++kd) {      \
                                for (int kh = 0; kh < kernel_size[1]; ++kh) {  \
                                    for (int kw = 0; kw < kernel_size[2];      \
                                         ++kw) {                               \
                                        int d = d_start + kd;                  \
                                        int h = h_start + kh;                  \
                                        int w = w_start + kw;                  \
                                        if (d >= 0 &&                          \
                                            d < static_cast<int>(depth) &&     \
                                            h >= 0 &&                          \
                                            h < static_cast<int>(height) &&    \
                                            w >= 0 &&                          \
                                            w < static_cast<int>(width)) {     \
                                            std::vector<size_t> coords =       \
                                                has_batch                      \
                                                    ? std::vector<             \
                                                          size_t>{b, c,        \
                                                                  static_cast< \
                                                                      size_t>( \
                                                                      d),      \
                                                                  static_cast< \
                                                                      size_t>( \
                                                                      h),      \
                                                                  static_cast< \
                                                                      size_t>( \
                                                                      w)}      \
                                                    : std::vector<size_t>{     \
                                                          c,                   \
                                                          static_cast<size_t>( \
                                                              d),              \
                                                          static_cast<size_t>( \
                                                              h),              \
                                                          static_cast<size_t>( \
                                                              w)};             \
                                            CTYPE val =                        \
                                                input_cpu.item<CTYPE>(coords); \
                                            if (val > max_val)                 \
                                                max_val = val;                 \
                                        }                                      \
                                    }                                          \
                                }                                              \
                            }                                                  \
                            std::vector<size_t> out_coords =                   \
                                has_batch                                      \
                                    ? std::vector<size_t>{b, c, od, oh, ow}    \
                                    : std::vector<size_t>{c, od, oh, ow};      \
                            result.set_item<CTYPE>(out_coords, max_val);       \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        break;                                                                 \
    }
        MAX_POOL3D_CASE(DType::Float32, float,
                        -std::numeric_limits<float>::infinity())
        MAX_POOL3D_CASE(DType::Float64, double,
                        -std::numeric_limits<double>::infinity())
#undef MAX_POOL3D_CASE
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "max_pool3d");
    }

    return input.device() == Device::GPU ? result.gpu() : result;
}

Tensor avg_pool3d(const Tensor &input, const std::vector<int> &kernel_size,
                  const std::vector<int> &stride,
                  const std::vector<int> &padding, bool count_include_pad) {
    if (kernel_size.size() != 3) {
        throw ValueError("avg_pool3d: kernel_size must have 3 elements");
    }

    std::vector<int> actual_stride = stride.empty() ? kernel_size : stride;
    std::vector<int> actual_padding =
        padding.empty() ? std::vector<int>{0, 0, 0} : padding;

    if (input.ndim() < 4 || input.ndim() > 5) {
        throw ShapeError("avg_pool3d: expected 4D or 5D input, got " +
                         std::to_string(input.ndim()) + "D");
    }

    bool has_batch = (input.ndim() == 5);
    size_t batch_size = has_batch ? input.shape()[0] : 1;
    size_t channels = has_batch ? input.shape()[1] : input.shape()[0];
    size_t depth = has_batch ? input.shape()[2] : input.shape()[1];
    size_t height = has_batch ? input.shape()[3] : input.shape()[2];
    size_t width = has_batch ? input.shape()[4] : input.shape()[3];

    size_t out_d = pool_output_size(depth, kernel_size[0], actual_stride[0],
                                    actual_padding[0]);
    size_t out_h = pool_output_size(height, kernel_size[1], actual_stride[1],
                                    actual_padding[1]);
    size_t out_w = pool_output_size(width, kernel_size[2], actual_stride[2],
                                    actual_padding[2]);

    Shape out_shape = has_batch
                          ? Shape{batch_size, channels, out_d, out_h, out_w}
                          : Shape{channels, out_d, out_h, out_w};

    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();
    Tensor result(out_shape, input.dtype(), Device::CPU);

    switch (input.dtype()) {
#define AVG_POOL3D_CASE(DTYPE, CTYPE)                                          \
    case DTYPE: {                                                              \
        for (size_t b = 0; b < batch_size; ++b) {                              \
            for (size_t c = 0; c < channels; ++c) {                            \
                for (size_t od = 0; od < out_d; ++od) {                        \
                    for (size_t oh = 0; oh < out_h; ++oh) {                    \
                        for (size_t ow = 0; ow < out_w; ++ow) {                \
                            CTYPE sum = 0;                                     \
                            int count = 0;                                     \
                            int d_start =                                      \
                                static_cast<int>(od) * actual_stride[0] -      \
                                actual_padding[0];                             \
                            int h_start =                                      \
                                static_cast<int>(oh) * actual_stride[1] -      \
                                actual_padding[1];                             \
                            int w_start =                                      \
                                static_cast<int>(ow) * actual_stride[2] -      \
                                actual_padding[2];                             \
                            for (int kd = 0; kd < kernel_size[0]; ++kd) {      \
                                for (int kh = 0; kh < kernel_size[1]; ++kh) {  \
                                    for (int kw = 0; kw < kernel_size[2];      \
                                         ++kw) {                               \
                                        int d = d_start + kd;                  \
                                        int h = h_start + kh;                  \
                                        int w = w_start + kw;                  \
                                        if (d >= 0 &&                          \
                                            d < static_cast<int>(depth) &&     \
                                            h >= 0 &&                          \
                                            h < static_cast<int>(height) &&    \
                                            w >= 0 &&                          \
                                            w < static_cast<int>(width)) {     \
                                            std::vector<size_t> coords =       \
                                                has_batch                      \
                                                    ? std::vector<             \
                                                          size_t>{b, c,        \
                                                                  static_cast< \
                                                                      size_t>( \
                                                                      d),      \
                                                                  static_cast< \
                                                                      size_t>( \
                                                                      h),      \
                                                                  static_cast< \
                                                                      size_t>( \
                                                                      w)}      \
                                                    : std::vector<size_t>{     \
                                                          c,                   \
                                                          static_cast<size_t>( \
                                                              d),              \
                                                          static_cast<size_t>( \
                                                              h),              \
                                                          static_cast<size_t>( \
                                                              w)};             \
                                            sum +=                             \
                                                input_cpu.item<CTYPE>(coords); \
                                            count++;                           \
                                        } else if (count_include_pad) {        \
                                            count++;                           \
                                        }                                      \
                                    }                                          \
                                }                                              \
                            }                                                  \
                            CTYPE avg = count > 0 ? sum / count : CTYPE(0);    \
                            std::vector<size_t> out_coords =                   \
                                has_batch                                      \
                                    ? std::vector<size_t>{b, c, od, oh, ow}    \
                                    : std::vector<size_t>{c, od, oh, ow};      \
                            result.set_item<CTYPE>(out_coords, avg);           \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        break;                                                                 \
    }
        AVG_POOL3D_CASE(DType::Float32, float)
        AVG_POOL3D_CASE(DType::Float64, double)
#undef AVG_POOL3D_CASE
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "avg_pool3d");
    }

    return input.device() == Device::GPU ? result.gpu() : result;
}

Tensor adaptive_max_pool1d(const Tensor &input, int output_size) {
    if (input.ndim() < 2 || input.ndim() > 3) {
        throw ShapeError("adaptive_max_pool1d: expected 2D or 3D input, got " +
                         std::to_string(input.ndim()) + "D");
    }

    bool has_batch = (input.ndim() == 3);
    size_t input_length = has_batch ? input.shape()[2] : input.shape()[1];

    // Calculate effective kernel size and stride
    int kernel_size = (input_length + output_size - 1) / output_size;
    int stride = input_length / output_size;

    return max_pool1d(input, kernel_size, stride, 0);
}

Tensor adaptive_avg_pool1d(const Tensor &input, int output_size) {
    if (input.ndim() < 2 || input.ndim() > 3) {
        throw ShapeError("adaptive_avg_pool1d: expected 2D or 3D input, got " +
                         std::to_string(input.ndim()) + "D");
    }

    bool has_batch = (input.ndim() == 3);
    size_t input_length = has_batch ? input.shape()[2] : input.shape()[1];

    int kernel_size = (input_length + output_size - 1) / output_size;
    int stride = input_length / output_size;

    return avg_pool1d(input, kernel_size, stride, 0, false);
}

Tensor adaptive_max_pool2d(const Tensor &input,
                           const std::vector<int> &output_size) {
    if (output_size.size() != 2) {
        throw ValueError(
            "adaptive_max_pool2d: output_size must have 2 elements");
    }

    if (input.ndim() < 3 || input.ndim() > 4) {
        throw ShapeError("adaptive_max_pool2d: expected 3D or 4D input, got " +
                         std::to_string(input.ndim()) + "D");
    }

    bool has_batch = (input.ndim() == 4);
    size_t input_h = has_batch ? input.shape()[2] : input.shape()[1];
    size_t input_w = has_batch ? input.shape()[3] : input.shape()[2];

    int kernel_h = (input_h + output_size[0] - 1) / output_size[0];
    int kernel_w = (input_w + output_size[1] - 1) / output_size[1];
    int stride_h = input_h / output_size[0];
    int stride_w = input_w / output_size[1];

    return max_pool2d(input, {kernel_h, kernel_w}, {stride_h, stride_w},
                      {0, 0});
}

Tensor adaptive_avg_pool2d(const Tensor &input,
                           const std::vector<int> &output_size) {
    if (output_size.size() != 2) {
        throw ValueError(
            "adaptive_avg_pool2d: output_size must have 2 elements");
    }

    if (input.ndim() < 3 || input.ndim() > 4) {
        throw ShapeError("adaptive_avg_pool2d: expected 3D or 4D input, got " +
                         std::to_string(input.ndim()) + "D");
    }

    bool has_batch = (input.ndim() == 4);
    size_t input_h = has_batch ? input.shape()[2] : input.shape()[1];
    size_t input_w = has_batch ? input.shape()[3] : input.shape()[2];

    int kernel_h = (input_h + output_size[0] - 1) / output_size[0];
    int kernel_w = (input_w + output_size[1] - 1) / output_size[1];
    int stride_h = input_h / output_size[0];
    int stride_w = input_w / output_size[1];

    return avg_pool2d(input, {kernel_h, kernel_w}, {stride_h, stride_w}, {0, 0},
                      false);
}

} // namespace ops
} // namespace axiom