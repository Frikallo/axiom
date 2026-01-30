#import "axiom/expr/base.hpp"
#import "axiom/expr/binary.hpp"
#import "axiom/expr/matmul.hpp"
#import "axiom/expr/traits.hpp"
#import "axiom/expr/unary.hpp"
#import "axiom/operations.hpp"
#import "axiom/tensor.hpp"

// Include Metal backend headers
#import "metal/metal_storage.hpp"
#import "metal/metal_common.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <map>
#include <vector>

namespace axiom {
namespace expr {

// ============================================================================
// MPSGraph Expression Builder
// Builds a single MPSGraph for an entire expression tree
// ============================================================================

// NOTE: MPSGraphExprBuilder is NOT in an anonymous namespace so that
// eval_gpu_fused can use it. The class is still internal to this TU.

class MPSGraphExprBuilder {
    MPSGraph *graph_;
    std::map<const Tensor *, MPSGraphTensor *> tensor_cache_;
    std::map<const Tensor *, MPSGraphTensorData *> tensor_data_cache_;

  public:
    MPSGraphExprBuilder() { graph_ = [[MPSGraph alloc] init]; }

    MPSGraph *graph() const { return graph_; }

    // Get or create placeholder for a tensor
    MPSGraphTensor *getPlaceholder(const Tensor &tensor) {
        auto it = tensor_cache_.find(&tensor);
        if (it != tensor_cache_.end()) {
            return it->second;
        }

        MPSDataType dtype = getMPSDataType(tensor.dtype());
        MPSShape *shape = getMPSShape(tensor.shape());
        MPSGraphTensor *placeholder =
            [graph_ placeholderWithShape:shape dataType:dtype name:nil];

        tensor_cache_[&tensor] = placeholder;
        return placeholder;
    }

    // Create tensor data for a tensor
    MPSGraphTensorData *getTensorData(const Tensor &tensor) {
        auto it = tensor_data_cache_.find(&tensor);
        if (it != tensor_data_cache_.end()) {
            return it->second;
        }

        auto *storage = static_cast<const axiom::backends::metal::MetalStorage *>(
            tensor.storage().get());
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)storage->buffer();

        MPSDataType dtype = getMPSDataType(tensor.dtype());
        MPSShape *shape = getMPSShape(tensor.shape());

        MPSGraphTensorData *data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                    shape:shape
                                                 dataType:dtype];

        tensor_data_cache_[&tensor] = data;
        return data;
    }

    // Build MPSGraphTensor for TensorRef (leaf)
    MPSGraphTensor *build(const TensorRef &ref) {
        return getPlaceholder(ref.tensor());
    }

    // Build MPSGraphTensor for ScalarExpr
    template <typename T>
    MPSGraphTensor *build(const ScalarExpr<T> &scalar) {
        return [graph_ constantWithScalar:static_cast<double>(scalar.value())
                                 dataType:getMPSDataType(scalar.dtype())];
    }

    // Build MPSGraphTensor for BinaryExpr
    template <typename Op, typename LHS, typename RHS>
    MPSGraphTensor *build(const BinaryExpr<Op, LHS, RHS> &expr) {
        MPSGraphTensor *lhs = build(expr.lhs());
        MPSGraphTensor *rhs = build(expr.rhs());

        // Type promotion
        DType common_dtype = ops::promote_types(expr.lhs().dtype(), expr.rhs().dtype());
        lhs = castToType(lhs, common_dtype);
        rhs = castToType(rhs, common_dtype);

        return applyBinaryOp<Op>(lhs, rhs);
    }

    // Build MPSGraphTensor for UnaryExpr
    template <typename Op, typename Operand>
    MPSGraphTensor *build(const UnaryExpr<Op, Operand> &expr) {
        MPSGraphTensor *operand = build(expr.operand());
        return applyUnaryOp<Op>(operand);
    }

    // Execute the graph with the given output tensor
    Tensor execute(MPSGraphTensor *output, const Shape &shape, DType dtype) {
        // Create output tensor
        Tensor result(shape, dtype, Device::GPU);
        auto *result_storage = static_cast<axiom::backends::metal::MetalStorage *>(
            result.storage().get());
        id<MTLBuffer> result_buffer =
            (__bridge id<MTLBuffer>)result_storage->buffer();

        MPSGraphTensorData *output_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:result_buffer
                                                    shape:getMPSShape(shape)
                                                 dataType:getMPSDataType(dtype)];

        // Build feeds dictionary from cached tensor data
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionary];
        for (const auto &pair : tensor_cache_) {
            MPSGraphTensorData *data = getTensorData(*pair.first);
            feeds[pair.second] = data;
        }

        // Execute
        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *targets =
            @{output : output_data};

        [graph_
            runWithMTLCommandQueue:(__bridge id<MTLCommandQueue>)
                                       backends::metal::MetalContext::instance()
                                           .command_queue()
                             feeds:feeds
                  targetOperations:nil
                 resultsDictionary:targets];

        return result;
    }

  private:
    static MPSDataType getMPSDataType(DType dtype) {
        switch (dtype) {
        case DType::Float32:
            return MPSDataTypeFloat32;
        case DType::Float16:
            return MPSDataTypeFloat16;
        case DType::Int32:
            return MPSDataTypeInt32;
        case DType::Int64:
            return MPSDataTypeInt64;
        case DType::Int16:
            return MPSDataTypeInt16;
        case DType::Int8:
            return MPSDataTypeInt8;
        case DType::UInt8:
            return MPSDataTypeUInt8;
        case DType::UInt16:
            return MPSDataTypeUInt16;
        case DType::UInt32:
            return MPSDataTypeUInt32;
        case DType::UInt64:
            return MPSDataTypeUInt64;
        case DType::Bool:
            return MPSDataTypeBool;
        default:
            @throw [NSException exceptionWithName:@"UnsupportedDType"
                                           reason:@"Unsupported dtype for MPSGraph"
                                         userInfo:nil];
        }
    }

    static MPSShape *getMPSShape(const Shape &shape) {
        NSMutableArray<NSNumber *> *mps_shape =
            [NSMutableArray arrayWithCapacity:shape.size()];
        for (size_t i = 0; i < shape.size(); ++i) {
            [mps_shape addObject:@(shape[i])];
        }
        return mps_shape;
    }

    MPSGraphTensor *castToType(MPSGraphTensor *tensor, DType dtype) {
        MPSDataType target = getMPSDataType(dtype);
        if (tensor.dataType != target) {
            return [graph_ castTensor:tensor toType:target name:nil];
        }
        return tensor;
    }

    // Binary operation dispatch
    template <typename Op>
    MPSGraphTensor *applyBinaryOp(MPSGraphTensor *lhs, MPSGraphTensor *rhs) {
        if constexpr (std::is_same_v<Op, AddOp>) {
            return [graph_ additionWithPrimaryTensor:lhs
                                     secondaryTensor:rhs
                                                name:nil];
        } else if constexpr (std::is_same_v<Op, SubOp>) {
            return [graph_ subtractionWithPrimaryTensor:lhs
                                        secondaryTensor:rhs
                                                   name:nil];
        } else if constexpr (std::is_same_v<Op, MulOp>) {
            return [graph_ multiplicationWithPrimaryTensor:lhs
                                           secondaryTensor:rhs
                                                      name:nil];
        } else if constexpr (std::is_same_v<Op, DivOp>) {
            return [graph_ divisionWithPrimaryTensor:lhs
                                     secondaryTensor:rhs
                                                name:nil];
        } else if constexpr (std::is_same_v<Op, ModOp>) {
            return [graph_ moduloWithPrimaryTensor:lhs
                                   secondaryTensor:rhs
                                              name:nil];
        } else if constexpr (std::is_same_v<Op, PowOp>) {
            return [graph_ powerWithPrimaryTensor:lhs
                                  secondaryTensor:rhs
                                             name:nil];
        } else if constexpr (std::is_same_v<Op, MaxOp>) {
            return [graph_ maximumWithPrimaryTensor:lhs
                                    secondaryTensor:rhs
                                               name:nil];
        } else if constexpr (std::is_same_v<Op, MinOp>) {
            return [graph_ minimumWithPrimaryTensor:lhs
                                    secondaryTensor:rhs
                                               name:nil];
        } else if constexpr (std::is_same_v<Op, EqOp>) {
            return [graph_ equalWithPrimaryTensor:lhs
                                  secondaryTensor:rhs
                                             name:nil];
        } else if constexpr (std::is_same_v<Op, NeOp>) {
            return [graph_ notEqualWithPrimaryTensor:lhs
                                     secondaryTensor:rhs
                                                name:nil];
        } else if constexpr (std::is_same_v<Op, LtOp>) {
            return [graph_ lessThanWithPrimaryTensor:lhs
                                     secondaryTensor:rhs
                                                name:nil];
        } else if constexpr (std::is_same_v<Op, LeOp>) {
            return [graph_ lessThanOrEqualToWithPrimaryTensor:lhs
                                              secondaryTensor:rhs
                                                         name:nil];
        } else if constexpr (std::is_same_v<Op, GtOp>) {
            return [graph_ greaterThanWithPrimaryTensor:lhs
                                        secondaryTensor:rhs
                                                   name:nil];
        } else if constexpr (std::is_same_v<Op, GeOp>) {
            return [graph_ greaterThanOrEqualToWithPrimaryTensor:lhs
                                                 secondaryTensor:rhs
                                                            name:nil];
        } else if constexpr (std::is_same_v<Op, AndOp>) {
            MPSGraphTensor *lhs_bool =
                [graph_ castTensor:lhs toType:MPSDataTypeBool name:nil];
            MPSGraphTensor *rhs_bool =
                [graph_ castTensor:rhs toType:MPSDataTypeBool name:nil];
            return [graph_ logicalANDWithPrimaryTensor:lhs_bool
                                       secondaryTensor:rhs_bool
                                                  name:nil];
        } else if constexpr (std::is_same_v<Op, OrOp>) {
            MPSGraphTensor *lhs_bool =
                [graph_ castTensor:lhs toType:MPSDataTypeBool name:nil];
            MPSGraphTensor *rhs_bool =
                [graph_ castTensor:rhs toType:MPSDataTypeBool name:nil];
            return [graph_ logicalORWithPrimaryTensor:lhs_bool
                                      secondaryTensor:rhs_bool
                                                 name:nil];
        } else if constexpr (std::is_same_v<Op, XorOp>) {
            MPSGraphTensor *lhs_bool =
                [graph_ castTensor:lhs toType:MPSDataTypeBool name:nil];
            MPSGraphTensor *rhs_bool =
                [graph_ castTensor:rhs toType:MPSDataTypeBool name:nil];
            return [graph_ logicalXORWithPrimaryTensor:lhs_bool
                                       secondaryTensor:rhs_bool
                                                  name:nil];
        } else if constexpr (std::is_same_v<Op, BitAndOp>) {
            return [graph_ bitwiseANDWithPrimaryTensor:lhs
                                       secondaryTensor:rhs
                                                  name:nil];
        } else if constexpr (std::is_same_v<Op, BitOrOp>) {
            return [graph_ bitwiseORWithPrimaryTensor:lhs
                                      secondaryTensor:rhs
                                                 name:nil];
        } else if constexpr (std::is_same_v<Op, BitXorOp>) {
            return [graph_ bitwiseXORWithPrimaryTensor:lhs
                                       secondaryTensor:rhs
                                                  name:nil];
        } else if constexpr (std::is_same_v<Op, LShiftOp>) {
            return [graph_ bitwiseLeftShiftWithPrimaryTensor:lhs
                                             secondaryTensor:rhs
                                                        name:nil];
        } else if constexpr (std::is_same_v<Op, RShiftOp>) {
            return [graph_ bitwiseRightShiftWithPrimaryTensor:lhs
                                              secondaryTensor:rhs
                                                         name:nil];
        } else {
            @throw [NSException exceptionWithName:@"UnsupportedOp"
                                           reason:@"Unsupported binary op"
                                         userInfo:nil];
        }
    }

    // Unary operation dispatch
    template <typename Op>
    MPSGraphTensor *applyUnaryOp(MPSGraphTensor *operand) {
        if constexpr (std::is_same_v<Op, NegOp>) {
            return [graph_ negativeWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, AbsOp>) {
            return [graph_ absoluteWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, SqrtOp>) {
            return [graph_ squareRootWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, ExpOp>) {
            return [graph_ exponentWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, LogOp>) {
            return [graph_ logarithmWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, SinOp>) {
            return [graph_ sinWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, CosOp>) {
            return [graph_ cosWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, TanOp>) {
            return [graph_ tanWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, ErfOp>) {
            return [graph_ erfWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, SignOp>) {
            return [graph_ signWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, FloorOp>) {
            return [graph_ floorWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, CeilOp>) {
            return [graph_ ceilWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, TruncOp>) {
            return [graph_ truncateWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, RoundOp>) {
            return [graph_ roundWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, ReciprocalOp>) {
            return [graph_ reciprocalWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, SquareOp>) {
            return [graph_ squareWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, CbrtOp>) {
            // cbrt(x) = sign(x) * |x|^(1/3)
            MPSGraphTensor *abs_op = [graph_ absoluteWithTensor:operand name:nil];
            MPSGraphTensor *one_third =
                [graph_ constantWithScalar:(1.0 / 3.0) dataType:operand.dataType];
            MPSGraphTensor *abs_result =
                [graph_ powerWithPrimaryTensor:abs_op
                               secondaryTensor:one_third
                                          name:nil];
            MPSGraphTensor *sign_op = [graph_ signWithTensor:operand name:nil];
            return [graph_ multiplicationWithPrimaryTensor:sign_op
                                           secondaryTensor:abs_result
                                                      name:nil];
        } else if constexpr (std::is_same_v<Op, IsNaNOp>) {
            return [graph_ isNaNWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, IsInfOp>) {
            return [graph_ isInfiniteWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, IsFiniteOp>) {
            return [graph_ isFiniteWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, NotOp>) {
            MPSGraphTensor *bool_tensor =
                [graph_ castTensor:operand toType:MPSDataTypeBool name:nil];
            return [graph_ notWithTensor:bool_tensor name:nil];
        } else if constexpr (std::is_same_v<Op, ReluOp>) {
            return [graph_ reLUWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, SigmoidOp>) {
            return [graph_ sigmoidWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, TanhActivationOp>) {
            return [graph_ tanhWithTensor:operand name:nil];
        } else if constexpr (std::is_same_v<Op, GeluOp>) {
            // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            MPSGraphTensor *sqrt2 =
                [graph_ constantWithScalar:M_SQRT2 dataType:operand.dataType];
            MPSGraphTensor *half =
                [graph_ constantWithScalar:0.5 dataType:operand.dataType];
            MPSGraphTensor *one =
                [graph_ constantWithScalar:1.0 dataType:operand.dataType];

            MPSGraphTensor *x_scaled =
                [graph_ divisionWithPrimaryTensor:operand
                                  secondaryTensor:sqrt2
                                             name:nil];
            MPSGraphTensor *erf_val = [graph_ erfWithTensor:x_scaled name:nil];
            MPSGraphTensor *inner =
                [graph_ additionWithPrimaryTensor:one
                                  secondaryTensor:erf_val
                                             name:nil];
            MPSGraphTensor *half_x =
                [graph_ multiplicationWithPrimaryTensor:half
                                        secondaryTensor:operand
                                                   name:nil];
            return [graph_ multiplicationWithPrimaryTensor:half_x
                                           secondaryTensor:inner
                                                      name:nil];
        } else if constexpr (std::is_same_v<Op, SiluOp>) {
            // SiLU(x) = x * sigmoid(x)
            MPSGraphTensor *sig = [graph_ sigmoidWithTensor:operand name:nil];
            return [graph_ multiplicationWithPrimaryTensor:operand
                                           secondaryTensor:sig
                                                      name:nil];
        } else if constexpr (std::is_same_v<Op, ConjOp>) {
            return [graph_ conjugateWithTensor:operand name:nil];
        } else {
            @throw [NSException exceptionWithName:@"UnsupportedOp"
                                           reason:@"Unsupported unary op"
                                         userInfo:nil];
        }
    }

  public:
    // Storage class accessor for MetalStorage (needed for buffer access)
    class MetalStorageAccessor {
      public:
        static void *getBuffer(const Tensor &tensor);
    };
};

// Check if all tensors in expression are on GPU and contiguous
template <typename Expr>
bool canFuseGPU(const Expr &expr) {
    if (expr.device() != Device::GPU) {
        return false;
    }
    // For now, only fuse relatively shallow expression trees
    if constexpr (expr_depth_v<Expr> > 8) {
        return false;
    }
    return true;
}

// Collect all TensorRefs from an expression tree
template <typename Expr>
void collectTensorRefs(const Expr &expr, std::vector<const Tensor *> &refs) {
    if constexpr (is_tensor_ref_v<Expr>) {
        refs.push_back(expr.tensor_ptr());
    } else if constexpr (is_scalar_expr_v<Expr>) {
        // No tensor refs in scalar expressions
    } else if constexpr (is_binary_expr_v<Expr>) {
        collectTensorRefs(expr.lhs(), refs);
        collectTensorRefs(expr.rhs(), refs);
    } else if constexpr (is_unary_expr_v<Expr>) {
        collectTensorRefs(expr.operand(), refs);
    } else if constexpr (is_matmul_expr_v<Expr>) {
        collectTensorRefs(expr.lhs(), refs);
        collectTensorRefs(expr.rhs(), refs);
    }
}

// Check if all referenced tensors are contiguous
template <typename Expr>
bool allTensorsContiguous(const Expr &expr) {
    std::vector<const Tensor *> refs;
    collectTensorRefs(expr, refs);
    for (const Tensor *t : refs) {
        if (!t->is_contiguous()) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// GPU Fused Evaluation Entry Point
// Builds and executes a single MPSGraph for the entire expression tree
// ============================================================================

template <typename Expr>
Tensor eval_gpu_fused_impl(const Expr &expr) {
    MPSGraphExprBuilder builder;
    MPSGraphTensor *output = builder.build(expr);
    return builder.execute(output, expr.shape(), expr.dtype());
}

// ============================================================================
// Explicit instantiations for common expression types
// Since this is Objective-C++, templates must be instantiated here
// ============================================================================

// Helper to check if GPU fusion is beneficial
template <typename Expr>
bool shouldFuseGPU(const Expr &expr) {
    // Only fuse if expression depth > 1 (no benefit for single ops)
    if constexpr (expr_depth_v<Expr> <= 1) {
        return false;
    }

    // Only fuse if on GPU
    if (expr.device() != Device::GPU) {
        return false;
    }

    // Only fuse if all tensors are contiguous
    if (!allTensorsContiguous(expr)) {
        return false;
    }

    // Don't fuse extremely deep expressions
    if constexpr (expr_depth_v<Expr> > 8) {
        return false;
    }

    return true;
}

// ============================================================================
// Type-erased GPU evaluation dispatcher
// This allows calling from headers without exposing MPSGraph types
// ============================================================================

// Forward declare the dispatcher class
class GPUFusedEvaluator {
public:
    // Evaluate a binary expression with GPU fusion
    template <typename Op, typename LHS, typename RHS>
    static Tensor eval(const BinaryExpr<Op, LHS, RHS> &expr) {
        if (shouldFuseGPU(expr)) {
            return eval_gpu_fused_impl(expr);
        }
        // Fallback to eager evaluation
        Tensor lhs_tensor = expr.lhs().eval();
        Tensor rhs_tensor = expr.rhs().eval();
        const ops::Operation *op =
            ops::OperationRegistry::get_operation(Op::type, lhs_tensor.device());
        return op->execute_binary(lhs_tensor, rhs_tensor);
    }

    // Evaluate a unary expression with GPU fusion
    template <typename Op, typename Operand>
    static Tensor eval(const UnaryExpr<Op, Operand> &expr) {
        if (shouldFuseGPU(expr)) {
            return eval_gpu_fused_impl(expr);
        }
        // Fallback to eager evaluation
        Tensor operand_tensor = expr.operand().eval();
        const ops::Operation *op =
            ops::OperationRegistry::get_operation(Op::type, operand_tensor.device());
        return op->execute_unary(operand_tensor);
    }
};

// ============================================================================
// Explicit template instantiations for BinaryExpr with depth 2+
// These cover the most common expression patterns
// ============================================================================

// Common binary ops: Add, Sub, Mul, Div
// Depth 2: (a op b) op c patterns

// BinaryExpr<BinaryExpr<TensorRef, TensorRef>, TensorRef>
#define INSTANTIATE_BINARY_DEPTH2(OP1, OP2) \
    template Tensor eval_gpu_fused_impl<BinaryExpr<OP2, BinaryExpr<OP1, TensorRef, TensorRef>, TensorRef>>( \
        const BinaryExpr<OP2, BinaryExpr<OP1, TensorRef, TensorRef>, TensorRef> &);

// Generate instantiations for common patterns
INSTANTIATE_BINARY_DEPTH2(AddOp, MulOp)  // (a + b) * c
INSTANTIATE_BINARY_DEPTH2(AddOp, AddOp)  // (a + b) + c
INSTANTIATE_BINARY_DEPTH2(MulOp, AddOp)  // (a * b) + c
INSTANTIATE_BINARY_DEPTH2(MulOp, MulOp)  // (a * b) * c
INSTANTIATE_BINARY_DEPTH2(SubOp, MulOp)  // (a - b) * c
INSTANTIATE_BINARY_DEPTH2(AddOp, SubOp)  // (a + b) - c

// Unary on binary: relu((a + b))
#define INSTANTIATE_UNARY_ON_BINARY(UNARY_OP, BINARY_OP) \
    template Tensor eval_gpu_fused_impl<UnaryExpr<UNARY_OP, BinaryExpr<BINARY_OP, TensorRef, TensorRef>>>( \
        const UnaryExpr<UNARY_OP, BinaryExpr<BINARY_OP, TensorRef, TensorRef>> &);

INSTANTIATE_UNARY_ON_BINARY(ReluOp, AddOp)
INSTANTIATE_UNARY_ON_BINARY(ReluOp, MulOp)
INSTANTIATE_UNARY_ON_BINARY(SigmoidOp, AddOp)
INSTANTIATE_UNARY_ON_BINARY(SigmoidOp, MulOp)
INSTANTIATE_UNARY_ON_BINARY(TanhActivationOp, AddOp)
INSTANTIATE_UNARY_ON_BINARY(GeluOp, AddOp)
INSTANTIATE_UNARY_ON_BINARY(SiluOp, AddOp)

// Chained unary: sigmoid(relu(x))
#define INSTANTIATE_CHAINED_UNARY(OP1, OP2) \
    template Tensor eval_gpu_fused_impl<UnaryExpr<OP2, UnaryExpr<OP1, TensorRef>>>( \
        const UnaryExpr<OP2, UnaryExpr<OP1, TensorRef>> &);

INSTANTIATE_CHAINED_UNARY(ReluOp, SigmoidOp)
INSTANTIATE_CHAINED_UNARY(SigmoidOp, ReluOp)
INSTANTIATE_CHAINED_UNARY(ExpOp, SigmoidOp)
INSTANTIATE_CHAINED_UNARY(ReluOp, TanhActivationOp)

#undef INSTANTIATE_BINARY_DEPTH2
#undef INSTANTIATE_UNARY_ON_BINARY
#undef INSTANTIATE_CHAINED_UNARY

} // namespace expr
} // namespace axiom
