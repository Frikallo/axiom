#import "metal_operations.hpp"
#import "mpsgraph_operations.hpp"

#import "metal_common.hpp"
#import "metal_storage.hpp"
#import "axiom/operations.hpp"
#import "axiom/error.hpp"
#import "axiom/shape.hpp"
#import "axiom/tensor.hpp"
#import "axiom/dtype.hpp"

#import <Metal/Metal.h>
#import <vector>
#import <numeric>
#import <algorithm>
#import <string>

namespace axiom {
namespace backends {
namespace metal {

// ============================================================================
// All operations have been migrated to MPSGraph
// ============================================================================
// This file now only contains the registration function that delegates to
// mpsgraph_operations.mm. All operation implementations use MPSGraph for:
// - Automatic kernel fusion
// - Apple Silicon optimizations
// - Simpler codebase
// - Easier maintenance
//
// Operations migrated:
// - Binary arithmetic (Add, Subtract, Multiply, Divide)
// - Unary operations (Negate, Abs, Sqrt, Exp, Log, Sin, Cos, Tan)
// - Reductions (Sum, Mean, Max, Min)
// - Comparisons (Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual)
// - Logical operations (LogicalAnd, LogicalOr, LogicalXor, LogicalNot)
// - Bitwise operations (BitwiseAnd, BitwiseOr, BitwiseXor, LeftShift, RightShift)
// - Math operations (Maximum, Minimum, Atan2, Power, Modulo)
// - Matrix multiplication (MatMul)
// - ArgMax, ArgMin
// ============================================================================

void register_metal_operations() {
    if (!is_metal_available()) return;
    
    // All GPU operations are now registered via MPSGraph
    register_mpsgraph_operations();
}

} // namespace metal
} // namespace backends
} // namespace axiom
