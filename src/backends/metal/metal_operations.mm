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

void register_metal_operations() {
    if (!is_metal_available()) return;
    
    register_mpsgraph_operations();
}

} // namespace metal
} // namespace backends
} // namespace axiom
