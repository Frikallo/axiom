#include "axiom/system.hpp"

// Forward-declare the internal function from the Metal backend.
// The actual implementation is in an Objective-C++ file.
namespace axiom::backends::metal {
bool is_metal_available();
}

namespace axiom::system {
bool is_metal_available() {
#ifdef AXIOM_METAL_SUPPORT
    return axiom::backends::metal::is_metal_available();
#else
    return false;
#endif
}
} // namespace axiom::system 