#include "axiom/dtype.hpp"

namespace axiom {

std::string dtype_name(DTypes dtype) {
    return std::visit(overload{[](auto t) { return t.name(); }}, dtype);
}

} // namespace axiom