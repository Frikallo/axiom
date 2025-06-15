#pragma once

#include "axiom/operations.hpp"

namespace axiom {
namespace backends {
namespace metal {

void add(Tensor& a, const Tensor& b);

void register_metal_operations();

} // namespace metal
} // namespace backends
} // namespace axiom 