#pragma once

namespace axiom {
namespace backends {
namespace ane {

// Initialize the ANE bridge.
// Unlike CPU/GPU, ANE does NOT register per-op operations in the
// OperationRegistry. ANE works at the graph level via ANECompiledModel.
void initialize_ane_backend();

// Check if ANE is available on this system.
bool is_ane_available();

} // namespace ane
} // namespace backends
} // namespace axiom
