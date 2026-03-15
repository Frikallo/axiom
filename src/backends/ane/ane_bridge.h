#pragma once

// C-callable wrapper around Apple's private ANE frameworks.
// Uses dlopen + objc_msgSend to access _ANEInMemoryModelDescriptor,
// _ANEInMemoryModel, _ANERequest, and _ANEIOSurfaceObject from
// /System/Library/PrivateFrameworks/AppleNeuralEngine.framework

#include <IOSurface/IOSurface.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the ANE bridge: dlopen the private framework and resolve classes.
// Thread-safe (dispatch_once). Returns 0 on success, -1 if unavailable.
int ane_init(void);

// Check if ANE hardware is available on this system.
// Calls ane_init() internally if not already initialized.
bool ane_is_available(void);

// Opaque handle to a compiled ANE model.
typedef struct ANEModelHandle ANEModelHandle;

// Compile MIL text into an ANE model.
// mil_text: null-terminated MIL program string
// Returns handle on success, NULL on failure.
// Each call increments the compile counter (~119 limit per process).
ANEModelHandle *ane_compile(const char *mil_text);

// Load a compiled model onto ANE hardware for execution.
// Must be called after ane_compile() and before ane_eval().
// Returns 0 on success.
int ane_load(ANEModelHandle *handle);

// Evaluate: run a loaded model with the given input/output IOSurfaces.
// Both surfaces must be locked before calling and unlocked after.
// input_surface: IOSurfaceRef containing input data in [1,C,1,S] FP16 format
// output_surface: IOSurfaceRef for output data in [1,C,1,S] FP16 format
// Returns 0 on success.
int ane_eval(ANEModelHandle *handle, IOSurfaceRef input_surface,
             IOSurfaceRef output_surface);

// Release a compiled model and free associated resources.
void ane_release(ANEModelHandle *handle);

// Get the current compile count (for budget tracking).
// The ANE private framework leaks resources after ~119 compilations.
int ane_compile_count(void);

// Maximum recommended compilations before resource exhaustion.
#define ANE_COMPILE_BUDGET_WARNING 95
#define ANE_COMPILE_BUDGET_LIMIT 115

#ifdef __cplusplus
}
#endif
