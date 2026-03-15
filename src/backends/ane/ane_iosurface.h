#pragma once

// IOSurface utilities for ANE data transfer.
// ANE uses IOSurface for zero-copy I/O in [1, C, 1, S] channel-first FP16 format.

#include <IOSurface/IOSurface.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Create an IOSurface for ANE I/O.
// Allocates a surface with dimensions suitable for [1, channels, 1, spatial_size]
// in FP16 (2 bytes per element).
// Returns IOSurfaceRef on success (caller must CFRelease), NULL on failure.
IOSurfaceRef ane_create_surface(int channels, int spatial_size);

// Write FP32 row-major data into an IOSurface, converting to FP16.
// Data is expected in row-major [channels, spatial_size] layout.
// The IOSurface stores it in ANE's native [1, C, 1, S] format.
// Returns 0 on success.
int ane_surface_write_f32(IOSurfaceRef surface, const float *data, int channels,
                          int spatial_size);

// Read FP16 data from IOSurface back to FP32 row-major.
// Output is written in [channels, spatial_size] row-major layout.
// Returns 0 on success.
int ane_surface_read_f32(IOSurfaceRef surface, float *data, int channels,
                         int spatial_size);

// Write raw FP16 data into IOSurface (no conversion).
// Data layout: [channels, spatial_size] in FP16.
// Returns 0 on success.
int ane_surface_write_f16(IOSurfaceRef surface, const uint16_t *data,
                          int channels, int spatial_size);

// Read raw FP16 data from IOSurface (no conversion).
// Returns 0 on success.
int ane_surface_read_f16(IOSurfaceRef surface, uint16_t *data, int channels,
                         int spatial_size);

// Get the total byte size of an IOSurface's allocation.
size_t ane_surface_size_bytes(IOSurfaceRef surface);

#ifdef __cplusplus
}
#endif
