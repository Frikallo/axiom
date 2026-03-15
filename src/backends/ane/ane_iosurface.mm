#import "ane_iosurface.h"

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

// ============================================================================
// IOSurface creation for ANE
// ============================================================================

IOSurfaceRef ane_create_surface(int channels, int spatial_size) {
    if (channels <= 0 || spatial_size <= 0) {
        return NULL;
    }

    // ANE IOSurface layout: each "row" is one channel, each "column" is one
    // spatial element. FP16 = 2 bytes per element.
    size_t bytes_per_element = 2; // FP16
    size_t bytes_per_row = (size_t)spatial_size * bytes_per_element;

    // Align row bytes to 64-byte boundary (ANE requirement)
    size_t aligned_bytes_per_row = (bytes_per_row + 63) & ~(size_t)63;
    size_t total_bytes = aligned_bytes_per_row * (size_t)channels;

    NSDictionary *props = @{
        (id)kIOSurfaceWidth : @(spatial_size),
        (id)kIOSurfaceHeight : @(channels),
        (id)kIOSurfaceBytesPerElement : @(bytes_per_element),
        (id)kIOSurfaceBytesPerRow : @(aligned_bytes_per_row),
        (id)kIOSurfaceAllocSize : @(total_bytes),
        (id)kIOSurfacePixelFormat : @(0x00000001), // One-component format
    };

    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
    return surface;
}

// ============================================================================
// FP32 <-> FP16 conversion using Accelerate
// ============================================================================

int ane_surface_write_f32(IOSurfaceRef surface, const float *data, int channels,
                          int spatial_size) {
    if (!surface || !data || channels <= 0 || spatial_size <= 0) {
        return -1;
    }

    IOSurfaceLock(surface, 0, NULL);

    void *base = IOSurfaceGetBaseAddress(surface);
    size_t row_bytes = IOSurfaceGetBytesPerRow(surface);
    size_t elements_per_row = spatial_size;

    // Convert FP32 -> FP16 row by row (each row = one channel)
    for (int c = 0; c < channels; c++) {
        const float *src_row = data + (size_t)c * elements_per_row;
        uint16_t *dst_row =
            (uint16_t *)((uint8_t *)base + (size_t)c * row_bytes);

        // Use vImage for FP32 -> FP16 conversion (Accelerate framework)
        vImage_Buffer src_buf = {
            .data = (void *)src_row,
            .height = 1,
            .width = (vImagePixelCount)elements_per_row,
            .rowBytes = elements_per_row * sizeof(float),
        };
        vImage_Buffer dst_buf = {
            .data = dst_row,
            .height = 1,
            .width = (vImagePixelCount)elements_per_row,
            .rowBytes = elements_per_row * sizeof(uint16_t),
        };

        vImageConvert_PlanarFtoPlanar16F(&src_buf, &dst_buf, 0);
    }

    IOSurfaceUnlock(surface, 0, NULL);
    return 0;
}

int ane_surface_read_f32(IOSurfaceRef surface, float *data, int channels,
                         int spatial_size) {
    if (!surface || !data || channels <= 0 || spatial_size <= 0) {
        return -1;
    }

    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);

    void *base = IOSurfaceGetBaseAddress(surface);
    size_t row_bytes = IOSurfaceGetBytesPerRow(surface);
    size_t elements_per_row = spatial_size;

    // Convert FP16 -> FP32 row by row
    for (int c = 0; c < channels; c++) {
        const uint16_t *src_row =
            (const uint16_t *)((const uint8_t *)base + (size_t)c * row_bytes);
        float *dst_row = data + (size_t)c * elements_per_row;

        vImage_Buffer src_buf = {
            .data = (void *)src_row,
            .height = 1,
            .width = (vImagePixelCount)elements_per_row,
            .rowBytes = elements_per_row * sizeof(uint16_t),
        };
        vImage_Buffer dst_buf = {
            .data = dst_row,
            .height = 1,
            .width = (vImagePixelCount)elements_per_row,
            .rowBytes = elements_per_row * sizeof(float),
        };

        vImageConvert_Planar16FtoPlanarF(&src_buf, &dst_buf, 0);
    }

    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return 0;
}

int ane_surface_write_f16(IOSurfaceRef surface, const uint16_t *data,
                          int channels, int spatial_size) {
    if (!surface || !data || channels <= 0 || spatial_size <= 0) {
        return -1;
    }

    IOSurfaceLock(surface, 0, NULL);

    void *base = IOSurfaceGetBaseAddress(surface);
    size_t row_bytes = IOSurfaceGetBytesPerRow(surface);
    size_t element_bytes = (size_t)spatial_size * sizeof(uint16_t);

    for (int c = 0; c < channels; c++) {
        const uint16_t *src_row = data + (size_t)c * spatial_size;
        void *dst_row = (uint8_t *)base + (size_t)c * row_bytes;
        memcpy(dst_row, src_row, element_bytes);
    }

    IOSurfaceUnlock(surface, 0, NULL);
    return 0;
}

int ane_surface_read_f16(IOSurfaceRef surface, uint16_t *data, int channels,
                         int spatial_size) {
    if (!surface || !data || channels <= 0 || spatial_size <= 0) {
        return -1;
    }

    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);

    void *base = IOSurfaceGetBaseAddress(surface);
    size_t row_bytes = IOSurfaceGetBytesPerRow(surface);
    size_t element_bytes = (size_t)spatial_size * sizeof(uint16_t);

    for (int c = 0; c < channels; c++) {
        const void *src_row =
            (const uint8_t *)base + (size_t)c * row_bytes;
        uint16_t *dst_row = data + (size_t)c * spatial_size;
        memcpy(dst_row, src_row, element_bytes);
    }

    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return 0;
}

IOSurfaceRef ane_create_flat_surface(size_t size_bytes) {
    if (size_bytes == 0) {
        return NULL;
    }

    NSDictionary *props = @{
        (id)kIOSurfaceWidth : @(size_bytes),
        (id)kIOSurfaceHeight : @1,
        (id)kIOSurfaceBytesPerElement : @1,
        (id)kIOSurfaceBytesPerRow : @(size_bytes),
        (id)kIOSurfaceAllocSize : @(size_bytes),
        (id)kIOSurfacePixelFormat : @0,
    };

    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

size_t ane_surface_size_bytes(IOSurfaceRef surface) {
    if (!surface) {
        return 0;
    }
    return IOSurfaceGetAllocSize(surface);
}
