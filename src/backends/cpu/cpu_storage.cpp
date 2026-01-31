#include "cpu_storage.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <malloc.h>
#endif

#include "axiom/error.hpp"

namespace axiom {
namespace backends {
namespace cpu {

// ============================================================================
// Aligned Memory Allocation
// ============================================================================

// Custom deleter for aligned memory
struct AlignedDeleter {
    void operator()(uint8_t *ptr) const {
        if (ptr) {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
    }
};

// Allocate aligned memory (64-byte alignment for optimal SIMD/cache
// performance)
static std::shared_ptr<uint8_t[]> allocate_aligned(size_t size_bytes) {
    constexpr size_t alignment = 64; // Cache line size

    if (size_bytes == 0) {
        return nullptr;
    }

    void *ptr = nullptr;

#ifdef _WIN32
    ptr = _aligned_malloc(size_bytes, alignment);
    if (ptr == nullptr) {
        throw MemoryError("Failed to allocate " + std::to_string(size_bytes) +
                          " bytes of aligned memory");
    }
#else
    int result = posix_memalign(&ptr, alignment, size_bytes);
    if (result != 0 || ptr == nullptr) {
        throw MemoryError("Failed to allocate " + std::to_string(size_bytes) +
                          " bytes of aligned memory");
    }
#endif

    // Use shared_ptr with custom deleter for proper cleanup
    return std::shared_ptr<uint8_t[]>(static_cast<uint8_t *>(ptr),
                                      AlignedDeleter{});
}

// ============================================================================
// CPUStorage Implementation
// ============================================================================

CPUStorage::CPUStorage(size_t size_bytes)
    : data_(allocate_aligned(size_bytes)), size_bytes_(size_bytes), offset_(0),
      base_storage_(nullptr) {}

void *CPUStorage::data() {
    if (data_ == nullptr) {
        throw MemoryError("Storage has no data");
    }
    return data_.get() + offset_;
}

const void *CPUStorage::data() const {
    if (data_ == nullptr) {
        throw MemoryError("Storage has no data");
    }
    return data_.get() + offset_;
}

size_t CPUStorage::size_bytes() const { return size_bytes_; }

void CPUStorage::copy_to(Storage &other) const {
    if (other.device() == Device::CPU) {
        // CPU to CPU copy
        std::memcpy(other.data(), data(),
                    std::min(size_bytes_, other.size_bytes()));
    } else {
        // CPU to GPU copy - delegate to the GPU storage
        other.copy_from(*this);
    }
}

void CPUStorage::copy_from(const Storage &other) {
    if (other.device() == Device::CPU) {
        // CPU to CPU copy
        std::memcpy(data(), other.data(),
                    std::min(size_bytes_, other.size_bytes()));
    } else {
        // GPU to CPU copy - delegate to the GPU storage
        const_cast<Storage &>(other).copy_to(*this);
    }
}

std::unique_ptr<Storage> CPUStorage::clone() const {
    auto new_storage = std::make_unique<CPUStorage>(size_bytes_);
    new_storage->copy_from(*this);
    return new_storage;
}

// ============================================================================
// Factory functions
// ============================================================================

std::unique_ptr<Storage> make_cpu_storage(size_t size_bytes) {
    return std::make_unique<CPUStorage>(size_bytes);
}

} // namespace cpu
} // namespace backends
} // namespace axiom