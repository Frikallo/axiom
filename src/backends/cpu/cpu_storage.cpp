#include "cpu_storage.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace axiom {
namespace backends {
namespace cpu {

// ============================================================================
// CPUStorage Implementation
// ============================================================================

CPUStorage::CPUStorage(size_t size_bytes) 
    : data_(new uint8_t[size_bytes], std::default_delete<uint8_t[]>())
    , size_bytes_(size_bytes)
    , offset_(0)
    , base_storage_(nullptr) {
}

CPUStorage::CPUStorage(std::shared_ptr<Storage> base, size_t offset, size_t size_bytes)
    : data_(nullptr)
    , size_bytes_(size_bytes)
    , offset_(offset)
    , base_storage_(base) {
    
    if (base->device() != Device::CPU) {
        throw std::runtime_error("Cannot create CPU view of non-CPU storage");
    }
    
    // Get the underlying CPU storage
    auto cpu_base = std::dynamic_pointer_cast<CPUStorage>(base);
    if (!cpu_base) {
        throw std::runtime_error("Invalid CPU storage cast");
    }
    
    if (offset + size_bytes > cpu_base->size_bytes()) {
        throw std::runtime_error("View exceeds base storage bounds");
    }
    
    // Share the underlying data pointer
    data_ = cpu_base->data_;
}

void* CPUStorage::data() {
    if (data_ == nullptr) {
        throw std::runtime_error("Storage has no data");
    }
    return data_.get() + offset_;
}

const void* CPUStorage::data() const {
    if (data_ == nullptr) {
        throw std::runtime_error("Storage has no data");
    }
    return data_.get() + offset_;
}

size_t CPUStorage::size_bytes() const {
    return size_bytes_;
}

void CPUStorage::copy_to(Storage& other) const {
    if (other.device() == Device::CPU) {
        // CPU to CPU copy
        std::memcpy(other.data(), data(), std::min(size_bytes_, other.size_bytes()));
    } else {
        // CPU to GPU copy - delegate to the GPU storage
        other.copy_from(*this);
    }
}

void CPUStorage::copy_from(const Storage& other) {
    if (other.device() == Device::CPU) {
        // CPU to CPU copy
        std::memcpy(data(), other.data(), std::min(size_bytes_, other.size_bytes()));
    } else {
        // GPU to CPU copy - delegate to the GPU storage
        const_cast<Storage&>(other).copy_to(*this);
    }
}

std::unique_ptr<Storage> CPUStorage::clone() const {
    auto new_storage = std::make_unique<CPUStorage>(size_bytes_);
    new_storage->copy_from(*this);
    return new_storage;
}

bool CPUStorage::is_view() const {
    return base_storage_ != nullptr;
}

std::shared_ptr<Storage> CPUStorage::base() const {
    return base_storage_ ? base_storage_ : nullptr;
}

// ============================================================================
// Factory functions
// ============================================================================

std::unique_ptr<Storage> make_cpu_storage(size_t size_bytes) {
    return std::make_unique<CPUStorage>(size_bytes);
}

std::unique_ptr<Storage> make_cpu_storage_view(std::shared_ptr<Storage> base, size_t offset, size_t size_bytes) {
    return std::make_unique<CPUStorage>(base, offset, size_bytes);
}

} // namespace cpu
} // namespace backends
} // namespace axiom