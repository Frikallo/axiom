#include "cpu_storage.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace axiom {
namespace backends {
namespace cpu {

// ============================================================================
// CPUStorage Implementation
// ============================================================================

CPUStorage::CPUStorage(size_t size_bytes)
    : data_(new uint8_t[size_bytes], std::default_delete<uint8_t[]>()),
      size_bytes_(size_bytes),
      offset_(0),
      base_storage_(nullptr) {}

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

size_t CPUStorage::size_bytes() const { return size_bytes_; }

void CPUStorage::copy_to(Storage& other) const {
  if (other.device() == Device::CPU) {
    // CPU to CPU copy
    std::memcpy(other.data(), data(),
                std::min(size_bytes_, other.size_bytes()));
  } else {
    // CPU to GPU copy - delegate to the GPU storage
    other.copy_from(*this);
  }
}

void CPUStorage::copy_from(const Storage& other) {
  if (other.device() == Device::CPU) {
    // CPU to CPU copy
    std::memcpy(data(), other.data(),
                std::min(size_bytes_, other.size_bytes()));
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

// ============================================================================
// Factory functions
// ============================================================================

std::unique_ptr<Storage> make_cpu_storage(size_t size_bytes) {
  return std::make_unique<CPUStorage>(size_bytes);
}

}  // namespace cpu
}  // namespace backends
}  // namespace axiom