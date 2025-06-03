#pragma once

#include <cstddef>
#include <memory>

namespace axiom {

// Device types for tensor storage
enum class Device {
  CPU,
  GPU  // Metal on Apple Silicon, future: CUDA, OpenCL, etc.
};

// Abstract base class for tensor storage
class Storage {
 public:
  virtual ~Storage() = default;

  // Get raw data pointer (nullptr for GPU storage)
  virtual void* data() = 0;
  virtual const void* data() const = 0;

  // Get size in bytes
  virtual size_t size_bytes() const = 0;

  // Get device type
  virtual Device device() const = 0;

  // Copy data to another storage
  virtual void copy_to(Storage& other) const = 0;

  // Copy data from another storage
  virtual void copy_from(const Storage& other) = 0;

  // Clone storage with same data
  virtual std::unique_ptr<Storage> clone() const = 0;

  // Check if this is a view of another storage
  virtual bool is_view() const = 0;

  // Get base storage if this is a view
  virtual std::shared_ptr<Storage> base() const = 0;
};

// Factory functions for creating storage
std::unique_ptr<Storage> make_storage(size_t size_bytes,
                                      Device device = Device::CPU);
std::unique_ptr<Storage> make_storage_view(std::shared_ptr<Storage> base,
                                           size_t offset, size_t size_bytes);

}  // namespace axiom