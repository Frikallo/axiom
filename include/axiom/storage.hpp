#pragma once

#include <cstddef>
#include <memory>

namespace axiom {

enum class Device {
  CPU,
  GPU  // Metal on Apple Silicon, future: CUDA, OpenCL, etc.
};

class Storage {
 public:
  virtual ~Storage() = default;

  virtual void* data() = 0;
  virtual const void* data() const = 0;

  virtual size_t size_bytes() const = 0;

  virtual Device device() const = 0;

  virtual void copy_to(Storage& other) const = 0;

  virtual void copy_from(const Storage& other) = 0;

  virtual std::unique_ptr<Storage> clone() const = 0;

  virtual bool is_view() const = 0;

  virtual std::shared_ptr<Storage> base() const = 0;
};

std::unique_ptr<Storage> make_storage(size_t size_bytes,
                                      Device device = Device::CPU);
std::unique_ptr<Storage> make_storage_view(std::shared_ptr<Storage> base,
                                           size_t offset, size_t size_bytes);

}  // namespace axiom