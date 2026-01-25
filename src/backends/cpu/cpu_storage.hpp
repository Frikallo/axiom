#pragma once

#include <memory>

#include "axiom/storage.hpp"

namespace axiom {
namespace backends {
namespace cpu {

class CPUStorage : public Storage {
  private:
    std::shared_ptr<uint8_t[]> data_;
    size_t size_bytes_;
    size_t offset_;
    std::shared_ptr<Storage> base_storage_;

  public:
    explicit CPUStorage(size_t size_bytes);

    void *data() override;
    const void *data() const override;
    size_t size_bytes() const override;
    Device device() const override { return Device::CPU; }
    void copy_to(Storage &other) const override;
    void copy_from(const Storage &other) override;
    std::unique_ptr<Storage> clone() const override;

    template <typename T> T *typed_data() {
        return reinterpret_cast<T *>(data());
    }

    template <typename T> const T *typed_data() const {
        return reinterpret_cast<const T *>(data());
    }
};

std::unique_ptr<Storage> make_cpu_storage(size_t size_bytes);

} // namespace cpu
} // namespace backends
} // namespace axiom