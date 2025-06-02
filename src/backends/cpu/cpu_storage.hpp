#pragma once

#include "axiom/storage.hpp"
#include <memory>

namespace axiom {
namespace backends {
namespace cpu {

// CPU storage implementation
class CPUStorage : public Storage {
private:
    std::shared_ptr<uint8_t[]> data_;
    size_t size_bytes_;
    size_t offset_;
    std::shared_ptr<Storage> base_storage_;
    
public:
    // Create new CPU storage
    explicit CPUStorage(size_t size_bytes);
    
    // Create view of existing storage
    CPUStorage(std::shared_ptr<Storage> base, size_t offset, size_t size_bytes);
    
    void* data() override;
    const void* data() const override;
    size_t size_bytes() const override;
    Device device() const override { return Device::CPU; }
    void copy_to(Storage& other) const override;
    void copy_from(const Storage& other) override;
    std::unique_ptr<Storage> clone() const override;
    bool is_view() const override;
    std::shared_ptr<Storage> base() const override;
    
    // CPU-specific methods
    template<typename T>
    T* typed_data() { return reinterpret_cast<T*>(data()); }
    
    template<typename T>
    const T* typed_data() const { return reinterpret_cast<const T*>(data()); }
};

// CPU backend factory functions
std::unique_ptr<Storage> make_cpu_storage(size_t size_bytes);
std::unique_ptr<Storage> make_cpu_storage_view(std::shared_ptr<Storage> base, size_t offset, size_t size_bytes);

} // namespace cpu
} // namespace backends
} // namespace axiom