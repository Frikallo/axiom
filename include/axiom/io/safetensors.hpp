#pragma once

#include <map>
#include <string>
#include <vector>

#include "axiom/tensor.hpp"

namespace axiom {
namespace io {
namespace safetensors {

// Check if file has .safetensors extension
bool is_safetensors_file(const std::string &filename);

// Load all tensors from a .safetensors file
std::map<std::string, Tensor> load(const std::string &filename,
                                   Device device = Device::CPU);

// List tensor names in a .safetensors file without loading data
std::vector<std::string> list_tensors(const std::string &filename);

// Load a specific tensor by name from a .safetensors file
Tensor load_tensor(const std::string &filename, const std::string &tensor_name,
                   Device device = Device::CPU);

} // namespace safetensors
} // namespace io
} // namespace axiom
