#include <iostream>
#include <iomanip>
#include "axiom/axiom.hpp"

using namespace axiom;

void print_tensor_info(const Tensor& tensor, const std::string& name) {
  std::cout << name << ":\n";
  std::cout << "  Shape: [";
  for (size_t i = 0; i < tensor.shape().size(); ++i) {
    std::cout << tensor.shape()[i];
    if (i < tensor.shape().size() - 1) std::cout << ", ";
  }
  std::cout << "]\n";
  std::cout << "  DType: " << tensor.dtype_name() << "\n";
  std::cout << "  Device: " << (tensor.device() == Device::CPU ? "CPU" : "GPU") << "\n";
  std::cout << "  Size: " << tensor.size() << " elements\n";
  
  // Print some values for small tensors
  if (tensor.size() <= 12 && tensor.device() == Device::CPU) {
    std::cout << "  Values: ";
    if (tensor.dtype() == DType::Float32) {
      const float* data = tensor.typed_data<float>();
      for (size_t i = 0; i < std::min(tensor.size(), size_t(12)); ++i) {
        std::cout << std::fixed << std::setprecision(3) << data[i];
        if (i < std::min(tensor.size(), size_t(12)) - 1) std::cout << ", ";
      }
    } else if (tensor.dtype() == DType::Bool) {
      const bool* data = tensor.typed_data<bool>();
      for (size_t i = 0; i < std::min(tensor.size(), size_t(12)); ++i) {
        std::cout << (data[i] ? "true" : "false");
        if (i < std::min(tensor.size(), size_t(12)) - 1) std::cout << ", ";
      }
    } else if (tensor.dtype() == DType::Int32) {
      const int32_t* data = tensor.typed_data<int32_t>();
      for (size_t i = 0; i < std::min(tensor.size(), size_t(12)); ++i) {
        std::cout << data[i];
        if (i < std::min(tensor.size(), size_t(12)) - 1) std::cout << ", ";
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

int main() {
  std::cout << "=== Axiom Tensor Operations Demo ===\n\n";
  
  // Initialize the operation registry
  ops::OperationRegistry::initialize_builtin_operations();
  
  try {
    // Create some test tensors
    std::cout << "1. Creating test tensors:\n";
    
    Tensor a = Tensor::full({2, 3}, 2.0f, Device::CPU);
    Tensor b = Tensor::full({2, 3}, 3.0f, Device::CPU);
    Tensor c = Tensor::full({1}, 5.0f, Device::CPU);  // Scalar for broadcasting
    
    print_tensor_info(a, "Tensor a");
    print_tensor_info(b, "Tensor b");
    print_tensor_info(c, "Tensor c (scalar)");
    
    // Test basic arithmetic operations
    std::cout << "2. Basic arithmetic operations:\n";
    
    Tensor add_result = a + b;
    print_tensor_info(add_result, "a + b");
    
    Tensor sub_result = a - b;
    print_tensor_info(sub_result, "a - b");
    
    Tensor mul_result = a * b;
    print_tensor_info(mul_result, "a * b");
    
    Tensor div_result = a / b;
    print_tensor_info(div_result, "a / b");
    
    // Test broadcasting
    std::cout << "3. Broadcasting operations:\n";
    
    Tensor broadcast_add = a + c;
    print_tensor_info(broadcast_add, "a + c (broadcasting)");
    
    Tensor broadcast_mul = a * c;
    print_tensor_info(broadcast_mul, "a * c (broadcasting)");
    
    // Test comparison operations
    std::cout << "4. Comparison operations:\n";
    
    Tensor eq_result = a == b;
    print_tensor_info(eq_result, "a == b");
    
    Tensor lt_result = a < b;
    print_tensor_info(lt_result, "a < b");
    
    Tensor gt_result = a > c;
    print_tensor_info(gt_result, "a > c (broadcasting)");
    
    // Test math operations
    std::cout << "5. Math operations:\n";
    
    Tensor max_result = ops::maximum(a, b);
    print_tensor_info(max_result, "maximum(a, b)");
    
    Tensor min_result = ops::minimum(a, b);
    print_tensor_info(min_result, "minimum(a, b)");
    
    // Test type promotion
    std::cout << "6. Type promotion:\n";
    
    Tensor int_tensor = Tensor::full({2, 3}, 4, Device::CPU);  // Int32
    Tensor float_tensor = Tensor::full({2, 3}, 2.5f, Device::CPU);  // Float32
    
    print_tensor_info(int_tensor, "int_tensor");
    print_tensor_info(float_tensor, "float_tensor");
    
    Tensor promoted_result = int_tensor + float_tensor;
    print_tensor_info(promoted_result, "int_tensor + float_tensor (promoted)");
    
    // Test scalar operations
    std::cout << "7. Scalar operations:\n";
    
    Tensor scalar_add = a + 10.0f;
    print_tensor_info(scalar_add, "a + 10.0");
    
    Tensor scalar_mul = 2.0f * a;
    print_tensor_info(scalar_mul, "2.0 * a");
    
    // Test in-place operations
    std::cout << "8. In-place operations:\n";
    
    Tensor inplace_test = Tensor::full({2, 3}, 1.0f, Device::CPU);
    print_tensor_info(inplace_test, "inplace_test (before)");
    
    inplace_test += a;
    print_tensor_info(inplace_test, "inplace_test += a");
    
    inplace_test = inplace_test * 2.0f;
    print_tensor_info(inplace_test, "inplace_test *= 2.0");
    
    std::cout << "=== Demo completed successfully! ===\n";
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
} 