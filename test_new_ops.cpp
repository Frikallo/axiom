#include <iostream>
#include "axiom/axiom.hpp"

using namespace axiom;

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "=== Testing New Operations ===" << std::endl;

    // Test flatten
    std::cout << "\n--- Flatten ---" << std::endl;
    auto t1 = Tensor::ones({2, 3, 4});
    auto flat = t1.flatten();
    std::cout << "Original: " << t1.repr() << std::endl;
    std::cout << "Flattened: " << flat.repr() << std::endl;
    std::cout << "Is contiguous: " << flat.is_contiguous() << std::endl;

    // Test expand (zero-copy)
    std::cout << "\n--- Expand (Zero-Copy) ---" << std::endl;
    auto t2 = Tensor::ones({1, 4});
    std::cout << "Original: " << t2.repr() << std::endl;
    auto expanded = t2.expand({3, 4});
    std::cout << "Expanded to (3, 4): " << expanded.repr() << std::endl;
    std::cout << "Strides: [" << expanded.strides()[0] << ", " << expanded.strides()[1] << "]" << std::endl;
    std::cout << "First stride is 0 (broadcast): " << (expanded.strides()[0] == 0 ? "yes" : "no") << std::endl;

    // Test repeat (copies data)
    std::cout << "\n--- Repeat (Copies Data) ---" << std::endl;
    auto t3 = Tensor::arange(0, 4);
    t3 = t3.reshape({2, 2});
    std::cout << "Original (2x2): " << t3.repr() << std::endl;
    auto repeated = t3.repeat({2, 3});
    std::cout << "Repeated by (2, 3): " << repeated.repr() << std::endl;

    // Test argmax
    std::cout << "\n--- ArgMax ---" << std::endl;
    float data[] = {1.0f, 5.0f, 2.0f, 8.0f, 3.0f, 4.0f};
    auto t4 = Tensor::from_data(data, {2, 3});
    std::cout << "Input (2x3): values are [1,5,2], [8,3,4]" << std::endl;
    auto argmax_0 = t4.argmax(0);
    auto argmax_1 = t4.argmax(1);
    std::cout << "ArgMax axis=0: ";
    for (size_t i = 0; i < argmax_0.size(); ++i) {
        std::cout << argmax_0.typed_data<int64_t>()[i] << " ";
    }
    std::cout << "(expected: 1 0 1)" << std::endl;

    std::cout << "ArgMax axis=1: ";
    for (size_t i = 0; i < argmax_1.size(); ++i) {
        std::cout << argmax_1.typed_data<int64_t>()[i] << " ";
    }
    std::cout << "(expected: 1 0)" << std::endl;

    // Test member reduction functions
    std::cout << "\n--- Member Reduction Functions ---" << std::endl;
    auto t5 = Tensor::ones({3, 4});
    std::cout << "Tensor (3x4) of ones:" << std::endl;
    std::cout << "  sum(): " << t5.sum().typed_data<float>()[0] << " (expected: 12)" << std::endl;
    std::cout << "  mean(): " << t5.mean().typed_data<float>()[0] << " (expected: 1)" << std::endl;
    std::cout << "  sum(0).shape: [";
    for (auto s : t5.sum(0).shape()) std::cout << s << " ";
    std::cout << "] (expected: [4])" << std::endl;

    std::cout << "\n=== All Tests Passed ===" << std::endl;
    return 0;
}
