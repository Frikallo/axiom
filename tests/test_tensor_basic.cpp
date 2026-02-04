//=============================================================================
// tests/test_comprehensive.cpp - Complete test suite for Axiom tensors
//=============================================================================

#include <axiom/axiom.hpp>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace axiom;

// Test utilities
class TestRunner {
  private:
    int tests_run = 0;
    int tests_passed = 0;
    int tests_failed = 0;
    std::vector<std::string> failed_tests;

  public:
    void run_test(const std::string &name, std::function<void()> test_func) {
        tests_run++;
        std::cout << "Running " << name << "..." << std::flush;

        try {
            test_func();
            tests_passed++;
            std::cout << " PASSED" << std::endl;
        } catch (const std::exception &e) {
            tests_failed++;
            failed_tests.push_back(name + ": " + e.what());
            std::cout << " FAILED: " << e.what() << std::endl;
        }
    }

    void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << tests_run << std::endl;
        std::cout << "Passed: " << tests_passed << std::endl;
        std::cout << "Failed: " << tests_failed << std::endl;

        if (tests_failed > 0) {
            std::cout << "\nFailed tests:" << std::endl;
            for (const auto &failure : failed_tests) {
                std::cout << "  - " << failure << std::endl;
            }
        }

        std::cout << "Success rate: " << std::fixed << std::setprecision(1)
                  << (100.0 * tests_passed / tests_run) << "%" << std::endl;
    }

    bool all_passed() const { return tests_failed == 0; }
};

// Device availability helper
bool is_gpu_available() {
    // Use the system function which checks both Metal availability
    // and the AXIOM_SKIP_GPU_TESTS environment variable
    return axiom::system::should_run_gpu_tests();
}

// Test data validation helper
template <typename T>
void validate_tensor_data(const Tensor &tensor,
                          const std::vector<T> &expected_data) {
    assert(tensor.device() == Device::CPU);
    assert(tensor.size() == expected_data.size());

    auto data = tensor.typed_data<T>();
    for (size_t i = 0; i < expected_data.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            assert(std::abs(data[i] - expected_data[i]) < 1e-6);
        } else {
            assert(data[i] == expected_data[i]);
        }
    }
}

//=============================================================================
// Basic Tensor Tests
//=============================================================================

void test_tensor_creation_cpu() {
    // Test basic constructors
    auto t1 = Tensor({3, 4}, DType::Float32, Device::CPU);
    assert(t1.ndim() == 2);
    assert(t1.shape()[0] == 3);
    assert(t1.shape()[1] == 4);
    assert(t1.size() == 12);
    assert(t1.dtype() == DType::Float32);
    assert(t1.device() == Device::CPU);
    assert(t1.memory_order() == MemoryOrder::RowMajor);

    // Test initializer list constructor
    auto t2 = Tensor({2, 3, 4}, DType::Int32, Device::CPU);
    assert(t2.ndim() == 3);
    assert(t2.size() == 24);
    assert(t2.dtype() == DType::Int32);

    // Test with memory order
    auto t3 =
        Tensor({2, 3}, DType::Float64, Device::CPU, MemoryOrder::ColMajor);
    assert(t3.memory_order() == MemoryOrder::ColMajor);
    assert(t3.is_f_contiguous());

    // Test empty tensor
    auto t4 = Tensor();
    assert(t4.ndim() == 0);
    assert(t4.size() == 1);
}

void test_tensor_creation_gpu() {
    auto t1 = Tensor({3, 4}, DType::Float32, Device::GPU);
    assert(t1.ndim() == 2);
    assert(t1.shape()[0] == 3);
    assert(t1.shape()[1] == 4);
    assert(t1.size() == 12);
    assert(t1.dtype() == DType::Float32);
    assert(t1.device() == Device::GPU);

    // Test with different memory orders
    auto t2 =
        Tensor({2, 3}, DType::Float32, Device::GPU, MemoryOrder::ColMajor);
    assert(t2.memory_order() == MemoryOrder::ColMajor);
    assert(t2.device() == Device::GPU);
}

//=============================================================================
// Factory Function Tests
//=============================================================================

void test_factory_functions_cpu() {
    // Test zeros
    auto z = Tensor::zeros({2, 3}, DType::Float32, Device::CPU);
    assert(z.shape()[0] == 2);
    assert(z.shape()[1] == 3);
    assert(z.is_c_contiguous());

    auto data = z.typed_data<float>();
    for (size_t i = 0; i < z.size(); ++i) {
        assert(data[i] == 0.0f);
    }

    // Test zeros with Fortran order
    auto z_f = Tensor::zeros({2, 3}, DType::Float32, Device::CPU,
                             MemoryOrder::ColMajor);
    assert(z_f.is_f_contiguous());
    assert(z_f.memory_order() == MemoryOrder::ColMajor);

    // Test ones
    auto o = Tensor::ones({2, 2}, DType::Float32, Device::CPU);
    auto ones_data = o.typed_data<float>();
    for (size_t i = 0; i < o.size(); ++i) {
        assert(ones_data[i] == 1.0f);
    }

    // Test ones with different dtypes
    auto o_int = Tensor::ones({2, 2}, DType::Int32, Device::CPU);
    auto int_data = o_int.typed_data<int32_t>();
    for (size_t i = 0; i < o_int.size(); ++i) {
        assert(int_data[i] == 1);
    }

    // Test eye/identity
    auto eye_mat = Tensor::eye(3, DType::Float32, Device::CPU);
    auto eye_data = eye_mat.typed_data<float>();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            assert(eye_data[i * 3 + j] == expected);
        }
    }

    // Test identity (alias for eye)
    auto id_mat = Tensor::identity(2, DType::Float64, Device::CPU);
    assert(id_mat.shape()[0] == 2);
    assert(id_mat.shape()[1] == 2);
    assert(id_mat.dtype() == DType::Float64);

    // Test empty
    auto e = Tensor::empty({3, 3}, DType::Int32, Device::CPU);
    assert(e.shape()[0] == 3);
    assert(e.shape()[1] == 3);
    assert(e.dtype() == DType::Int32);

    // Test full
    auto f = Tensor::full<float>({2, 2}, 42.0f, Device::CPU);
    auto full_data = f.typed_data<float>();
    for (size_t i = 0; i < f.size(); ++i) {
        assert(full_data[i] == 42.0f);
    }
}

void test_factory_functions_gpu() {
    // Test zeros on GPU
    auto z = Tensor::zeros({2, 3}, DType::Float32, Device::GPU);
    assert(z.device() == Device::GPU);
    assert(z.shape()[0] == 2);
    assert(z.shape()[1] == 3);

    // Test ones on GPU
    auto o = Tensor::ones({2, 2}, DType::Float32, Device::GPU);
    assert(o.device() == Device::GPU);

    // Test eye on GPU
    auto eye_mat = Tensor::eye(3, DType::Float32, Device::GPU);
    assert(eye_mat.device() == Device::GPU);
    assert(eye_mat.shape()[0] == 3);
    assert(eye_mat.shape()[1] == 3);

    // Test empty on GPU
    auto e = Tensor::empty({3, 3}, DType::Int32, Device::GPU);
    assert(e.device() == Device::GPU);

    // Test with different memory orders
    auto z_f = Tensor::zeros({2, 3}, DType::Float32, Device::GPU,
                             MemoryOrder::ColMajor);
    assert(z_f.device() == Device::GPU);
    assert(z_f.memory_order() == MemoryOrder::ColMajor);
}

void test_creation_routines_cpu() {
    // Test linspace
    auto lin = Tensor::linspace(0.0, 10.0, 5);
    assert(lin.size() == 5);
    auto lin_data = lin.typed_data<double>();
    assert(std::abs(lin_data[0] - 0.0) < 1e-10);
    assert(std::abs(lin_data[1] - 2.5) < 1e-10);
    assert(std::abs(lin_data[2] - 5.0) < 1e-10);
    assert(std::abs(lin_data[3] - 7.5) < 1e-10);
    assert(std::abs(lin_data[4] - 10.0) < 1e-10);

    // Test linspace without endpoint
    auto lin_noend = Tensor::linspace(0.0, 10.0, 5, false);
    auto lin_noend_data = lin_noend.typed_data<double>();
    assert(std::abs(lin_noend_data[0] - 0.0) < 1e-10);
    assert(std::abs(lin_noend_data[1] - 2.0) < 1e-10);

    // Test logspace
    auto logs = Tensor::logspace(0.0, 2.0, 3);
    auto logs_data = logs.typed_data<double>();
    assert(std::abs(logs_data[0] - 1.0) < 1e-10);   // 10^0 = 1
    assert(std::abs(logs_data[1] - 10.0) < 1e-10);  // 10^1 = 10
    assert(std::abs(logs_data[2] - 100.0) < 1e-10); // 10^2 = 100

    // Test geomspace
    auto geom = Tensor::geomspace(1.0, 1000.0, 4);
    auto geom_data = geom.typed_data<double>();
    assert(std::abs(geom_data[0] - 1.0) < 1e-10);
    assert(std::abs(geom_data[1] - 10.0) < 1e-10);
    assert(std::abs(geom_data[2] - 100.0) < 1e-10);
    assert(std::abs(geom_data[3] - 1000.0) < 1e-10);

    // Test zeros_like, ones_like, empty_like
    auto proto = Tensor::zeros({2, 3}, DType::Float32, Device::CPU);
    auto zl = Tensor::zeros_like(proto);
    assert(zl.shape() == proto.shape());
    assert(zl.dtype() == proto.dtype());
    assert(zl.device() == proto.device());

    auto ol = Tensor::ones_like(proto);
    auto ol_data = ol.typed_data<float>();
    for (size_t i = 0; i < ol.size(); ++i) {
        assert(ol_data[i] == 1.0f);
    }

    auto el = Tensor::empty_like(proto);
    assert(el.shape() == proto.shape());

    auto fl = Tensor::full_like(proto, 7.5f);
    auto fl_data = fl.typed_data<float>();
    for (size_t i = 0; i < fl.size(); ++i) {
        assert(fl_data[i] == 7.5f);
    }

    // Test diag (construct from 1D)
    auto vec = Tensor::from_data(std::vector<float>{1, 2, 3}.data(), {3});
    auto diag_mat = Tensor::diag(vec);
    assert(diag_mat.shape()[0] == 3);
    assert(diag_mat.shape()[1] == 3);
    auto diag_data = diag_mat.typed_data<float>();
    assert(diag_data[0] == 1.0f); // [0,0]
    assert(diag_data[4] == 2.0f); // [1,1]
    assert(diag_data[8] == 3.0f); // [2,2]
    assert(diag_data[1] == 0.0f); // [0,1]

    // Test diag with offset
    auto diag_off = Tensor::diag(vec, 1);
    assert(diag_off.shape()[0] == 4);
    assert(diag_off.shape()[1] == 4);

    // Test diag (extract from 2D)
    auto mat = Tensor::eye(3, DType::Float32);
    auto extracted = Tensor::diag(mat);
    assert(extracted.size() == 3);
    auto ext_data = extracted.typed_data<float>();
    for (size_t i = 0; i < 3; ++i) {
        assert(ext_data[i] == 1.0f);
    }

    // Test tri
    auto tri_mat = Tensor::tri(3, 3, 0, DType::Float64);
    auto tri_data = tri_mat.typed_data<double>();
    assert(tri_data[0] == 1.0); // [0,0]
    assert(tri_data[1] == 0.0); // [0,1]
    assert(tri_data[3] == 1.0); // [1,0]
    assert(tri_data[4] == 1.0); // [1,1]
    assert(tri_data[5] == 0.0); // [1,2]

    // Test tril
    auto full_mat = Tensor::full<float>({3, 3}, 1.0f);
    auto lower = Tensor::tril(full_mat);
    auto lower_data = lower.typed_data<float>();
    assert(lower_data[0] == 1.0f); // [0,0]
    assert(lower_data[1] == 0.0f); // [0,1]
    assert(lower_data[2] == 0.0f); // [0,2]
    assert(lower_data[3] == 1.0f); // [1,0]
    assert(lower_data[4] == 1.0f); // [1,1]
    assert(lower_data[5] == 0.0f); // [1,2]

    // Test triu
    auto upper = Tensor::triu(full_mat);
    auto upper_data = upper.typed_data<float>();
    assert(upper_data[0] == 1.0f); // [0,0]
    assert(upper_data[1] == 1.0f); // [0,1]
    assert(upper_data[2] == 1.0f); // [0,2]
    assert(upper_data[3] == 0.0f); // [1,0]
    assert(upper_data[4] == 1.0f); // [1,1]
    assert(upper_data[5] == 1.0f); // [1,2]

    // Test tril/triu with offset
    auto lower_k1 = Tensor::tril(full_mat, 1);
    auto lower_k1_data = lower_k1.typed_data<float>();
    assert(lower_k1_data[2] == 0.0f); // [0,2] still zero
    assert(lower_k1_data[1] == 1.0f); // [0,1] now included

    auto upper_km1 = Tensor::triu(full_mat, -1);
    auto upper_km1_data = upper_km1.typed_data<float>();
    assert(upper_km1_data[3] == 1.0f); // [1,0] now included

    // Test unflatten (inverse of flatten)
    auto t3d = Tensor::zeros({2, 3, 4}, DType::Float32);
    auto flattened = t3d.flatten(1, 2); // shape: [2, 12]
    assert(flattened.shape()[0] == 2);
    assert(flattened.shape()[1] == 12);

    auto unflattened = flattened.unflatten(1, {3, 4}); // back to [2, 3, 4]
    assert(unflattened.shape()[0] == 2);
    assert(unflattened.shape()[1] == 3);
    assert(unflattened.shape()[2] == 4);

    // Test unflatten with different split
    auto unflat2 = flattened.unflatten(1, {2, 6}); // [2, 2, 6]
    assert(unflat2.shape()[0] == 2);
    assert(unflat2.shape()[1] == 2);
    assert(unflat2.shape()[2] == 6);

    // Test unflatten with three dimensions
    auto unflat3 = flattened.unflatten(1, {2, 2, 3}); // [2, 2, 2, 3]
    assert(unflat3.ndim() == 4);
    assert(unflat3.shape()[0] == 2);
    assert(unflat3.shape()[1] == 2);
    assert(unflat3.shape()[2] == 2);
    assert(unflat3.shape()[3] == 3);
}

//=============================================================================
// Data Access and Indexing Tests
//=============================================================================

void test_data_access_cpu() {
    auto t = Tensor::zeros({2, 3}, DType::Float32, Device::CPU);

    // Test direct data access
    void *raw_data = t.data();
    assert(raw_data != nullptr);

    const void *const_raw_data = const_cast<const Tensor &>(t).data();
    assert(const_raw_data != nullptr);

    // Test typed data access
    auto typed_data = t.typed_data<float>();
    assert(typed_data != nullptr);

    // Test element access and modification
    t.set_item<float>({0, 1}, 5.0f);
    float val = t.item<float>({0, 1});
    assert(val == 5.0f);

    t.set_item<float>({1, 2}, 3.14f);
    val = t.item<float>({1, 2});
    assert(std::abs(val - 3.14f) < 1e-6);

    // Test fill operation
    auto t2 = Tensor::empty({3, 3}, DType::Int32, Device::CPU);
    t2.fill<int32_t>(42);
    auto data = t2.typed_data<int32_t>();
    for (size_t i = 0; i < t2.size(); ++i) {
        assert(data[i] == 42);
    }

    // Test bounds checking
    bool caught_exception = false;
    try {
        t.item<float>({5, 5}); // Out of bounds
    } catch (const std::exception &) {
        caught_exception = true;
    }
    assert(caught_exception);
}

void test_data_access_gpu() {
    auto t = Tensor::zeros({2, 3}, DType::Float32, Device::GPU);

    // GPU tensors should not allow direct data access
    bool caught_exception = false;
    try {
        t.data();
    } catch (const std::exception &) {
        caught_exception = true;
    }
    assert(caught_exception);

    // Test that GPU tensor element access throws
    caught_exception = false;
    try {
        t.item<float>({0, 1});
    } catch (const std::exception &) {
        caught_exception = true;
    }
    assert(caught_exception);
}

//=============================================================================
// Shape Manipulation Tests
//=============================================================================

void test_shape_manipulation_cpu() {
    auto t = Tensor::zeros({2, 3, 4}, DType::Float32, Device::CPU);

    // Test reshape
    auto reshaped = t.reshape({6, 4});
    assert(reshaped.shape()[0] == 6);
    assert(reshaped.shape()[1] == 4);
    assert(reshaped.size() == t.size());
    assert(reshaped.memory_order() == MemoryOrder::RowMajor);

    // Test reshape with memory order
    auto reshaped_f = t.reshape({6, 4}, MemoryOrder::ColMajor);
    assert(reshaped_f.memory_order() == MemoryOrder::ColMajor);

    // Test reshape with -1 (inferred dimension)
    auto reshaped_infer = t.reshape({static_cast<unsigned long>(-1), 4});
    assert(reshaped_infer.shape()[0] == 6);
    assert(reshaped_infer.shape()[1] == 4);

    // Test transpose
    auto t2d = Tensor::zeros({3, 4}, DType::Float32, Device::CPU);

    // Test squeeze on a tensor with no dimensions of size 1 (should be a no-op)
    auto no_squeeze = t2d.squeeze(0);
    assert(no_squeeze.ndim() == 2);
    assert(no_squeeze.shape() == t2d.shape());

    auto transposed = t2d.transpose();
    assert(transposed.shape()[0] == 4);
    assert(transposed.shape()[1] == 3);

    // Test transpose with specific axes
    auto t3d = Tensor::zeros({2, 3, 4}, DType::Float32, Device::CPU);
    auto transposed_axes = t3d.transpose({2, 0, 1});
    assert(transposed_axes.shape()[0] == 4);
    assert(transposed_axes.shape()[1] == 2);
    assert(transposed_axes.shape()[2] == 3);

    // Test squeeze
    auto t_with_ones = Tensor::zeros({1, 3, 1, 4}, DType::Float32, Device::CPU);
    auto squeezed = t_with_ones.squeeze();
    assert(squeezed.ndim() == 2);
    assert(squeezed.shape()[0] == 3);
    assert(squeezed.shape()[1] == 4);

    // Test squeeze specific axis
    auto squeezed_axis = t_with_ones.squeeze(0);
    assert(squeezed_axis.ndim() == 3);
    assert(squeezed_axis.shape()[0] == 3);
    assert(squeezed_axis.shape()[1] == 1);
    assert(squeezed_axis.shape()[2] == 4);

    // Test unsqueeze
    auto t2 = Tensor::zeros({3, 4}, DType::Float32, Device::CPU);
    auto unsqueezed = t2.unsqueeze(1);
    assert(unsqueezed.ndim() == 3);
    assert(unsqueezed.shape()[0] == 3);
    assert(unsqueezed.shape()[1] == 1);
    assert(unsqueezed.shape()[2] == 4);

    // Test unsqueeze with negative axis
    auto unsqueezed_neg = t2.unsqueeze(-1);
    assert(unsqueezed_neg.ndim() == 3);
    assert(unsqueezed_neg.shape()[2] == 1);

    // Test view
    auto view = t.view({24});
    assert(view.ndim() == 1);
    assert(view.shape()[0] == 24);
    assert(view.size() == t.size());
}

void test_shape_manipulation_gpu() {
    auto t = Tensor::zeros({2, 3, 4}, DType::Float32, Device::GPU);

    // Test reshape on GPU
    auto reshaped = t.reshape({6, 4});
    assert(reshaped.device() == Device::GPU);
    assert(reshaped.shape()[0] == 6);
    assert(reshaped.shape()[1] == 4);

    // Test transpose on GPU
    auto t2d = Tensor::zeros({3, 4}, DType::Float32, Device::GPU);
    auto transposed = t2d.transpose();
    assert(transposed.device() == Device::GPU);
    assert(transposed.shape()[0] == 4);
    assert(transposed.shape()[1] == 3);

    // Test squeeze on GPU
    auto t_with_ones = Tensor::zeros({1, 3, 1, 4}, DType::Float32, Device::GPU);
    auto squeezed = t_with_ones.squeeze();
    assert(squeezed.device() == Device::GPU);
    assert(squeezed.ndim() == 2);

    // Test unsqueeze on GPU
    auto unsqueezed = t2d.unsqueeze(1);
    assert(unsqueezed.device() == Device::GPU);
    assert(unsqueezed.ndim() == 3);
}

//=============================================================================
// Memory Operations Tests
//=============================================================================

void test_memory_operations_cpu() {
    auto t1 = Tensor::ones({2, 3}, DType::Float32, Device::CPU);

    // Test copy
    auto t2 = t1.copy();
    assert(t2.same_shape(t1));
    assert(t2.same_dtype(t1));
    assert(t2.same_device(t1));
    assert(t2.same_memory_order(t1));

    // Modify original, copy should be unchanged
    t1.fill<float>(5.0f);
    auto t1_data = t1.typed_data<float>();
    auto t2_data = t2.typed_data<float>();

    assert(t1_data[0] == 5.0f);
    assert(t2_data[0] == 1.0f);

    // Test copy with different memory order
    auto t3 = t1.copy(MemoryOrder::ColMajor);
    assert(t3.memory_order() == MemoryOrder::ColMajor);
    assert(t3.is_f_contiguous());

    // Test clone (alias for copy)
    auto t4 = t1.clone();
    assert(t4.same_shape(t1));

    // Test device transfer (CPU to CPU should be no-op)
    auto t5 = t1.to(Device::CPU);
    assert(t5.device() == Device::CPU);

    // Test cpu() method
    auto t6 = t1.cpu();
    assert(t6.device() == Device::CPU);
}

void test_memory_operations_gpu() {
    auto cpu_tensor = Tensor::ones({2, 3}, DType::Float32, Device::CPU);

    // Test CPU to GPU transfer
    auto gpu_tensor = cpu_tensor.gpu();
    assert(gpu_tensor.device() == Device::GPU);
    assert(gpu_tensor.same_shape(cpu_tensor));
    assert(gpu_tensor.same_dtype(cpu_tensor));

    // Test GPU copy
    auto gpu_copy = gpu_tensor.copy();
    assert(gpu_copy.device() == Device::GPU);
    assert(gpu_copy.same_shape(gpu_tensor));

    // Test GPU to CPU transfer
    auto back_to_cpu = gpu_tensor.cpu();
    assert(back_to_cpu.device() == Device::CPU);

    // Verify data integrity through round trip
    auto cpu_data = back_to_cpu.typed_data<float>();
    for (size_t i = 0; i < back_to_cpu.size(); ++i) {
        assert(cpu_data[i] == 1.0f);
    }

    // Test direct device transfer
    auto gpu_tensor2 = cpu_tensor.to(Device::GPU);
    assert(gpu_tensor2.device() == Device::GPU);

    // Test transfer with memory order change
    auto gpu_f = cpu_tensor.to(Device::GPU, MemoryOrder::ColMajor);
    assert(gpu_f.device() == Device::GPU);
    assert(gpu_f.memory_order() == MemoryOrder::ColMajor);
}

//=============================================================================
// Memory Order Tests
//=============================================================================

void test_memory_order_cpu() {
    // Test C-order creation and properties
    auto c_tensor = Tensor::zeros({3, 4}, DType::Float32, Device::CPU,
                                  MemoryOrder::RowMajor);
    assert(c_tensor.memory_order() == MemoryOrder::RowMajor);
    assert(c_tensor.is_c_contiguous());
    assert(!c_tensor.is_f_contiguous());

    // Test F-order creation and properties
    auto f_tensor = Tensor::zeros({3, 4}, DType::Float32, Device::CPU,
                                  MemoryOrder::ColMajor);
    assert(f_tensor.memory_order() == MemoryOrder::ColMajor);
    assert(f_tensor.is_f_contiguous());
    assert(!f_tensor.is_c_contiguous());

    // Test stride patterns
    auto c_strides = c_tensor.strides();
    auto f_strides = f_tensor.strides();

    // C-order: [16, 4] for 3x4 float32 matrix
    assert(c_strides[1] == 4);  // Column stride = itemsize
    assert(c_strides[0] == 16); // Row stride = 4 * itemsize

    // F-order: [4, 12] for 3x4 float32 matrix
    assert(f_strides[0] == 4);  // Row stride = itemsize
    assert(f_strides[1] == 12); // Column stride = 3 * itemsize

    // Test conversion functions
    auto c_to_f = c_tensor.asfortranarray();
    assert(c_to_f.is_f_contiguous());
    assert(c_to_f.memory_order() == MemoryOrder::ColMajor);

    auto f_to_c = f_tensor.ascontiguousarray();
    assert(f_to_c.is_c_contiguous());
    assert(f_to_c.memory_order() == MemoryOrder::RowMajor);

    // Test NumPy-style functions
    auto c_result = Tensor::ascontiguousarray(f_tensor);
    assert(c_result.is_c_contiguous());

    auto f_result = Tensor::asfortranarray(c_tensor);
    assert(f_result.is_f_contiguous());

    // Test idempotence
    auto c_result2 = Tensor::ascontiguousarray(c_tensor);
    assert(c_result2.storage() == c_tensor.storage());

    auto f_result2 = Tensor::asfortranarray(f_tensor);
    assert(f_result2.storage() == f_tensor.storage());
}

void test_memory_order_gpu() {
    // Test memory order preservation on GPU
    auto c_gpu = Tensor::zeros({3, 4}, DType::Float32, Device::GPU,
                               MemoryOrder::RowMajor);
    assert(c_gpu.memory_order() == MemoryOrder::RowMajor);
    assert(c_gpu.device() == Device::GPU);

    auto f_gpu = Tensor::zeros({3, 4}, DType::Float32, Device::GPU,
                               MemoryOrder::ColMajor);
    assert(f_gpu.memory_order() == MemoryOrder::ColMajor);
    assert(f_gpu.device() == Device::GPU);

    // Test conversion on GPU
    auto c_to_f_gpu = c_gpu.asfortranarray();
    assert(c_to_f_gpu.device() == Device::GPU);
    assert(c_to_f_gpu.memory_order() == MemoryOrder::ColMajor);

    // Test device transfer with memory order
    auto cpu_c = Tensor::ones({2, 3}, DType::Float32, Device::CPU,
                              MemoryOrder::RowMajor);
    auto gpu_f = cpu_c.to(Device::GPU, MemoryOrder::ColMajor);
    assert(gpu_f.device() == Device::GPU);
    assert(gpu_f.memory_order() == MemoryOrder::ColMajor);
}

//=============================================================================
// Data Type System Tests
//=============================================================================

void test_dtype_system() {
    // Test different data types
    auto t_f32 = Tensor::zeros({2, 2}, DType::Float32, Device::CPU);
    auto t_f64 = Tensor::zeros({2, 2}, DType::Float64, Device::CPU);
    auto t_i32 = Tensor::zeros({2, 2}, DType::Int32, Device::CPU);
    auto t_i64 = Tensor::zeros({2, 2}, DType::Int64, Device::CPU);
    auto t_bool = Tensor::zeros({2, 2}, DType::Bool, Device::CPU);

    // Test itemsize
    assert(t_f32.itemsize() == 4);
    assert(t_f64.itemsize() == 8);
    assert(t_i32.itemsize() == 4);
    assert(t_i64.itemsize() == 8);
    assert(t_bool.itemsize() == 1);

    // Test dtype names
    assert(t_f32.dtype_name() == "float32");
    assert(t_f64.dtype_name() == "float64");
    assert(t_i32.dtype_name() == "int32");
    assert(t_i64.dtype_name() == "int64");
    assert(t_bool.dtype_name() == "bool");

    // Test nbytes calculation
    assert(t_f32.nbytes() == 16); // 4 elements * 4 bytes
    assert(t_f64.nbytes() == 32); // 4 elements * 8 bytes

    // Test automatic dtype deduction (compile-time)
    static_assert(dtype_of_v<float> == DType::Float32);
    static_assert(dtype_of_v<double> == DType::Float64);
    static_assert(dtype_of_v<int32_t> == DType::Int32);
    static_assert(dtype_of_v<int64_t> == DType::Int64);
    static_assert(dtype_of_v<bool> == DType::Bool);

    // Test dtype comparison
    assert(t_f32.same_dtype(t_f32));
    assert(!t_f32.same_dtype(t_f64));
    assert(!t_f32.same_dtype(t_i32));
}

//=============================================================================
// Views and Memory Sharing Tests
//=============================================================================

void test_views_and_sharing_cpu() {
    auto base = Tensor::zeros({2, 3, 4}, DType::Float32, Device::CPU);
    base.fill<float>(1.0f);

    // Test reshape view (should share memory for contiguous tensors)
    auto reshaped = base.reshape({6, 4});

    // Modify via reshaped view
    reshaped.set_item<float>({0, 0}, 99.0f);
    float val = base.item<float>({0, 0, 0});
    assert(val == 99.0f); // Change should be visible in base

    // Test view method
    auto view = base.view({24});
    assert(view.ndim() == 1);
    assert(view.shape()[0] == 24);

    // Modify via view
    view.set_item<float>({1}, 77.0f);
    float val2 = base.item<float>({0, 0, 1});
    assert(val2 == 77.0f);

    // Test transpose view
    auto t2d = Tensor::zeros({3, 4}, DType::Float32, Device::CPU);
    t2d.set_item<float>({1, 2}, 42.0f);

    auto transposed = t2d.transpose();
    float val3 = transposed.item<float>({2, 1});
    assert(val3 == 42.0f);

    // Test squeeze/unsqueeze views
    auto with_ones = Tensor::zeros({1, 3, 1}, DType::Float32, Device::CPU);
    with_ones.set_item<float>({0, 1, 0}, 123.0f);

    auto squeezed = with_ones.squeeze();
    float val4 = squeezed.item<float>({1});
    assert(val4 == 123.0f);

    auto unsqueezed = squeezed.unsqueeze(0);
    float val5 = unsqueezed.item<float>({0, 1});
    assert(val5 == 123.0f);

    // Test that copy creates independent data
    auto copied = base.copy();
    copied.set_item<float>({0, 0, 0}, 555.0f);
    float original_val = base.item<float>({0, 0, 0});
    float copied_val = copied.item<float>({0, 0, 0});
    assert(original_val == 99.0f); // Original unchanged
    assert(copied_val == 555.0f);  // Copy modified
}

void test_views_and_sharing_gpu() {
    auto base = Tensor::zeros({2, 3, 4}, DType::Float32, Device::GPU);

    // Test reshape on GPU (should work for views)
    auto reshaped = base.reshape({6, 4});
    assert(reshaped.device() == Device::GPU);
    assert(reshaped.same_shape(Tensor({6, 4})));

    // Test view on GPU
    auto view = base.view({24});
    assert(view.device() == Device::GPU);
    assert(view.ndim() == 1);

    // Test transpose on GPU
    auto t2d = Tensor::zeros({3, 4}, DType::Float32, Device::GPU);
    auto transposed = t2d.transpose();
    assert(transposed.device() == Device::GPU);
    assert(transposed.shape()[0] == 4);
    assert(transposed.shape()[1] == 3);
}

//=============================================================================
// Error Handling Tests
//=============================================================================

void test_error_handling() {
    // Test invalid shape
    bool caught = false;
    try {
        auto t = Tensor({0}, DType::Float32, Device::CPU); // Zero dimension
    } catch (const std::exception &) {
        // This might be valid for some implementations
    }

    // Test out of bounds access
    auto t = Tensor::zeros({2, 3}, DType::Float32, Device::CPU);
    caught = false;
    try {
        t.item<float>({5, 5});
    } catch (const std::exception &) {
        caught = true;
    }
    assert(caught);

    // Test invalid reshape
    caught = false;
    try {
        t.reshape({7}); // 6 elements can't reshape to 7
    } catch (const std::exception &) {
        caught = true;
    }
    assert(caught);

    // Test invalid squeeze (should be a no-op)
    auto no_squeeze = t.squeeze(0);
    assert(no_squeeze.shape() == t.shape());

    // Test GPU data access
    if (is_gpu_available()) {
        auto gpu_tensor = Tensor::zeros({2, 2}, DType::Float32, Device::GPU);
        caught = false;
        try {
            gpu_tensor.data();
        } catch (const std::exception &) {
            caught = true;
        }
        assert(caught);
    }

    // Test invalid view
    caught = false;
    try {
        t.view({7}); // Can't view 6 elements as 7
    } catch (const std::exception &) {
        caught = true;
    }
    assert(caught);
}

//=============================================================================
// Performance and Stress Tests
//=============================================================================

void test_large_tensors() {
    // Test large tensor creation and basic operations
    const size_t large_size = 1000;

    auto large_tensor =
        Tensor::zeros({large_size, large_size}, DType::Float32, Device::CPU);
    assert(large_tensor.size() == large_size * large_size);
    assert(large_tensor.nbytes() == large_size * large_size * 4);

    // Test reshape of large tensor
    auto reshaped = large_tensor.reshape({large_size * large_size});
    assert(reshaped.ndim() == 1);
    assert(reshaped.size() == large_size * large_size);

    // Test copy of large tensor
    auto copied = large_tensor.copy();
    assert(copied.same_shape(large_tensor));

    if (is_gpu_available()) {
        // Test GPU transfer of large tensor
        auto gpu_large = large_tensor.gpu();
        assert(gpu_large.device() == Device::GPU);
        assert(gpu_large.same_shape(large_tensor));
    }
}

void test_memory_order_performance() {
    const size_t size = 200;

    // Create matrices in both orders
    auto c_matrix = Tensor::zeros({size, size}, DType::Float32, Device::CPU,
                                  MemoryOrder::RowMajor);
    auto f_matrix = Tensor::zeros({size, size}, DType::Float32, Device::CPU,
                                  MemoryOrder::ColMajor);

    // Test that memory orders are correct
    assert(c_matrix.is_c_contiguous());
    assert(f_matrix.is_f_contiguous());

    // Test conversion performance (not timing, just functionality)
    auto c_to_f = c_matrix.asfortranarray();
    assert(c_to_f.is_f_contiguous());

    auto f_to_c = f_matrix.ascontiguousarray();
    assert(f_to_c.is_c_contiguous());

    // Test that data is preserved during conversion
    c_matrix.set_item<float>({10, 20}, 42.0f);
    auto converted = c_matrix.asfortranarray();
    assert(converted.item<float>({10, 20}) == 42.0f);
}

//=============================================================================
// Integration Tests
//=============================================================================

void test_cross_device_workflow() {
    if (!is_gpu_available())
        return;

    // Create data on CPU
    auto cpu_data = Tensor::ones({10, 10}, DType::Float32, Device::CPU);

    // Modify some data
    for (size_t i = 0; i < 10; ++i) {
        cpu_data.set_item<float>({i, i}, static_cast<float>(i));
    }

    // Transfer to GPU
    auto gpu_data = cpu_data.gpu();
    assert(gpu_data.device() == Device::GPU);

    // Reshape on GPU
    auto gpu_reshaped = gpu_data.reshape({100});
    assert(gpu_reshaped.device() == Device::GPU);
    assert(gpu_reshaped.ndim() == 1);

    // Transfer back to CPU
    auto result = gpu_reshaped.cpu();
    assert(result.device() == Device::CPU);
    assert(result.ndim() == 1);
    assert(result.size() == 100);

    // Verify data integrity
    for (size_t i = 0; i < 10; ++i) {
        float expected = static_cast<float>(i);
        float actual = result.item<float>({i * 10 + i}); // Diagonal elements
        assert(actual == expected);
    }
}

void test_mixed_memory_orders() {
    // Test operations mixing different memory orders
    auto c_tensor = Tensor::ones({3, 4}, DType::Float32, Device::CPU,
                                 MemoryOrder::RowMajor);
    auto f_tensor = Tensor::ones({3, 4}, DType::Float32, Device::CPU,
                                 MemoryOrder::ColMajor);

    // Both should have same logical values
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            assert(c_tensor.item<float>({i, j}) == 1.0f);
            assert(f_tensor.item<float>({i, j}) == 1.0f);
        }
    }

    // Test conversions preserve data
    auto c_to_f = c_tensor.copy(MemoryOrder::ColMajor);
    auto f_to_c = f_tensor.copy(MemoryOrder::RowMajor);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            assert(c_to_f.item<float>({i, j}) == 1.0f);
            assert(f_to_c.item<float>({i, j}) == 1.0f);
        }
    }
}

void test_tensor_printing() {
    auto a =
        axiom::Tensor::arange(4).reshape({2, 2}).astype(axiom::DType::Float32);
    std::stringstream ss;
    ss << a;
    std::string expected = "[[0.0000 1.0000]\n [2.0000 3.0000]] "
                           "Tensor(shape=[2, 2], dtype=float32, device=CPU, "
                           "order=RowMajor)";
    assert(ss.str() == expected);
}

void test_tensor_printing_ellipsis() {
    auto a = axiom::Tensor::arange(100).reshape({10, 10});
    std::stringstream ss;
    ss << a;
    std::string expected = "[[0 1 2 ... 7 8 9]\n"
                           " [10 11 12 ... 17 18 19]\n"
                           " [20 21 22 ... 27 28 29]\n"
                           " ...\n"
                           " [70 71 72 ... 77 78 79]\n"
                           " [80 81 82 ... 87 88 89]\n"
                           " [90 91 92 ... 97 98 99]] "
                           "Tensor(shape=[10, 10], dtype=int32, device=CPU, "
                           "order=RowMajor)";
    assert(ss.str() == expected);
}

//=============================================================================
// Main Test Runner
//=============================================================================

int main() {
    std::cout << "=== Axiom Comprehensive Test Suite ===" << std::endl;
    std::cout << "GPU Available: " << (is_gpu_available() ? "Yes" : "No")
              << std::endl;
    std::cout << std::endl;

    TestRunner runner;

    // Basic functionality tests
    runner.run_test("Tensor Creation (CPU)", test_tensor_creation_cpu);
    if (is_gpu_available()) {
        runner.run_test("Tensor Creation (GPU)", test_tensor_creation_gpu);
    }

    // Factory function tests
    runner.run_test("Factory Functions (CPU)", test_factory_functions_cpu);
    if (is_gpu_available()) {
        runner.run_test("Factory Functions (GPU)", test_factory_functions_gpu);
    }

    // Creation routines tests (linspace, logspace, diag, tri, tril, triu, etc.)
    runner.run_test("Creation Routines (CPU)", test_creation_routines_cpu);

    // Data access tests
    runner.run_test("Data Access (CPU)", test_data_access_cpu);
    if (is_gpu_available()) {
        runner.run_test("Data Access (GPU)", test_data_access_gpu);
    }

    // Shape manipulation tests
    runner.run_test("Shape Manipulation (CPU)", test_shape_manipulation_cpu);
    if (is_gpu_available()) {
        runner.run_test("Shape Manipulation (GPU)",
                        test_shape_manipulation_gpu);
    }

    // Memory operation tests
    runner.run_test("Memory Operations (CPU)", test_memory_operations_cpu);
    if (is_gpu_available()) {
        runner.run_test("Memory Operations (GPU)", test_memory_operations_gpu);
    }

    // Memory order tests
    runner.run_test("Memory Order (CPU)", test_memory_order_cpu);
    if (is_gpu_available()) {
        runner.run_test("Memory Order (GPU)", test_memory_order_gpu);
    }

    // Data type tests
    runner.run_test("Data Type System", test_dtype_system);

    // Views and sharing tests
    runner.run_test("Views and Sharing (CPU)", test_views_and_sharing_cpu);
    if (is_gpu_available()) {
        runner.run_test("Views and Sharing (GPU)", test_views_and_sharing_gpu);
    }

    // Error handling tests
    runner.run_test("Error Handling", test_error_handling);

    // Performance and stress tests
    runner.run_test("Large Tensors", test_large_tensors);
    runner.run_test("Memory Order Performance", test_memory_order_performance);

    // Integration tests
    if (is_gpu_available()) {
        runner.run_test("Cross-Device Workflow", test_cross_device_workflow);
    }
    runner.run_test("Mixed Memory Orders", test_mixed_memory_orders);
    runner.run_test("Tensor Printing", test_tensor_printing);
    runner.run_test("Tensor Printing Ellipsis", test_tensor_printing_ellipsis);

    // Print final results
    std::cout << std::endl;
    runner.print_summary();

    return runner.all_passed() ? 0 : 1;
}