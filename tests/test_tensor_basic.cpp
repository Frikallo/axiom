//=============================================================================
// tests/test_tensor_basic.cpp - Complete test suite for Axiom tensors
//=============================================================================

#include "axiom_test_utils.hpp"
#include <iomanip>
#include <sstream>
#include <vector>

using namespace axiom;

//=============================================================================
// Basic Tensor Tests
//=============================================================================

TEST(TensorBasic, TensorCreationCpu) {
    // Test basic constructors
    auto t1 = Tensor({3, 4}, DType::Float32, Device::CPU);
    ASSERT_EQ(t1.ndim(), 2u);
    ASSERT_EQ(t1.shape()[0], 3u);
    ASSERT_EQ(t1.shape()[1], 4u);
    ASSERT_EQ(t1.size(), 12u);
    ASSERT_EQ(t1.dtype(), DType::Float32);
    ASSERT_EQ(t1.device(), Device::CPU);
    ASSERT_EQ(t1.memory_order(), MemoryOrder::RowMajor);

    // Test initializer list constructor
    auto t2 = Tensor({2, 3, 4}, DType::Int32, Device::CPU);
    ASSERT_EQ(t2.ndim(), 3u);
    ASSERT_EQ(t2.size(), 24u);
    ASSERT_EQ(t2.dtype(), DType::Int32);

    // Test with memory order
    auto t3 =
        Tensor({2, 3}, DType::Float64, Device::CPU, MemoryOrder::ColMajor);
    ASSERT_EQ(t3.memory_order(), MemoryOrder::ColMajor);
    ASSERT_TRUE(t3.is_f_contiguous());

    // Test empty tensor
    auto t4 = Tensor();
    ASSERT_EQ(t4.ndim(), 0u);
    ASSERT_EQ(t4.size(), 1u);
}

TEST(TensorBasic, TensorCreationGpu) {
    SKIP_IF_NO_GPU();

    auto t1 = Tensor({3, 4}, DType::Float32, Device::GPU);
    ASSERT_EQ(t1.ndim(), 2u);
    ASSERT_EQ(t1.shape()[0], 3u);
    ASSERT_EQ(t1.shape()[1], 4u);
    ASSERT_EQ(t1.size(), 12u);
    ASSERT_EQ(t1.dtype(), DType::Float32);
    ASSERT_EQ(t1.device(), Device::GPU);

    // Test with different memory orders
    auto t2 =
        Tensor({2, 3}, DType::Float32, Device::GPU, MemoryOrder::ColMajor);
    ASSERT_EQ(t2.memory_order(), MemoryOrder::ColMajor);
    ASSERT_EQ(t2.device(), Device::GPU);
}

//=============================================================================
// Factory Function Tests
//=============================================================================

TEST(TensorBasic, FactoryFunctionsCpu) {
    // Test zeros
    auto z = Tensor::zeros({2, 3}, DType::Float32, Device::CPU);
    ASSERT_EQ(z.shape()[0], 2u);
    ASSERT_EQ(z.shape()[1], 3u);
    ASSERT_TRUE(z.is_c_contiguous());

    auto data = z.typed_data<float>();
    for (size_t i = 0; i < z.size(); ++i) {
        ASSERT_EQ(data[i], 0.0f);
    }

    // Test zeros with Fortran order
    auto z_f = Tensor::zeros({2, 3}, DType::Float32, Device::CPU,
                             MemoryOrder::ColMajor);
    ASSERT_TRUE(z_f.is_f_contiguous());
    ASSERT_EQ(z_f.memory_order(), MemoryOrder::ColMajor);

    // Test ones
    auto o = Tensor::ones({2, 2}, DType::Float32, Device::CPU);
    auto ones_data = o.typed_data<float>();
    for (size_t i = 0; i < o.size(); ++i) {
        ASSERT_EQ(ones_data[i], 1.0f);
    }

    // Test ones with different dtypes
    auto o_int = Tensor::ones({2, 2}, DType::Int32, Device::CPU);
    auto int_data = o_int.typed_data<int32_t>();
    for (size_t i = 0; i < o_int.size(); ++i) {
        ASSERT_EQ(int_data[i], 1);
    }

    // Test eye/identity
    auto eye_mat = Tensor::eye(3, DType::Float32, Device::CPU);
    auto eye_data = eye_mat.typed_data<float>();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            ASSERT_EQ(eye_data[i * 3 + j], expected);
        }
    }

    // Test identity (alias for eye)
    auto id_mat = Tensor::identity(2, DType::Float64, Device::CPU);
    ASSERT_EQ(id_mat.shape()[0], 2u);
    ASSERT_EQ(id_mat.shape()[1], 2u);
    ASSERT_EQ(id_mat.dtype(), DType::Float64);

    // Test empty
    auto e = Tensor::empty({3, 3}, DType::Int32, Device::CPU);
    ASSERT_EQ(e.shape()[0], 3u);
    ASSERT_EQ(e.shape()[1], 3u);
    ASSERT_EQ(e.dtype(), DType::Int32);

    // Test full
    auto f = Tensor::full<float>({2, 2}, 42.0f, Device::CPU);
    auto full_data = f.typed_data<float>();
    for (size_t i = 0; i < f.size(); ++i) {
        ASSERT_EQ(full_data[i], 42.0f);
    }
}

TEST(TensorBasic, FactoryFunctionsGpu) {
    SKIP_IF_NO_GPU();

    // Test zeros on GPU
    auto z = Tensor::zeros({2, 3}, DType::Float32, Device::GPU);
    ASSERT_EQ(z.device(), Device::GPU);
    ASSERT_EQ(z.shape()[0], 2u);
    ASSERT_EQ(z.shape()[1], 3u);

    // Test ones on GPU
    auto o = Tensor::ones({2, 2}, DType::Float32, Device::GPU);
    ASSERT_EQ(o.device(), Device::GPU);

    // Test eye on GPU
    auto eye_mat = Tensor::eye(3, DType::Float32, Device::GPU);
    ASSERT_EQ(eye_mat.device(), Device::GPU);
    ASSERT_EQ(eye_mat.shape()[0], 3u);
    ASSERT_EQ(eye_mat.shape()[1], 3u);

    // Test empty on GPU
    auto e = Tensor::empty({3, 3}, DType::Int32, Device::GPU);
    ASSERT_EQ(e.device(), Device::GPU);

    // Test with different memory orders
    auto z_f = Tensor::zeros({2, 3}, DType::Float32, Device::GPU,
                             MemoryOrder::ColMajor);
    ASSERT_EQ(z_f.device(), Device::GPU);
    ASSERT_EQ(z_f.memory_order(), MemoryOrder::ColMajor);
}

TEST(TensorBasic, CreationRoutinesCpu) {
    // Test linspace
    auto lin = Tensor::linspace(0.0, 10.0, 5);
    ASSERT_EQ(lin.size(), 5u);
    auto lin_data = lin.typed_data<double>();
    ASSERT_NEAR(lin_data[0], 0.0, 1e-10);
    ASSERT_NEAR(lin_data[1], 2.5, 1e-10);
    ASSERT_NEAR(lin_data[2], 5.0, 1e-10);
    ASSERT_NEAR(lin_data[3], 7.5, 1e-10);
    ASSERT_NEAR(lin_data[4], 10.0, 1e-10);

    // Test linspace without endpoint
    auto lin_noend = Tensor::linspace(0.0, 10.0, 5, false);
    auto lin_noend_data = lin_noend.typed_data<double>();
    ASSERT_NEAR(lin_noend_data[0], 0.0, 1e-10);
    ASSERT_NEAR(lin_noend_data[1], 2.0, 1e-10);

    // Test logspace
    auto logs = Tensor::logspace(0.0, 2.0, 3);
    auto logs_data = logs.typed_data<double>();
    ASSERT_NEAR(logs_data[0], 1.0, 1e-10);   // 10^0 = 1
    ASSERT_NEAR(logs_data[1], 10.0, 1e-10);  // 10^1 = 10
    ASSERT_NEAR(logs_data[2], 100.0, 1e-10); // 10^2 = 100

    // Test geomspace
    auto geom = Tensor::geomspace(1.0, 1000.0, 4);
    auto geom_data = geom.typed_data<double>();
    ASSERT_NEAR(geom_data[0], 1.0, 1e-10);
    ASSERT_NEAR(geom_data[1], 10.0, 1e-10);
    ASSERT_NEAR(geom_data[2], 100.0, 1e-10);
    ASSERT_NEAR(geom_data[3], 1000.0, 1e-10);

    // Test zeros_like, ones_like, empty_like
    auto proto = Tensor::zeros({2, 3}, DType::Float32, Device::CPU);
    auto zl = Tensor::zeros_like(proto);
    ASSERT_TRUE(zl.shape() == proto.shape());
    ASSERT_EQ(zl.dtype(), proto.dtype());
    ASSERT_EQ(zl.device(), proto.device());

    auto ol = Tensor::ones_like(proto);
    auto ol_data = ol.typed_data<float>();
    for (size_t i = 0; i < ol.size(); ++i) {
        ASSERT_EQ(ol_data[i], 1.0f);
    }

    auto el = Tensor::empty_like(proto);
    ASSERT_TRUE(el.shape() == proto.shape());

    auto fl = Tensor::full_like(proto, 7.5f);
    auto fl_data = fl.typed_data<float>();
    for (size_t i = 0; i < fl.size(); ++i) {
        ASSERT_EQ(fl_data[i], 7.5f);
    }

    // Test diag (construct from 1D)
    auto vec = Tensor::from_data(std::vector<float>{1, 2, 3}.data(), {3});
    auto diag_mat = Tensor::diag(vec);
    ASSERT_EQ(diag_mat.shape()[0], 3u);
    ASSERT_EQ(diag_mat.shape()[1], 3u);
    auto diag_data = diag_mat.typed_data<float>();
    ASSERT_EQ(diag_data[0], 1.0f); // [0,0]
    ASSERT_EQ(diag_data[4], 2.0f); // [1,1]
    ASSERT_EQ(diag_data[8], 3.0f); // [2,2]
    ASSERT_EQ(diag_data[1], 0.0f); // [0,1]

    // Test diag with offset
    auto diag_off = Tensor::diag(vec, 1);
    ASSERT_EQ(diag_off.shape()[0], 4u);
    ASSERT_EQ(diag_off.shape()[1], 4u);

    // Test diag (extract from 2D)
    auto mat = Tensor::eye(3, DType::Float32);
    auto extracted = Tensor::diag(mat);
    ASSERT_EQ(extracted.size(), 3u);
    auto ext_data = extracted.typed_data<float>();
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(ext_data[i], 1.0f);
    }

    // Test tri
    auto tri_mat = Tensor::tri(3, 3, 0, DType::Float64);
    auto tri_data = tri_mat.typed_data<double>();
    ASSERT_EQ(tri_data[0], 1.0); // [0,0]
    ASSERT_EQ(tri_data[1], 0.0); // [0,1]
    ASSERT_EQ(tri_data[3], 1.0); // [1,0]
    ASSERT_EQ(tri_data[4], 1.0); // [1,1]
    ASSERT_EQ(tri_data[5], 0.0); // [1,2]

    // Test tril
    auto full_mat = Tensor::full<float>({3, 3}, 1.0f);
    auto lower = Tensor::tril(full_mat);
    auto lower_data = lower.typed_data<float>();
    ASSERT_EQ(lower_data[0], 1.0f); // [0,0]
    ASSERT_EQ(lower_data[1], 0.0f); // [0,1]
    ASSERT_EQ(lower_data[2], 0.0f); // [0,2]
    ASSERT_EQ(lower_data[3], 1.0f); // [1,0]
    ASSERT_EQ(lower_data[4], 1.0f); // [1,1]
    ASSERT_EQ(lower_data[5], 0.0f); // [1,2]

    // Test triu
    auto upper = Tensor::triu(full_mat);
    auto upper_data = upper.typed_data<float>();
    ASSERT_EQ(upper_data[0], 1.0f); // [0,0]
    ASSERT_EQ(upper_data[1], 1.0f); // [0,1]
    ASSERT_EQ(upper_data[2], 1.0f); // [0,2]
    ASSERT_EQ(upper_data[3], 0.0f); // [1,0]
    ASSERT_EQ(upper_data[4], 1.0f); // [1,1]
    ASSERT_EQ(upper_data[5], 1.0f); // [1,2]

    // Test tril/triu with offset
    auto lower_k1 = Tensor::tril(full_mat, 1);
    auto lower_k1_data = lower_k1.typed_data<float>();
    ASSERT_EQ(lower_k1_data[2], 0.0f); // [0,2] still zero
    ASSERT_EQ(lower_k1_data[1], 1.0f); // [0,1] now included

    auto upper_km1 = Tensor::triu(full_mat, -1);
    auto upper_km1_data = upper_km1.typed_data<float>();
    ASSERT_EQ(upper_km1_data[3], 1.0f); // [1,0] now included

    // Test unflatten (inverse of flatten)
    auto t3d = Tensor::zeros({2, 3, 4}, DType::Float32);
    auto flattened = t3d.flatten(1, 2); // shape: [2, 12]
    ASSERT_EQ(flattened.shape()[0], 2u);
    ASSERT_EQ(flattened.shape()[1], 12u);

    auto unflattened = flattened.unflatten(1, {3, 4}); // back to [2, 3, 4]
    ASSERT_EQ(unflattened.shape()[0], 2u);
    ASSERT_EQ(unflattened.shape()[1], 3u);
    ASSERT_EQ(unflattened.shape()[2], 4u);

    // Test unflatten with different split
    auto unflat2 = flattened.unflatten(1, {2, 6}); // [2, 2, 6]
    ASSERT_EQ(unflat2.shape()[0], 2u);
    ASSERT_EQ(unflat2.shape()[1], 2u);
    ASSERT_EQ(unflat2.shape()[2], 6u);

    // Test unflatten with three dimensions
    auto unflat3 = flattened.unflatten(1, {2, 2, 3}); // [2, 2, 2, 3]
    ASSERT_EQ(unflat3.ndim(), 4u);
    ASSERT_EQ(unflat3.shape()[0], 2u);
    ASSERT_EQ(unflat3.shape()[1], 2u);
    ASSERT_EQ(unflat3.shape()[2], 2u);
    ASSERT_EQ(unflat3.shape()[3], 3u);
}

//=============================================================================
// Data Access and Indexing Tests
//=============================================================================

TEST(TensorBasic, DataAccessCpu) {
    auto t = Tensor::zeros({2, 3}, DType::Float32, Device::CPU);

    // Test direct data access
    void *raw_data = t.data();
    ASSERT_TRUE(raw_data != nullptr);

    const void *const_raw_data = const_cast<const Tensor &>(t).data();
    ASSERT_TRUE(const_raw_data != nullptr);

    // Test typed data access
    auto typed_data = t.typed_data<float>();
    ASSERT_TRUE(typed_data != nullptr);

    // Test element access and modification
    t.set_item<float>({0, 1}, 5.0f);
    float val = t.item<float>({0, 1});
    ASSERT_EQ(val, 5.0f);

    t.set_item<float>({1, 2}, 3.14f);
    val = t.item<float>({1, 2});
    ASSERT_NEAR(val, 3.14f, 1e-6);

    // Test fill operation
    auto t2 = Tensor::empty({3, 3}, DType::Int32, Device::CPU);
    t2.fill<int32_t>(42);
    auto data = t2.typed_data<int32_t>();
    for (size_t i = 0; i < t2.size(); ++i) {
        ASSERT_EQ(data[i], 42);
    }

    // Test bounds checking
    bool caught_exception = false;
    try {
        t.item<float>({5, 5}); // Out of bounds
    } catch (const std::exception &) {
        caught_exception = true;
    }
    ASSERT_TRUE(caught_exception);
}

TEST(TensorBasic, DataAccessGpu) {
    SKIP_IF_NO_GPU();

    auto t = Tensor::zeros({2, 3}, DType::Float32, Device::GPU);

    // GPU tensors should not allow direct data access
    bool caught_exception = false;
    try {
        t.data();
    } catch (const std::exception &) {
        caught_exception = true;
    }
    ASSERT_TRUE(caught_exception);

    // Test that GPU tensor element access throws
    caught_exception = false;
    try {
        t.item<float>({0, 1});
    } catch (const std::exception &) {
        caught_exception = true;
    }
    ASSERT_TRUE(caught_exception);
}

//=============================================================================
// Shape Manipulation Tests
//=============================================================================

TEST(TensorBasic, ShapeManipulationCpu) {
    auto t = Tensor::zeros({2, 3, 4}, DType::Float32, Device::CPU);

    // Test reshape
    auto reshaped = t.reshape({6, 4});
    ASSERT_EQ(reshaped.shape()[0], 6u);
    ASSERT_EQ(reshaped.shape()[1], 4u);
    ASSERT_EQ(reshaped.size(), t.size());
    ASSERT_EQ(reshaped.memory_order(), MemoryOrder::RowMajor);

    // Test reshape with memory order
    auto reshaped_f = t.reshape({6, 4}, MemoryOrder::ColMajor);
    ASSERT_EQ(reshaped_f.memory_order(), MemoryOrder::ColMajor);

    // Test reshape with -1 (inferred dimension)
    auto reshaped_infer = t.reshape({static_cast<unsigned long>(-1), 4});
    ASSERT_EQ(reshaped_infer.shape()[0], 6u);
    ASSERT_EQ(reshaped_infer.shape()[1], 4u);

    // Test transpose
    auto t2d = Tensor::zeros({3, 4}, DType::Float32, Device::CPU);

    // Test squeeze on a tensor with no dimensions of size 1 (should be no-op)
    auto no_squeeze = t2d.squeeze(0);
    ASSERT_EQ(no_squeeze.ndim(), 2u);
    ASSERT_TRUE(no_squeeze.shape() == t2d.shape());

    auto transposed = t2d.transpose();
    ASSERT_EQ(transposed.shape()[0], 4u);
    ASSERT_EQ(transposed.shape()[1], 3u);

    // Test transpose with specific axes
    auto t3d = Tensor::zeros({2, 3, 4}, DType::Float32, Device::CPU);
    auto transposed_axes = t3d.transpose({2, 0, 1});
    ASSERT_EQ(transposed_axes.shape()[0], 4u);
    ASSERT_EQ(transposed_axes.shape()[1], 2u);
    ASSERT_EQ(transposed_axes.shape()[2], 3u);

    // Test squeeze
    auto t_with_ones = Tensor::zeros({1, 3, 1, 4}, DType::Float32, Device::CPU);
    auto squeezed = t_with_ones.squeeze();
    ASSERT_EQ(squeezed.ndim(), 2u);
    ASSERT_EQ(squeezed.shape()[0], 3u);
    ASSERT_EQ(squeezed.shape()[1], 4u);

    // Test squeeze specific axis
    auto squeezed_axis = t_with_ones.squeeze(0);
    ASSERT_EQ(squeezed_axis.ndim(), 3u);
    ASSERT_EQ(squeezed_axis.shape()[0], 3u);
    ASSERT_EQ(squeezed_axis.shape()[1], 1u);
    ASSERT_EQ(squeezed_axis.shape()[2], 4u);

    // Test unsqueeze
    auto t2 = Tensor::zeros({3, 4}, DType::Float32, Device::CPU);
    auto unsqueezed = t2.unsqueeze(1);
    ASSERT_EQ(unsqueezed.ndim(), 3u);
    ASSERT_EQ(unsqueezed.shape()[0], 3u);
    ASSERT_EQ(unsqueezed.shape()[1], 1u);
    ASSERT_EQ(unsqueezed.shape()[2], 4u);

    // Test unsqueeze with negative axis
    auto unsqueezed_neg = t2.unsqueeze(-1);
    ASSERT_EQ(unsqueezed_neg.ndim(), 3u);
    ASSERT_EQ(unsqueezed_neg.shape()[2], 1u);

    // Test view
    auto view = t.view({24});
    ASSERT_EQ(view.ndim(), 1u);
    ASSERT_EQ(view.shape()[0], 24u);
    ASSERT_EQ(view.size(), t.size());
}

TEST(TensorBasic, ShapeManipulationGpu) {
    SKIP_IF_NO_GPU();

    auto t = Tensor::zeros({2, 3, 4}, DType::Float32, Device::GPU);

    // Test reshape on GPU
    auto reshaped = t.reshape({6, 4});
    ASSERT_EQ(reshaped.device(), Device::GPU);
    ASSERT_EQ(reshaped.shape()[0], 6u);
    ASSERT_EQ(reshaped.shape()[1], 4u);

    // Test transpose on GPU
    auto t2d = Tensor::zeros({3, 4}, DType::Float32, Device::GPU);
    auto transposed = t2d.transpose();
    ASSERT_EQ(transposed.device(), Device::GPU);
    ASSERT_EQ(transposed.shape()[0], 4u);
    ASSERT_EQ(transposed.shape()[1], 3u);

    // Test squeeze on GPU
    auto t_with_ones = Tensor::zeros({1, 3, 1, 4}, DType::Float32, Device::GPU);
    auto squeezed = t_with_ones.squeeze();
    ASSERT_EQ(squeezed.device(), Device::GPU);
    ASSERT_EQ(squeezed.ndim(), 2u);

    // Test unsqueeze on GPU
    auto unsqueezed = t2d.unsqueeze(1);
    ASSERT_EQ(unsqueezed.device(), Device::GPU);
    ASSERT_EQ(unsqueezed.ndim(), 3u);
}

//=============================================================================
// Memory Operations Tests
//=============================================================================

TEST(TensorBasic, MemoryOperationsCpu) {
    auto t1 = Tensor::ones({2, 3}, DType::Float32, Device::CPU);

    // Test copy
    auto t2 = t1.copy();
    ASSERT_TRUE(t2.same_shape(t1));
    ASSERT_TRUE(t2.same_dtype(t1));
    ASSERT_TRUE(t2.same_device(t1));
    ASSERT_TRUE(t2.same_memory_order(t1));

    // Modify original, copy should be unchanged
    t1.fill<float>(5.0f);
    auto t1_data = t1.typed_data<float>();
    auto t2_data = t2.typed_data<float>();

    ASSERT_EQ(t1_data[0], 5.0f);
    ASSERT_EQ(t2_data[0], 1.0f);

    // Test copy with different memory order
    auto t3 = t1.copy(MemoryOrder::ColMajor);
    ASSERT_EQ(t3.memory_order(), MemoryOrder::ColMajor);
    ASSERT_TRUE(t3.is_f_contiguous());

    // Test clone (alias for copy)
    auto t4 = t1.clone();
    ASSERT_TRUE(t4.same_shape(t1));

    // Test device transfer (CPU to CPU should be no-op)
    auto t5 = t1.to(Device::CPU);
    ASSERT_EQ(t5.device(), Device::CPU);

    // Test cpu() method
    auto t6 = t1.cpu();
    ASSERT_EQ(t6.device(), Device::CPU);
}

TEST(TensorBasic, MemoryOperationsGpu) {
    SKIP_IF_NO_GPU();

    auto cpu_tensor = Tensor::ones({2, 3}, DType::Float32, Device::CPU);

    // Test CPU to GPU transfer
    auto gpu_tensor = cpu_tensor.gpu();
    ASSERT_EQ(gpu_tensor.device(), Device::GPU);
    ASSERT_TRUE(gpu_tensor.same_shape(cpu_tensor));
    ASSERT_TRUE(gpu_tensor.same_dtype(cpu_tensor));

    // Test GPU copy
    auto gpu_copy = gpu_tensor.copy();
    ASSERT_EQ(gpu_copy.device(), Device::GPU);
    ASSERT_TRUE(gpu_copy.same_shape(gpu_tensor));

    // Test GPU to CPU transfer
    auto back_to_cpu = gpu_tensor.cpu();
    ASSERT_EQ(back_to_cpu.device(), Device::CPU);

    // Verify data integrity through round trip
    auto cpu_data = back_to_cpu.typed_data<float>();
    for (size_t i = 0; i < back_to_cpu.size(); ++i) {
        ASSERT_EQ(cpu_data[i], 1.0f);
    }

    // Test direct device transfer
    auto gpu_tensor2 = cpu_tensor.to(Device::GPU);
    ASSERT_EQ(gpu_tensor2.device(), Device::GPU);

    // Test transfer with memory order change
    auto gpu_f = cpu_tensor.to(Device::GPU, MemoryOrder::ColMajor);
    ASSERT_EQ(gpu_f.device(), Device::GPU);
    ASSERT_EQ(gpu_f.memory_order(), MemoryOrder::ColMajor);
}

//=============================================================================
// Memory Order Tests
//=============================================================================

TEST(TensorBasic, MemoryOrderCpu) {
    // Test C-order creation and properties
    auto c_tensor = Tensor::zeros({3, 4}, DType::Float32, Device::CPU,
                                  MemoryOrder::RowMajor);
    ASSERT_EQ(c_tensor.memory_order(), MemoryOrder::RowMajor);
    ASSERT_TRUE(c_tensor.is_c_contiguous());
    ASSERT_TRUE(!c_tensor.is_f_contiguous());

    // Test F-order creation and properties
    auto f_tensor = Tensor::zeros({3, 4}, DType::Float32, Device::CPU,
                                  MemoryOrder::ColMajor);
    ASSERT_EQ(f_tensor.memory_order(), MemoryOrder::ColMajor);
    ASSERT_TRUE(f_tensor.is_f_contiguous());
    ASSERT_TRUE(!f_tensor.is_c_contiguous());

    // Test stride patterns
    auto c_strides = c_tensor.strides();
    auto f_strides = f_tensor.strides();

    // C-order: [16, 4] for 3x4 float32 matrix
    ASSERT_EQ(c_strides[1], 4u);  // Column stride = itemsize
    ASSERT_EQ(c_strides[0], 16u); // Row stride = 4 * itemsize

    // F-order: [4, 12] for 3x4 float32 matrix
    ASSERT_EQ(f_strides[0], 4u);  // Row stride = itemsize
    ASSERT_EQ(f_strides[1], 12u); // Column stride = 3 * itemsize

    // Test conversion functions
    auto c_to_f = c_tensor.asfortranarray();
    ASSERT_TRUE(c_to_f.is_f_contiguous());
    ASSERT_EQ(c_to_f.memory_order(), MemoryOrder::ColMajor);

    auto f_to_c = f_tensor.ascontiguousarray();
    ASSERT_TRUE(f_to_c.is_c_contiguous());
    ASSERT_EQ(f_to_c.memory_order(), MemoryOrder::RowMajor);

    // Test NumPy-style functions
    auto c_result = Tensor::ascontiguousarray(f_tensor);
    ASSERT_TRUE(c_result.is_c_contiguous());

    auto f_result = Tensor::asfortranarray(c_tensor);
    ASSERT_TRUE(f_result.is_f_contiguous());

    // Test idempotence
    auto c_result2 = Tensor::ascontiguousarray(c_tensor);
    ASSERT_TRUE(c_result2.storage() == c_tensor.storage());

    auto f_result2 = Tensor::asfortranarray(f_tensor);
    ASSERT_TRUE(f_result2.storage() == f_tensor.storage());
}

TEST(TensorBasic, MemoryOrderGpu) {
    SKIP_IF_NO_GPU();

    // Test memory order preservation on GPU
    auto c_gpu = Tensor::zeros({3, 4}, DType::Float32, Device::GPU,
                               MemoryOrder::RowMajor);
    ASSERT_EQ(c_gpu.memory_order(), MemoryOrder::RowMajor);
    ASSERT_EQ(c_gpu.device(), Device::GPU);

    auto f_gpu = Tensor::zeros({3, 4}, DType::Float32, Device::GPU,
                               MemoryOrder::ColMajor);
    ASSERT_EQ(f_gpu.memory_order(), MemoryOrder::ColMajor);
    ASSERT_EQ(f_gpu.device(), Device::GPU);

    // Test conversion on GPU
    auto c_to_f_gpu = c_gpu.asfortranarray();
    ASSERT_EQ(c_to_f_gpu.device(), Device::GPU);
    ASSERT_EQ(c_to_f_gpu.memory_order(), MemoryOrder::ColMajor);

    // Test device transfer with memory order
    auto cpu_c = Tensor::ones({2, 3}, DType::Float32, Device::CPU,
                              MemoryOrder::RowMajor);
    auto gpu_f = cpu_c.to(Device::GPU, MemoryOrder::ColMajor);
    ASSERT_EQ(gpu_f.device(), Device::GPU);
    ASSERT_EQ(gpu_f.memory_order(), MemoryOrder::ColMajor);
}

//=============================================================================
// Data Type System Tests
//=============================================================================

TEST(TensorBasic, DtypeSystem) {
    // Test different data types
    auto t_f32 = Tensor::zeros({2, 2}, DType::Float32, Device::CPU);
    auto t_f64 = Tensor::zeros({2, 2}, DType::Float64, Device::CPU);
    auto t_i32 = Tensor::zeros({2, 2}, DType::Int32, Device::CPU);
    auto t_i64 = Tensor::zeros({2, 2}, DType::Int64, Device::CPU);
    auto t_bool = Tensor::zeros({2, 2}, DType::Bool, Device::CPU);

    // Test itemsize
    ASSERT_EQ(t_f32.itemsize(), 4u);
    ASSERT_EQ(t_f64.itemsize(), 8u);
    ASSERT_EQ(t_i32.itemsize(), 4u);
    ASSERT_EQ(t_i64.itemsize(), 8u);
    ASSERT_EQ(t_bool.itemsize(), 1u);

    // Test dtype names
    ASSERT_EQ(t_f32.dtype_name(), "float32");
    ASSERT_EQ(t_f64.dtype_name(), "float64");
    ASSERT_EQ(t_i32.dtype_name(), "int32");
    ASSERT_EQ(t_i64.dtype_name(), "int64");
    ASSERT_EQ(t_bool.dtype_name(), "bool");

    // Test nbytes calculation
    ASSERT_EQ(t_f32.nbytes(), 16u); // 4 elements * 4 bytes
    ASSERT_EQ(t_f64.nbytes(), 32u); // 4 elements * 8 bytes

    // Test automatic dtype deduction (compile-time)
    static_assert(dtype_of_v<float> == DType::Float32);
    static_assert(dtype_of_v<double> == DType::Float64);
    static_assert(dtype_of_v<int32_t> == DType::Int32);
    static_assert(dtype_of_v<int64_t> == DType::Int64);
    static_assert(dtype_of_v<bool> == DType::Bool);

    // Test dtype comparison
    ASSERT_TRUE(t_f32.same_dtype(t_f32));
    ASSERT_TRUE(!t_f32.same_dtype(t_f64));
    ASSERT_TRUE(!t_f32.same_dtype(t_i32));
}

//=============================================================================
// Views and Memory Sharing Tests
//=============================================================================

TEST(TensorBasic, ViewsAndSharingCpu) {
    auto base = Tensor::zeros({2, 3, 4}, DType::Float32, Device::CPU);
    base.fill<float>(1.0f);

    // Test reshape view (should share memory for contiguous tensors)
    auto reshaped = base.reshape({6, 4});

    // Modify via reshaped view
    reshaped.set_item<float>({0, 0}, 99.0f);
    float val = base.item<float>({0, 0, 0});
    ASSERT_EQ(val, 99.0f); // Change should be visible in base

    // Test view method
    auto view = base.view({24});
    ASSERT_EQ(view.ndim(), 1u);
    ASSERT_EQ(view.shape()[0], 24u);

    // Modify via view
    view.set_item<float>({1}, 77.0f);
    float val2 = base.item<float>({0, 0, 1});
    ASSERT_EQ(val2, 77.0f);

    // Test transpose view
    auto t2d = Tensor::zeros({3, 4}, DType::Float32, Device::CPU);
    t2d.set_item<float>({1, 2}, 42.0f);

    auto transposed = t2d.transpose();
    float val3 = transposed.item<float>({2, 1});
    ASSERT_EQ(val3, 42.0f);

    // Test squeeze/unsqueeze views
    auto with_ones = Tensor::zeros({1, 3, 1}, DType::Float32, Device::CPU);
    with_ones.set_item<float>({0, 1, 0}, 123.0f);

    auto squeezed = with_ones.squeeze();
    float val4 = squeezed.item<float>({1});
    ASSERT_EQ(val4, 123.0f);

    auto unsqueezed = squeezed.unsqueeze(0);
    float val5 = unsqueezed.item<float>({0, 1});
    ASSERT_EQ(val5, 123.0f);

    // Test that copy creates independent data
    auto copied = base.copy();
    copied.set_item<float>({0, 0, 0}, 555.0f);
    float original_val = base.item<float>({0, 0, 0});
    float copied_val = copied.item<float>({0, 0, 0});
    ASSERT_EQ(original_val, 99.0f); // Original unchanged
    ASSERT_EQ(copied_val, 555.0f);  // Copy modified
}

TEST(TensorBasic, ViewsAndSharingGpu) {
    SKIP_IF_NO_GPU();

    auto base = Tensor::zeros({2, 3, 4}, DType::Float32, Device::GPU);

    // Test reshape on GPU (should work for views)
    auto reshaped = base.reshape({6, 4});
    ASSERT_EQ(reshaped.device(), Device::GPU);
    ASSERT_TRUE(reshaped.same_shape(Tensor({6, 4})));

    // Test view on GPU
    auto view = base.view({24});
    ASSERT_EQ(view.device(), Device::GPU);
    ASSERT_EQ(view.ndim(), 1u);

    // Test transpose on GPU
    auto t2d = Tensor::zeros({3, 4}, DType::Float32, Device::GPU);
    auto transposed = t2d.transpose();
    ASSERT_EQ(transposed.device(), Device::GPU);
    ASSERT_EQ(transposed.shape()[0], 4u);
    ASSERT_EQ(transposed.shape()[1], 3u);
}

//=============================================================================
// Error Handling Tests
//=============================================================================

TEST(TensorBasic, ErrorHandling) {
    // Test invalid shape
    try {
        auto t = Tensor({0}, DType::Float32, Device::CPU); // Zero dimension
    } catch (const std::exception &) {
        // This might be valid for some implementations
    }

    // Test out of bounds access
    auto t = Tensor::zeros({2, 3}, DType::Float32, Device::CPU);
    bool caught = false;
    try {
        t.item<float>({5, 5});
    } catch (const std::exception &) {
        caught = true;
    }
    ASSERT_TRUE(caught);

    // Test invalid reshape
    caught = false;
    try {
        t.reshape({7}); // 6 elements can't reshape to 7
    } catch (const std::exception &) {
        caught = true;
    }
    ASSERT_TRUE(caught);

    // Test invalid squeeze (should be a no-op)
    auto no_squeeze = t.squeeze(0);
    ASSERT_TRUE(no_squeeze.shape() == t.shape());

    // Test GPU data access
    if (axiom::system::should_run_gpu_tests()) {
        auto gpu_tensor = Tensor::zeros({2, 2}, DType::Float32, Device::GPU);
        caught = false;
        try {
            gpu_tensor.data();
        } catch (const std::exception &) {
            caught = true;
        }
        ASSERT_TRUE(caught);
    }

    // Test invalid view
    caught = false;
    try {
        t.view({7}); // Can't view 6 elements as 7
    } catch (const std::exception &) {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

//=============================================================================
// Performance and Stress Tests
//=============================================================================

TEST(TensorBasic, LargeTensors) {
    // Test large tensor creation and basic operations
    const size_t large_size = 1000;

    auto large_tensor =
        Tensor::zeros({large_size, large_size}, DType::Float32, Device::CPU);
    ASSERT_EQ(large_tensor.size(), large_size * large_size);
    ASSERT_EQ(large_tensor.nbytes(), large_size * large_size * 4);

    // Test reshape of large tensor
    auto reshaped = large_tensor.reshape({large_size * large_size});
    ASSERT_EQ(reshaped.ndim(), 1u);
    ASSERT_EQ(reshaped.size(), large_size * large_size);

    // Test copy of large tensor
    auto copied = large_tensor.copy();
    ASSERT_TRUE(copied.same_shape(large_tensor));

    if (axiom::system::should_run_gpu_tests()) {
        // Test GPU transfer of large tensor
        auto gpu_large = large_tensor.gpu();
        ASSERT_EQ(gpu_large.device(), Device::GPU);
        ASSERT_TRUE(gpu_large.same_shape(large_tensor));
    }
}

TEST(TensorBasic, MemoryOrderPerformance) {
    const size_t size = 200;

    // Create matrices in both orders
    auto c_matrix = Tensor::zeros({size, size}, DType::Float32, Device::CPU,
                                  MemoryOrder::RowMajor);
    auto f_matrix = Tensor::zeros({size, size}, DType::Float32, Device::CPU,
                                  MemoryOrder::ColMajor);

    // Test that memory orders are correct
    ASSERT_TRUE(c_matrix.is_c_contiguous());
    ASSERT_TRUE(f_matrix.is_f_contiguous());

    // Test conversion performance (not timing, just functionality)
    auto c_to_f = c_matrix.asfortranarray();
    ASSERT_TRUE(c_to_f.is_f_contiguous());

    auto f_to_c = f_matrix.ascontiguousarray();
    ASSERT_TRUE(f_to_c.is_c_contiguous());

    // Test that data is preserved during conversion
    c_matrix.set_item<float>({10, 20}, 42.0f);
    auto converted = c_matrix.asfortranarray();
    ASSERT_EQ(converted.item<float>({10, 20}), 42.0f);
}

//=============================================================================
// Integration Tests
//=============================================================================

TEST(TensorBasic, CrossDeviceWorkflow) {
    SKIP_IF_NO_GPU();

    // Create data on CPU
    auto cpu_data = Tensor::ones({10, 10}, DType::Float32, Device::CPU);

    // Modify some data
    for (size_t i = 0; i < 10; ++i) {
        cpu_data.set_item<float>({i, i}, static_cast<float>(i));
    }

    // Transfer to GPU
    auto gpu_data = cpu_data.gpu();
    ASSERT_EQ(gpu_data.device(), Device::GPU);

    // Reshape on GPU
    auto gpu_reshaped = gpu_data.reshape({100});
    ASSERT_EQ(gpu_reshaped.device(), Device::GPU);
    ASSERT_EQ(gpu_reshaped.ndim(), 1u);

    // Transfer back to CPU
    auto result = gpu_reshaped.cpu();
    ASSERT_EQ(result.device(), Device::CPU);
    ASSERT_EQ(result.ndim(), 1u);
    ASSERT_EQ(result.size(), 100u);

    // Verify data integrity
    for (size_t i = 0; i < 10; ++i) {
        float expected = static_cast<float>(i);
        float actual = result.item<float>({i * 10 + i}); // Diagonal elements
        ASSERT_EQ(actual, expected);
    }
}

TEST(TensorBasic, MixedMemoryOrders) {
    // Test operations mixing different memory orders
    auto c_tensor = Tensor::ones({3, 4}, DType::Float32, Device::CPU,
                                 MemoryOrder::RowMajor);
    auto f_tensor = Tensor::ones({3, 4}, DType::Float32, Device::CPU,
                                 MemoryOrder::ColMajor);

    // Both should have same logical values
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            ASSERT_EQ(c_tensor.item<float>({i, j}), 1.0f);
            ASSERT_EQ(f_tensor.item<float>({i, j}), 1.0f);
        }
    }

    // Test conversions preserve data
    auto c_to_f = c_tensor.copy(MemoryOrder::ColMajor);
    auto f_to_c = f_tensor.copy(MemoryOrder::RowMajor);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            ASSERT_EQ(c_to_f.item<float>({i, j}), 1.0f);
            ASSERT_EQ(f_to_c.item<float>({i, j}), 1.0f);
        }
    }
}

TEST(TensorBasic, TensorPrinting) {
    auto a =
        axiom::Tensor::arange(4).reshape({2, 2}).astype(axiom::DType::Float32);
    std::stringstream ss;
    ss << a;
    std::string expected = "[[0.0000 1.0000]\n [2.0000 3.0000]] "
                           "Tensor(shape=[2, 2], dtype=float32, device=CPU, "
                           "order=RowMajor)";
    ASSERT_EQ(ss.str(), expected);
}

TEST(TensorBasic, TensorPrintingEllipsis) {
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
    ASSERT_EQ(ss.str(), expected);
}
