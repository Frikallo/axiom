#include <axiom/axiom.hpp>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ==================================
//
//      TEST HARNESS
//
// ==================================

static int tests_run = 0;
static int tests_passed = 0;
static std::string current_test_name;

#define RUN_TEST(test_func, device)                                            \
    run_test([&]() { test_func(device); }, #test_func,                         \
             std::string(" (") + axiom::system::device_to_string(device) +     \
                 ")")

void run_test(const std::function<void()> &test_func,
              const std::string &test_name, const std::string &device_str) {
    tests_run++;
    current_test_name = test_name;
    std::cout << "--- Running: " << test_name << device_str << " ---"
              << std::endl;
    try {
        test_func();
        std::cout << "--- PASSED: " << test_name << device_str << " ---"
                  << std::endl;
        tests_passed++;
    } catch (const std::exception &e) {
        std::cerr << "--- FAILED: " << test_name << device_str << " ---"
                  << std::endl;
        std::cerr << "    Error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "--- FAILED: " << test_name << device_str << " ---"
                  << std::endl;
        std::cerr << "    Error: Unknown exception caught." << std::endl;
    }
    std::cout << std::endl;
}

#define ASSERT(condition, msg)                                                 \
    do {                                                                       \
        if (!(condition)) {                                                    \
            throw std::runtime_error("Assertion failed: (" #condition ") - " + \
                                     std::string(msg));                        \
        }                                                                      \
    } while (0)

template <typename T>
void assert_tensor_equals_cpu(const axiom::Tensor &t,
                              const std::vector<T> &expected_data,
                              double epsilon = 1e-6) {
    auto t_cpu = t.cpu();
    ASSERT(t_cpu.device() == axiom::Device::CPU, "Tensor is not on CPU");
    ASSERT(t_cpu.size() == expected_data.size(), "Tensor size mismatch");

    const T *t_data = t_cpu.template typed_data<T>();
    for (size_t i = 0; i < expected_data.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::abs(static_cast<double>(t_data[i]) -
                         static_cast<double>(expected_data[i])) >= epsilon) {
                throw std::runtime_error(
                    "Tensor data mismatch at index " + std::to_string(i) +
                    ": got " + std::to_string(t_data[i]) + ", expected " +
                    std::to_string(expected_data[i]));
            }
        } else {
            if (t_data[i] != expected_data[i]) {
                throw std::runtime_error("Tensor data mismatch at index " +
                                         std::to_string(i));
            }
        }
    }
}

// ==================================
//
//      MATH OPERATION TESTS
//
// ==================================

void test_sign(axiom::Device device) {
    auto data = std::vector<float>({-3.0f, -1.0f, 0.0f, 1.0f, 3.0f});
    auto t = axiom::Tensor::from_data<float>(data.data(), {5}).to(device);
    auto result = t.sign();
    assert_tensor_equals_cpu<float>(result, {-1.0f, -1.0f, 0.0f, 1.0f, 1.0f});
}

void test_floor_ceil_trunc(axiom::Device device) {
    auto data = std::vector<float>({-2.7f, -0.5f, 0.5f, 2.7f});
    auto t = axiom::Tensor::from_data<float>(data.data(), {4}).to(device);

    auto fl = t.floor();
    assert_tensor_equals_cpu<float>(fl, {-3.0f, -1.0f, 0.0f, 2.0f});

    auto ce = t.ceil();
    assert_tensor_equals_cpu<float>(ce, {-2.0f, 0.0f, 1.0f, 3.0f});

    auto tr = t.trunc();
    assert_tensor_equals_cpu<float>(tr, {-2.0f, 0.0f, 0.0f, 2.0f});
}

void test_reciprocal(axiom::Device device) {
    auto data = std::vector<float>({1.0f, 2.0f, 4.0f, 0.5f});
    auto t = axiom::Tensor::from_data<float>(data.data(), {4}).to(device);
    auto result = t.reciprocal();
    assert_tensor_equals_cpu<float>(result, {1.0f, 0.5f, 0.25f, 2.0f});
}

void test_square_cbrt(axiom::Device device) {
    auto data = std::vector<float>({2.0f, 3.0f, 4.0f});
    auto t = axiom::Tensor::from_data<float>(data.data(), {3}).to(device);

    auto sq = t.square();
    assert_tensor_equals_cpu<float>(sq, {4.0f, 9.0f, 16.0f});

    auto data2 = std::vector<float>({8.0f, 27.0f, 64.0f});
    auto t2 = axiom::Tensor::from_data<float>(data2.data(), {3}).to(device);
    auto cb = t2.cbrt();
    assert_tensor_equals_cpu<float>(cb, {2.0f, 3.0f, 4.0f}, 1e-5);
}

void test_isnan_isinf_isfinite(axiom::Device device) {
    auto data = std::vector<float>({1.0f, NAN, INFINITY, -INFINITY, 0.0f});
    auto t = axiom::Tensor::from_data<float>(data.data(), {5}).to(device);

    auto nan_result = t.isnan();
    assert_tensor_equals_cpu<bool>(nan_result,
                                   {false, true, false, false, false});

    auto inf_result = t.isinf();
    assert_tensor_equals_cpu<bool>(inf_result,
                                   {false, false, true, true, false});

    auto fin_result = t.isfinite();
    assert_tensor_equals_cpu<bool>(fin_result,
                                   {true, false, false, false, true});
}

void test_clip(axiom::Device device) {
    auto data = std::vector<float>({-5.0f, -1.0f, 0.0f, 1.0f, 5.0f});
    auto t = axiom::Tensor::from_data<float>(data.data(), {5}).to(device);
    auto result = t.clip(-2.0, 2.0);
    assert_tensor_equals_cpu<float>(result, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
}

// ==================================
//
//      REDUCTION TESTS
//
// ==================================

void test_prod(axiom::Device device) {
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto t = axiom::Tensor::from_data<float>(data.data(), {2, 2}).to(device);

    auto prod_all = t.prod();
    ASSERT(std::abs(prod_all.item<float>() - 24.0f) < 1e-5,
           "prod should be 24");

    auto prod_axis0 = t.prod(0);
    assert_tensor_equals_cpu<float>(prod_axis0, {3.0f, 8.0f});
}

void test_var_std(axiom::Device device) {
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto t = axiom::Tensor::from_data<float>(data.data(), {5}).to(device);

    // var = mean((x - mean(x))^2) = 2.0
    auto v = t.var();
    ASSERT(std::abs(v.item<float>() - 2.0f) < 1e-4, "var should be 2.0");

    // std = sqrt(var) = sqrt(2) ~= 1.414
    auto s = t.std();
    ASSERT(std::abs(s.item<float>() - std::sqrt(2.0f)) < 1e-4,
           "std should be sqrt(2)");
}

void test_ptp(axiom::Device device) {
    auto data = std::vector<float>({1.0f, 5.0f, 3.0f, 2.0f, 8.0f});
    auto t = axiom::Tensor::from_data<float>(data.data(), {5}).to(device);
    auto result = t.ptp();
    ASSERT(std::abs(result.item<float>() - 7.0f) < 1e-5, "ptp should be 7");
}

void test_any_all(axiom::Device device) {
    // Use uint8_t arrays since std::vector<bool> is specialized and has no
    // .data()
    uint8_t data1[] = {1, 1, 1};
    auto t1 = axiom::Tensor::from_data<uint8_t>(data1, {3})
                  .astype(axiom::DType::Bool)
                  .to(device);
    ASSERT(t1.all().item<bool>() == true, "all should be true");
    ASSERT(t1.any().item<bool>() == true, "any should be true");

    uint8_t data2[] = {1, 0, 1};
    auto t2 = axiom::Tensor::from_data<uint8_t>(data2, {3})
                  .astype(axiom::DType::Bool)
                  .to(device);
    ASSERT(t2.all().item<bool>() == false, "all should be false");
    ASSERT(t2.any().item<bool>() == true, "any should be true");

    uint8_t data3[] = {0, 0, 0};
    auto t3 = axiom::Tensor::from_data<uint8_t>(data3, {3})
                  .astype(axiom::DType::Bool)
                  .to(device);
    ASSERT(t3.all().item<bool>() == false, "all should be false");
    ASSERT(t3.any().item<bool>() == false, "any should be false");
}

// ==================================
//
//      COMPARISON TESTS
//
// ==================================

void test_isclose_allclose(axiom::Device device) {
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto data2 = std::vector<float>({1.0f, 2.00001f, 3.0f});
    auto t1 = axiom::Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = axiom::Tensor::from_data<float>(data2.data(), {3}).to(device);

    ASSERT(t1.allclose(t2, 1e-4, 1e-4) == true, "tensors should be close");

    auto data3 = std::vector<float>({1.0f, 3.0f, 3.0f});
    auto t3 = axiom::Tensor::from_data<float>(data3.data(), {3}).to(device);
    ASSERT(t1.allclose(t3) == false, "tensors should not be close");
}

void test_array_equal(axiom::Device device) {
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto t1 = axiom::Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = axiom::Tensor::from_data<float>(data1.data(), {3}).to(device);

    ASSERT(t1.array_equal(t2) == true, "identical tensors should be equal");

    auto data2 = std::vector<float>({1.0f, 2.1f, 3.0f});
    auto t3 = axiom::Tensor::from_data<float>(data2.data(), {3}).to(device);
    ASSERT(t1.array_equal(t3) == false,
           "different tensors should not be equal");
}

// ==================================
//
//      STACKING TESTS
//
// ==================================

void test_concatenate(axiom::Device device) {
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto data2 = std::vector<float>({4.0f, 5.0f, 6.0f});
    auto t1 = axiom::Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = axiom::Tensor::from_data<float>(data2.data(), {3}).to(device);

    // Using vector
    auto result = axiom::Tensor::concatenate({t1, t2}, 0);
    ASSERT(result.size() == 6, "concatenate size should be 6");
    assert_tensor_equals_cpu<float>(result,
                                    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Using cat alias with initializer list
    auto result2 = axiom::Tensor::cat({t1, t2}, 0);
    assert_tensor_equals_cpu<float>(result2,
                                    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Member function cat
    auto result3 = t1.cat(t2, 0);
    assert_tensor_equals_cpu<float>(result3,
                                    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

void test_concatenate_2d(axiom::Device device) {
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto data2 = std::vector<float>({5.0f, 6.0f, 7.0f, 8.0f});
    auto t1 = axiom::Tensor::from_data<float>(data1.data(), {2, 2}).to(device);
    auto t2 = axiom::Tensor::from_data<float>(data2.data(), {2, 2}).to(device);

    // Concat along axis 0
    auto result0 = axiom::Tensor::cat({t1, t2}, 0);
    ASSERT(result0.shape() == axiom::Shape({4, 2}), "shape should be (4, 2)");

    // Concat along axis 1
    auto result1 = axiom::Tensor::cat({t1, t2}, 1);
    ASSERT(result1.shape() == axiom::Shape({2, 4}), "shape should be (2, 4)");
}

void test_stack(axiom::Device device) {
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto data2 = std::vector<float>({4.0f, 5.0f, 6.0f});
    auto t1 = axiom::Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = axiom::Tensor::from_data<float>(data2.data(), {3}).to(device);

    // Stack along axis 0 (creates new first dimension)
    auto result = axiom::Tensor::stack({t1, t2}, 0);
    ASSERT(result.shape() == axiom::Shape({2, 3}), "shape should be (2, 3)");

    // Stack along axis 1
    auto result1 = axiom::Tensor::stack({t1, t2}, 1);
    ASSERT(result1.shape() == axiom::Shape({3, 2}), "shape should be (3, 2)");
}

void test_vstack_hstack(axiom::Device device) {
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto data2 = std::vector<float>({4.0f, 5.0f, 6.0f});
    auto t1 = axiom::Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = axiom::Tensor::from_data<float>(data2.data(), {3}).to(device);

    // vstack 1D arrays
    auto vresult = axiom::Tensor::vstack({t1, t2});
    ASSERT(vresult.shape() == axiom::Shape({2, 3}),
           "vstack shape should be (2, 3)");

    // hstack 1D arrays
    auto hresult = axiom::Tensor::hstack({t1, t2});
    ASSERT(hresult.shape() == axiom::Shape({6}), "hstack shape should be (6,)");
    assert_tensor_equals_cpu<float>(hresult,
                                    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

void test_split(axiom::Device device) {
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto t = axiom::Tensor::from_data<float>(data.data(), {6}).to(device);

    // Split into 3 equal parts
    auto parts = t.split(3, 0);
    ASSERT(parts.size() == 3, "should have 3 parts");
    ASSERT(parts[0].size() == 2, "each part should have 2 elements");
    assert_tensor_equals_cpu<float>(parts[0], {1.0f, 2.0f});
    assert_tensor_equals_cpu<float>(parts[1], {3.0f, 4.0f});
    assert_tensor_equals_cpu<float>(parts[2], {5.0f, 6.0f});
}

void test_chunk(axiom::Device device) {
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto t = axiom::Tensor::from_data<float>(data.data(), {5}).to(device);

    // Chunk into 3 parts (unequal)
    auto chunks = t.chunk(3, 0);
    ASSERT(chunks.size() == 3, "should have 3 chunks");
    // First 2 chunks get 2 elements, last gets 1
    ASSERT(chunks[0].size() == 2, "first chunk should have 2 elements");
    ASSERT(chunks[1].size() == 2, "second chunk should have 2 elements");
    ASSERT(chunks[2].size() == 1, "third chunk should have 1 element");
}

// ==================================
//
//      FLIP OPERATIONS TESTS
//
// ==================================

void test_flip_1d_zero_copy(axiom::Device device) {
    // Test that flip returns a zero-copy view with negative strides
    auto data =
        std::vector<float>({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    auto a = axiom::Tensor::from_data<float>(data.data(), {8}).to(device);
    auto b = a.flip(0);

    // Verify flip is zero-copy (shares storage)
    ASSERT(a.shares_storage(b), "flipped tensor should share storage");

    // Verify flip has negative stride
    ASSERT(b.has_negative_stride(),
           "flipped tensor should have negative stride");

    // Verify flipped tensor is not contiguous
    ASSERT(!b.is_contiguous(), "flipped tensor should not be contiguous");

    if (device == axiom::Device::CPU) {
        // Verify values are correct
        // a = [0, 1, 2, 3, 4, 5, 6, 7]
        // b = [7, 6, 5, 4, 3, 2, 1, 0]
        ASSERT(std::abs(b.item<float>({0}) - 7.0f) < 1e-5, "b[0] should be 7");
        ASSERT(std::abs(b.item<float>({7}) - 0.0f) < 1e-5, "b[7] should be 0");
        ASSERT(std::abs(b.item<float>({3}) - 4.0f) < 1e-5, "b[3] should be 4");
    }
}

void test_flip_2d(axiom::Device device) {
    // Test 2D flip along different axes
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto a = axiom::Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    // a = [[1, 2, 3],
    //      [4, 5, 6]]

    // Flip along axis 0 (rows)
    auto b = a.flip(0);
    ASSERT(a.shares_storage(b), "flip(0) should share storage");
    ASSERT(b.has_negative_stride(), "flip(0) should have negative stride");

    if (device == axiom::Device::CPU) {
        // b = [[4, 5, 6],
        //      [1, 2, 3]]
        ASSERT(std::abs(b.item<float>({0, 0}) - 4.0f) < 1e-5,
               "b[0,0] should be 4");
        ASSERT(std::abs(b.item<float>({0, 2}) - 6.0f) < 1e-5,
               "b[0,2] should be 6");
        ASSERT(std::abs(b.item<float>({1, 0}) - 1.0f) < 1e-5,
               "b[1,0] should be 1");
    }

    // Flip along axis 1 (cols)
    auto c = a.flip(1);
    ASSERT(a.shares_storage(c), "flip(1) should share storage");

    if (device == axiom::Device::CPU) {
        // c = [[3, 2, 1],
        //      [6, 5, 4]]
        ASSERT(std::abs(c.item<float>({0, 0}) - 3.0f) < 1e-5,
               "c[0,0] should be 3");
        ASSERT(std::abs(c.item<float>({1, 2}) - 4.0f) < 1e-5,
               "c[1,2] should be 4");
    }
}

void test_flipud_fliplr(axiom::Device device) {
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto a = axiom::Tensor::from_data<float>(data.data(), {2, 3}).to(device);

    // flipud = flip(0)
    auto ud = a.flipud();
    ASSERT(a.shares_storage(ud), "flipud should share storage");

    // fliplr = flip(1)
    auto lr = a.fliplr();
    ASSERT(a.shares_storage(lr), "fliplr should share storage");

    if (device == axiom::Device::CPU) {
        // Same value checks as test_flip_2d
        ASSERT(std::abs(ud.item<float>({0, 0}) - 4.0f) < 1e-5,
               "flipud[0,0] should be 4");
        ASSERT(std::abs(lr.item<float>({0, 0}) - 3.0f) < 1e-5,
               "fliplr[0,0] should be 3");
    }
}

void test_flip_multi_axis(axiom::Device device) {
    // Test flipping multiple axes at once
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto a = axiom::Tensor::from_data<float>(data.data(), {2, 3}).to(device);

    // Flip both axes
    auto b = a.flip({0, 1});
    ASSERT(a.shares_storage(b), "multi-axis flip should share storage");

    if (device == axiom::Device::CPU) {
        // b = [[6, 5, 4],
        //      [3, 2, 1]]
        ASSERT(std::abs(b.item<float>({0, 0}) - 6.0f) < 1e-5,
               "b[0,0] should be 6");
        ASSERT(std::abs(b.item<float>({1, 2}) - 1.0f) < 1e-5,
               "b[1,2] should be 1");
    }
}

void test_flip_operations_on_flipped(axiom::Device device) {
    // Test that arithmetic operations work on flipped tensors
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto data2 = std::vector<float>({10.0f, 20.0f, 30.0f, 40.0f});
    auto a = axiom::Tensor::from_data<float>(data1.data(), {4}).to(device);
    auto b = axiom::Tensor::from_data<float>(data2.data(), {4}).to(device);

    // Flip a: [4, 3, 2, 1]
    auto a_flip = a.flip(0);

    // Add flipped tensor to normal tensor
    // [4, 3, 2, 1] + [10, 20, 30, 40] = [14, 23, 32, 41]
    auto result = axiom::ops::add(a_flip, b);

    if (device == axiom::Device::CPU) {
        ASSERT(std::abs(result.item<float>({0}) - 14.0f) < 1e-5,
               "result[0] should be 14");
        ASSERT(std::abs(result.item<float>({1}) - 23.0f) < 1e-5,
               "result[1] should be 23");
        ASSERT(std::abs(result.item<float>({2}) - 32.0f) < 1e-5,
               "result[2] should be 32");
        ASSERT(std::abs(result.item<float>({3}) - 41.0f) < 1e-5,
               "result[3] should be 41");
    }
}

void test_flip_contiguous_copy(axiom::Device device) {
    // Test that ascontiguousarray() correctly materializes a flipped tensor
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto a = axiom::Tensor::from_data<float>(data.data(), {4}).to(device);
    auto b = a.flip(0);

    // b is a view (non-contiguous with negative stride)
    ASSERT(!b.is_contiguous(), "flipped should not be contiguous");
    ASSERT(a.shares_storage(b), "should share storage before copy");

    // Make contiguous copy
    auto c = b.ascontiguousarray();
    ASSERT(c.is_contiguous(), "ascontiguousarray result should be contiguous");
    ASSERT(!a.shares_storage(c), "contiguous copy should not share storage");

    if (device == axiom::Device::CPU) {
        // Verify values are preserved
        ASSERT(std::abs(c.item<float>({0}) - 4.0f) < 1e-5, "c[0] should be 4");
        ASSERT(std::abs(c.item<float>({3}) - 1.0f) < 1e-5, "c[3] should be 1");
    }
}

void test_rot90(axiom::Device device) {
    // Test rot90 which uses flip internally
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto a = axiom::Tensor::from_data<float>(data.data(), {2, 2}).to(device);
    // a = [[1, 2],
    //      [3, 4]]

    // 90 degree rotation = flip + transpose
    auto b = a.rot90(1, {0, 1});
    // Expected: [[2, 4],
    //            [1, 3]]

    if (device == axiom::Device::CPU) {
        ASSERT(std::abs(b.item<float>({0, 0}) - 2.0f) < 1e-5,
               "b[0,0] should be 2");
        ASSERT(std::abs(b.item<float>({0, 1}) - 4.0f) < 1e-5,
               "b[0,1] should be 4");
        ASSERT(std::abs(b.item<float>({1, 0}) - 1.0f) < 1e-5,
               "b[1,0] should be 1");
        ASSERT(std::abs(b.item<float>({1, 1}) - 3.0f) < 1e-5,
               "b[1,1] should be 3");
    }
}

// ==================================
//
//      MAIN
//
// ==================================

int main(int argc, char **argv) {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "\n=== NumPy-like Math Operations ===\n" << std::endl;
    RUN_TEST(test_sign, axiom::Device::CPU);
    RUN_TEST(test_floor_ceil_trunc, axiom::Device::CPU);
    RUN_TEST(test_reciprocal, axiom::Device::CPU);
    RUN_TEST(test_square_cbrt, axiom::Device::CPU);
    RUN_TEST(test_isnan_isinf_isfinite, axiom::Device::CPU);
    RUN_TEST(test_clip, axiom::Device::CPU);

    std::cout << "\n=== Reduction Operations ===\n" << std::endl;
    RUN_TEST(test_prod, axiom::Device::CPU);
    RUN_TEST(test_var_std, axiom::Device::CPU);
    RUN_TEST(test_ptp, axiom::Device::CPU);
    RUN_TEST(test_any_all, axiom::Device::CPU);

    std::cout << "\n=== Comparison Operations ===\n" << std::endl;
    RUN_TEST(test_isclose_allclose, axiom::Device::CPU);
    RUN_TEST(test_array_equal, axiom::Device::CPU);

    std::cout << "\n=== Stacking Operations ===\n" << std::endl;
    RUN_TEST(test_concatenate, axiom::Device::CPU);
    RUN_TEST(test_concatenate_2d, axiom::Device::CPU);
    RUN_TEST(test_stack, axiom::Device::CPU);
    RUN_TEST(test_vstack_hstack, axiom::Device::CPU);
    RUN_TEST(test_split, axiom::Device::CPU);
    RUN_TEST(test_chunk, axiom::Device::CPU);

    std::cout << "\n=== Flip Operations (Zero-Copy) ===\n" << std::endl;
    RUN_TEST(test_flip_1d_zero_copy, axiom::Device::CPU);
    RUN_TEST(test_flip_2d, axiom::Device::CPU);
    RUN_TEST(test_flipud_fliplr, axiom::Device::CPU);
    RUN_TEST(test_flip_multi_axis, axiom::Device::CPU);
    RUN_TEST(test_flip_operations_on_flipped, axiom::Device::CPU);
    RUN_TEST(test_flip_contiguous_copy, axiom::Device::CPU);
    RUN_TEST(test_rot90, axiom::Device::CPU);

    if (axiom::system::should_run_gpu_tests()) {
        std::cout << "\n--- Running GPU tests ---\n" << std::endl;

        // NumPy-like math ops
        RUN_TEST(test_sign, axiom::Device::GPU);
        RUN_TEST(test_floor_ceil_trunc, axiom::Device::GPU);
        RUN_TEST(test_reciprocal, axiom::Device::GPU);
        RUN_TEST(test_square_cbrt, axiom::Device::GPU);
        RUN_TEST(test_isnan_isinf_isfinite, axiom::Device::GPU);
        RUN_TEST(test_clip, axiom::Device::GPU);

        // Reductions
        RUN_TEST(test_prod, axiom::Device::GPU);
        RUN_TEST(test_var_std, axiom::Device::GPU);
        RUN_TEST(test_ptp, axiom::Device::GPU);

        // Stacking
        RUN_TEST(test_concatenate, axiom::Device::GPU);
        RUN_TEST(test_concatenate_2d, axiom::Device::GPU);
        RUN_TEST(test_stack, axiom::Device::GPU);
        RUN_TEST(test_vstack_hstack, axiom::Device::GPU);

        // Flip operations (GPU materializes via ensureContiguous)
        RUN_TEST(test_flip_1d_zero_copy, axiom::Device::GPU);
        RUN_TEST(test_flip_2d, axiom::Device::GPU);
        RUN_TEST(test_flipud_fliplr, axiom::Device::GPU);
        RUN_TEST(test_flip_operations_on_flipped, axiom::Device::GPU);
    }

    std::cout << "\n----------------------------------\n";
    std::cout << "         TEST SUMMARY\n";
    std::cout << "----------------------------------\n";
    std::cout << "TOTAL TESTS: " << tests_run << std::endl;
    std::cout << "PASSED:      " << tests_passed << std::endl;
    std::cout << "FAILED:      " << tests_run - tests_passed << std::endl;
    std::cout << "----------------------------------\n";

    return (tests_run == tests_passed) ? 0 : 1;
}
