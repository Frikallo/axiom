#include "axiom_test_utils.hpp"
#include <axiom/axiom.hpp>

using namespace axiom;
using namespace axiom::nn;

TEST(NnPositional, BasicShape) {
    int seq_len = 10;
    int d_model = 64;
    auto pe = sinusoidal_position_embedding(seq_len, d_model);
    ASSERT_EQ(pe.ndim(), 2);
    ASSERT_EQ(pe.shape()[0], 2 * seq_len - 1);
    ASSERT_EQ(pe.shape()[1], static_cast<size_t>(d_model));
    ASSERT_EQ(pe.dtype(), DType::Float32);
    ASSERT_EQ(pe.device(), Device::CPU);
}

TEST(NnPositional, DtypeCast) {
    auto pe = sinusoidal_position_embedding(5, 16, DType::Float16);
    ASSERT_EQ(pe.dtype(), DType::Float16);
    ASSERT_EQ(pe.shape()[0], 9); // 2*5 - 1
}

TEST(NnPositional, ValuesInRange) {
    auto pe = sinusoidal_position_embedding(8, 32);
    auto pe_data = pe.typed_data<float>();
    size_t n = pe.size();
    for (size_t i = 0; i < n; ++i) {
        ASSERT_GE(pe_data[i], -1.0f);
        ASSERT_LE(pe_data[i], 1.0f);
    }
}

TEST(NnPositional, SeqLen1) {
    auto pe = sinusoidal_position_embedding(1, 8);
    ASSERT_EQ(pe.shape()[0], 1); // 2*1 - 1
    ASSERT_EQ(pe.shape()[1], 8);
}

TEST(NnPositional, GpuTransfer) {
    SKIP_IF_NO_GPU();
    auto pe = sinusoidal_position_embedding(4, 16, DType::Float32, Device::GPU);
    ASSERT_EQ(pe.device(), Device::GPU);
    ASSERT_EQ(pe.shape()[0], 7);
}
