#pragma once

#include "axiom/operations.hpp"

#include <cstddef>

namespace axiom {
namespace graph {

struct OpTraits {
    bool is_unary : 1;
    bool is_binary : 1;
    bool is_elementwise : 1;
    bool is_reduction : 1;
    bool is_comparison : 1;
    bool is_logical : 1;
};

// clang-format off
inline constexpr OpTraits OP_TRAITS[] = {
    // Binary arithmetic
    /*Add*/             {0,1,1,0,0,0},
    /*Subtract*/        {0,1,1,0,0,0},
    /*Multiply*/        {0,1,1,0,0,0},
    /*Divide*/          {0,1,1,0,0,0},
    /*Power*/           {0,1,1,0,0,0},
    /*Modulo*/          {0,1,1,0,0,0},
    // Comparison
    /*Equal*/           {0,1,1,0,1,0},
    /*NotEqual*/        {0,1,1,0,1,0},
    /*Less*/            {0,1,1,0,1,0},
    /*LessEqual*/       {0,1,1,0,1,0},
    /*Greater*/         {0,1,1,0,1,0},
    /*GreaterEqual*/    {0,1,1,0,1,0},
    // Logical
    /*LogicalAnd*/      {0,1,1,0,0,1},
    /*LogicalOr*/       {0,1,1,0,0,1},
    /*LogicalXor*/      {0,1,1,0,0,1},
    /*LogicalNot*/      {1,0,1,0,0,1},
    // Bitwise
    /*BitwiseAnd*/      {0,1,1,0,0,0},
    /*BitwiseOr*/       {0,1,1,0,0,0},
    /*BitwiseXor*/      {0,1,1,0,0,0},
    /*LeftShift*/       {0,1,1,0,0,0},
    /*RightShift*/      {0,1,1,0,0,0},
    // Binary math
    /*Maximum*/         {0,1,1,0,0,0},
    /*Minimum*/         {0,1,1,0,0,0},
    /*Atan2*/           {0,1,1,0,0,0},
    /*Hypot*/           {0,1,1,0,0,0},
    // Unary math
    /*Negate*/          {1,0,1,0,0,0},
    /*Abs*/             {1,0,1,0,0,0},
    /*Sqrt*/            {1,0,1,0,0,0},
    /*Exp*/             {1,0,1,0,0,0},
    /*Log*/             {1,0,1,0,0,0},
    /*Sin*/             {1,0,1,0,0,0},
    /*Cos*/             {1,0,1,0,0,0},
    /*Tan*/             {1,0,1,0,0,0},
    /*Erf*/             {1,0,1,0,0,0},
    // NumPy-like math
    /*Sign*/            {1,0,1,0,0,0},
    /*Floor*/           {1,0,1,0,0,0},
    /*Ceil*/            {1,0,1,0,0,0},
    /*Trunc*/           {1,0,1,0,0,0},
    /*Round*/           {1,0,1,0,0,0},
    /*Reciprocal*/      {1,0,1,0,0,0},
    /*Square*/          {1,0,1,0,0,0},
    /*Cbrt*/            {1,0,1,0,0,0},
    // Element-wise testing
    /*IsNaN*/           {1,0,1,0,0,0},
    /*IsInf*/           {1,0,1,0,0,0},
    /*IsFinite*/        {1,0,1,0,0,0},
    // Complex
    /*Conj*/            {1,0,1,0,0,0},
    /*Real*/            {1,0,1,0,0,0},
    /*Imag*/            {1,0,1,0,0,0},
    // Activations
    /*ReLU*/            {1,0,1,0,0,0},
    /*LeakyReLU*/       {1,0,1,0,0,0},
    /*SiLU*/            {1,0,1,0,0,0},
    /*Sigmoid*/         {1,0,1,0,0,0},
    /*Tanh*/            {1,0,1,0,0,0},
    /*GELU*/            {1,0,1,0,0,0},
    /*Softmax*/         {0,0,0,0,0,0},
    /*LogSoftmax*/      {0,0,0,0,0,0},
    // Reductions
    /*Sum*/             {0,0,0,1,0,0},
    /*Mean*/            {0,0,0,1,0,0},
    /*Max*/             {0,0,0,1,0,0},
    /*Min*/             {0,0,0,1,0,0},
    /*ArgMax*/          {0,0,0,1,0,0},
    /*ArgMin*/          {0,0,0,1,0,0},
    /*Any*/             {0,0,0,1,0,0},
    /*All*/             {0,0,0,1,0,0},
    /*Prod*/            {0,0,0,1,0,0},
    // Matrix ops
    /*MatMul*/          {0,0,0,0,0,0},
    /*BatchMatMul*/     {0,0,0,0,0,0},
    // Conditional
    /*Where*/           {0,0,0,0,0,0},
    // Masking
    /*MaskedFill*/      {0,0,0,0,0,0},
    /*MaskedSelect*/    {0,0,0,0,0,0},
    // Indexing
    /*Gather*/          {0,0,0,0,0,0},
    /*Scatter*/         {0,0,0,0,0,0},
    /*IndexSelect*/     {0,0,0,0,0,0},
    /*Take*/            {0,0,0,0,0,0},
    /*TakeAlongAxis*/   {0,0,0,0,0,0},
    // Normalization
    /*LayerNorm*/       {0,0,0,0,0,0},
    /*RMSNorm*/         {0,0,0,0,0,0},
    // Dropout
    /*Dropout*/         {0,0,0,0,0,0},
    // Cast
    /*Cast*/            {0,0,0,0,0,0},
    // Pooling
    /*MaxPool1D*/       {0,0,0,0,0,0},
    /*MaxPool2D*/       {0,0,0,0,0,0},
    /*MaxPool3D*/       {0,0,0,0,0,0},
    /*AvgPool1D*/       {0,0,0,0,0,0},
    /*AvgPool2D*/       {0,0,0,0,0,0},
    /*AvgPool3D*/       {0,0,0,0,0,0},
    /*AdaptiveMaxPool2D*/ {0,0,0,0,0,0},
    /*AdaptiveAvgPool2D*/ {0,0,0,0,0,0},
    // Convolution
    /*Conv1D*/            {0,0,0,0,0,0},
    /*Conv2D*/            {0,0,0,0,0,0},
    /*ConvTranspose1D*/   {0,0,0,0,0,0},
    /*ConvTranspose2D*/   {0,0,0,0,0,0},
    // Fused attention
    /*ScaledDotProductAttention*/ {0,0,0,0,0,0},
};
// clang-format on

static_assert(sizeof(OP_TRAITS) / sizeof(OP_TRAITS[0]) ==
                  static_cast<size_t>(ops::OpType::_Count),
              "OP_TRAITS table must have one entry per OpType");

inline constexpr bool is_unary_op(ops::OpType op) {
    return OP_TRAITS[static_cast<size_t>(op)].is_unary;
}

inline constexpr bool is_binary_op(ops::OpType op) {
    return OP_TRAITS[static_cast<size_t>(op)].is_binary;
}

inline constexpr bool is_elementwise_op(ops::OpType op) {
    return OP_TRAITS[static_cast<size_t>(op)].is_elementwise;
}

inline constexpr bool is_reduction_op(ops::OpType op) {
    return OP_TRAITS[static_cast<size_t>(op)].is_reduction;
}

inline constexpr bool is_comparison_op(ops::OpType op) {
    return OP_TRAITS[static_cast<size_t>(op)].is_comparison;
}

inline constexpr bool is_logical_op(ops::OpType op) {
    return OP_TRAITS[static_cast<size_t>(op)].is_logical;
}

} // namespace graph
} // namespace axiom
