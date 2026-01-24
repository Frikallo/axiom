#pragma once

#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"

// Forward-declare Objective-C types
#ifdef __OBJC__
@class MPSGraph;
@class MPSGraphTensor;
@class MPSGraphTensorData;
#else
typedef void MPSGraph;
typedef void MPSGraphTensor;
typedef void MPSGraphTensorData;
#endif

namespace axiom {
namespace backends {
namespace metal {

// ============================================================================
// MPSGraph Helper Infrastructure
// ============================================================================

// Type to represent a block that builds an MPSGraph operation
// Takes MPSGraph and input tensors, returns output tensor
using MPSGraphBinaryOpBlock = MPSGraphTensor* (*)(MPSGraph*, MPSGraphTensor*, MPSGraphTensor*);
using MPSGraphUnaryOpBlock = MPSGraphTensor* (*)(MPSGraph*, MPSGraphTensor*);
using MPSGraphTernaryOpBlock = MPSGraphTensor* (*)(MPSGraph*, MPSGraphTensor*, MPSGraphTensor*, MPSGraphTensor*);

// Base class for all MPSGraph-based operations
class MPSGraphOperation : public ops::Operation {
protected:
    ops::OpType op_type_;
    std::string op_name_;

public:
    MPSGraphOperation(ops::OpType op_type, std::string op_name)
        : op_type_(op_type), op_name_(std::move(op_name)) {}

    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return op_name_; }
    Device device() const override { return Device::GPU; }

    bool supports_binary(const Tensor& lhs, const Tensor& rhs) const override {
        // MPSGraph handles broadcasting automatically
        return true;
    }
};

// ============================================================================
// MPSGraph Binary Operations
// ============================================================================

class MPSGraphBinaryOperation : public MPSGraphOperation {
private:
    MPSGraphBinaryOpBlock op_block_;

public:
    MPSGraphBinaryOperation(ops::OpType op_type, std::string op_name, 
                           MPSGraphBinaryOpBlock op_block);

    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override;
    
    Tensor execute_unary(const Tensor& input) const override {
        throw RuntimeError::internal("execute_unary called on binary operation");
    }
    
    Tensor execute_reduction(const Tensor& input, const std::vector<int>& axis, bool keep_dims) const override {
        throw RuntimeError::internal("execute_reduction called on binary operation");
    }
};

// ============================================================================
// MPSGraph Unary Operations
// ============================================================================

class MPSGraphUnaryOperation : public MPSGraphOperation {
private:
    MPSGraphUnaryOpBlock op_block_;

public:
    MPSGraphUnaryOperation(ops::OpType op_type, std::string op_name,
                          MPSGraphUnaryOpBlock op_block);

    Tensor execute_unary(const Tensor& input) const override;
    
    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        throw RuntimeError::internal("execute_binary called on unary operation");
    }
    
    Tensor execute_reduction(const Tensor& input, const std::vector<int>& axis, bool keep_dims) const override {
        throw RuntimeError::internal("execute_reduction called on unary operation");
    }
};

// ============================================================================
// MPSGraph Ternary Operations (for 'where')
// ============================================================================

class MPSGraphTernaryOperation : public MPSGraphOperation {
private:
    MPSGraphTernaryOpBlock op_block_;

public:
    MPSGraphTernaryOperation(ops::OpType op_type, std::string op_name,
                            MPSGraphTernaryOpBlock op_block);

    // 'where' is a special ternary operation
    Tensor execute_where(const Tensor& condition, const Tensor& a, const Tensor& b) const;
    
    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        throw RuntimeError::internal("execute_binary called on ternary operation");
    }
    
    Tensor execute_unary(const Tensor& input) const override {
        throw RuntimeError::internal("execute_unary called on ternary operation");
    }
    
    Tensor execute_reduction(const Tensor& input, const std::vector<int>& axis, bool keep_dims) const override {
        throw RuntimeError::internal("execute_reduction called on ternary operation");
    }
};

// ============================================================================
// MPSGraph MatMul Operation
// ============================================================================

// Note: MPSGraphMatMulOperation is defined in the .mm file since it's a simple
// wrapper that delegates to the executeMatMul function. Unlike binary/unary ops,
// it doesn't need a block-based approach.

// ============================================================================
// MPSGraph ArgMax/ArgMin Operations
// ============================================================================

// Note: MPSGraphArgMaxMinOperation is defined in the .mm file since it's a 
// simple wrapper for the reduction operations.

// ============================================================================
// Registration Functions
// ============================================================================

void register_mpsgraph_operations();

} // namespace metal
} // namespace backends
} // namespace axiom
