#include "axiom/indexing.hpp"
#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"

namespace axiom {

// ============================================================================
// TensorIndex Implementation
// ============================================================================

TensorIndex::TensorIndex(const Tensor &t)
    : indices(std::make_shared<Tensor>(t)) {}

// ============================================================================
// MaskedView Implementation
// ============================================================================

MaskedView::MaskedView(Tensor &parent, const Tensor &mask)
    : parent_(&parent), mask_(std::make_shared<Tensor>(mask)),
      is_const_(false) {}

MaskedView::MaskedView(const Tensor &parent, const Tensor &mask)
    : parent_(const_cast<Tensor *>(&parent)),
      mask_(std::make_shared<Tensor>(mask)), is_const_(true) {}

MaskedView::operator Tensor() const { return select(); }

MaskedView &MaskedView::operator=(float value) {
    if (is_const_) {
        throw MemoryError("Cannot assign to const MaskedView");
    }
    *parent_ = parent_->masked_fill(*mask_, value);
    return *this;
}

MaskedView &MaskedView::operator=(double value) {
    if (is_const_) {
        throw MemoryError("Cannot assign to const MaskedView");
    }
    *parent_ = parent_->masked_fill(*mask_, static_cast<float>(value));
    return *this;
}

MaskedView &MaskedView::operator=(int32_t value) {
    if (is_const_) {
        throw MemoryError("Cannot assign to const MaskedView");
    }
    *parent_ = parent_->masked_fill(*mask_, static_cast<float>(value));
    return *this;
}

MaskedView &MaskedView::operator=(int64_t value) {
    if (is_const_) {
        throw MemoryError("Cannot assign to const MaskedView");
    }
    *parent_ = parent_->masked_fill(*mask_, static_cast<float>(value));
    return *this;
}

MaskedView &MaskedView::operator=(const Tensor &values) {
    if (is_const_) {
        throw MemoryError("Cannot assign to const MaskedView");
    }
    // For tensor assignment, we need to scatter values back
    // This requires masked_scatter which we'll implement as a simple where
    // For now, use masked_fill with the first element if it's a scalar-like tensor
    if (values.size() == 1) {
        float val = values.cpu().typed_data<float>()[0];
        *parent_ = parent_->masked_fill(*mask_, val);
    } else {
        // Full masked_scatter semantics: place values where mask is true
        // This is more complex - for now throw
        throw RuntimeError::not_implemented(
            "MaskedView assignment with non-scalar tensor");
    }
    return *this;
}

Tensor MaskedView::select() const { return parent_->masked_select(*mask_); }

const Tensor &MaskedView::mask() const { return *mask_; }

const Tensor &MaskedView::parent() const { return *parent_; }

} // namespace axiom
