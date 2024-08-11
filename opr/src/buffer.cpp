#include "tensor.h"

namespace opr {

bool tensor_shape::operator==(const tensor_shape& other) {
    if (this->size() != other.size()) return false;
    for (size_t i = 0; i < this->size(); i++) {
        if (this->at(i) != other.at(i)) {
            return false;
        }
    }
    return true;
}

uint32_t tensor_shape::elems() {
    uint32_t total_elem = 1;
    for (size_t i = 0; i < this->size(); i++) {
        total_elem *= this->at(i);
    }

    return total_elem;
}

uint32_t tensor_shape::elems() const {
    uint32_t total_elem = 1;
    for (size_t i = 0; i < this->size(); i++) {
        total_elem *= this->at(i);
    }

    return total_elem;
}

const tensor_shape& buffer::shape() const {
    return shape_;
}
uint32_t buffer::size() const {
    return size_;
}
uint32_t buffer::size_bytes() const {
    return size_in_bytes_;
}

buffer::buffer(const tensor_shape& shape) : shape_(shape), size_(shape.elems()) {}
buffer::buffer(const tensor_shape& shape, uint32_t type_size_in_bytes)
    : shape_(shape),
      size_(shape.elems()),
      size_in_bytes_(type_size_in_bytes * shape.elems()) {}

buffer::buffer(const buffer& other)
    : shape_(other.shape_),
      size_(other.size_),
      size_in_bytes_(other.size_in_bytes_) {}

buffer::buffer(buffer&& other) noexcept
    : shape_(other.shape_),
      size_(other.size_),
      size_in_bytes_(other.size_in_bytes_) {
    other.shape_.resize(0);
    other.size_          = 0;
    other.size_in_bytes_ = 0;
}
buffer& buffer::operator=(const buffer& other) {
    if (this != &other) {
        shape_         = other.shape_;
        size_          = other.size_;
        size_in_bytes_ = other.size_in_bytes_;
    }
    return *this;
}
buffer& buffer::operator=(buffer&& other) noexcept {
    if (this != &other) {
        shape_         = other.shape_;
        size_          = other.size_;
        size_in_bytes_ = other.size_in_bytes_;
        other.shape_.resize(0);
        other.size_          = 0;
        other.size_in_bytes_ = 0;
    }
    return *this;
}

}  // namespace opr
