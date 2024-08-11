#pragma once

#include <vector>

#include "defines.h"

namespace opr {

class tensor_shape : public std::vector<uint32_t> {
public:
    using std::vector<uint32_t>::vector;
    bool operator==(const tensor_shape& other);
    uint32_t elems();
    uint32_t elems() const;
};

class buffer {
public:
    buffer() = default;
    const tensor_shape& shape() const;

    // return num elements of the buffer
    uint32_t size() const;

    // return size in byte of the buffer
    uint32_t size_bytes() const;
    void* data = nullptr;

protected:
    explicit buffer(const tensor_shape& shape);
    explicit buffer(const tensor_shape& shape, uint32_t type_size_in_bytes);

    buffer(const buffer& other);
    buffer(buffer&& other) noexcept;
    buffer& operator=(const buffer& other);
    buffer& operator=(buffer&& other) noexcept;

    tensor_shape shape_;
    uint32_t size_          = 0;
    uint32_t size_in_bytes_ = 0;
};

}  // namespace opr
