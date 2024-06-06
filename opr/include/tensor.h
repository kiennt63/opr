#pragma once

#include <variant>
#include <vector>

#include "defines.h"

namespace opr {

class tensor_shape : public std::vector<uint32_t>
{
public:
    bool operator==(const tensor_shape& other)
    {
        if (this->size() != other.size()) return false;
        for (size_t i = 0; i < this->size(); i++)
        {
            if (this->at(i) != other.at(i))
            {
                return false;
            }
        }
        return true;
    }
};

template <typename t>
class buffer
{
public:
    tensor_shape shape() { return shape_; }
    uint32_t size() { return size_; }
    uint32_t size_bytes() { return size_bytes_; }

private:
    tensor_shape shape_;
    uint32_t size_       = 0;
    uint32_t size_bytes_ = 0;
};

class cpu_buffer : public buffer<cpu_buffer>
{
    cpu_buffer() = default;
    cpu_buffer(void* new_data) : data(new_data) {}
    void* data = nullptr;
};

struct cuda_buffer : public buffer<cuda_buffer>
{
    cuda_buffer() = default;
    cuda_buffer(void* other) : data(other) {}
    void* data = nullptr;
};

using tensor = std::variant<cpu_buffer, cuda_buffer>;

}  // namespace opr
