#pragma once

#include <cstdlib>
#include <variant>
#include <vector>

#include "defines.h"
#include "logger.h"

namespace opr {

class tensor_shape : public std::vector<uint32_t>
{
public:
    using std::vector<uint32_t>::vector;
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
    uint32_t elems()
    {
        uint32_t total_elem = 1;
        for (size_t i = 0; i < this->size(); i++)
        {
            total_elem *= this->at(i);
        }

        return total_elem;
    }
    uint32_t elems() const
    {
        uint32_t total_elem = 1;
        for (size_t i = 0; i < this->size(); i++)
        {
            total_elem *= this->at(i);
        }

        return total_elem;
    }
};

class buffer
{
public:
    buffer() = default;
    tensor_shape shape() { return shape_; }
    uint32_t size() { return size_; }
    uint32_t size_bytes() { return size_in_bytes_; }
    void* data = nullptr;

protected:
    explicit buffer(const tensor_shape& shape) : shape_(shape) {}
    explicit buffer(const tensor_shape& shape, uint32_t type_size_in_bytes)
        : shape_(shape),
          size_in_bytes_(type_size_in_bytes * shape.elems())
    {
    }

    buffer(const buffer& other)
        : shape_(other.shape_),
          size_(other.size_),
          size_in_bytes_(other.size_in_bytes_)
    {
    }

    buffer(buffer&& other) noexcept
        : shape_(other.shape_),
          size_(other.size_),
          size_in_bytes_(other.size_in_bytes_)
    {
        other.shape_.resize(0);
        other.size_          = 0;
        other.size_in_bytes_ = 0;
    }
    buffer& operator=(const buffer& other)
    {
        if (this != &other)
        {
            shape_         = other.shape_;
            size_          = other.size_;
            size_in_bytes_ = other.size_in_bytes_;
        }
        return *this;
    }
    buffer& operator=(buffer&& other) noexcept
    {
        if (this != &other)
        {
            shape_         = other.shape_;
            size_          = other.size_;
            size_in_bytes_ = other.size_in_bytes_;
            other.shape_.resize(0);
            other.size_          = 0;
            other.size_in_bytes_ = 0;
        }
        return *this;
    }

    tensor_shape shape_;
    uint32_t size_          = 0;
    uint32_t size_in_bytes_ = 0;
};

class cpu_buffer : public buffer
{
public:
    cpu_buffer() = default;

    cpu_buffer(const tensor_shape& shape) : buffer(shape) {}

    cpu_buffer(const tensor_shape& shape, uint32_t type_size_in_bytes)
        : buffer(shape, type_size_in_bytes)
    {
        if (data != nullptr)
        {
            free(data);
        }
        data = malloc(size_in_bytes_);
    }

    // copy constructor
    cpu_buffer(const cpu_buffer& other) : buffer(other)
    {
        if (data != nullptr)
        {
            free(data);
        }
        data = malloc(size_in_bytes_);
        memcpy(data, other.data, size_in_bytes_);
        log_wrn("[buffer]: copy constructor called");
    }

    // move constructor
    cpu_buffer(cpu_buffer&& other) noexcept : buffer(std::move(other))
    {
        data       = other.data;
        other.data = nullptr;
        log_wrn("[buffer]: move constructor called");
    }

    // copy assignment
    cpu_buffer& operator=(const cpu_buffer& other)
    {
        if (this != &other)
        {
            if (data)
            {
                free(data);
            }
            buffer::operator=(other);
            if (size_ > 0)
            {
                memcpy(data, other.data, size_in_bytes_);
            }
            else
            {
                data = nullptr;
            }
        }
        log_wrn("[buffer]: copy assignment called");
        return *this;
    }

    // move assignment
    cpu_buffer& operator=(cpu_buffer&& other) noexcept
    {
        if (this != &other)
        {
            if (data)
            {
                free(data);
            }
            buffer::operator=(std::move(other));

            data       = other.data;
            other.data = nullptr;
        }
        return *this;
    }

    template <typename t>
    t* get()
    {
        return reinterpret_cast<t*>(data);
    }

    ~cpu_buffer() { free(data); }
};

class cuda_buffer : public buffer
{
public:
    cuda_buffer() = default;
    cuda_buffer(const tensor_shape& shape) : buffer(shape) {}
    cuda_buffer(const tensor_shape& shape, uint32_t type_size_in_bytes)
        : buffer(shape, type_size_in_bytes)
    {
    }
};

using tensor = std::variant<cpu_buffer, cuda_buffer>;

}  // namespace opr
