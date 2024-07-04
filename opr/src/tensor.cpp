#include <cuda_runtime_api.h>

#include "logger.h"
#include "tensor.h"

namespace opr {

bool tensor_shape::operator==(const tensor_shape& other)
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

uint32_t tensor_shape::elems()
{
    uint32_t total_elem = 1;
    for (size_t i = 0; i < this->size(); i++)
    {
        total_elem *= this->at(i);
    }

    return total_elem;
}

uint32_t tensor_shape::elems() const
{
    uint32_t total_elem = 1;
    for (size_t i = 0; i < this->size(); i++)
    {
        total_elem *= this->at(i);
    }

    return total_elem;
}

const tensor_shape& buffer::shape() const
{
    return shape_;
}
uint32_t buffer::size() const
{
    return size_;
}
uint32_t buffer::size_bytes() const
{
    return size_in_bytes_;
}

buffer::buffer(const tensor_shape& shape) : shape_(shape), size_(shape.elems()) {}
buffer::buffer(const tensor_shape& shape, uint32_t type_size_in_bytes)
    : shape_(shape),
      size_(shape.elems()),
      size_in_bytes_(type_size_in_bytes * shape.elems())
{
}

buffer::buffer(const buffer& other)
    : shape_(other.shape_),
      size_(other.size_),
      size_in_bytes_(other.size_in_bytes_)
{
}

buffer::buffer(buffer&& other) noexcept
    : shape_(other.shape_),
      size_(other.size_),
      size_in_bytes_(other.size_in_bytes_)
{
    other.shape_.resize(0);
    other.size_          = 0;
    other.size_in_bytes_ = 0;
}
buffer& buffer::operator=(const buffer& other)
{
    if (this != &other)
    {
        shape_         = other.shape_;
        size_          = other.size_;
        size_in_bytes_ = other.size_in_bytes_;
    }
    return *this;
}
buffer& buffer::operator=(buffer&& other) noexcept
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

cpu_buffer::cpu_buffer(const tensor_shape& shape) : buffer(shape) {}

cpu_buffer::cpu_buffer(const tensor_shape& shape, uint32_t type_size_in_bytes)
    : buffer(shape, type_size_in_bytes)
{
    if (data != nullptr)
    {
        free(data);
    }
    data = malloc(size_in_bytes_);
}

// copy constructor
cpu_buffer::cpu_buffer(const cpu_buffer& other) : buffer(other)
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
cpu_buffer::cpu_buffer(cpu_buffer&& other) noexcept : buffer(std::move(other))
{
    data       = other.data;
    other.data = nullptr;
}

// copy assignment
cpu_buffer& cpu_buffer::operator=(const cpu_buffer& other)
{
    if (this != &other)
    {
        if (data)
        {
            free(data);
        }
        buffer::operator=(other);
        data            = malloc(size_in_bytes_);
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
cpu_buffer& cpu_buffer::operator=(cpu_buffer&& other) noexcept
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
t* cpu_buffer::get() const
{
    return reinterpret_cast<t*>(data);
}

cpu_buffer::~cpu_buffer()
{
    free(data);
}

std::ostream& operator<<(std::ostream& os, const cpu_buffer& obj)
{
    for (size_t i = 0; i < obj.size(); i++)
    {
        os << obj.get<int32_t>()[i] << " ";
    }
    os << "\n";
    return os;
}

cuda_buffer::cuda_buffer(const tensor_shape& shape) : buffer(shape) {}
cuda_buffer::cuda_buffer(const tensor_shape& shape, uint32_t type_size_in_bytes)
    : buffer(shape, type_size_in_bytes)
{
    // TODO: allocate a cuda buffer in here with shape and size
    if (data)
    {
        cudaFree(data);
    }
    cudaMalloc((void**)&data, size_in_bytes_);
}
// copy constructor
cuda_buffer::cuda_buffer(const cuda_buffer& other) : buffer(other)
{
    if (data)
    {
        cudaFree(data);
    }
    cudaMalloc((void**)&data, size_in_bytes_);
    cudaMemcpy(data, other.data, size_in_bytes_, cudaMemcpyDeviceToDevice);
    log_wrn("[cuda_buffer]: copy constructor called");
}

// move constructor
cuda_buffer::cuda_buffer(cuda_buffer&& other) noexcept : buffer(std::move(other))
{
    data       = other.data;
    other.data = nullptr;
}

// copy assignment
cuda_buffer& cuda_buffer::operator=(const cuda_buffer& other)
{
    if (this != &other)
    {
        if (data)
        {
            cudaFree(data);
        }
        buffer::operator=(other);
        cudaMalloc((void**)&data, size_in_bytes_);

        if (size_ > 0)
        {
            cudaMemcpy(data, other.data, size_in_bytes_, cudaMemcpyDeviceToDevice);
        }
        else
        {
            data = nullptr;
        }
    }
    log_wrn("[cuda_buffer]: copy assignment called");
    return *this;
}

// move assignment
cuda_buffer& cuda_buffer::operator=(cuda_buffer&& other) noexcept
{
    if (this != &other)
    {
        if (data)
        {
            cudaFree(data);
        }
        buffer::operator=(std::move(other));

        data       = other.data;
        other.data = nullptr;
    }
    return *this;
}

cuda_buffer::~cuda_buffer()
{
    cudaFree(data);
}

}  // namespace opr
