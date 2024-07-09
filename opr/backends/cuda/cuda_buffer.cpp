#include "cuda_buffer.h"
#include "logger.h"

namespace opr {

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
