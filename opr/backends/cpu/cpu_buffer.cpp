#include "cpu_buffer.h"
#include "logger.h"

namespace opr {

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

        data = malloc(size_in_bytes_);

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

void* cpu_buffer::get() const
{
    return data;
}

cpu_buffer::~cpu_buffer()
{
    free(data);
}

std::ostream& operator<<(std::ostream& os, const cpu_buffer& obj)
{
    for (size_t i = 0; i < obj.size(); i++)
    {
        os << static_cast<float32_t*>(obj.data)[i] << " ";
    }
    os << "\n";
    return os;
}

}  // namespace opr
