#pragma once

#include <variant>

namespace opr {

struct cpu_memory
{
    cpu_memory() = default;
    cpu_memory(void* new_data, size_t new_size_bytes) : data(new_data), size_bytes(new_size_bytes)
    {
    }

    void* data        = nullptr;
    size_t size_bytes = 0;
};

struct cuda_memory
{
    cuda_memory() = default;
    cuda_memory(void* other) : data(other) {}
    void* data        = nullptr;
    size_t size_bytes = 0;
};

using buffer = std::variant<cpu_memory, cuda_memory>;

}  // namespace opr
