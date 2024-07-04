#pragma once

#include <cstdlib>
#include <variant>
#include <vector>

#include <spdlog/fmt/bundled/format.h>

#include "defines.h"

#define get_last_cuda_errors()                                                 \
    {                                                                          \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            log_inf("{}: {}", cudaGetErrorName(err), cudaGetErrorString(err)); \
        }                                                                      \
    }

namespace opr {

class tensor_shape : public std::vector<uint32_t>
{
public:
    using std::vector<uint32_t>::vector;
    bool operator==(const tensor_shape& other);
    uint32_t elems();
    uint32_t elems() const;
};

class buffer
{
public:
    buffer() = default;
    const tensor_shape& shape() const;
    uint32_t size() const;
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

class cpu_buffer : public buffer
{
public:
    cpu_buffer() = default;

    cpu_buffer(const tensor_shape& shape);

    cpu_buffer(const tensor_shape& shape, uint32_t type_size_in_bytes);

    // copy constructor
    cpu_buffer(const cpu_buffer& other);

    // move constructor
    cpu_buffer(cpu_buffer&& other) noexcept;

    // copy assignment
    cpu_buffer& operator=(const cpu_buffer& other);

    // move assignment
    cpu_buffer& operator=(cpu_buffer&& other) noexcept;

    template <typename t>
    t* get() const;

    ~cpu_buffer();
};

std::ostream& operator<<(std::ostream& os, const cpu_buffer& obj);

class cuda_buffer : public buffer
{
public:
    cuda_buffer() = default;
    cuda_buffer(const tensor_shape& shape);
    cuda_buffer(const tensor_shape& shape, uint32_t type_size_in_bytes);
    // copy constructor
    cuda_buffer(const cuda_buffer& other);

    // move constructor
    cuda_buffer(cuda_buffer&& other) noexcept;

    // copy assignment
    cuda_buffer& operator=(const cuda_buffer& other);

    // move assignment
    cuda_buffer& operator=(cuda_buffer&& other) noexcept;

    ~cuda_buffer();
};

using tensor = std::variant<cpu_buffer, cuda_buffer>;

}  // namespace opr

// Specialize fmt::formatter for your type
namespace fmt {
template <>
struct formatter<opr::cpu_buffer>
{
    // Parses format specifications of the form 'int' and 'string'.
    constexpr auto parse(format_parse_context& ctx)
    {
        // Parse until '}' or ':' (if no format spec is given)
        // auto it = ctx.begin(), end = ctx.end();
        // if (it != end && (*it == '}' || *it == ':')) return it;
        // throw format_error("invalid format");
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const opr::cpu_buffer& obj, FormatContext& ctx)
    {
        constexpr int print_truncation_length = 4;

        std::string values_str = "[";
        const auto values      = obj.get<int32_t>();
        for (size_t i = 0; i < obj.size(); ++i)
        {
            if (i > print_truncation_length && i != obj.size() - 1) continue;

            values_str += fmt::format("{}", values[i]);

            if (i == obj.size() - 1) continue;

            if (i == print_truncation_length)
            {
                values_str += ", .., ";
            }
            else
            {
                values_str += ", ";
            }
        }
        values_str += "]";
        return format_to(ctx.out(), "{}", values_str);
    }
};

}  // namespace fmt
