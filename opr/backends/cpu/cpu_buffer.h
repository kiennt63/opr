#include <spdlog/fmt/bundled/format.h>
#include <iostream>

#include "buffer.h"

namespace opr {

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

    void* get() const;

    ~cpu_buffer();
};

std::ostream& operator<<(std::ostream& os, const cpu_buffer& obj);

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
        const auto values      = static_cast<float32_t*>(obj.get());
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
