#include "subtract_node.h"
#include "logger.h"

namespace opr {

subtract_node::subtract_node(int id, const tensor_shape& shape) : node(id, shape)
{
    cpu_buffer buffer(shape, sizeof(int32_t));
    output = std::make_shared<tensor>(std::move(buffer));
}

subtract_node::~subtract_node() {}

status subtract_node::exec()
{
    check_err(input.size() == 2, "[subtract_node]: inputs size must be 2");

    auto& first  = std::get<cpu_buffer>(*input[0]);
    auto& second = std::get<cpu_buffer>(*input[1]);
    auto& out    = std::get<cpu_buffer>(*output);

    check_err(first.size() == second.size(), "[subtract_node]: two inputs must have same size");
    check_err(first.size() == out.size(), "[subtract_node]: input and output must have same size");

    auto buf0    = first.get<int32_t>();
    auto buf1    = second.get<int32_t>();
    auto buf_out = out.get<int32_t>();
    for (size_t i = 0; i < out.size(); i++)
    {
        buf_out[i] = buf0[i] - buf1[i];
    }

    log_inf("[subtract_node]: output: {}", buf_out[out.size() - 1]);
    return status::ok;
}

}  // namespace opr
