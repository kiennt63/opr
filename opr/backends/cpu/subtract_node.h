#pragma once

#include "logger.h"
#include "node.h"

namespace opr::cpu {

template <typename t>
class subtract_node : public node
{
public:
    subtract_node(int id, const tensor_shape& shape) : node(id, shape)
    {
        cpu_buffer buffer(shape, sizeof(t));
        output = std::make_shared<tensor>(std::move(buffer));
    }

    virtual ~subtract_node() {}

    status exec() override
    {
        check_err(input.size() == 2, "[subtract_node]: inputs size must be 2");

        auto& first  = std::get<cpu_buffer>(*input[0]);
        auto& second = std::get<cpu_buffer>(*input[1]);
        auto& out    = std::get<cpu_buffer>(*output);

        check_err(first.size() == second.size(), "[subtract_node]: two inputs must have same size");
        check_err(first.size() == out.size(),
                  "[subtract_node]: input and output must have same size");

        auto buf0    = static_cast<t*>(first.get());
        auto buf1    = static_cast<t*>(second.get());
        auto buf_out = static_cast<t*>(out.get());
        for (size_t i = 0; i < out.size(); i++)
        {
            buf_out[i] = buf0[i] - buf1[i];
        }

        log_inf("[subtract_node] output: {}", out);
        return status::ok;
    }
};

}  // namespace opr::cpu
