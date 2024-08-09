#pragma once

#include "kernel.h"
#include "node.h"

namespace opr::cuda {

template <typename t>
class add_node : public node
{
public:
    add_node(int id, const tensor_shape& shape) : node(id, shape)
    {
        cuda_buffer buffer(shape, sizeof(int32_t));
        output = std::make_shared<tensor>(std::move(buffer));
        cuda_check_err();
    }
    virtual ~add_node() {}
    status exec() override
    {
        check_err(input.size() == 2, "[add_node]: inputs size must be 2");

        auto& first  = std::get<cuda_buffer>(*input[0]);
        auto& second = std::get<cuda_buffer>(*input[1]);
        auto& out    = std::get<cuda_buffer>(*output);
        cuda_check_err();

        check_err(first.size() == second.size(), "[add_node]: two inputs must have same size");
        check_err(first.size() == out.size(), "[add_node]: input and output must have same size");

        t* buf0    = static_cast<t*>(first.data);
        t* buf1    = static_cast<t*>(second.data);
        t* buf_out = static_cast<t*>(out.data);

        vec_add<t>(buf0, buf1, buf_out, out.size());

        return status::ok;
    }
};

}  // namespace opr::cuda
