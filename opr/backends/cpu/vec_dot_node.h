#pragma once

#include "logger.h"
#include "node.h"

namespace opr::cpu {

template <typename t>
class vec_dot_node : public node {
public:
    vec_dot_node(int id, const tensor_shape& shape) : node(id, shape) {
        cpu_buffer buffer(shape, sizeof(t));
        output = std::make_shared<tensor>(std::move(buffer));
    }

    virtual ~vec_dot_node() {}

    status exec() override {
        check_err(input.size() == 2, "[vec_dot_node]: inputs size must be 2");

        auto& first  = std::get<cpu_buffer>(*input[0]);
        auto& second = std::get<cpu_buffer>(*input[1]);
        auto& out    = std::get<cpu_buffer>(*output);

        check_err(first.size() == second.size(), "[vec_dot_node]: two inputs must have same size");
        check_err(out.size() == 1, "[vec_dot_node]: output must be single value");

        auto buf0    = static_cast<t*>(first.get());
        auto buf1    = static_cast<t*>(second.get());
        auto buf_out = static_cast<t*>(out.get());
        for (size_t i = 0; i < out.size(); i++) {
            buf_out[0] += buf0[i] * buf1[i];
        }

        log_inf("[vec_dot_node] output: {}", out);
        return status::ok;
    }
};

}  // namespace opr::cpu
