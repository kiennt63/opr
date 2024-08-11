#pragma once

#include "logger.h"
#include "node.h"

namespace opr::cpu {

template <typename t>
class const_node : public node {
public:
    const_node(int id, const tensor_shape& shape, int value) : node(id, shape) {
        cpu_buffer buffer(shape, sizeof(t));

        auto data = static_cast<t*>(buffer.get());
        for (size_t i = 0; i < shape.elems(); i++) {
            data[i] = value;
        }
        output = std::make_shared<tensor>(std::move(buffer));
    }

    const_node(int id, const tensor_shape& shape, void* value) : node(id, shape) {
        cpu_buffer buffer(shape, sizeof(t));

        memcpy(buffer.data, value, buffer.size_bytes());
        output = std::make_shared<tensor>(std::move(buffer));
    }

    virtual ~const_node() {}

    // do absly nothing
    status exec() override {
        log_inf("[const_node] output: {}", std::get<cpu_buffer>(*output));
        return status::ok;
    }
};

}  // namespace opr::cpu
