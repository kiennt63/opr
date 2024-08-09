#pragma once

#include "logger.h"
#include "node.h"

namespace opr::cuda {

template <typename t>
class const_node : public node
{
public:
    const_node(int id, const tensor_shape& shape, int value) : node(id, shape)
    {
        cuda_buffer buffer(shape, sizeof(int32_t));
        get_last_cuda_errors();

        t* tmp = new t[buffer.size()];
        for (size_t i = 0; i < buffer.size(); ++i)
        {
            tmp[i] = value;
        }

        cudaMemcpy(buffer.data, tmp, buffer.size_bytes(), cudaMemcpyHostToDevice);

        free(tmp);

        get_last_cuda_errors();

        output = std::make_shared<tensor>(std::move(buffer));
        get_last_cuda_errors();
    }
    virtual ~const_node() {}
    status exec() override
    {
        log_inf("[const_node]: exec()");
        return status::ok;
    }
};

}  // namespace opr::cuda
