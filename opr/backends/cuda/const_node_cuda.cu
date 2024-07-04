#include "const_node_cuda.h"
#include "logger.h"

namespace opr {

const_node_cuda::const_node_cuda(int id, const tensor_shape& shape, int value) : node(id, shape)
{
    cuda_buffer buffer(shape, sizeof(int32_t));
    get_last_cuda_errors();

    float* tmp = new float[buffer.size()];
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

const_node_cuda::~const_node_cuda() {}

// do absly nothing
status const_node_cuda::exec()
{
    log_inf("[const_node_cuda]: exec()");
    // log_inf("[const_node_cuda] output: {}", std::get<cpu_buffer>(*output));
    return status::ok;
}

}  // namespace opr
