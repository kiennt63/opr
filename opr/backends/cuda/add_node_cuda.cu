#include "add_node_cuda.h"
#include "logger.h"

__global__ void vec_add(const float* buf0, const float* buf1, float* out, const int size)
{
    int index_1d = blockIdx.y * blockDim.y + blockIdx.x;
    // printf("index: %d - buf0: %.2f - buf1: %.2f\n", index_1d, buf0[index_1d], buf1[index_1d]);
    if (index_1d >= 0 && index_1d < size)
    {
        out[index_1d] = buf0[index_1d] + buf1[index_1d];
    }
}

namespace opr {

add_node_cuda::add_node_cuda(int id, const tensor_shape& shape) : node(id, shape)
{
    cuda_buffer buffer(shape, sizeof(int32_t));
    output = std::make_shared<tensor>(std::move(buffer));
    get_last_cuda_errors();
}

add_node_cuda::~add_node_cuda() {}

status add_node_cuda::exec()
{
    check_err(input.size() == 2, "[add_node_cuda]: inputs size must be 2");

    auto& first  = std::get<cuda_buffer>(*input[0]);
    auto& second = std::get<cuda_buffer>(*input[1]);
    auto& out    = std::get<cuda_buffer>(*output);
    get_last_cuda_errors();

    float* buf0    = static_cast<float*>(first.data);
    float* buf1    = static_cast<float*>(second.data);
    float* buf_out = static_cast<float*>(out.data);
    get_last_cuda_errors();

    check_err(first.size() == second.size(), "[add_node_cuda]: two inputs must have same size");
    check_err(first.size() == out.size(), "[add_node_cuda]: input and output must have same size");

    cudaStream_t add_stream;
    get_last_cuda_errors();
    cudaStreamCreate(&add_stream);
    get_last_cuda_errors();

    // TODO: handle multi-dimension
    dim3 thread_dim = {9 * 2 * 3, 1, 1};
    dim3 block_dim  = {out.size() - 1 / thread_dim.x + 1, 1, 1};

    get_last_cuda_errors();
    vec_add<<<block_dim, thread_dim>>>(buf0, buf1, buf_out, out.size());
    get_last_cuda_errors();

    cudaDeviceSynchronize();
    get_last_cuda_errors();
    cudaStreamDestroy(add_stream);
    get_last_cuda_errors();

    log_inf("[add_node_cuda]: exec()");
    // log_inf("[add_node_cuda] output: {}", out);
    return status::ok;
}

}  // namespace opr
