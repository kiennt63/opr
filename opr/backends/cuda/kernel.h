#pragma once

#include <cuda_runtime_api.h>

#include "logger.h"

#define cuda_check_err()                                                       \
    {                                                                          \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            log_inf("{}: {}", cudaGetErrorName(err), cudaGetErrorString(err)); \
        }                                                                      \
    }

template <typename t>
__global__ void vec_add_kern(const t* buf0, const t* buf1, t* out, const unsigned long size)
{
    int index_1d = blockIdx.y * blockDim.x + threadIdx.x;
    if (index_1d >= 0 && index_1d < size)
    {
        out[index_1d] = buf0[index_1d] + buf1[index_1d];
    }
}

template <typename t>
__global__ void vec_sub_kern(const t* buf0, const t* buf1, t* out, const unsigned long size)
{
    int index_1d = blockIdx.y * blockDim.x + threadIdx.x;
    if (index_1d >= 0 && index_1d < size)
    {
        out[index_1d] = buf0[index_1d] - buf1[index_1d];
    }
}

template <typename t>
void vec_add(const t* buf0, const t* buf1, t* out, const unsigned long num_elem)
{
    cudaStream_t add_stream;
    cuda_check_err();
    cudaStreamCreate(&add_stream);
    cuda_check_err();

    // TODO: handle multi-dimension
    dim3 block_dim(512, 1, 1);
    dim3 grid_dim = ((num_elem + block_dim.x - 1) / block_dim.x);

    cuda_check_err();
    log_inf("running vec_add for size = {}", num_elem);
    vec_add_kern<t><<<grid_dim, block_dim>>>(buf0, buf1, out, num_elem);
    cuda_check_err();

    cudaDeviceSynchronize();
    cuda_check_err();
    cudaStreamDestroy(add_stream);
    cuda_check_err();
}

template <typename t>
void vec_sub(const t* buf0, const t* buf1, t* out, const unsigned long num_elem)
{
    cudaStream_t sub_stream;
    cuda_check_err();
    cudaStreamCreate(&sub_stream);
    cuda_check_err();

    // TODO: handle multi-dimension
    dim3 block_dim(512, 1, 1);
    dim3 grid_dim = ((num_elem + block_dim.x - 1) / block_dim.x);

    cuda_check_err();
    log_inf("running vec_sub for size = {}", num_elem);
    vec_sub_kern<t><<<grid_dim, block_dim>>>(buf0, buf1, out, num_elem);
    cuda_check_err();

    cudaDeviceSynchronize();
    cuda_check_err();
    cudaStreamDestroy(sub_stream);
    cuda_check_err();
}
