#pragma once

#include <cuda_runtime_api.h>

#include "logger.h"

#define cuda_check_err()                                                       \
    {                                                                          \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            log_inf("{}: {}", cudaGetErrorName(err), cudaGetErrorString(err)); \
        }                                                                      \
    }

template <typename t>
inline void read_and_print(const std::string& buf_name, t* in, size_t num_elem) {
    t* h_buf = new t[num_elem];
    cudaMemcpy(h_buf, in, num_elem * sizeof(t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("----------------------------- %s -----------------------------\n", buf_name.c_str());
    for (size_t i = 0; i < num_elem; i++) {
        printf("%.1f ", static_cast<float>(h_buf[i]));
    }
    printf("\n------------------------------------------------------------\n");
    delete[] h_buf;
}

template <typename t>
__global__ void vec_add_kern(const t* buf0, const t* buf1, t* out, const unsigned long num_elem) {
    size_t index_1d = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_1d < num_elem) {
        out[index_1d] = buf0[index_1d] + buf1[index_1d];
    }
}

template <typename t>
__global__ void vec_sub_kern(const t* buf0, const t* buf1, t* out, const unsigned long num_elem) {
    size_t index_1d = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_1d < num_elem) {
        out[index_1d] = buf0[index_1d] - buf1[index_1d];
    }
}

template <typename t>
__global__ void vec_mul_kern(const t* buf0, const t* buf1, t* out, const unsigned long num_elem) {
    size_t index_1d = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_1d < num_elem) {
        out[index_1d] = buf0[index_1d] * buf1[index_1d];
    }
}

template <typename t>
__device__ void warp_sum(volatile t* s_data, size_t tid) {
    s_data[tid] += s_data[tid + 32];
    s_data[tid] += s_data[tid + 16];
    s_data[tid] += s_data[tid + 8];
    s_data[tid] += s_data[tid + 4];
    s_data[tid] += s_data[tid + 2];
    s_data[tid] += s_data[tid + 1];
}

template <typename t>
__global__ void vec_sum_kern(const t* in, t* out, const unsigned long num_elem) {
    extern __shared__ t s_data[];

    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * (blockDim.x * 2) + tid;

    t value = 0;
    if (gid < num_elem) {
        value += in[gid];
    }
    if (gid + blockDim.x < num_elem) {
        value += in[gid + blockDim.x];
    }
    s_data[tid] = value;

    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warp_sum(s_data, tid);
    }

    if (tid == 0) {
        out[blockIdx.x] = s_data[0];
    }
}

template <typename t>
void vec_add(const t* buf0, const t* buf1, t* out, const unsigned long num_elem) {
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
void vec_mul(const t* buf0, const t* buf1, t* out, const unsigned long num_elem) {
    cudaStream_t mul_stream;
    cuda_check_err();
    cudaStreamCreate(&mul_stream);
    cuda_check_err();

    // TODO: handle multi-dimension
    dim3 block_dim(512, 1, 1);
    dim3 grid_dim = ((num_elem + block_dim.x - 1) / block_dim.x);

    cuda_check_err();
    log_inf("running vec_mul for size = {}", num_elem);
    vec_mul_kern<t><<<grid_dim, block_dim>>>(buf0, buf1, out, num_elem);
    cuda_check_err();

    cudaDeviceSynchronize();
    cuda_check_err();
    cudaStreamDestroy(mul_stream);
    cuda_check_err();
}

template <typename t>
void vec_sub(const t* buf0, const t* buf1, t* out, const unsigned long num_elem) {
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

// expect out and in to be same size
template <typename t>
void vec_sum(const t* in, t* out, const unsigned long num_elem) {
    cudaStream_t sum_stream;
    cuda_check_err();
    cudaStreamCreate(&sum_stream);
    cuda_check_err();

    log_inf("running vec_sum for size = {}", num_elem);
    dim3 block_dim(512, 1, 1);
    dim3 grid_dim = ((num_elem + (block_dim.x * 2) - 1) / (block_dim.x * 2));

    t* tmp0;
    t* tmp1;
    cudaMalloc((void**)&tmp0, num_elem * sizeof(t));
    cudaMalloc((void**)&tmp1, num_elem * sizeof(t));
    cudaMemcpy(tmp0, in, num_elem * sizeof(t), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    cuda_check_err();

    size_t new_num_elem = num_elem;
    while (grid_dim.x > 1) {
        // printf("kernel call with grid_dim (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
        vec_sum_kern<<<grid_dim, block_dim, block_dim.x * sizeof(int)>>>(tmp0, tmp1, new_num_elem);

        tmp0         = tmp1;
        new_num_elem = grid_dim.x;
        grid_dim     = ((new_num_elem + (block_dim.x * 2) - 1) / (block_dim.x * 2));
    }

    // printf("kernel call with grid_dim (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
    vec_sum_kern<<<grid_dim, block_dim, block_dim.x * sizeof(int)>>>(tmp0, tmp1, new_num_elem);

    cudaMemcpy(out, tmp1, sizeof(t), cudaMemcpyDeviceToDevice);
}

template <typename t>
void vec_dot(const t* buf0, const t* buf1, t* out, const unsigned long num_elem) {
    cudaStream_t dot_stream;
    cuda_check_err();
    cudaStreamCreate(&dot_stream);
    cuda_check_err();

    log_inf("running vec_dot for size = {}", num_elem);
    t* buf_mult;
    cudaMalloc((void**)&buf_mult, num_elem * sizeof(t));
    vec_mul(buf0, buf1, buf_mult, num_elem);
    vec_sum(buf_mult, out, num_elem);
    cudaDeviceSynchronize();
    cuda_check_err();
    cudaStreamDestroy(dot_stream);
    cuda_check_err();
}
