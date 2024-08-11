#include <cuda_runtime_api.h>

#include "buffer.h"

#define get_last_cuda_errors()                                                 \
    {                                                                          \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            log_inf("{}: {}", cudaGetErrorName(err), cudaGetErrorString(err)); \
        }                                                                      \
    }

namespace opr {

class cuda_buffer : public buffer {
public:
    cuda_buffer() = default;
    cuda_buffer(const tensor_shape& shape);
    cuda_buffer(const tensor_shape& shape, uint32_t type_size_in_bytes);
    // copy constructor
    cuda_buffer(const cuda_buffer& other);

    // move constructor
    cuda_buffer(cuda_buffer&& other) noexcept;

    // copy assignment
    cuda_buffer& operator=(const cuda_buffer& other);

    // move assignment
    cuda_buffer& operator=(cuda_buffer&& other) noexcept;

    ~cuda_buffer();
};

}  // namespace opr
