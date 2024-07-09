#include <variant>

#include "backends/cpu/cpu_buffer.h"

#ifdef __APPLE__
using tensor = std::variant<opr::cpu_buffer>;
#else
#include "backends/cuda/cuda_buffer.h"
using tensor = std::variant<opr::cpu_buffer, opr::cuda_buffer>;
#endif
