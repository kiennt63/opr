find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
endif()

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
    option(OPT_CUDA_ENABLED "" ON)
else()
    option(OPT_CUDA_ENABLED "" OFF)
endif()

option(OPT_MESSAGE_LIB "" ON)

print_option(OPT_CUDA_ENABLED)
print_option(OPT_MESSAGE_LIB)
