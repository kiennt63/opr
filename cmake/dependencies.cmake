set(VENDOR_PREFIX ${CMAKE_SOURCE_DIR}/vendor)

vendor_add(SPDLOG)

if(NOT APPLE)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
endif()
