file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/backends/cpu/*.cpp)

if(NOT APPLE)
    file(GLOB_RECURSE CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/*.cu
         ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/*.cpp)
    list(APPEND SRCS ${CUDA_SRCS})
endif()

add_library(opr ${SRCS})

target_include_directories(
    opr SYSTEM PUBLIC ${_VENDOR_INCLUDE_DIR_}
                      /usr/local/cuda-12.2/targets/x86_64-linux/include)

target_include_directories(opr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                      ${CMAKE_CURRENT_SOURCE_DIR}/include)
