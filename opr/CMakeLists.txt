file(GLOB_RECURSE SRCS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/cpu/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/*.cu
)

add_library(opr ${SRCS})

print(_VENDOR_INCLUDE_DIR_)
target_include_directories(opr SYSTEM PUBLIC
    ${_VENDOR_INCLUDE_DIR_}
    /usr/local/cuda-12.2/targets/x86_64-linux/include
)

target_include_directories(opr  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
