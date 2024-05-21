file(GLOB_RECURSE SRCS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/cpu/*.cpp
)

add_library(opr ${SRCS})

target_include_directories(opr SYSTEM PUBLIC
    SYSTEM ${_VENDOR_INCLUDE_DIR_}
)

target_include_directories(opr  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
