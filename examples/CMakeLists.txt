cmake_minimum_required(VERSION 3.5)

add_executable(example main.cpp)

target_link_libraries(example opr
    ${_VENDOR_LIBRARIES_}
)
