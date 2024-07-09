cmake_minimum_required(VERSION 3.5)

add_executable(cpu cpu.cpp)

target_link_libraries(cpu opr ${_VENDOR_LIBRARIES_})

if(NOT APPLE)
    add_executable(cuda cuda.cpp)

    target_link_libraries(cuda opr ${_VENDOR_LIBRARIES_})
endif()
