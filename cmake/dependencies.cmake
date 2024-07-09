set(VENDOR_PREFIX ${CMAKE_SOURCE_DIR}/vendor)

vendor_add(SPDLOG)

if(NOT APPLE)
    enable_language(CUDA)
endif()
