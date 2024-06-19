macro(vendor_add lib_name)
    find_package(${lib_name})
    link_directories(${${lib_name}_LIBRARY_DIR})
    list(APPEND _VENDOR_INCLUDE_DIR_ ${${lib_name}_INCLUDE_DIR})
    list(APPEND _VENDOR_LIBRARY_DIR_ ${${lib_name}_LIBRARY_DIR})
    list(APPEND _VENDOR_LIBRARIES_ ${${lib_name}_LIBRARIES})
    print_lib(${lib_name})
endmacro()
