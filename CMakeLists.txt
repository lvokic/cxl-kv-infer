cmake_minimum_required(VERSION 3.20)
project(cxl_kv_native LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The NUMA bridge requires Linux; no placement fallback")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_path(NUMA_INCLUDE_DIR numaif.h REQUIRED)
find_library(NUMA_LIBRARY numa REQUIRED)

pybind11_add_module(_numa src/storage/numa.cpp)
target_compile_features(_numa PRIVATE cxx_std_17)
target_compile_options(_numa PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(_numa PRIVATE ${NUMA_INCLUDE_DIR})
target_link_libraries(_numa PRIVATE ${NUMA_LIBRARY})
set_target_properties(_numa PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/storage")
