#
# a library included as a git submodule that we do not control
#
cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0063 NEW)
project(submod)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_ARCHITECTURES "60;70;75")
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
enable_language(CUDA)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(submod devut.cu)
set_target_properties(submod PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(submod PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(submod PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
#set_target_properties(submod PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(submod PROPERTIES CUDA_ARCHITECTURES "60;70;75")

target_include_directories(submod
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_PREFIX}>
    )

