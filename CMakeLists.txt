cmake_minimum_required(VERSION 3.28)
project(andrea CUDA CXX)

list(APPEND CMAKE_PREFIX_PATH "/usr/local/cuda-12")
set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(include)

add_library(cuda_utils STATIC src/cuda_utils.cu)
add_library(tensor STATIC src/tensor.cpp)
add_library(tensor_ops STATIC src/tensor_ops.cu)

set_target_properties(cuda_utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tensor_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE cuda_utils tensor tensor_ops)