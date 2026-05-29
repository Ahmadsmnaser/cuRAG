cmake_minimum_required(VERSION 3.18)

project(cuRAG LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(curag_dummy
    src/main.cpp
    kernels/dummy_kernel.cu
)

target_include_directories(curag_dummy PRIVATE
    include
)

set_target_properties(curag_dummy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)