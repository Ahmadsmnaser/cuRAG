cmake_minimum_required(VERSION 3.18)

project(cuRAG LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_testing()

add_executable(test_cosine_similarity
    tests/test_cosine_similarity.cpp
    kernels/cosine_similarity.cu
)

target_include_directories(test_cosine_similarity PRIVATE include)

set_target_properties(test_cosine_similarity PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(
    NAME cosine_similarity
    COMMAND test_cosine_similarity
)