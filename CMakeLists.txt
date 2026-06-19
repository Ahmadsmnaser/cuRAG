cmake_minimum_required(VERSION 3.18)

project(cuRAG LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_testing()

add_library(curag STATIC
    kernels/cosine_similarity.cu
    kernels/normalize.cu
    kernels/topk_bitonic.cu
    src/index.cu
)

target_include_directories(curag PUBLIC include)

set_target_properties(curag PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(curag_demo
    src/main.cpp
)

target_link_libraries(curag_demo PRIVATE curag)

add_executable(test_cosine_similarity
    tests/test_cosine_similarity.cpp
)

target_link_libraries(test_cosine_similarity PRIVATE curag)

add_test(
    NAME cosine_similarity
    COMMAND test_cosine_similarity
)

add_executable(test_normalize
    tests/test_normalize.cpp
)

target_link_libraries(test_normalize PRIVATE curag)

add_test(
    NAME normalize
    COMMAND test_normalize
)

add_executable(test_search_pipeline
    tests/test_search_pipeline.cpp
)

target_link_libraries(test_search_pipeline PRIVATE curag)

add_test(
    NAME search_pipeline
    COMMAND test_search_pipeline
)

add_executable(test_topk
    tests/test_topk.cpp
)

target_link_libraries(test_topk PRIVATE curag)

add_test(
    NAME topk
    COMMAND test_topk
)

add_executable(benchmark_cosine_similarity
    benchmarks/benchmark_cosine_similarity.cpp
)

target_link_libraries(benchmark_cosine_similarity PRIVATE curag)

add_executable(benchmark_topk
    benchmarks/benchmark_topk.cpp
)

target_link_libraries(benchmark_topk PRIVATE curag)

add_executable(test_search_topk_pipeline
    tests/test_search_topk_pipeline.cpp
)

target_link_libraries(test_search_topk_pipeline PRIVATE curag)

add_test(
    NAME search_topk_pipeline
    COMMAND test_search_topk_pipeline
)

add_executable(benchmark_search_pipeline
    benchmarks/benchmark_search_pipeline.cpp
)

target_link_libraries(benchmark_search_pipeline PRIVATE curag)

add_executable(test_index
    tests/test_index.cpp
)

target_link_libraries(test_index PRIVATE curag)

add_test(
    NAME index
    COMMAND test_index
)
add_executable(test_device_buffer
    tests/test_device_buffer.cpp
)

target_link_libraries(test_device_buffer PRIVATE curag)

add_test(
    NAME device_buffer
    COMMAND test_device_buffer
)

add_executable(benchmark_index_search
    benchmarks/benchmark_index_search.cpp
)

target_link_libraries(benchmark_index_search PRIVATE curag)

add_executable(test_index_save_load
    tests/test_index_save_load.cpp
)

target_link_libraries(test_index_save_load PRIVATE curag)

add_test(
    NAME index_save_load
    COMMAND test_index_save_load
)

add_executable(test_index_batch
    tests/test_index_batch.cpp
)

target_link_libraries(test_index_batch PRIVATE curag)

add_test(
    NAME index_batch
    COMMAND test_index_batch
)