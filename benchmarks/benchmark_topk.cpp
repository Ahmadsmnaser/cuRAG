#include "curag/topk.hpp"
#include "curag/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <vector>

void run_benchmark(int num_scores, int k)
{
    std::vector<float> scores(num_scores);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &score : scores)
    {
        score = dist(rng);
    }

    float *d_scores = nullptr;
    float *d_topk_values = nullptr;
    int *d_topk_indices = nullptr;

    CURAG_CUDA_CHECK(cudaMalloc(&d_scores, num_scores * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_topk_values, k * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_topk_indices, k * sizeof(int)));

    CURAG_CUDA_CHECK(cudaMemcpy(
        d_scores,
        scores.data(),
        num_scores * sizeof(float),
        cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 5; ++i)
    {
        curag::topk(d_scores, d_topk_values, d_topk_indices, num_scores, k);
    }

    CURAG_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CURAG_CUDA_CHECK(cudaEventCreate(&start));
    CURAG_CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int iterations = 20;

    CURAG_CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i)
    {
        curag::topk(d_scores, d_topk_values, d_topk_indices, num_scores, k);
    }

    CURAG_CUDA_CHECK(cudaEventRecord(stop));
    CURAG_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CURAG_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float avg_ms = elapsed_ms / iterations;

    std::cout << "N=" << num_scores
              << " K=" << k
              << " topk_time=" << avg_ms << " ms"
              << std::endl;

    CURAG_CUDA_CHECK(cudaEventDestroy(start));
    CURAG_CUDA_CHECK(cudaEventDestroy(stop));

    CURAG_CUDA_CHECK(cudaFree(d_scores));
    CURAG_CUDA_CHECK(cudaFree(d_topk_values));
    CURAG_CUDA_CHECK(cudaFree(d_topk_indices));
}

int main()
{
    run_benchmark(10'000, 10);
    run_benchmark(100'000, 10);

    return 0;
}