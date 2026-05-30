#include "curag/cosine_similarity.hpp"
#include "curag/cuda_utils.hpp"
#include "curag/normalize.hpp"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace
{

constexpr int kDim = 768;
constexpr int kWarmupRuns = 5;
constexpr int kMeasuredRuns = 20;

void fill_random(std::vector<float> &values)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &value : values)
    {
        value = dist(rng);
    }
}

void run_benchmark(int num_vectors)
{
    std::vector<float> query(kDim);
    std::vector<float> corpus(static_cast<size_t>(num_vectors) * kDim);

    fill_random(query);
    fill_random(corpus);

    float *d_query = nullptr;
    float *d_corpus = nullptr;
    float *d_scores = nullptr;

    size_t query_bytes = query.size() * sizeof(float);
    size_t corpus_bytes = corpus.size() * sizeof(float);
    size_t scores_bytes = static_cast<size_t>(num_vectors) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_query, query_bytes));
    CUDA_CHECK(cudaMalloc(&d_corpus, corpus_bytes));
    CUDA_CHECK(cudaMalloc(&d_scores, scores_bytes));

    CUDA_CHECK(cudaMemcpy(
        d_query, query.data(), query_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_corpus, corpus.data(), corpus_bytes, cudaMemcpyHostToDevice));

    curag::l2_normalize(d_query, 1, kDim);
    curag::l2_normalize(d_corpus, num_vectors, kDim);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < kWarmupRuns; ++i)
    {
        curag::cosine_similarity(
            d_query, d_corpus, d_scores, num_vectors, kDim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kMeasuredRuns; ++i)
    {
        curag::cosine_similarity(
            d_query, d_corpus, d_scores, num_vectors, kDim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float average_ms = elapsed_ms / kMeasuredRuns;
    double vectors_per_second = num_vectors / (average_ms / 1000.0);

    std::cout << std::setw(10) << num_vectors
              << std::setw(10) << kDim
              << std::setw(16) << std::fixed << std::setprecision(3)
              << average_ms
              << std::setw(20) << std::fixed << std::setprecision(0)
              << vectors_per_second << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_corpus));
    CUDA_CHECK(cudaFree(d_scores));
}

} // namespace

int main()
{
    std::cout << "Cosine similarity kernel baseline\n";
    std::cout << std::setw(10) << "vectors"
              << std::setw(10) << "dim"
              << std::setw(16) << "kernel_ms"
              << std::setw(20) << "vectors_per_sec" << '\n';

    run_benchmark(10'000);
    run_benchmark(100'000);
    return 0;
}
