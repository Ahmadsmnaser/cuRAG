#include "curag/cosine_similarity.hpp"
#include "curag/cuda_utils.hpp"
#include "curag/normalize.hpp"
#include "curag/topk.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <vector>

struct TimingResult
{
    float query_normalize_ms;
    float cosine_ms;
    float topk_ms;
    float total_ms;
};

void fill_random(std::vector<float> &values)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &value : values)
    {
        value = dist(rng);
    }
}

float time_event_pair(cudaEvent_t start, cudaEvent_t stop)
{
    float elapsed_ms = 0.0f;
    CURAG_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return elapsed_ms;
}

TimingResult run_once(
    float *d_query,
    float *d_corpus,
    float *d_scores,
    float *d_topk_values,
    int *d_topk_indices,
    int num_vectors,
    int dim,
    int k)
{
    cudaEvent_t total_start, total_stop;
    cudaEvent_t norm_start, norm_stop;
    cudaEvent_t cosine_start, cosine_stop;
    cudaEvent_t topk_start, topk_stop;

    CURAG_CUDA_CHECK(cudaEventCreate(&total_start));
    CURAG_CUDA_CHECK(cudaEventCreate(&total_stop));
    CURAG_CUDA_CHECK(cudaEventCreate(&norm_start));
    CURAG_CUDA_CHECK(cudaEventCreate(&norm_stop));
    CURAG_CUDA_CHECK(cudaEventCreate(&cosine_start));
    CURAG_CUDA_CHECK(cudaEventCreate(&cosine_stop));
    CURAG_CUDA_CHECK(cudaEventCreate(&topk_start));
    CURAG_CUDA_CHECK(cudaEventCreate(&topk_stop));

    CURAG_CUDA_CHECK(cudaEventRecord(total_start));

    CURAG_CUDA_CHECK(cudaEventRecord(norm_start));
    curag::l2_normalize(d_query, 1, dim);
    CURAG_CUDA_CHECK(cudaEventRecord(norm_stop));

    CURAG_CUDA_CHECK(cudaEventRecord(cosine_start));
    curag::cosine_similarity(d_query, d_corpus, d_scores, num_vectors, dim);
    CURAG_CUDA_CHECK(cudaEventRecord(cosine_stop));

    CURAG_CUDA_CHECK(cudaEventRecord(topk_start));
    curag::topk(d_scores, d_topk_values, d_topk_indices, num_vectors, k);
    CURAG_CUDA_CHECK(cudaEventRecord(topk_stop));

    CURAG_CUDA_CHECK(cudaEventRecord(total_stop));
    CURAG_CUDA_CHECK(cudaEventSynchronize(total_stop));

    TimingResult result{};
    result.query_normalize_ms = time_event_pair(norm_start, norm_stop);
    result.cosine_ms = time_event_pair(cosine_start, cosine_stop);
    result.topk_ms = time_event_pair(topk_start, topk_stop);
    result.total_ms = time_event_pair(total_start, total_stop);

    CURAG_CUDA_CHECK(cudaEventDestroy(total_start));
    CURAG_CUDA_CHECK(cudaEventDestroy(total_stop));
    CURAG_CUDA_CHECK(cudaEventDestroy(norm_start));
    CURAG_CUDA_CHECK(cudaEventDestroy(norm_stop));
    CURAG_CUDA_CHECK(cudaEventDestroy(cosine_start));
    CURAG_CUDA_CHECK(cudaEventDestroy(cosine_stop));
    CURAG_CUDA_CHECK(cudaEventDestroy(topk_start));
    CURAG_CUDA_CHECK(cudaEventDestroy(topk_stop));

    return result;
}

void run_benchmark(int num_vectors, int dim, int k)
{
    std::vector<float> query(dim);
    std::vector<float> corpus(static_cast<size_t>(num_vectors) * dim);

    fill_random(query);
    fill_random(corpus);

    float *d_query = nullptr;
    float *d_corpus = nullptr;
    float *d_scores = nullptr;
    float *d_topk_values = nullptr;
    int *d_topk_indices = nullptr;

    size_t query_bytes = dim * sizeof(float);
    size_t corpus_bytes = static_cast<size_t>(num_vectors) * dim * sizeof(float);
    size_t scores_bytes = num_vectors * sizeof(float);
    size_t topk_values_bytes = k * sizeof(float);
    size_t topk_indices_bytes = k * sizeof(int);

    CURAG_CUDA_CHECK(cudaMalloc(&d_query, query_bytes));
    CURAG_CUDA_CHECK(cudaMalloc(&d_corpus, corpus_bytes));
    CURAG_CUDA_CHECK(cudaMalloc(&d_scores, scores_bytes));
    CURAG_CUDA_CHECK(cudaMalloc(&d_topk_values, topk_values_bytes));
    CURAG_CUDA_CHECK(cudaMalloc(&d_topk_indices, topk_indices_bytes));

    CURAG_CUDA_CHECK(cudaMemcpy(d_query, query.data(), query_bytes, cudaMemcpyHostToDevice));
    CURAG_CUDA_CHECK(cudaMemcpy(d_corpus, corpus.data(), corpus_bytes, cudaMemcpyHostToDevice));

    // Normalize corpus once, like index build time.
    curag::l2_normalize(d_corpus, num_vectors, dim);
    CURAG_CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int warmup_iterations = 5;
    for (int i = 0; i < warmup_iterations; ++i)
    {
        CURAG_CUDA_CHECK(cudaMemcpy(d_query, query.data(), query_bytes, cudaMemcpyHostToDevice));
        curag::l2_normalize(d_query, 1, dim);
        curag::cosine_similarity(d_query, d_corpus, d_scores, num_vectors, dim);
        curag::topk(d_scores, d_topk_values, d_topk_indices, num_vectors, k);
    }

    CURAG_CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int iterations = 20;

    TimingResult total{};
    for (int i = 0; i < iterations; ++i)
    {
        // Reset query each iteration because l2_normalize modifies it in place.
        CURAG_CUDA_CHECK(cudaMemcpy(d_query, query.data(), query_bytes, cudaMemcpyHostToDevice));

        TimingResult current = run_once(
            d_query,
            d_corpus,
            d_scores,
            d_topk_values,
            d_topk_indices,
            num_vectors,
            dim,
            k);

        total.query_normalize_ms += current.query_normalize_ms;
        total.cosine_ms += current.cosine_ms;
        total.topk_ms += current.topk_ms;
        total.total_ms += current.total_ms;
    }

    std::cout << "N=" << num_vectors
              << " dim=" << dim
              << " K=" << k << "\n";

    std::cout << "  query_normalize: "
              << total.query_normalize_ms / iterations
              << " ms\n";

    std::cout << "  cosine_similarity: "
              << total.cosine_ms / iterations
              << " ms\n";

    std::cout << "  topk: "
              << total.topk_ms / iterations
              << " ms\n";

    std::cout << "  total_search: "
              << total.total_ms / iterations
              << " ms\n\n";

    CURAG_CUDA_CHECK(cudaFree(d_query));
    CURAG_CUDA_CHECK(cudaFree(d_corpus));
    CURAG_CUDA_CHECK(cudaFree(d_scores));
    CURAG_CUDA_CHECK(cudaFree(d_topk_values));
    CURAG_CUDA_CHECK(cudaFree(d_topk_indices));
}

int main()
{
    constexpr int dim = 768;
    constexpr int k = 10;

    run_benchmark(10'000, dim, k);
    run_benchmark(100'000, dim, k);

    return 0;
}