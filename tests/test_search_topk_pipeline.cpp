#include "curag/normalize.hpp"
#include "curag/cosine_similarity.hpp"
#include "curag/topk.hpp"
#include "curag/cuda_utils.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

void expect_near(float actual, float expected, float eps = 1e-5f)
{
    assert(std::fabs(actual - expected) <= eps);
}

void cpu_normalize(std::vector<float> &vectors, int num_vectors, int dim)
{
    for (int i = 0; i < num_vectors; ++i)
    {
        int base = i * dim;
        float sum = 0.0f;

        for (int j = 0; j < dim; ++j)
        {
            sum += vectors[base + j] * vectors[base + j];
        }

        float norm = std::sqrt(sum);
        if (norm == 0.0f)
            continue;

        for (int j = 0; j < dim; ++j)
        {
            vectors[base + j] /= norm;
        }
    }
}

int main()
{
    constexpr int dim = 2;
    constexpr int num_vectors = 5;
    constexpr int k = 2;

    std::vector<float> query = {
        1.0f, 0.0f};

    std::vector<float> corpus = {
        1.0f, 0.0f,  // score 1.0
        0.0f, 1.0f,  // score 0.0
        -1.0f, 0.0f, // score -1.0
        0.8f, 0.6f,  // score 0.8 after normalization
        0.6f, 0.8f   // score 0.6 after normalization
    };

    std::vector<float> cpu_query = query;
    std::vector<float> cpu_corpus = corpus;

    cpu_normalize(cpu_query, 1, dim);
    cpu_normalize(cpu_corpus, num_vectors, dim);

    std::vector<std::pair<float, int>> expected;

    for (int i = 0; i < num_vectors; ++i)
    {
        float score = 0.0f;

        for (int j = 0; j < dim; ++j)
        {
            score += cpu_query[j] * cpu_corpus[i * dim + j];
        }

        expected.push_back({score, i});
    }

    std::sort(expected.begin(), expected.end(),
              [](const auto &a, const auto &b)
              {
                  return a.first > b.first;
              });

    float *d_query = nullptr;
    float *d_corpus = nullptr;
    float *d_scores = nullptr;
    float *d_topk_values = nullptr;
    int *d_topk_indices = nullptr;

    CURAG_CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_corpus, num_vectors * dim * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_scores, num_vectors * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_topk_values, k * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_topk_indices, k * sizeof(int)));

    CURAG_CUDA_CHECK(cudaMemcpy(d_query, query.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
    CURAG_CUDA_CHECK(cudaMemcpy(d_corpus, corpus.data(), num_vectors * dim * sizeof(float), cudaMemcpyHostToDevice));

    curag::l2_normalize(d_query, 1, dim);
    curag::l2_normalize(d_corpus, num_vectors, dim);

    curag::cosine_similarity(d_query, d_corpus, d_scores, num_vectors, dim);

    curag::topk(d_scores, d_topk_values, d_topk_indices, num_vectors, k);

    CURAG_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual_values(k);
    std::vector<int> actual_indices(k);

    CURAG_CUDA_CHECK(cudaMemcpy(actual_values.data(), d_topk_values, k * sizeof(float), cudaMemcpyDeviceToHost));
    CURAG_CUDA_CHECK(cudaMemcpy(actual_indices.data(), d_topk_indices, k * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < k; ++i)
    {
        expect_near(actual_values[i], expected[i].first);
        assert(actual_indices[i] == expected[i].second);
    }

    CURAG_CUDA_CHECK(cudaFree(d_query));
    CURAG_CUDA_CHECK(cudaFree(d_corpus));
    CURAG_CUDA_CHECK(cudaFree(d_scores));
    CURAG_CUDA_CHECK(cudaFree(d_topk_values));
    CURAG_CUDA_CHECK(cudaFree(d_topk_indices));

    std::cout << "Search + Top-K pipeline test passed.\n";
    return 0;
}