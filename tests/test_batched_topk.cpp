#include "curag/batched_topk.hpp"
#include "curag/cuda_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

void expect_near(float actual, float expected, float eps = 1e-5f)
{
    assert(std::fabs(actual - expected) <= eps);
}

std::vector<std::pair<float, int>> cpu_topk_for_row(
    const std::vector<float> &scores,
    int row,
    int num_vectors,
    int k)
{
    std::vector<std::pair<float, int>> pairs;

    int base = row * num_vectors;

    for (int i = 0; i < num_vectors; ++i)
    {
        pairs.push_back({scores[base + i], i});
    }

    std::sort(
        pairs.begin(),
        pairs.end(),
        [](const auto &a, const auto &b)
        {
            if (a.first != b.first)
            {
                return a.first > b.first;
            }
            return a.second < b.second;
        });

    pairs.resize(k);
    return pairs;
}

void run_batched_topk_test(
    const std::vector<float> &scores,
    int num_queries,
    int num_vectors,
    int k)
{
    std::vector<float> expected_values(static_cast<std::size_t>(num_queries) * k);
    std::vector<int> expected_indices(static_cast<std::size_t>(num_queries) * k);

    for (int q = 0; q < num_queries; ++q)
    {
        auto expected = cpu_topk_for_row(scores, q, num_vectors, k);

        for (int j = 0; j < k; ++j)
        {
            std::size_t out = static_cast<std::size_t>(q) * k + j;
            expected_values[out] = expected[j].first;
            expected_indices[out] = expected[j].second;
        }
    }

    float *d_scores = nullptr;
    float *d_values = nullptr;
    int *d_indices = nullptr;

    std::size_t scores_count =
        static_cast<std::size_t>(num_queries) * num_vectors;

    std::size_t output_count =
        static_cast<std::size_t>(num_queries) * k;

    CURAG_CUDA_CHECK(cudaMalloc(&d_scores, scores_count * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_values, output_count * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_indices, output_count * sizeof(int)));

    CURAG_CUDA_CHECK(cudaMemcpy(
        d_scores,
        scores.data(),
        scores_count * sizeof(float),
        cudaMemcpyHostToDevice));

    curag::batched_topk(
        d_scores,
        d_values,
        d_indices,
        num_queries,
        num_vectors,
        k);

    CURAG_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual_values(output_count);
    std::vector<int> actual_indices(output_count);

    CURAG_CUDA_CHECK(cudaMemcpy(
        actual_values.data(),
        d_values,
        output_count * sizeof(float),
        cudaMemcpyDeviceToHost));

    CURAG_CUDA_CHECK(cudaMemcpy(
        actual_indices.data(),
        d_indices,
        output_count * sizeof(int),
        cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < output_count; ++i)
    {
        expect_near(actual_values[i], expected_values[i]);
        assert(actual_indices[i] == expected_indices[i]);
    }

    CURAG_CUDA_CHECK(cudaFree(d_scores));
    CURAG_CUDA_CHECK(cudaFree(d_values));
    CURAG_CUDA_CHECK(cudaFree(d_indices));
}

void test_small_known_case()
{
    constexpr int num_queries = 2;
    constexpr int num_vectors = 5;
    constexpr int k = 2;

    std::vector<float> scores = {
        0.1f, 0.9f, 0.3f, 0.7f, -0.1f,
        0.8f, 0.2f, 0.5f, 0.4f, 0.9f};

    run_batched_topk_test(scores, num_queries, num_vectors, k);
}

void test_random_case()
{
    constexpr int num_queries = 4;
    constexpr int num_vectors = 1000;
    constexpr int k = 10;

    std::vector<float> scores(
        static_cast<std::size_t>(num_queries) * num_vectors);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &score : scores)
    {
        score = dist(rng);
    }

    run_batched_topk_test(scores, num_queries, num_vectors, k);
}

void test_k100()
{
    constexpr int num_queries = 3;
    constexpr int num_vectors = 2000;
    constexpr int k = 100;

    std::vector<float> scores(
        static_cast<std::size_t>(num_queries) * num_vectors);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &score : scores)
    {
        score = dist(rng);
    }

    run_batched_topk_test(scores, num_queries, num_vectors, k);
}

void test_invalid_inputs()
{
    float *d_scores = nullptr;
    float *d_values = nullptr;
    int *d_indices = nullptr;

    CURAG_CUDA_CHECK(cudaMalloc(&d_scores, sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_values, sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_indices, sizeof(int)));

    bool caught_queries = false;
    try
    {
        curag::batched_topk(d_scores, d_values, d_indices, 0, 10, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_queries = true;
    }
    assert(caught_queries);

    bool caught_vectors = false;
    try
    {
        curag::batched_topk(d_scores, d_values, d_indices, 1, 0, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_vectors = true;
    }
    assert(caught_vectors);

    bool caught_k = false;
    try
    {
        curag::batched_topk(d_scores, d_values, d_indices, 1, 10, 0);
    }
    catch (const std::runtime_error &)
    {
        caught_k = true;
    }
    assert(caught_k);

    bool caught_k_too_large = false;
    try
    {
        curag::batched_topk(d_scores, d_values, d_indices, 1, 10, 11);
    }
    catch (const std::runtime_error &)
    {
        caught_k_too_large = true;
    }
    assert(caught_k_too_large);

    CURAG_CUDA_CHECK(cudaFree(d_scores));
    CURAG_CUDA_CHECK(cudaFree(d_values));
    CURAG_CUDA_CHECK(cudaFree(d_indices));
}

int main()
{
    test_small_known_case();
    test_random_case();
    test_k100();
    test_invalid_inputs();

    std::cout << "All batched top-K tests passed.\n";
    return 0;
}