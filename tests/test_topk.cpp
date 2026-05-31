#include "curag/topk.hpp"
#include "curag/cuda_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <random>
#include <vector>

void expect_near(float actual, float expected, float tolerance = 1e-5f)
{
    assert(std::fabs(actual - expected) <= tolerance);
}

std::vector<std::pair<float, int>> cpu_topk(
    const std::vector<float> &scores,
    int k)
{
    std::vector<std::pair<float, int>> pairs;

    for (int i = 0; i < static_cast<int>(scores.size()); ++i)
    {
        pairs.push_back({scores[i], i});
    }

    std::sort(
        pairs.begin(),
        pairs.end(),
        [](const auto &a, const auto &b)
        {
            return a.first > b.first;
        });

    pairs.resize(k);
    return pairs;
}

void run_topk_test(const std::vector<float> &scores, int k)
{
    int num_scores = static_cast<int>(scores.size());

    std::vector<float> expected_values(k);
    std::vector<int> expected_indices(k);

    auto expected = cpu_topk(scores, k);

    for (int i = 0; i < k; ++i)
    {
        expected_values[i] = expected[i].first;
        expected_indices[i] = expected[i].second;
    }

    float *d_scores = nullptr;
    float *d_topk_values = nullptr;
    int *d_topk_indices = nullptr;

    std::vector<float> actual_values(k);
    std::vector<int> actual_indices(k);

    CURAG_CUDA_CHECK(cudaMalloc(&d_scores, num_scores * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_topk_values, k * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_topk_indices, k * sizeof(int)));

    CURAG_CUDA_CHECK(cudaMemcpy(
        d_scores,
        scores.data(),
        num_scores * sizeof(float),
        cudaMemcpyHostToDevice));

    curag::topk(
        d_scores,
        d_topk_values,
        d_topk_indices,
        num_scores,
        k);

    CURAG_CUDA_CHECK(cudaDeviceSynchronize());

    CURAG_CUDA_CHECK(cudaMemcpy(
        actual_values.data(),
        d_topk_values,
        k * sizeof(float),
        cudaMemcpyDeviceToHost));

    CURAG_CUDA_CHECK(cudaMemcpy(
        actual_indices.data(),
        d_topk_indices,
        k * sizeof(int),
        cudaMemcpyDeviceToHost));

    for (int i = 0; i < k; ++i)
    {
        expect_near(actual_values[i], expected_values[i]);
        assert(actual_indices[i] == expected_indices[i]);
    }

    CURAG_CUDA_CHECK(cudaFree(d_scores));
    CURAG_CUDA_CHECK(cudaFree(d_topk_values));
    CURAG_CUDA_CHECK(cudaFree(d_topk_indices));
}

void test_basic_topk()
{
    std::vector<float> scores = {
        0.1f, 0.9f, -0.2f, 0.7f, 0.3f};

    run_topk_test(scores, 3);
}

void test_k_1()
{
    std::vector<float> scores = {
        -1.0f, 2.0f, 0.5f, 4.0f, 3.0f};

    run_topk_test(scores, 1);
}

void test_k_equals_num_scores()
{
    std::vector<float> scores = {
        3.0f, 1.0f, 2.0f};

    run_topk_test(scores, 3);
}

void test_invalid_inputs()
{
    float *d_scores = nullptr;
    float *d_values = nullptr;
    int *d_indices = nullptr;

    CURAG_CUDA_CHECK(cudaMalloc(&d_scores, sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_values, sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_indices, sizeof(int)));

    bool caught_k_zero = false;
    try
    {
        curag::topk(d_scores, d_values, d_indices, 10, 0);
    }
    catch (const std::runtime_error &)
    {
        caught_k_zero = true;
    }
    assert(caught_k_zero);

    bool caught_num_scores_zero = false;
    try
    {
        curag::topk(d_scores, d_values, d_indices, 0, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_num_scores_zero = true;
    }
    assert(caught_num_scores_zero);

    bool caught_k_too_large = false;
    try
    {
        curag::topk(d_scores, d_values, d_indices, 10, 11);
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
void test_large_random_topk()
{
    constexpr int num_scores = 10000;
    constexpr int k = 10;

    std::vector<float> scores(num_scores);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &score : scores)
    {
        score = dist(rng);
    }

    run_topk_test(scores, k);
}
int main()
{
    test_basic_topk();
    test_k_1();
    test_k_equals_num_scores();
    test_invalid_inputs();
    test_large_random_topk();

    std::cout << "All top-K tests passed.\n";
    return 0;
}