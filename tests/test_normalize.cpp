#include "curag/cuda_utils.hpp"
#include "curag/normalize.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void expect_near(float actual, float expected, float tolerance = 1e-5f)
{
    assert(std::fabs(actual - expected) <= tolerance);
}

void run_normalize_test(
    const std::vector<float> &input,
    const std::vector<float> &expected,
    int num_vectors,
    int dim)
{
    assert(input.size() == expected.size());

    float *d_vectors = nullptr;
    size_t bytes = input.size() * sizeof(float);
    std::vector<float> actual(input.size());

    CURAG_CUDA_CHECK(cudaMalloc(&d_vectors, bytes));
    CURAG_CUDA_CHECK(cudaMemcpy(
        d_vectors, input.data(), bytes, cudaMemcpyHostToDevice));

    curag::l2_normalize(d_vectors, num_vectors, dim);

    CURAG_CUDA_CHECK(cudaDeviceSynchronize());
    CURAG_CUDA_CHECK(cudaMemcpy(
        actual.data(), d_vectors, bytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < actual.size(); ++i)
    {
        expect_near(actual[i], expected[i]);
    }

    CURAG_CUDA_CHECK(cudaFree(d_vectors));
}

void test_known_vectors()
{
    std::vector<float> input = {
        3.0f, 4.0f,
        0.0f, 5.0f,
        -3.0f, -4.0f};

    std::vector<float> expected = {
        0.6f, 0.8f,
        0.0f, 1.0f,
        -0.6f, -0.8f};

    run_normalize_test(input, expected, 3, 2);
}

void test_zero_vector_stays_zero()
{
    run_normalize_test(
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        1,
        3);
}

void test_dimension_1024()
{
    constexpr int dim = 1024;

    std::vector<float> input(dim, 1.0f);
    std::vector<float> expected(dim, 1.0f / std::sqrt(1024.0f));

    run_normalize_test(input, expected, 1, dim);
}

void test_invalid_inputs()
{
    bool caught_null = false;
    try
    {
        curag::l2_normalize(nullptr, 1, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_null = true;
    }
    assert(caught_null);

    float *d_vector = nullptr;
    CURAG_CUDA_CHECK(cudaMalloc(&d_vector, sizeof(float)));

    bool caught_invalid_count = false;
    try
    {
        curag::l2_normalize(d_vector, 0, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_invalid_count = true;
    }
    assert(caught_invalid_count);

    bool caught_invalid_dim = false;
    try
    {
        curag::l2_normalize(d_vector, 1, 0);
    }
    catch (const std::runtime_error &)
    {
        caught_invalid_dim = true;
    }
    assert(caught_invalid_dim);

    bool caught_large_dim = false;
    try
    {
        curag::l2_normalize(d_vector, 1, 1025);
    }
    catch (const std::runtime_error &)
    {
        caught_large_dim = true;
    }
    assert(caught_large_dim);

    CURAG_CUDA_CHECK(cudaFree(d_vector));
}

int main()
{
    test_known_vectors();
    test_zero_vector_stays_zero();
    test_dimension_1024();
    test_invalid_inputs();

    std::cout << "All normalization tests passed.\n";
    return 0;
}