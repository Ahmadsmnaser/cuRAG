#include "curag/cosine_similarity.hpp"
#include "curag/cuda_utils.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

void expect_near(float actual, float expected, float tolerance = 1e-4f)
{
    assert(std::fabs(actual - expected) <= tolerance);
}

float cpu_dot(const float *query, const float *corpus_vector, int dim)
{
    float result = 0.0f;

    for (int j = 0; j < dim; ++j)
    {
        result += query[j] * corpus_vector[j];
    }

    return result;
}

void cpu_l2_normalize(float *vectors, int num_vectors, int dim)
{
    for (int i = 0; i < num_vectors; ++i)
    {
        float sum = 0.0f;
        int base = i * dim;

        for (int j = 0; j < dim; ++j)
        {
            float value = vectors[base + j];
            sum += value * value;
        }

        float norm = std::sqrt(sum);

        // Keep zero vectors unchanged.
        if (norm == 0.0f)
        {
            continue;
        }

        for (int j = 0; j < dim; ++j)
        {
            vectors[base + j] /= norm;
        }
    }
}

void run_gpu_similarity_test(
    const std::vector<float> &query,
    const std::vector<float> &corpus,
    const std::vector<float> &expected,
    int num_vectors,
    int dim)
{
    assert(static_cast<int>(query.size()) == dim);
    assert(static_cast<int>(corpus.size()) == num_vectors * dim);
    assert(static_cast<int>(expected.size()) == num_vectors);

    float *d_query = nullptr;
    float *d_corpus = nullptr;
    float *d_scores = nullptr;

    std::vector<float> actual(num_vectors, 0.0f);

    size_t query_bytes = dim * sizeof(float);
    size_t corpus_bytes = static_cast<size_t>(num_vectors) * dim * sizeof(float);
    size_t scores_bytes = num_vectors * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_query, query_bytes));
    CUDA_CHECK(cudaMalloc(&d_corpus, corpus_bytes));
    CUDA_CHECK(cudaMalloc(&d_scores, scores_bytes));

    CUDA_CHECK(cudaMemcpy(
        d_query,
        query.data(),
        query_bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_corpus,
        corpus.data(),
        corpus_bytes,
        cudaMemcpyHostToDevice));

    curag::cosine_similarity(
        d_query,
        d_corpus,
        d_scores,
        num_vectors,
        dim);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        actual.data(),
        d_scores,
        scores_bytes,
        cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_vectors; ++i)
    {
        expect_near(actual[i], expected[i]);
    }

    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_corpus));
    CUDA_CHECK(cudaFree(d_scores));
}

void test_single_query_against_several_vectors()
{
    constexpr int dim = 2;
    constexpr int num_vectors = 3;

    std::vector<float> query = {1.0f, 0.0f};

    std::vector<float> corpus =
    {
        1.0f, 0.0f,
        0.0f, 1.0f,
        -1.0f, 0.0f
    };

    std::vector<float> expected = {
        1.0f,
        0.0f,
        -1.0f};

    run_gpu_similarity_test(query, corpus, expected, num_vectors, dim);
}

void test_more_than_256_vectors()
{
    constexpr int dim = 2;
    constexpr int num_vectors = 300;

    std::vector<float> query = {1.0f, 0.0f};

    std::vector<float> corpus(num_vectors * dim);
    std::vector<float> expected(num_vectors);

    for (int i = 0; i < num_vectors; ++i)
    {
        int base = i * dim;

        if (i % 3 == 0)
        {
            corpus[base + 0] = 1.0f;
            corpus[base + 1] = 0.0f;
            expected[i] = 1.0f;
        }
        else if (i % 3 == 1)
        {
            corpus[base + 0] = 0.0f;
            corpus[base + 1] = 1.0f;
            expected[i] = 0.0f;
        }
        else
        {
            corpus[base + 0] = -1.0f;
            corpus[base + 1] = 0.0f;
            expected[i] = -1.0f;
        }
    }

    // This launches one kernel over 300 vectors, so it exercises multiple blocks.
    run_gpu_similarity_test(query, corpus, expected, num_vectors, dim);
}

void test_dimension_1()
{
    constexpr int dim = 1;
    constexpr int num_vectors = 2;

    std::vector<float> query = {
        1.0f};

    std::vector<float> corpus = {
        1.0f,
        -1.0f};

    std::vector<float> expected = {
        1.0f,
        -1.0f};

    run_gpu_similarity_test(query, corpus, expected, num_vectors, dim);
}

void test_fixed_dimension_identical_vectors(int dim)
{
    constexpr int num_vectors = 4;

    std::vector<float> query(dim, 1.0f);
    std::vector<float> corpus(num_vectors * dim, 1.0f);

    cpu_l2_normalize(query.data(), 1, dim);
    cpu_l2_normalize(corpus.data(), num_vectors, dim);

    std::vector<float> expected(num_vectors, 1.0f);

    run_gpu_similarity_test(query, corpus, expected, num_vectors, dim);
}

void test_random_normalized_vectors()
{
    constexpr int dim = 768;
    constexpr int num_vectors = 512;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> query(dim);
    std::vector<float> corpus(num_vectors * dim);
    std::vector<float> expected(num_vectors);

    for (float &value : query)
    {
        value = dist(rng);
    }

    for (float &value : corpus)
    {
        value = dist(rng);
    }

    cpu_l2_normalize(query.data(), 1, dim);
    cpu_l2_normalize(corpus.data(), num_vectors, dim);

    for (int i = 0; i < num_vectors; ++i)
    {
        expected[i] = cpu_dot(
            query.data(),
            corpus.data() + i * dim,
            dim);
    }

    run_gpu_similarity_test(query, corpus, expected, num_vectors, dim);
}

void test_invalid_inputs()
{
    float *d_query = nullptr;
    float *d_corpus = nullptr;
    float *d_scores = nullptr;

    CUDA_CHECK(cudaMalloc(&d_query, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corpus, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, sizeof(float)));

    bool caught_null_pointer = false;
    try
    {
        curag::cosine_similarity(nullptr, d_corpus, d_scores, 1, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_null_pointer = true;
    }

    assert(caught_null_pointer);

    bool caught_invalid_dim = false;
    try
    {
        curag::cosine_similarity(d_query, d_corpus, d_scores, 1, 0);
    }
    catch (const std::runtime_error &)
    {
        caught_invalid_dim = true;
    }

    assert(caught_invalid_dim);

    bool caught_invalid_num_vectors = false;
    try
    {
        curag::cosine_similarity(d_query, d_corpus, d_scores, 0, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_invalid_num_vectors = true;
    }

    assert(caught_invalid_num_vectors);

    bool caught_too_large_dim = false;
    try
    {
        curag::cosine_similarity(d_query, d_corpus, d_scores, 1, 1025);
    }
    catch (const std::runtime_error &)
    {
        caught_too_large_dim = true;
    }

    assert(caught_too_large_dim);

    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_corpus));
    CUDA_CHECK(cudaFree(d_scores));
}

int main()
{
    test_single_query_against_several_vectors();
    test_more_than_256_vectors();

    test_dimension_1();
    test_fixed_dimension_identical_vectors(384);
    test_fixed_dimension_identical_vectors(768);
    test_fixed_dimension_identical_vectors(1024);

    test_invalid_inputs();
    test_random_normalized_vectors();

    std::cout << "All cosine similarity tests passed.\n";
    return 0;
}
