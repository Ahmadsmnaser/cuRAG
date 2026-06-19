#include "curag/batched_cosine_similarity.hpp"
#include "curag/cuda_utils.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

void expect_near(float actual, float expected, float eps = 1e-4f)
{
    assert(std::fabs(actual - expected) <= eps);
}

void cpu_l2_normalize(float *vectors, int num_vectors, int dim)
{
    for (int i = 0; i < num_vectors; ++i)
    {
        int base = i * dim;
        float sum = 0.0f;

        for (int j = 0; j < dim; ++j)
        {
            float value = vectors[base + j];
            sum += value * value;
        }

        float norm = std::sqrt(sum);
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

float cpu_dot(
    const float *query,
    const float *corpus_vector,
    int dim)
{
    float result = 0.0f;

    for (int j = 0; j < dim; ++j)
    {
        result += query[j] * corpus_vector[j];
    }

    return result;
}

void run_batched_test(
    std::vector<float> queries,
    std::vector<float> corpus,
    int num_queries,
    int num_vectors,
    int dim)
{
    cpu_l2_normalize(queries.data(), num_queries, dim);
    cpu_l2_normalize(corpus.data(), num_vectors, dim);

    std::vector<float> expected(
        static_cast<std::size_t>(num_queries) * num_vectors);

    for (int q = 0; q < num_queries; ++q)
    {
        for (int i = 0; i < num_vectors; ++i)
        {
            expected[static_cast<std::size_t>(q) * num_vectors + i] =
                cpu_dot(
                    queries.data() + static_cast<std::size_t>(q) * dim,
                    corpus.data() + static_cast<std::size_t>(i) * dim,
                    dim);
        }
    }

    float *d_queries = nullptr;
    float *d_corpus = nullptr;
    float *d_scores = nullptr;

    std::size_t query_count =
        static_cast<std::size_t>(num_queries) * dim;
    std::size_t corpus_count =
        static_cast<std::size_t>(num_vectors) * dim;
    std::size_t score_count =
        static_cast<std::size_t>(num_queries) * num_vectors;

    CURAG_CUDA_CHECK(cudaMalloc(&d_queries, query_count * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_corpus, corpus_count * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_scores, score_count * sizeof(float)));

    CURAG_CUDA_CHECK(cudaMemcpy(
        d_queries,
        queries.data(),
        query_count * sizeof(float),
        cudaMemcpyHostToDevice));

    CURAG_CUDA_CHECK(cudaMemcpy(
        d_corpus,
        corpus.data(),
        corpus_count * sizeof(float),
        cudaMemcpyHostToDevice));

    curag::batched_cosine_similarity(
        d_queries,
        d_corpus,
        d_scores,
        num_queries,
        num_vectors,
        dim);

    CURAG_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(score_count);

    CURAG_CUDA_CHECK(cudaMemcpy(
        actual.data(),
        d_scores,
        score_count * sizeof(float),
        cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < score_count; ++i)
    {
        expect_near(actual[i], expected[i]);
    }

    CURAG_CUDA_CHECK(cudaFree(d_queries));
    CURAG_CUDA_CHECK(cudaFree(d_corpus));
    CURAG_CUDA_CHECK(cudaFree(d_scores));
}

void test_small_known_case()
{
    constexpr int num_queries = 2;
    constexpr int num_vectors = 3;
    constexpr int dim = 2;

    std::vector<float> queries = {
        1.0f, 0.0f,
        0.0f, 1.0f};

    std::vector<float> corpus = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        -1.0f, 0.0f};

    run_batched_test(
        queries,
        corpus,
        num_queries,
        num_vectors,
        dim);
}

void test_random_384()
{
    constexpr int num_queries = 4;
    constexpr int num_vectors = 512;
    constexpr int dim = 384;

    std::vector<float> queries(static_cast<std::size_t>(num_queries) * dim);
    std::vector<float> corpus(static_cast<std::size_t>(num_vectors) * dim);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &value : queries)
    {
        value = dist(rng);
    }

    for (float &value : corpus)
    {
        value = dist(rng);
    }

    run_batched_test(
        queries,
        corpus,
        num_queries,
        num_vectors,
        dim);
}

void test_random_768()
{
    constexpr int num_queries = 3;
    constexpr int num_vectors = 300;
    constexpr int dim = 768;

    std::vector<float> queries(static_cast<std::size_t>(num_queries) * dim);
    std::vector<float> corpus(static_cast<std::size_t>(num_vectors) * dim);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &value : queries)
    {
        value = dist(rng);
    }

    for (float &value : corpus)
    {
        value = dist(rng);
    }

    run_batched_test(
        queries,
        corpus,
        num_queries,
        num_vectors,
        dim);
}

void test_invalid_inputs()
{
    float *d_queries = nullptr;
    float *d_corpus = nullptr;
    float *d_scores = nullptr;

    CURAG_CUDA_CHECK(cudaMalloc(&d_queries, sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_corpus, sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_scores, sizeof(float)));

    bool caught_queries = false;
    try
    {
        curag::batched_cosine_similarity(nullptr, d_corpus, d_scores, 1, 1, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_queries = true;
    }
    assert(caught_queries);

    bool caught_num_queries = false;
    try
    {
        curag::batched_cosine_similarity(d_queries, d_corpus, d_scores, 0, 1, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_num_queries = true;
    }
    assert(caught_num_queries);

    bool caught_num_vectors = false;
    try
    {
        curag::batched_cosine_similarity(d_queries, d_corpus, d_scores, 1, 0, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_num_vectors = true;
    }
    assert(caught_num_vectors);

    bool caught_dim = false;
    try
    {
        curag::batched_cosine_similarity(d_queries, d_corpus, d_scores, 1, 1, 0);
    }
    catch (const std::runtime_error &)
    {
        caught_dim = true;
    }
    assert(caught_dim);

    CURAG_CUDA_CHECK(cudaFree(d_queries));
    CURAG_CUDA_CHECK(cudaFree(d_corpus));
    CURAG_CUDA_CHECK(cudaFree(d_scores));
}

int main()
{
    test_small_known_case();
    test_random_384();
    test_random_768();
    test_invalid_inputs();

    std::cout << "All batched cosine similarity tests passed.\n";
    return 0;
}