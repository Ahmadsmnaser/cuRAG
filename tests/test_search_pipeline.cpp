#include "curag/cosine_similarity.hpp"
#include "curag/cuda_utils.hpp"
#include "curag/normalize.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void expect_near(float actual, float expected, float tolerance = 1e-5f)
{
    assert(std::fabs(actual - expected) <= tolerance);
}

int main()
{
    constexpr int dim = 2;
    constexpr int num_vectors = 4;

    std::vector<float> query = {
        3.0f, 4.0f};

    std::vector<float> corpus = {
        3.0f, 4.0f,
        4.0f, -3.0f,
        -3.0f, -4.0f,
        0.0f, 0.0f};

    std::vector<float> expected = {
        1.0f,
        0.0f,
        -1.0f,
        0.0f};

    std::vector<float> actual(num_vectors);

    float *d_query = nullptr;
    float *d_corpus = nullptr;
    float *d_scores = nullptr;

    size_t query_bytes = query.size() * sizeof(float);
    size_t corpus_bytes = corpus.size() * sizeof(float);
    size_t scores_bytes = actual.size() * sizeof(float);

    CURAG_CUDA_CHECK(cudaMalloc(&d_query, query_bytes));
    CURAG_CUDA_CHECK(cudaMalloc(&d_corpus, corpus_bytes));
    CURAG_CUDA_CHECK(cudaMalloc(&d_scores, scores_bytes));

    CURAG_CUDA_CHECK(cudaMemcpy(
        d_query, query.data(), query_bytes, cudaMemcpyHostToDevice));
    CURAG_CUDA_CHECK(cudaMemcpy(
        d_corpus, corpus.data(), corpus_bytes, cudaMemcpyHostToDevice));

    curag::l2_normalize(d_query, 1, dim);
    curag::l2_normalize(d_corpus, num_vectors, dim);
    curag::cosine_similarity(d_query, d_corpus, d_scores, num_vectors, dim);

    CURAG_CUDA_CHECK(cudaDeviceSynchronize());
    CURAG_CUDA_CHECK(cudaMemcpy(
        actual.data(), d_scores, scores_bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_vectors; ++i)
    {
        expect_near(actual[i], expected[i]);
    }

    CURAG_CUDA_CHECK(cudaFree(d_query));
    CURAG_CUDA_CHECK(cudaFree(d_corpus));
    CURAG_CUDA_CHECK(cudaFree(d_scores));

    std::cout << "Search pipeline test passed.\n";
    return 0;
}
