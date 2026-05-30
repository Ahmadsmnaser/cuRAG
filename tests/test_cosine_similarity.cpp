#include "curag/cosine_similarity.hpp"
#include "curag/cuda_utils.hpp"

#include <cassert>
#include <cmath>

void expect_near(float actual, float expected, float tolerance = 1e-5f)
{
    assert(std::fabs(actual - expected) <= tolerance);
}

void run_test(const float *query, const float *corpus, float expected)
{
    constexpr int dim = 2;

    float *d_query = nullptr;
    float *d_corpus = nullptr;
    float *d_score = nullptr;
    float score = 0.0f;

    CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corpus, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_score, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_query, query, dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_corpus, corpus, dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    curag::cosine_similarity(d_query, d_corpus, d_score, 1, dim);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&score, d_score, sizeof(float),
                          cudaMemcpyDeviceToHost));

    expect_near(score, expected);

    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_corpus));
    CUDA_CHECK(cudaFree(d_score));
}

int main()
{
    float orthogonal_query[] = {1.0f, 0.0f};
    float orthogonal_corpus[] = {0.0f, 1.0f};
    run_test(orthogonal_query, orthogonal_corpus, 0.0f);

    float identical_query[] = {1.0f, 0.0f};
    float identical_corpus[] = {1.0f, 0.0f};
    run_test(identical_query, identical_corpus, 1.0f);

    float opposite_query[] = {1.0f, 0.0f};
    float opposite_corpus[] = {-1.0f, 0.0f};
    run_test(opposite_query, opposite_corpus, -1.0f);
}