#include "curag/cosine_similarity.hpp"
#include "curag/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main()
{
    const int num_vectors = 3;
    const int dim = 2;

    std::vector<float> query = {1.0f, 0.0f};
    std::vector<float> corpus = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        -1.0f, 0.0f};
    std::vector<float> scores(num_vectors);

    float *d_query = nullptr;
    float *d_corpus = nullptr;
    float *d_scores = nullptr;

    CURAG_CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_corpus, corpus.size() * sizeof(float)));
    CURAG_CUDA_CHECK(cudaMalloc(&d_scores, num_vectors * sizeof(float)));

    CURAG_CUDA_CHECK(cudaMemcpy(d_query, query.data(), dim * sizeof(float),
                                cudaMemcpyHostToDevice));
    CURAG_CUDA_CHECK(cudaMemcpy(d_corpus, corpus.data(), corpus.size() * sizeof(float),
                                cudaMemcpyHostToDevice));

    curag::cosine_similarity(d_query, d_corpus, d_scores, num_vectors, dim);

    CURAG_CUDA_CHECK(cudaDeviceSynchronize());
    CURAG_CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, num_vectors * sizeof(float),
                                cudaMemcpyDeviceToHost));

    for (float score : scores)
    {
        std::cout << score << '\n';
    }

    CURAG_CUDA_CHECK(cudaFree(d_query));
    CURAG_CUDA_CHECK(cudaFree(d_corpus));
    CURAG_CUDA_CHECK(cudaFree(d_scores));
}