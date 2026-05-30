// cosine_similarity.cu
#include <cuda_runtime.h>
#include "curag/cosine_similarity.hpp"
#include "curag/cuda_utils.hpp"

constexpr int MAX_QUERY_DIM = 1024;
namespace curag
{

    // each Thread computes the cosine similarity between the query vector and his assigned corpus vector
    __global__ void cosine_similarity_kernel(const float *d_query, const float *d_corpus,
                                             float *d_scores, int num_vectors, int dim)
    {

        __shared__ float shared_query[MAX_QUERY_DIM]; // assuming dim <= MAX_QUERY_DIM, adjust if necessary
        for (int j = threadIdx.x; j < dim; j += blockDim.x)
        {
            shared_query[j] = d_query[j];
        }
        __syncthreads();

        int vector_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (vector_id < num_vectors)
        {
            float dot = 0.0f;
            int start_pos = vector_id * dim;

            for (int j = 0; j < dim; j++)
            {
                dot += shared_query[j] * d_corpus[start_pos + j];
            }
            d_scores[vector_id] = dot; // since the vectors are normalized, the dot product is the cosine similarity
        }
    }

    void cosine_similarity(const float *d_query, const float *d_corpus,
                           float *d_scores, int num_vectors, int dim)
    {
        if (d_query == nullptr || d_corpus == nullptr || d_scores == nullptr)
        {
            throw std::runtime_error("Device pointers must not be null");
        }
        if (dim <= 0)
        {
            throw std::runtime_error("Dimension must be greater than 0");
        }
        if (num_vectors <= 0)
        {
            throw std::runtime_error("Number of vectors must be greater than 0");
        }
        if (dim > MAX_QUERY_DIM)
        {
            throw std::runtime_error("Dimension exceeds maximum supported size of 1024");
        }
        constexpr int blockSize = 256;
        int num_blocks = (num_vectors + blockSize - 1) / blockSize;
        cosine_similarity_kernel<<<num_blocks, blockSize>>>(d_query, d_corpus, d_scores, num_vectors, dim);

        CURAG_CUDA_CHECK(cudaGetLastError());
    }
}
