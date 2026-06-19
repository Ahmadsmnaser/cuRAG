#include "curag/batched_cosine_similarity.hpp"
#include "curag/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace curag
{

    namespace
    {

        constexpr int BLOCK_SIZE = 256;
        constexpr int MAX_QUERY_DIM = 1024;

        // Grid layout:
        // blockIdx.x -> query id
        // blockIdx.y + threadIdx.x -> corpus vector id
        //
        // Each thread computes one dot product:
        // queries[query_id] · corpus[vector_id]
        __global__ void batched_cosine_similarity_kernel(
            const float *d_queries,
            const float *d_corpus,
            float *d_scores,
            int num_queries,
            int num_vectors,
            int dim)
        {
            __shared__ float shared_query[MAX_QUERY_DIM];

            int query_id = blockIdx.x;
            int local_thread = threadIdx.x;

            for (int j = local_thread; j < dim; j += blockDim.x)
            {
                shared_query[j] = d_queries[query_id * dim + j];
            }

            __syncthreads();

            int vector_id = blockIdx.y * blockDim.x + threadIdx.x;

            if (query_id >= num_queries || vector_id >= num_vectors)
            {
                return;
            }

            float dot = 0.0f;
            int corpus_base = vector_id * dim;

            for (int j = 0; j < dim; ++j)
            {
                dot += shared_query[j] * d_corpus[corpus_base + j];
            }

            d_scores[query_id * num_vectors + vector_id] = dot;
        }

    } // namespace

    void batched_cosine_similarity(
        const float *d_queries,
        const float *d_corpus,
        float *d_scores,
        int num_queries,
        int num_vectors,
        int dim)
    {
        if (d_queries == nullptr || d_corpus == nullptr || d_scores == nullptr)
        {
            throw std::runtime_error("batched_cosine_similarity device pointers must not be null");
        }

        if (num_queries <= 0)
        {
            throw std::runtime_error("num_queries must be positive");
        }

        if (num_vectors <= 0)
        {
            throw std::runtime_error("num_vectors must be positive");
        }

        if (dim <= 0)
        {
            throw std::runtime_error("dim must be positive");
        }

        if (dim > MAX_QUERY_DIM)
        {
            throw std::runtime_error("dim exceeds maximum supported dimension 1024");
        }

        dim3 block(BLOCK_SIZE);
        dim3 grid(
            num_queries,
            (num_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE);

        batched_cosine_similarity_kernel<<<grid, block>>>(
            d_queries,
            d_corpus,
            d_scores,
            num_queries,
            num_vectors,
            dim);

        CURAG_CUDA_CHECK(cudaGetLastError());
    }

} // namespace curag