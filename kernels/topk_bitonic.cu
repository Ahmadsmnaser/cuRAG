#include "curag/topk.hpp"
#include "curag/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <stdexcept>

namespace curag{

namespace{

    constexpr int MAX_TOPK = 1024;
    constexpr int BLOCK_SIZE = 256;

    __device__ void insert_candidate(
    float score,
    int index,float* values,
    int* indices,int k){
        if(score <= values[k-1]){
            return; // Not a top-k candidate
        }
        values[k-1] = score;
        indices[k-1] = index;

        // Bubble up the new score to maintain sorted order
        for(int j = k-1; j > 0; --j){
            if(values[j] > values[j-1]){
                // Swap values                
                float temp_val = values[j];
                int temp_idx = indices[j];
                values[j] = values[j-1];
                indices[j] = indices[j-1];
                values[j-1] = temp_val;
                indices[j-1] = temp_idx;
            } else {
                break; // Already in correct position
            }
        }
    }

    // Kernel 1 : 
    // Each block scans a chunk of scores and produces local top-K
    __global__ void local_topk_kernel(
        const float *d_scores,
        float *d_partial_values,
        int *d_partial_indices,
        int num_scores,
        int k){
        int block_id = blockIdx.x;
        int tid = threadIdx.x;

        extern __shared__ unsigned char shared_mem[];

        float *shared_values = reinterpret_cast<float *>(shared_mem);
        int *shared_indices = reinterpret_cast<int *>(shared_values + BLOCK_SIZE * k);

        float *my_values = shared_values + tid * k;
        int *my_indices = shared_indices + tid * k;

        // Initialize local top-k buffer
        for (int i = 0; i < k; ++i) {
            my_values[i] = -CUDART_INF_F;
            my_indices[i] = -1;
        }
        // Grid-stride loop over scores assigned to this block.
        for (int i = block_id * blockDim.x + tid; i < num_scores; i += blockDim.x * gridDim.x)
        {
            insert_candidate(d_scores[i], i, my_values, my_indices, k);
        }

        __syncthreads();
        
        // Thread 0 merges all thread-local top-k into block-local top-k
        if (tid == 0){
            float block_values[MAX_TOPK];
            int block_indices[MAX_TOPK];

            for (int i = 0; i < k; ++i) {
                block_values[i] = -CUDART_INF_F;
                block_indices[i] = -1;
            }

            for (int t = 0; t < BLOCK_SIZE; ++t)
            {
                float *values = shared_values + t * k;
                int *indices = shared_indices + t * k;

                for (int j = 0; j < k; ++j)
                {
                    if (indices[j] != -1)
                    {
                        insert_candidate(values[j], indices[j], block_values, block_indices, k);
                    }
                }
            }
            int output_base = block_id * k;
            for (int i = 0; i < k; ++i)
            {
                d_partial_values[output_base + i] = block_values[i];
                d_partial_indices[output_base + i] = block_indices[i];
            }
        }
    }

    // Kernel 2:
    // Merge partial top-k results from all blocks to get final top-k
    // Current version use only one block / one thread for final correctness.
    __global__ void merge_topk_kernel(
        const float *d_partial_values,
        const int *d_partial_indices,
        float *d_topk_values,
        int *d_topk_indices,
        int num_candidates,
        int k){
            if(threadIdx.x != 0 || blockIdx.x != 0){
                return; // Only one thread does the merging for simplicity
            }
            float final_values[MAX_TOPK];
            int final_indices[MAX_TOPK];

            for(int i = 0; i < k; ++i){
                final_values[i] = -CUDART_INF_F;
                final_indices[i] = -1;
            }

            for(int i = 0; i < num_candidates; ++i){
                float score = d_partial_values[i];
                int index = d_partial_indices[i];
                if(index != -1){
                    insert_candidate(score, index, final_values, final_indices, k);
                }
            }
            for (int i = 0; i < k; ++i){
                d_topk_values[i] = final_values[i];
                d_topk_indices[i] = final_indices[i];
            }

        }
    } // namespace

    void topk(
        const float *d_scores,
        float *d_topk_values,
        int *d_topk_indices,
        int num_scores,
        int k)
    {
        if (d_scores == nullptr || d_topk_values == nullptr || d_topk_indices == nullptr)
        {
            throw std::runtime_error("topk device pointers must not be null");
        }

        if (num_scores <= 0)
        {
            throw std::runtime_error("num_scores must be positive");
        }

        if (k <= 0)
        {
            throw std::runtime_error("k must be positive");
        }

        if (k > num_scores)
        {
            throw std::runtime_error("k must be <= num_scores");
        }

        if (k > MAX_TOPK)
        {
            throw std::runtime_error("k exceeds MAX_TOPK=1024");
        }

        // Keep number of blocks reasonable for now.
        int num_blocks = (num_scores + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Avoid launching insane number of blocks for this early version.
        if (num_blocks > 1024)
        {
            num_blocks = 1024;
        }

        float *d_partial_values = nullptr;
        int *d_partial_indices = nullptr;

        int num_candidates = num_blocks * k;

        CURAG_CUDA_CHECK(cudaMalloc(&d_partial_values, num_candidates * sizeof(float)));
        CURAG_CUDA_CHECK(cudaMalloc(&d_partial_indices, num_candidates * sizeof(int)));

        size_t shared_bytes =
            BLOCK_SIZE * k * sizeof(float) +
            BLOCK_SIZE * k * sizeof(int);

        local_topk_kernel<<<num_blocks, BLOCK_SIZE, shared_bytes>>>(
            d_scores,
            d_partial_values,
            d_partial_indices,
            num_scores,
            k);

        CURAG_CUDA_CHECK(cudaGetLastError());

        merge_topk_kernel<<<1, 1>>>(
            d_partial_values,
            d_partial_indices,
            d_topk_values,
            d_topk_indices,
            num_candidates,
            k);

        CURAG_CUDA_CHECK(cudaGetLastError());

        CURAG_CUDA_CHECK(cudaFree(d_partial_values));
        CURAG_CUDA_CHECK(cudaFree(d_partial_indices));
    }

} // namespace curag