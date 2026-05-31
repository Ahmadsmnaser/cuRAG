#include "curag/topk.hpp"
#include "curag/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace curag
{

    namespace
    {

        constexpr int BLOCK_SIZE = 256;
        constexpr int MAX_TOPK = 1024;

        __device__ bool better_pair(
            float a_value,
            int a_index,
            float b_value,
            int b_index)
        {
            if (a_value > b_value)
            {
                return true;
            }

            if (a_value < b_value)
            {
                return false;
            }

            // Tie-breaker: smaller index wins for deterministic output.
            return a_index < b_index;
        }

        __device__ void swap_pair(
            float &a_value,
            int &a_index,
            float &b_value,
            int &b_index)
        {
            float temp_value = a_value;
            int temp_index = a_index;

            a_value = b_value;
            a_index = b_index;

            b_value = temp_value;
            b_index = temp_index;
        }

        __device__ void insert_candidate(
            float score,
            int index,
            float *values,
            int *indices,
            int k)
        {
            if (index < 0)
            {
                return;
            }

            if (!better_pair(score, index, values[k - 1], indices[k - 1]))
            {
                return;
            }

            values[k - 1] = score;
            indices[k - 1] = index;

            // Bubble up to keep descending order.
            for (int j = k - 1; j > 0; --j)
            {
                if (better_pair(values[j], indices[j], values[j - 1], indices[j - 1]))
                {
                    swap_pair(values[j], indices[j], values[j - 1], indices[j - 1]);
                }
                else
                {
                    break;
                }
            }
        }

        // Kernel 1:
        // Each block loads up to 256 scores, bitonic-sorts them in shared memory,
        // then emits the best local_k candidates.
        __global__ void local_topk_bitonic_kernel(
            const float *d_scores,
            float *d_partial_values,
            int *d_partial_indices,
            int num_scores,
            int local_k)
        {
            __shared__ float values[BLOCK_SIZE];
            __shared__ int indices[BLOCK_SIZE];

            int tid = threadIdx.x;
            int global_index = blockIdx.x * BLOCK_SIZE + tid;

            if (global_index < num_scores)
            {
                values[tid] = d_scores[global_index];
                indices[tid] = global_index;
            }
            else
            {
                values[tid] = -INFINITY;
                indices[tid] = -1;
            }

            __syncthreads();

            // Bitonic sort, descending by score.
            for (int size = 2; size <= BLOCK_SIZE; size <<= 1)
            {
                for (int stride = size >> 1; stride > 0; stride >>= 1)
                {
                    int partner = tid ^ stride;

                    if (partner > tid)
                    {
                        bool descending_phase = ((tid & size) == 0);

                        if (descending_phase)
                        {
                            // Larger/better pair should move left.
                            if (better_pair(
                                    values[partner],
                                    indices[partner],
                                    values[tid],
                                    indices[tid]))
                            {
                                swap_pair(
                                    values[tid],
                                    indices[tid],
                                    values[partner],
                                    indices[partner]);
                            }
                        }
                        else
                        {
                            // Smaller/worse pair should move left for bitonic merge.
                            if (better_pair(
                                    values[tid],
                                    indices[tid],
                                    values[partner],
                                    indices[partner]))
                            {
                                swap_pair(
                                    values[tid],
                                    indices[tid],
                                    values[partner],
                                    indices[partner]);
                            }
                        }
                    }

                    __syncthreads();
                }
            }

            if (tid < local_k)
            {
                int output_index = blockIdx.x * local_k + tid;
                d_partial_values[output_index] = values[tid];
                d_partial_indices[output_index] = indices[tid];
            }
        }

        // Kernel 2:
        // Merge all partial candidates into final top-K.
        // Current merge is correctness-first: one thread scans candidates.
        __global__ void merge_topk_kernel(
            const float *d_partial_values,
            const int *d_partial_indices,
            float *d_topk_values,
            int *d_topk_indices,
            int num_candidates,
            int k)
        {
            if (blockIdx.x != 0 || threadIdx.x != 0)
            {
                return;
            }

            float final_values[MAX_TOPK];
            int final_indices[MAX_TOPK];

            for (int i = 0; i < k; ++i)
            {
                final_values[i] = -INFINITY;
                final_indices[i] = -1;
            }

            for (int i = 0; i < num_candidates; ++i)
            {
                insert_candidate(
                    d_partial_values[i],
                    d_partial_indices[i],
                    final_values,
                    final_indices,
                    k);
            }

            for (int i = 0; i < k; ++i)
            {
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

        int num_blocks = (num_scores + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Each block can only emit up to BLOCK_SIZE sorted candidates.
        int local_k = std::min(k, BLOCK_SIZE);

        int num_candidates = num_blocks * local_k;

        float *d_partial_values = nullptr;
        int *d_partial_indices = nullptr;

        CURAG_CUDA_CHECK(cudaMalloc(&d_partial_values, num_candidates * sizeof(float)));
        CURAG_CUDA_CHECK(cudaMalloc(&d_partial_indices, num_candidates * sizeof(int)));

        local_topk_bitonic_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_scores,
            d_partial_values,
            d_partial_indices,
            num_scores,
            local_k);

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