#include "curag/topk.hpp"
#include "curag/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdexcept>

namespace curag{

namespace{

    constexpr int MAX_TOPK = 1024;


    __device__ void swap_pair(float &a, int &a_idx, float &b, int &b_idx) {
        float temp_val = a;
        int temp_idx = a_idx;
        a = b;
        a_idx = b_idx;
        b = temp_val;
        b_idx = temp_idx;
    }

    // Baseline top-k kernel
    // One block processes the full score array
    // This is intentionally simple for correctness before block-wise scaling
    __global__ void topk_baseline_kernel(
        const float *d_scores,
        float *d_topk_values,
        int *d_topk_indices,
        int num_scores, int k){

            __shared__ float values[MAX_TOPK];
            __shared__ int indices[MAX_TOPK];

            int tid = threadIdx.x;
            
            // Initialize local top-k buffer
            if (tid < k) {
                values[tid] = -CUDART_INF_F;
                indices[tid] = -1;
            }
            __syncthreads();

            // Single-thread baseline selection.
            // Slow but correct. We replace this with block-wise bitonic next.
            if (tid == 0){
                for (int i = 0; i < num_scores; ++i) {
                    float score = d_scores[i];
                    // Check if this score should be in the top-k
                    if (score > values[k-1]) {
                        // Insert into the correct position in the top-k list
                        values[k-1] = score;
                        indices[k-1] = i;
                        // Bubble up the new score to maintain sorted order
                        for (int j = k-1; j > 0; --j) {
                            if (values[j] > values[j-1]) {
                                swap_pair(values[j], indices[j], values[j-1], indices[j-1]);
                            } else {
                                break;
                            }
                        }
                    }
                }

                // Write results to global memory
                for (int i = 0; i < k; ++i) {
                    d_topk_values[i] = values[i];
                    d_topk_indices[i] = indices[i];
                }
            }
    }
} //namespace

void topk(
    const float *d_scores,
    float *d_topk_values,
    int *d_topk_indices,
    int num_scores,int k){
        if (d_scores == nullptr || d_topk_values == nullptr || d_topk_indices == nullptr){
            throw std::runtime_error("topk device pointers must not be null");
        }
        if (num_scores <= 0){
            throw std::runtime_error("num_scores must be positive");
        }
        if (k <= 0){
            throw std::runtime_error("k must be positive");
        }
        if (k > num_scores){
            throw std::runtime_error("k must be <= num_scores");
        }
        if (k > MAX_TOPK){
            throw std::runtime_error("k exceeds MAX_TOPK=1024");
        }
        constexpr int block_size = 256;

        topk_baseline_kernel<<<1, block_size>>>(d_scores, d_topk_values, d_topk_indices, num_scores, k);

        CURAG_CUDA_CHECK(cudaGetLastError());
    }
}
