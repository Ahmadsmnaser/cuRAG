#include "curag/normalize.hpp"
#include "curag/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace curag
{

    namespace
    {

        __global__ void l2_normalize_kernel(float *d_vectors, int num_vectors, int dim){
            int vector_id = blockIdx.x;
            int tid = threadIdx.x;

            extern __shared__ float shared[];

            float partial_sum = 0.0f;
            int base = vector_id * dim;

            // Each thread accumulates part of the squared norm.
            for (int j = tid; j < dim; j += blockDim.x)
            {
                float value = d_vectors[base + j];
                partial_sum += value * value;
            }

            shared[tid] = partial_sum;
            __syncthreads();

            // Parallel reduction inside the block.
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
            {
                if (tid < stride)
                {
                    shared[tid] += shared[tid + stride];
                }
                __syncthreads();
            }

            float norm = sqrtf(shared[0]);

            // Zero vector stays zero.
            if (norm == 0.0f)
            {
                return;
            }

            // Normalize vector elements.
            for (int j = tid; j < dim; j += blockDim.x)
            {
                d_vectors[base + j] /= norm;
            }
        }

    } // namespace

    void l2_normalize(float *d_vectors, int num_vectors, int dim){
        if (d_vectors == nullptr)
        {
            throw std::runtime_error("d_vectors must not be null");
        }

        if (num_vectors <= 0)
        {
            throw std::runtime_error("num_vectors must be positive");
        }

        if (dim <= 0)
        {
            throw std::runtime_error("dim must be positive");
        }

        if (dim > 1024)
        {
            throw std::runtime_error("dim exceeds maximum supported dimension 1024");
        }

        constexpr int block_size = 256;
        int num_blocks = num_vectors;
        int shared_bytes = block_size * sizeof(float);

        l2_normalize_kernel<<<num_blocks, block_size, shared_bytes>>>(
            d_vectors,
            num_vectors,
            dim);

        CUDA_CHECK(cudaGetLastError());
    }

} // namespace curag