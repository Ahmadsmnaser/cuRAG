#include <cuda_runtime.h>
#include <stdio.h>

/*
    Normalize vectors to L2 Normalization
    
    Input:
        (Vectors A)
    Output:
        (Normalized Vectors A')
    where A' = A / ||A||_2
*/

__global__ void l2_normalize(float* vectors, int num_vectors, int vector_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vectors) {
        float* vector = vectors + idx * vector_dim;
        float norm = 0.0f;

        // Compute the L2 norm
        for (int i = 0; i < vector_dim; ++i) {
            norm += vector[i] * vector[i];
        }
        norm = sqrtf(norm);

        // Normalize the vector
        for (int i = 0; i < vector_dim; ++i) {
            vector[i] /= norm;
        }
    }
}

// i will not use this kernel, but it is here for reference
__global__ void normalize_correct(float *A, float *B, float *A_norm, float *B_norm, float norm_A, float norm_B, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        A_norm[idx] = A[idx] / norm_A;
        B_norm[idx] = B[idx] / norm_B;
    }
}