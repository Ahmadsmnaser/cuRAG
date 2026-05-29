#include <cuda_runtime.h>

__global__ void dummy_kernel(float* data){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0){
        data[0] = 42.0f; // Just a dummy operation
    }
}

void launch_dummy_kernel(float* d_data){
    dummy_kernel<<<1, 32>>>(d_data);
}