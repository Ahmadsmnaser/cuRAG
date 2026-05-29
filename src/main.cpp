#include <cuda_runtime.h>
#include <iostream>

void launch_dummy_kernel(float* d_data);

int main(){
    float h_value = 0.0f;
    float* d_value = nullptr;

    cudaMalloc(&d_value, sizeof(float));
    cudaMemcpy(d_value, &h_value, sizeof(float), cudaMemcpyHostToDevice);
    launch_dummy_kernel(d_value);
    cudaMemcpy(&h_value, d_value, sizeof(float), cudaMemcpyDeviceToHost);   
    std::cout << "Value after kernel execution: " << h_value << std::endl;
    cudaFree(d_value);
    return 0;
}