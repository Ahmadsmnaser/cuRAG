#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CURAG_CUDA_CHECK(call)                                                               \
    do                                                                                       \
    {                                                                                        \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess)                                                              \
        {                                                                                    \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        }                                                                                    \
    } while (0)
