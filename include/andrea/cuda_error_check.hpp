#ifndef CUDA_ERROR_CHECK_HPP
#define CUDA_ERROR_CHECK_HPP

#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + \
                                     cudaGetErrorString(error) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

#endif // CUDA_ERROR_CHECK_HPP