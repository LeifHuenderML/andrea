#ifndef CUDA_HPP
#define CUDA_HPP


#include "andrea/cuda_error_check.hpp"
#include <cuda_runtime_api.h>
#include <math_functions.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <stdexcept>
#include <curand.h>
#include <vector>
#include <cuda.h>
#include <ctime>

__global__ void fillWithOnesKernel(float* data, int size);
__global__ void randomFillKernel(float* data, int size, unsigned int seed);
__global__ void addKernel(const float* a, const float* b, float* result, int size);
//finish these 
void launchFillWithOnesKernel(float* data, int size);
void launchRandomFillKernel(float* data, int size, unsigned int seed);
void launchAddKernel(const float* a, const float* b, float* result, int size);
#endif // CUDA_HPP