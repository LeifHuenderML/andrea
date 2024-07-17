#include "andrea/cuda.hpp"

__global__ void fillWithOnesKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f;
    }
}

__global__ void randomFillKernel(float* data, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = curand_uniform(&state);
    }
}

__global__ void addKernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}


void launchAddKernel(const float* a, const float* b, float* result, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(a, b, result, size);
    cudaDeviceSynchronize();
}
void launchFillWithOnesKernel(float* data, int size){
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    fillWithOnesKernel<<<numBlocks, blockSize>>>(data, size);
    cudaDeviceSynchronize();
}
void launchRandomFillKernel(float* data, int size, unsigned int seed){
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    randomFillKernel<<<numBlocks, blockSize>>>(data, size, seed);
    cudaDeviceSynchronize();
}


