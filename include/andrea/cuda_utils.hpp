#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

void cudaCheck(){
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        throw std::runtime_error("CUDA error: " + cudaGetErrorString(error) + "\n");
    }
}


























