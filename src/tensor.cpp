#include "andrea/tensor.hpp"
#include "andrea/cuda_utils.hpp"
#include <numeric>
#include <stdexcept>


void Tensor::toHost() {
    if (device == Device::CUDA && d_data != nullptr && h_data == nullptr) {
        h_data = static_cast<float*>(malloc(size * sizeof(float)));
        if (h_data == nullptr) {
            throw std::runtime_error("Failed to allocate host memory");
        }
        
        cudaError_t cudaStatus = cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            free(h_data);
            throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }
        
        cudaFree(d_data);
        d_data = nullptr;
        device = Device::CPU
    } else {
        throw std::runtime_error("Tensor is not on CUDA or is in an invalid state");
    }
}

void Tensor::toDevice(){
    if (device == Device::CPU && d_data == nullptr && h_data != nullptr){
        cudaMalloc((void**)*d_data, size * sizeof(float));
        if (d_data == nullptr){
            throw std::runtime_error("Failed to allocate device memory");
        }
        cudaError_t cudaStatus = cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess){
            cudaFree(d_data);
            throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(cudaStatus)));
        }

        free(h_data);
        h_data = nullptr;

        device = Device::CUDA;
    } else{
        throw std::runtime_error("Tensor is not on Host or is in an invalid state");
    }
}


    // float* h_data=nullptr;
    // float* d_data=nullptr;
    // std::vector<int> shape;
    // std::vector<int> strides;
    // std::string device;
    // int size;
    // int dimensions;

Tensor::Tensor(std::vector<int> shape, Device device){
    if(len(shape == 0)){
        throw std::invalid_argument("Shape cannot be len 0\n");
    }


}

Tensor::Tensor(std::vector<int> shape, float* data, Device device){
    
}

void Tensor::to(Device device){
    if(device == Device::CUDA){
        if(this->device == CPU){
            toDevice();
        }
    } else if(device == Device::CPU){
        if(this->device == CUDA){
            toHost();
        }
    } else{
        throw std::invalid_argument("Device must be either CUDA or CPU");
    }

}


