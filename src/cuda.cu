#include "andrea/cuda.hpp"

#define THREADS_PER_BLOCK 128
#define TILE_SIZE 32

namespace andrea{

__host__ void initialize_cuda() {
    cudaError_t err = cudaSetDevice(0);  // Use the first CUDA device
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to initialize CUDA device: " + std::string(cudaGetErrorString(err)));
    }
}

__host__ void cpu_to_cuda(Tensor& tensor) {
    initialize_cuda();
    if (tensor.device != "cpu") {
        throw std::runtime_error("Tensor is not on CPU. Cannot move to CUDA.");
    }

    if (tensor.data == nullptr) {
        throw std::runtime_error("Tensor data is null. Cannot move to CUDA.");
    }

    float* data_tmp;
    cudaError_t err = cudaMalloc((void **)&data_tmp, tensor.size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(data_tmp, tensor.data, tensor.size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(data_tmp);
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
    }

    // Free the CPU memory
    delete[] tensor.data;

    tensor.data = data_tmp;
    tensor.device = "cuda";

    // Synchronize to ensure all operations are complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA synchronization failed: " + std::string(cudaGetErrorString(err)));
    }
}

__host__ void cuda_to_cpu(Tensor& tensor) {
    float* data_tmp = (float*)malloc(tensor.size * sizeof(float));

    cudaMemcpy(data_tmp, tensor.data, tensor.size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(tensor.data);

    tensor.data = data_tmp;
    tensor.device = "cpu";
}


__global__ void add_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data1[i] + data2[i];
    }
}

__host__ void add_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}


} // namespace andr