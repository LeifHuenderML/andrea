#ifndef CUDA_HPP
#define CUDA_HPP

#include <cuda_runtime.h>
#include "andrea/common.hpp"
#include <stdexcept>
#include <vector>
#include <iostream>


namespace andrea {

__host__ void initialize_cuda();
__host__ void cpu_to_cuda(Tensor& tensor);
__host__ void cuda_to_cpu(Tensor& tensor);

__host__ void add_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);

}
#endif