#ifndef ANDREA_TENSOR_HPP
#define ANDREA_TENSOR_HPP

#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <string>
#include <random>
#include <cuda_runtime.h>
#include "andrea/common.hpp"
#include "andrea/cuda.hpp"
#include <vector>

namespace andrea{

Tensor* create_tensor(const float* data, std::vector<int> shape, std::string device="cuda");
Tensor* create_ones(std::vector<int> shape, std::string device="cuda");
Tensor* create_zeros(std::vector<int> shape, std::string device="cuda");
Tensor* create_random(std::vector<int> shape, std::string device="cuda");

void delete_tensor(Tensor* tensor);
void delete_strides(Tensor* tensor);
void delete_shape(Tensor* tensor);
void delete_data(Tensor* tensor);

void to_device(Tensor& tensor, const std::string device);

Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2);

void add_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);

// float get_item(Tensor* tensor, int* indices);
// Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2);
// void make_contiguous(Tensor* tensor);

// Tensor* sum_tensor(Tensor* tensor, int axis, bool keepdims);
// Tensor* max_tensor(Tensor* tensor, int axis, bool keepdim);
// Tensor* min_tensor(Tensor* tensor, int axis, bool keepdim);
// Tensor* sub_tensor(Tensor* tensor1, Tensor* tensor2);
// Tensor* elementwise_mul_tensor(Tensor* tensor1, Tensor* tensor2);
// Tensor* scalar_mul_tensor(Tensor* tensor, float scalar);
// Tensor* scalar_div_tensor(float scalar, Tensor* tensor);
// Tensor* tensor_div_scalar(Tensor* tensor, float scalar);
// Tensor* tensor_div_tensor(Tensor* tensor1, Tensor* tensor2);
// Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim);
// Tensor* matmul_tensor(Tensor* tensor1, Tensor* tensor2);
// Tensor* tensor_pow_scalar(Tensor* tensor, float exponent);
// Tensor* scalar_pow_tensor(float base, Tensor* tensor);
// Tensor* log_tensor(Tensor* tensor);
// Tensor* equal_tensor(Tensor* tensor1, Tensor* tensor2);
// Tensor* equal_broadcasted_tensor(Tensor* tensor1, Tensor* tensor2);
// Tensor* ones_like_tensor(Tensor* tensor);
// Tensor* zeros_like_tensor(Tensor* tensor);
// Tensor* sin_tensor(Tensor* tensor);
// Tensor* cos_tensor(Tensor* tensor);
// Tensor* transpose_tensor(Tensor* tensor);
// Tensor* transpose_axes_tensor(Tensor* tensor, int axis1, int axis2);

// Helper Functions
//only used to throw an error if device is not cpu or cuda does not set the device

} // namespace andrea

#endif // ANDREA_TENSOR_HPP