#ifndef TENSOR_OPS_HPP
#define TENSOR_OPS_HPP

#include "andrea/tensor.hpp"
#include <vector>
#include <random>

Tensor uninitialized_tensor(std::vector<int> shape, DeviceType device);
Tensor initialized_tensor(std::vector<int> shape, std::vector<float> data, DeviceType device);
Tensor ones_initiliazed_tensor(std::vector<int> shape, DeviceType device);
Tensor random_initized_tensor(std::vector<int> shape, DeviceType device);

Tensor add_two_tensors(const );

#endif // TENSOR_OPS_HPP