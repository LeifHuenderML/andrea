/**
 * @file Tensor.hpp
 * @brief Defines a Tensor class for efficient multi-dimensional array operations on CPU and CUDA devices.
 * 
 * This Tensor class provides a flexible and memory-safe way to work with multi-dimensional arrays
 * (tensors) in both CPU and CUDA environments. It utilizes modern C++ features for resource
 * management and CUDA integration.
 * 
 * Key features:
 * - Supports both CPU and CUDA memory allocation
 * - Automatic memory management using smart pointers
 * - Custom CUDA memory deallocation
 * - Move semantics for efficient ownership transfer
 * - Deleted copy constructor and assignment to prevent accidental copies
 * - Automatic calculation of size and strides based on shape
 * 
 * Usage:
 * Tensor cpu_tensor({2, 3, 4}, DeviceType::CPU);  // Creates a 2x3x4 tensor on CPU
 * Tensor gpu_tensor({2, 3, 4}, DeviceType::CUDA); // Creates a 2x3x4 tensor on CUDA device
 * 
 * Note: This class currently supports float data type. Extend using templates for other types.
 * 
 * @author Leif Huender
 * @date 2024
 * @version 1.0
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "cuda_error_check.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include "cuda.hpp"
#include <random>
#include <memory>
#include <vector>
#include <string>

// Custom deleter for CUDA device memory
struct cuda_deleter {
    void operator()(void* ptr) const {
        cudaFree(ptr);
    }
};

enum class DeviceType { CPU, CUDA };

class Tensor {
private:
    std::unique_ptr<float, cuda_deleter> data_;
    std::vector<int> shape_;
    std::vector<int> strides_;
    int size_;
    int ndim_;
    DeviceType device_;

    void allocate_memory();
    std::vector<int> calculate_strides(const std::vector<int>& shape); 
public:
    Tensor(const std::vector<int>& shape, DeviceType device = DeviceType::CUDA);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data, DeviceType device = DeviceType::CUDA);
       
    Tensor create(const std::vector<int>& shape, DeviceType device = DeviceType::CUDA); 
    Tensor create_with_data(const std::vector<int>& shape, const std::vector<float>& data, DeviceType device = DeviceType::CUDA);
    Tensor ones(const std::vector<int>& shape, DeviceType device = DeviceType::CUDA);
    Tensor random(const std::vector<int>& shape, DeviceType device = DeviceType::CUDA);
    
    // Operations
    Tensor add(const Tensor& other) const; 


    // Move constructor
    Tensor(Tensor&& other) noexcept = default;
    // Move assignment operator
    Tensor& operator=(Tensor&& other) noexcept = default;
    // Disable copy constructor and assignment
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Accessor methods
    const float* data() const { return data_.get(); }
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    int size() const { return size_; }
    int ndim() const { return ndim_; }
    DeviceType device() const { return device_; }

};

#endif // TENSOR_HPP