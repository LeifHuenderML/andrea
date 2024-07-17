#include "andrea/tensor.hpp"


void Tensor::allocate_memory() {
    if (device_ == DeviceType::CPU) {
        data_.reset(new float[size_]);
    } else {
        float* device_ptr;
        cudaMalloc(&device_ptr, size_ * sizeof(float));
        data_.reset(device_ptr);
    }
}

std::vector<int> Tensor::calculate_strides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size());
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

Tensor::Tensor(const std::vector<int>& shape, DeviceType device = DeviceType::CUDA)
    : shape_(shape), device_(device) {
    size_ = 1;
    for (int dim : shape_) {
        size_ *= dim;
    }
    ndim_ = shape_.size();
    strides_ = calculate_strides(shape_);
    allocate_memory();
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data, DeviceType device = DeviceType::CUDA)
    : Tensor(shape, device) {
    if (data.size() != size_) {
        throw std::invalid_argument("Data size does not match tensor shape");
    }
    if (device_ == DeviceType::CPU) {
        std::copy(data.begin(), data.end(), data_.get());
    } else {
        cudaMemcpy(data_.get(), data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);
    }
}

Tensor Tensor::create(const std::vector<int>& shape, DeviceType device = DeviceType::CUDA) {
    return Tensor(shape, device);
}

Tensor Tensor::create_with_data(const std::vector<int>& shape, const std::vector<float>& data, DeviceType device = DeviceType::CUDA) {
    return Tensor(shape, data, device);
}

Tensor Tensor::ones(const std::vector<int>& shape, DeviceType device = DeviceType::CUDA) {
    Tensor tensor(shape, device);

    if (device == DeviceType::CPU) {
        std::fill(tensor.data_.get(), tensor.data_.get() + tensor.size_, 1.0f);
    } else {
        launchFillWithOnesKernel(tensor.data_.get(), tensor.size_);
    }
    return tensor;
}

Tensor Tensor::random(const std::vector<int>& shape, DeviceType device = DeviceType::CUDA) {
    Tensor tensor(shape, device);
    if (device == DeviceType::CPU) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < tensor.size_; ++i) {
            tensor.data_.get()[i] = dis(gen);
        }
    } else {
        launchRandomFillKernel(tensor.data_.get(), tensor.size_, 42);
    }
    return tensor;
}

// Operations
Tensor Tensor::add(const Tensor& other) const {
    if (shape_ != other.shape_ || device_ != other.device_) {
        throw std::invalid_argument("Tensor shapes or devices do not match for addition");
    }
    Tensor result(shape_, device_);
    if (device_ == DeviceType::CPU) {
        for (int i = 0; i < size_; ++i) {
            result.data_.get()[i] = data_.get()[i] + other.data_.get()[i];
        }
    } else {
        launchAddKernel(data_.get(), other.data_.get(), result.data_.get(), size_);
    }
    return result;
}