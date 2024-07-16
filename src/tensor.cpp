#include "andrea/tensor.hpp"

namespace andrea{

Tensor* create_tensor(const float* data, std::vector<int> shape, std::string device) {
    // Allocate memory for tensor structure
    Tensor* tensor = new Tensor;
    if (!tensor) {
        throw std::runtime_error("Failed to allocate memory for tensor");
    }

    tensor->ndim = shape.size();
    tensor->size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    tensor->device = "cpu";

    // Allocate and copy shape
    tensor->shape = new int[tensor->ndim];
    std::memcpy(tensor->shape, shape.data(), tensor->ndim * sizeof(int));

    // Calculate and allocate strides
    tensor->strides = new int[tensor->ndim];
    int stride = 1;
    for (int i = tensor->ndim - 1; i >= 0; --i) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    // Allocate and copy/initialize data
    tensor->data = new float[tensor->size];
    if (data) {
        std::memcpy(tensor->data, data, tensor->size * sizeof(float));
    } else {
        std::fill(tensor->data, tensor->data + tensor->size, 0.0f);
    }

    to_device(*tensor, device);
    

    return tensor;
}


Tensor* create_ones(std::vector<int> shape, std::string device){
    Tensor* tensor = create_tensor(nullptr, shape, device);
    std::fill(tensor->data, tensor->data + tensor->size, 1.0f);
    return tensor;
}

Tensor* create_zeros(std::vector<int> shape, std::string device){
    // create tensor by default creates a zeros tensor
    Tensor* tensor = create_tensor(nullptr, shape, device);
    return tensor;
}

Tensor* create_random(std::vector<int> shape, std::string device) {
    // Fills the tensor with a uniform real distribution of numbers between 0 and 1
    Tensor* tensor = create_tensor(nullptr, shape, device);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    std::generate(tensor->data, tensor->data + tensor->size, [&]() { return dis(gen); });

    return tensor;
}


void delete_tensor(Tensor* tensor) {
    if (tensor) {
        delete_strides(tensor);
        delete_shape(tensor);
        delete_data(tensor);
        delete tensor;
    }
}

void delete_strides(Tensor* tensor){
    if (tensor->strides){
        delete[] tensor->strides;
    }
}
void delete_shape(Tensor* tensor){
    if (tensor->shape){
        delete[] tensor->shape;
    }
}

void delete_data(Tensor* tensor){
    if (tensor->data){
        delete[] tensor->data;
    }
}

void to_device(Tensor& tensor, const std::string device) {
    if (tensor.device == device) {
        return;
    }
    else if (device == "cpu"){
        cuda_to_cpu(tensor);
    }
    else if (device == "cuda"){
        std::cout << "took this path";
        cpu_to_cuda(tensor);
    }
    else{
        throw std::invalid_argument("Input for device must be cuda or cpu.\n");
    }
}

Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2) {
    if (tensor1->ndim != tensor2->ndim) {
        throw std::runtime_error("Tensors must be the same dimension");
    }
    if (tensor1->device != tensor2->device) {
        throw std::runtime_error("Tensors must be on the same device");
    }

    std::vector<int> shape;
    for (int i = 0; i < tensor1->ndim; i++) {
        if (tensor1->shape[i] != tensor2->shape[i]) {
            throw std::runtime_error("Tensors must have the same shape");
        }
        shape.push_back(tensor1->shape[i]);
    }

    float* result_data = nullptr;
    if (tensor1->device == "cpu") {
        result_data = new float[tensor1->size];
        add_tensor_cpu(tensor1, tensor2, result_data);
    }
    else if (tensor1->device == "cuda") {
        cudaError_t err = cudaMalloc(&result_data, tensor1->size * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)));
        }
        add_tensor_cuda(tensor1, tensor2, result_data);
    }
    else {
        throw std::runtime_error("Unknown device: " + tensor1->device);
    }

    return create_tensor(result_data, shape, tensor1->device);
}

void add_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] + tensor2->data[i];
    }
}

// float get_item(Tensor* tensor, int* indices);
// Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2);
// void make_contiguous(Tensor* tensor);

// // Tensor* sum_tensor(Tensor* tensor, int axis, bool keepdims);
// // Tensor* max_tensor(Tensor* tensor, int axis, bool keepdim);
// // Tensor* min_tensor(Tensor* tensor, int axis, bool keepdim);
// // Tensor* sub_tensor(Tensor* tensor1, Tensor* tensor2);
// // Tensor* elementwise_mul_tensor(Tensor* tensor1, Tensor* tensor2);
// // Tensor* scalar_mul_tensor(Tensor* tensor, float scalar);
// // Tensor* scalar_div_tensor(float scalar, Tensor* tensor);
// // Tensor* tensor_div_scalar(Tensor* tensor, float scalar);
// // Tensor* tensor_div_tensor(Tensor* tensor1, Tensor* tensor2);
// // Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim);
// // Tensor* matmul_tensor(Tensor* tensor1, Tensor* tensor2);
// // Tensor* tensor_pow_scalar(Tensor* tensor, float exponent);
// // Tensor* scalar_pow_tensor(float base, Tensor* tensor);
// // Tensor* log_tensor(Tensor* tensor);
// // Tensor* equal_tensor(Tensor* tensor1, Tensor* tensor2);
// // Tensor* equal_broadcasted_tensor(Tensor* tensor1, Tensor* tensor2);
// // Tensor* ones_like_tensor(Tensor* tensor);
// // Tensor* zeros_like_tensor(Tensor* tensor);
// // Tensor* sin_tensor(Tensor* tensor);
// // Tensor* cos_tensor(Tensor* tensor);
// // Tensor* transpose_tensor(Tensor* tensor);
// // Tensor* transpose_axes_tensor(Tensor* tensor, int axis1, int axis2);

} // namespace andrea