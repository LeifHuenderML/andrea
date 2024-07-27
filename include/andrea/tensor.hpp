#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <string>

enumerate class Device{CUDA, CPU};

class Tensor{
private:
    float* h_data=nullptr;
    float* d_data=nullptr;
    std::vector<int> shape;
    std::vector<int> strides;
    std::string device;
    int size;
    int dimensions;

    void toHost();
    void toDevice();
public:
    Tensor(std::vector<int> shape, std::string device=Device::CUDA);
    Tensor(std::vector<int> shape, float* data, std::string device=Device::CUDA);
    void to(Device device);
};
