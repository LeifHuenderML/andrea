#ifndef ANDREA_COMMON_HPP
#define ANDREA_COMMON_HPP


#include <string>

namespace andrea {

struct Tensor{
    float* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    std::string device;
};

}  // namespace andrea

#endif // ANDREA_COMMON_HPP