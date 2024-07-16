#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "andrea/tensor.hpp"
#include "andrea/cuda.hpp"
#include "andrea/common.hpp"

namespace py = pybind11;
using namespace andrea;

PYBIND11_MODULE(_andrea, m) {
    // Tensor dtype that houses data, shape, strides, ndim, size, device
    py::class_<andrea::Tensor>(m, "Tensor") 
        .def_property_readonly("data", [](const andrea::Tensor& t) {
            return py::array_t<float>({t.size}, {sizeof(float)}, t.data, py::cast(t));
        })
        .def_property_readonly("shape", [](const andrea::Tensor& t) {
            return py::array_t<int>({t.ndim}, {sizeof(int)}, t.shape, py::cast(t));
        })
        .def_property_readonly("strides", [](const andrea::Tensor& t) {
            return py::array_t<int>({t.ndim}, {sizeof(int)}, t.strides, py::cast(t));
        })
        .def_readonly("ndim", &andrea::Tensor::ndim)
        .def_readonly("size", &andrea::Tensor::size)
        .def_readonly("device", &andrea::Tensor::device);

    // Factory functions used to create tensors
    m.def("create_tensor", [](py::array_t<float, py::array::c_style | py::array::forcecast> data,
                              std::vector<int> shape, std::string device="cuda") {
        if (!data.is_none()) {
            if (data.size() != std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>())) {
                throw std::runtime_error("Data size does not match the specified shape");
            }
            return andrea::create_tensor(static_cast<const float*>(data.data()), shape, device);
        } else {
            return andrea::create_tensor(nullptr, shape, device);
        }
    }, py::arg("data") = py::none(), py::arg("shape"), py::arg("device") = "cuda",
    "Create a new tensor with given data, shape, and device");
    m.def("create_ones", &andrea::create_ones, py::arg("shape"), py::arg("device") = "cuda",
          "Create a new tensor filled with ones");
    m.def("create_zeros", &andrea::create_zeros, py::arg("shape"), py::arg("device") = "cuda",
          "Create a new tensor filled with zeros");
    m.def("create_random", &andrea::create_random, py::arg("shape"), py::arg("device") = "cuda",
          "Create a new tensor filled with random values");

    // Device transfer
      m.def("to_device", [](andrea::Tensor& tensor, const std::string& device) {
        try {
            andrea::to_device(tensor, device);
        } catch (const std::exception& e) {
            throw py::error_already_set();
        }
    }, py::arg("tensor"), py::arg("device"), "Move tensor to specified device (cpu or cuda)");



    //Factory functions used to delete tensors;     
    m.def("delete_tensor", &andrea::delete_tensor, "Delete a tensor and free its memory");

    //Tensor arithmetic ops
    m.def("add_tensor", &andrea::add_tensor, py::arg("tensor1"), py::arg("tensor1"), "Add 2 tensors of same shape");
    
}
