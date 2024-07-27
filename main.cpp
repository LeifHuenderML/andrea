#include "andrea/tensor.hpp"
#include "andrea/tensor_ops.hpp"
#include <iostream>

int main() {
    try {
        Tensor A({1000});
        Tensor B({1000});
        Tensor C({1000});

        // Initialize A and B...

        addTensors(A, B, C);

        C.toHost();  // Bring result back to host if needed

        std::cout << "Tensors added successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}