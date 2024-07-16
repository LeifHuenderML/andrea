import andrea

def test_tensor_to_cuda():
    # Create a tensor on CPU
    tensor = andrea.create_tensor([1.0, 2.0, 3.0], [3], "cpu")
    print(tensor.device)
    
    # Move to CUDA
    andrea.to_device(tensor, "cuda")
    print(tensor.device)
    print("Tensor successfully moved to CUDA")

def to(tensor, device):
    andrea.to_device(tensor, device)
    print(tensor.device)
    print("Tensor successfully moved to CUDA")
    return tensor

def add_two_tensor_on_cuda():
    a = andrea.create_ones([3000,3000], 'cpu')
    b = andrea.create_ones([3000,3000], 'cpu')
    a = to(a, "cuda")
    b = to(b, "cuda")



    result = andrea.add_tensor(a, b)
    print(result.data)

if __name__ == "__main__":
    test_tensor_to_cuda()
    add_two_tensor_on_cuda()