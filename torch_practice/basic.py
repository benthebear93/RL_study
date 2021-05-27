import torch
import numpy as np

#tensor create
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

#tensor from numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

#tensor from other tensor
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지합니다.
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씁니다.
print(f"Random Tensor: \n {x_rand} \n")

#random and constant

shape =(2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#tensor attribute
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#tensor operation each operation can be run on GPU!
# USE " .to " method

if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print("GPU used")

tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

#simillar as stack but little different. 
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#tensor multiplication
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

#element-wise product

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# not sure wht this means
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#inplace operation has prefix "_"
#not recomanded to use
print(tensor,"\n")
tensor.add_(5)
print(tensor)

#tensor to numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

#chages of tensor will affect numpy values too
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

#getting GPU!
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
