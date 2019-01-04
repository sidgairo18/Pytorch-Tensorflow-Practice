from __future__ import print_function
import torch

#unintialized matrix
x = torch.empty(5,3)
print(x)

#randomly initialized matrix
x = torch.rand(5,3)
print(x)

#matrix initialized with zeros
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

#Tensor from data
x = torch.tensor([5.5, 3])
print(x)

#Create tensor from existing tensor
x = x.new_ones(5,3,dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())

y = torch.rand(5,3)
print (x+y)

print(torch.add(x,y))

#output tensor as an argument
result = torch.empty(5,3)
torch.add(x,y,out=result)

print(result)

result = x+y

print (result)

#inplace addition
y.add_(x)
print (y)

#Resizing

x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8)
print (x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())


#Converting a torch tensor to a numpy array

a = torch.ones(5)
print (a)

b = a.numpy()
print (b)

a.add_(1)
print (a)
print (b)

b = b+1
print (a)
print(b)

#Converting a numpy array to a torch tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)

#Cuda Tensors

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)

    z = x+y
    print(z)
    print(z.to("cpu", torch.double))
