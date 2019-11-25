#https://blog.csdn.net/out_of_memory_error/article/details/81258809
#PyTorch基础入门一：PyTorch基本数据类型

import torch
import numpy as np

##############################################(1)Tensor(张量)########################################
#Pytorch最基本的操作对象就是Tensor（张量），它表示的其实就是一个多维矩阵，并有矩阵相关的运算操作。
#在使用上和numpy是对应的，它和numpy唯一的不同就是，pytorch可以在GPU上运行，而numpy不可以。
#所以，我们也可以使用Tensor来代替numpy的使用。当然，二者也可以相互转换。

a = torch.Tensor([[1,2],[2,3],[3,5]])
print(a.shape)

b = torch.zeros((3,2))
c = torch.ones(3,4)
d = torch.randn((3,3))
print(b,c,d)

#tensor 转为numpy
numpy_b = b.numpy()
print(type(numpy_b),numpy_b)

#numpy 转为tensor
numpy_e = np.array([[1,2],[3,4],[8,9]])
torch_e = torch.from_numpy(numpy_e)
print(numpy_e,torch_e)
 
# 定义一个3行2列的全为0的矩阵
tmp = torch.randn((3, 2))
 
# 如果支持GPU，则定义为GPU类型
if torch.cuda.is_available():
    inputs = tmp.cuda()
# 否则，定义为一般的Tensor类型
else:
    inputs = tmp

##############################################(2)Variable(变量)########################################
#Pytorch里面的Variable类型数据功能更加强大，相当于是在Tensor外层套了一个壳子
#这个壳子赋予了前向传播，反向传播，自动求导等功能，在计算图的构建中起的很重要的作用
#Variable最重要的两个属性是：data和grad。Data表示该变量保存的实际数据，通过该属性可以访问到它所保存的原始张量类型
#而关于该 variable（变量）的梯度会被累计到.grad 上去，在使用Variable的时候需要从torch.autograd中导入

from torch.autograd import Variable

#定义三个Variable变量
x = Variable(torch.Tensor([1,2,3]),requires_grad=True)
w = Variable(torch.Tensor([2,3,4]),requires_grad=True)
b = Variable(torch.Tensor([3,4,5]),requires_grad=True)

# 构建计算图，公式为：y = w * x^2 + b
y = w * x * x + b

#自动求导，计算梯度
y.backward(torch.Tensor([1,1,1]))

print(x.grad)
print(w.grad)
print(b.grad)
