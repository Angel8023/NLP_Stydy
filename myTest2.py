#https://blog.csdn.net/out_of_memory_error/article/details/81262309
#PyTorch基础入门二：PyTorch搭建一维线性回归模型

import torch
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn

#生成y = 2*x + 10 的数据集，使用随机数产生噪音
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = 3*x + 10 + torch.rand(x.size())

#绘制散点图
#plt.scatter(x,y)
#plt.show()

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(1,1) #输入和输出的维度都是1
    def forward(self,x):
        out = self.linear(x)
        return out

if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 1e-2)

num_epochs = 1000
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(x).cuda()
        target = Variable(y).cuda()
    else:
        inputs = Variable(x)
        target = Variable(y)
    
    #向前传播
    out = model(inputs)
    loss = criterion(out,target)
    #向后传播
    optimizer.zero_grad() #每次迭代都需要清零
    loss.backward()
    optimizer.step()

    if(epoch + 1)%20==0:
        print("Epoch[{}/{}],loss:{:.6f}".format(epoch+1,num_epochs,loss.item()))

model.eval()
if torch.cuda.is_available():
    predict = model(Variable(x).cuda())
    predict = predict.data.cpu().numpy()
else:
    predict = model(Variable(x))
    predict = predict.data.numpy()

plt.plot(x,y,"ro",label="Original Data")
plt.plot(x,predict,label="Fitting Line")
plt.show()

