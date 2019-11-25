#https://blog.csdn.net/out_of_memory_error/article/details/81275651
# PyTorch基础入门四：PyTorch搭建逻辑回归模型进行分类
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable


#Logistic起源于对人口数量增长情况的研究，后来又被应用到了对于微生物生长情况的研究，以及解决经济学相关的问题，现在作为一种回归分析的分支来处理分类问题。
#所以，虽然名字上听着是“回归”，但实际上处理的问题是“分类”问题。

#create data
n_data = torch.ones(100,2)  # 数据的基本形态
#print(n_data)
x0 = torch.normal(2*n_data,1)   # 类型0 x data (tensor), shape=(100, 2)
#print(x0)
y0 = torch.zeros(100)   # 类型0 y data (tensor), shape=(100, 1)
#print(y0)
x1 = torch.normal(-2*n_data,1)  # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)     # 类型1 y data (tensor), shape=(100, 1)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
# x存储的是点坐标，y存储的是每个点的类别信息
x = torch.cat((x0,x1),0).type(torch.FloatTensor)    # FloatTensor = 32-bit floating
y = torch.cat((y0,y1), 0).type(torch.FloatTensor)    # LongTensor = 64-bit integer
#print(len(x))

#绘图显示数据点
#plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0,cmap="RdYlGn")
#plt.show()

#接下来我们定义Logistic回归模型，以及二分类问题的损失函数和优化器。
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.lr = nn.Linear(2,1)
        self.sm = nn.Sigmoid()
    
    def forward(self,x):
        x = self.lr(x)
        x = self.sm(x)
        return x

logistic_model = LogisticRegression()
if torch.cuda.is_available():
    logistic_model.cuda()

#定义损失函数和优化器
#需要值得注意的是，这里定义的损失函数为BCE损失函数
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(logistic_model.parameters(),lr=1e-3,momentum=0.9)

#开始训练
for epoch in range(10000):
    if torch.cuda.is_available():
        x_data = Variable(x).cuda()
        y_data = Variable(y).cuda()
    else:
        x_data = Variable(x)
        y_data = Variable(y)
    
    out = logistic_model(x_data)
    loss = criterion(out,y_data)
    print_loss = loss.data.item()
    mask = out.ge(0.5).float() #以0.5为阈值进行分类
    correct = (mask == y_data).sum() #计算正确预测的样本个数
    acc = correct.item()/x_data.size(0) #计算精度
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #每隔20轮打印一下当前的误差和精度
    if (epoch + 1) % 20 == 0:
        print('*'*10)
        print('epoch {}'.format(epoch+1)) # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        print('acc is {:.4f}'.format(acc))  # 精度





