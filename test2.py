#https://blog.csdn.net/out_of_memory_error/article/details/81262309

#PyTorch基础入门二：PyTorch搭建一维线性回归模型

#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG
 
import torch
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn

 
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 3*x + 10 + torch.rand(x.size())
# 上面这行代码是制造出接近y=3x+10的数据集，后面加上torch.rand()函数制造噪音
 
# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1) # 输入和输出的维度都是1
    def forward(self, x):
        out = self.linear(x)
        return out
 
if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()
 
#然后我们定义出损失函数和优化函数，这里使用均方误差作为损失函数，使用梯度下降进行优化：
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

#首先定义了迭代的次数，这里为1000次，先向前传播计算出损失函数，然后向后传播计算梯度，
#这里需要注意的是，每次计算梯度前都要记得将梯度归零，不然梯度会累加到一起造成结果不收敛。为了便于看到结果，每隔一段时间输出当前的迭代轮数和损失函数。
num_epochs = 1000
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(x).cuda()
        target = Variable(y).cuda()
    else:
        inputs = Variable(x)
        target = Variable(y)
 
    # 向前传播
    out = model(inputs)
    loss = criterion(out, target)
 
    # 向后传播
    optimizer.zero_grad() # 注意每次迭代都需要清零
    loss.backward()
    optimizer.step()
 
    if (epoch+1) %20 == 0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch+1, num_epochs, loss.item()))

#通过model.eval()函数将模型变为测试模式，然后将数据放入模型中进行预测
model.eval()
if torch.cuda.is_available():
    predict = model(Variable(x).cuda())
    predict = predict.data.cpu().numpy()
else:
    predict = model(Variable(x))
    predict = predict.data.numpy()
plt.plot(x.numpy(), y.numpy(), 'ro', label='Original Data')
plt.plot(x.numpy(), predict, label='Fitting Line')
plt.show()
