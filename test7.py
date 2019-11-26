#https://blog.csdn.net/out_of_memory_error/article/details/81456501
#PyTorch基础入门七：PyTorch搭建循环神经网络(RNN)


##############################################(1)任务介绍########################################
#当我们以sin值作为输入，其对应的cos作为输出的时候，你会发现，即使输入值sin相同，其输出结果也可以是不同的，
#这样的话，以前学过的FC, CNN就难以处理，因为你的输出结果不仅仅依赖于输出，而且还依赖于之前的程序结果。所以说，RNN在这里就派上了用场。

import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

class Rnn(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(Rnn, self).__init__()
 
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
 
        self.out = nn.Linear(32, 1)
 
    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_state

#定义超参数
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

#创造一些数据
steps = np.linspace(0, np.pi*2, 100, dtype=np.float)
x_np = np.sin(steps)
y_np = np.cos(steps)

#查看数据
# plt.plot(steps, y_np, 'r-', label='target(cos)')
# plt.plot(steps, x_np, 'b-', label='input(sin)')
# plt.legend(loc='best')
# plt.show()


#选择模型
model = Rnn(INPUT_SIZE)
print(model)

#定义优化器和损失函数
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LR)

h_state = None #第一次暂存为0

for step in range(300):
    start, end = step * np.pi, (step+1)*np.pi
 
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
 
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
 
    prediction, h_state = model(x, h_state)
    h_state = h_state.data
 
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(steps, y_np.flatten(), 'r-')
plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
plt.show()
