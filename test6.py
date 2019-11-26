#https://blog.csdn.net/out_of_memory_error/article/details/81434907
#PyTorch基础入门六：PyTorch搭建卷积神经网络实现MNIST手写数字识别

##############################################(1)卷积神经网络(CNN)########################################
#可以用CNN解决的问题所具备的三个性质：
#局部性:对于一张图片而言，需要检测图片中的特征来决定图片的类别，通常情况下这些特征都不是由整张图片决定的，而是由一些局部的区域决定的。
       #例如在某张图片中的某个局部检测出了鸟喙，那么基本可以判定图片中有鸟这种动物。
#相同性:对于不同的图片，它们具有同样的特征，这些特征会出现在图片的不同位置，也就是说可以用同样的检测模式去检测不同图片的相同特征，
       #只不过这些特征处于图片中不同的位置，但是特征检测所做的操作几乎一样。
       #例如在不同的图片中，虽然鸟喙处于不同的位置，但是我们可以用相同的模式去检测。
#不变性:对于一张图片，如果我们进行下采样，那么图片的性质基本保持不变。

##############################################(2)PyTorch中的卷积神经网络########################################
#卷积层：nn.Conv2d()
#池化层：nn.MaxPool2d()

##############################################(3)实现MNIST手写数字识别########################################
#一共定义了五层，其中两层卷积层，两层池化层，最后一层为FC层进行分类输出。

from torch import nn,optim
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader #导入了DataLoader用于加载数据
from torchvision import datasets,transforms #使用了torchvision进行图片的预处理
from matplotlib import pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=3),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=3),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(50 * 5 * 5, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#定义超参数
batch_size = 64
learning_rate = 0.02
num_epoches = 20

#在torchvision中提供了transforms用于帮我们对图片进行预处理和标准化。其中我们需要用到的有两个：ToTensor()和Normalize()。
#前者用于将图片转换成Tensor格式的数据，并且进行了标准化处理。
#后者用均值和标准偏差对张量图像进行归一化：给定均值: (M1,...,Mn) 和标准差: (S1,..,Sn) 用于 n 个通道, 
#该变换将标准化输入 torch.*Tensor 的每一个通道

# 数据预处理。transforms.ToTensor()将图片转换成PyTorch中处理的对象Tensor,并且进行标准化（数据在0~1之间）
# transforms.Normalize()做归一化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差
# transforms.Compose()函数则是将各种预处理的操作组合到了一起
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
# 数据集的下载器
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#然后选择相应的神经网络模型来进行训练和测试，我们这里定义的神经网络输入层为28*28，因为我们处理过的图片像素为28*28，
#两个隐层分别为300和100，输出层为10，因为我们识别0~9十个数字，需要分为十类。损失函数和优化器这里采用了交叉熵和梯度下降。

# 选择模型
#model = SimpleNet(28 * 28, 300, 100, 10)
#model = ActivationNet(28 * 28, 300, 100, 10)
model = CNN()
if torch.cuda.is_available():
    model = model.cuda()
 
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#将图片显示出来
def showPic(npimg):
    x = [i for i in range(28)]*28
    y = [i//28 for i in range(28*28)]    
    co = [npimg[i//28][i%28]+2 for i in range(28*28)]
    #c参数设定点的颜色，它的维度必须和x*y相等，传入的是一个list
    plt.scatter(x,y,c=co)
    plt.show()
    
# 定义一些超参数
batch_size = 64
learning_rate = 0.02
num_epoches = 20

# 数据预处理。transforms.ToTensor()将图片转换成PyTorch中处理的对象Tensor,并且进行标准化（数据在0~1之间）
# transforms.Normalize()做归一化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差
# transforms.Compose()函数则是将各种预处理的操作组合到了一起
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

# 数据集的下载器
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 选择模型
model = CNN()
# model = net.Activation_Net(28 * 28, 300, 100, 10)
# model = net.Batch_Net(28 * 28, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
epoch = 0
for data in train_loader:
    img, label = data
    # img = img.view(img.size(0), -1)
    img = Variable(img)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    out = model(img)
    loss = criterion(out, label)
    print_loss = loss.data.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch+=1
    if epoch%50 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

# 模型评估
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    # img = img.view(img.size(0), -1)
    img = Variable(img)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data.item()*label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(test_dataset)),
    eval_acc / (len(test_dataset))
))
