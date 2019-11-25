#https://blog.csdn.net/out_of_memory_error/article/details/81414986
#PyTorch基础入门五：PyTorch搭建多层全连接神经网络实现MNIST手写数字识别分类

##############################################(1)全连接神经网络（FC）########################################
#全连接神经网络是一种最基本的神经网络结构，英文为Full Connection，所以一般简称FC。
#FC的准则很简单：神经网络中除输入层之外的每个节点都和上一层的所有节点有连接。

##############################################(2)三层FC实现MNIST手写数字分类########################################
#我们定义了三个不层次的神经网络模型：简单的FC，加激活函数的FC，加激活函数和批标准化的FC。

from torch import nn,optim
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader #导入了DataLoader用于加载数据
from torchvision import datasets,transforms #使用了torchvision进行图片的预处理
from matplotlib import pyplot as plt


#简单FC
class SimpleNet(nn.Module):
    #定义一个简单的三层全连接神经网络，每层都是线性的
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(SimpleNet,self).__init__()
        self.layer1 = nn.Linear(in_dim,n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2,out_dim)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

#加激活函数的FC
class ActivationNet(nn.Module):
    #在上面的SimpleNet的基础上，在每层的输出部分添加了激活函数
    #这里的Sequential()函数的功能是将网络的层组合到一起
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(ActivationNet,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))
    
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

#加激活函数和批标准化的FC
class BatchNet(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(BatchNet,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))
    
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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
model = BatchNet(28 * 28, 300, 100, 10)
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
    

# 训练模型
epoch = 0
#print(len(train_loader))   #train_loader 中有938个batch，一个batch中是64张图片和64个标签
for data in train_loader:
    #print(data)
    img, label = data   #每次读取64张图片，每张图片是28×28维的Tensor，label是一个list，包含64个范围0到9之间的整数数字
    #print(img.size())  #([64, 1, 28, 28])
    #showPic(img[0][0].numpy())

    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    #将图片传入网络中 进行前向传播计算
    out = model(img)
    #print(out.size())  #[64, 10]  64行10列，每行代表一张图片，10列一共是10种类别
    #计算损失    
    loss = criterion(out, label)
    #loss是一个tensor(tensor(data),tensor(grad)),loss.data是tensor(item)，item是float型的具体数值
    #print(type(loss.data.item()),loss.data.item()) 
    break
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
    img = img.view(img.size(0), -1)
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


    
