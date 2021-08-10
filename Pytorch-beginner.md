# Pytorch-beginner

## 1. Python深度学习框架的选择

在学习完深度学习的理论后你可能已经跃跃欲试，想要动手开始写自己的深度学习模型，由于深度学习的复杂性，在实际工程中不可能完全从头开始写网络结构，这时候就要借助于框架，目前开源的深度学习框架大大小小有十几种，其中用户数最多的是Tensorflow，Pytorch和Keras。以下是这三个框架在全球的google trend。

![1628262444759](C:\Users\skywater\AppData\Roaming\Typora\typora-user-images\1628262444759.png)

其中Tensorflow出现最早，在2015年由Google开源，随后Keras在Tensorflow的基础上应运而生，而Pytorch则于2016年有Facebook开源。

在最开始学习时应该如何选择合适的框架，下面就对这三个框架进行简单的介绍。

其中Keras是基于Tensorflow开发的，是一种更高级的API，代码写法十分简洁，但同时也丧失了灵活性，因此更适合用于简单的原型展示和验证，对于希望在深度学习领域深入研究的不推荐使用Keras。

而Tensorflow和Pytorch在近年来可谓是学术和工业界并驾齐驱的两个主流框架，实际上这两个框架本身也在互相学习，借鉴了对方的优点加以改善自身（从下表就可以看出），用一张表格来总结这两个框架。

| **框架**      | **PyTorch**                                                  | **TensorFlow**                                               |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 计算机制      | 动态图 - 用户可以在模型运行时执行节点，也就是在运行过程中调节模型，更适合自定义，因此也使得pytorch在学术界更受欢迎 | 状态图 - 在模型运行前就被静态定义，在2019年TensorFlow 2.0也开始引入动态图的概念 |
| **模型部署**  | 在2020年引入TorchServe，功能上不如Tensorflow                 | TensorFlow Serving是其自带的模型部署工具，提供了高灵活性和 高性能的部署方式 |
| **可视化**    | PyTorch 在1.2.0 版本开始引入Tensorboard                      | Tensorboard提供了帮助用户理解神经网络的可视化工具箱          |
| **Debugging** | 更容易使用Python的标准debugger工具，如pycharm                | 较为复杂，需要学习TensorFlow的原生debugger工具tfdbg          |

（来自https://www.imaginarycloud.com/blog/pytorch-vs-tensorflow/#TensorFlow）

那么对于一个新手入门者来说，最开始应该如何选择呢？

首先一个大前提是这两个框架都是十分强大的工具，都有着完善的社区环境，在学习之前我们应该确保在理论方面对神经网络有足够的了解，这才是真正能帮助我们脱颖而出的内核，而框架只是帮助我们实现目的的工具。

如果你对python这门编程语言更熟悉，那么在当下更推荐从pytorch开始入手，你会发现在写代码时更为习惯，并且pytorch的学习曲线是更平缓的。另外对于以学术型为主的开发者来说pytorch也是更为推荐的，这得益于pytorch更为直接的debug体验。

Tensorflow通常认为是一个更为成熟的框架，在模型部署方面相比于pytorch有自己的优势，对移动平台的支持也更好。

其实也不用太过于纠结如何选择，随着这两个框架互相学习，很多特征都已经越来越像，等熟悉以后有需要再转向另一个框架也不需要太大的学习成本。

所以现在就开始奇幻的Pytorch学习之旅吧！



## 2. Pytorch搭建神经网络的基本流程

个人推荐的入门pytorch的流程：

1. 首先学习神经网络的理论知识，至少先要了解权重，学习率，激活函数，反向传播等基本概念，了解最基础的网络结构ANN,CNN,RNN等。在这里推荐一些学习材料：

   - 台湾大学李宏毅的深度学习视频 https://www.bilibili.com/video/BV1JE411g7XF

   - Coursera创始人吴恩达的深度学习视频https://www.bilibili.com/video/BV1FT4y1E74V

   - 一本从零开始徒手写神经网络的入门书籍《深度学习图解》https://book.douban.com/subject/34932968/

2. 在具有基础的理论之后，选择合适的框架（这里我们选择pytorch），通过一个基本的神经网络搭建流程把框架的使用骨架先搭出来，在这一步时先不要太深挖细节。先画骨架图了解全貌后，再往骨架中填内容，深入研究细节，这样的学习曲线是较为平缓的，不至于在一开始就劝退初学者。

   顺便说一句，Pytorch的官网提供了很多对新手很友好的例子，有时间可以选择性阅读。

   那么根据官网的[quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)，总结出一个基本的Pytorch搭建神经网络的基本流程分为以下几步：

   1. 数据准备
   2. 定义模型
   3. 模型训练与验证
   4. 模型保存与加载

3. 在了解了基本流程后，开始深入填充细节，列举如下，从一个完整的流程来说加上了第五步模型部署

   1. 数据准备
      - 使用Dataset API下载pytorch提供的数据集，图片，文字，音频等
      - 使用Transform对rawdata转化
      - 使用Dataloader API加载数据集
      - 如何自定义数据集
   2. 定义模型
      - GPU和CPU的选择
      - 如何定义神经网络类
      - 模型层的参数详解
      - 定义序列化网络容器
      - 查看模型参数
      - 激活函数
      - 正则化
   3. 模型训练与验证
      - 损失函数
      - 模型优化器
      - 训练loop
      - 验证loop
      - 超参数调节
      - 权重初始化
      - 模型可视化
   4. 模型保存与加载
      - 模型保存
      - 模型加载
   5. 模型部署
      - API封装
      - 容器化



现在假定你已经有了第一步的理论知识的基础，开始进入第二步的流程，了解一个基本的Pytorch训练神经网络的流程。

精炼版的四个步骤的代码如下，如果你对传统机器学习了解就会发现大体上的流程是极为类似的，现在先大致浏览一遍代码，后面会详细讲解并扩展。

```python
%matplotlib inline
```


```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
```

**一、数据准备**


```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```



```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
```

    Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
    Shape of y:  torch.Size([64]) torch.int64

**二、定义模型**


```python
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

    Using cpu device
    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
        (5): ReLU()
      )
    )

**三、模型训练与验证**


```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```


```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```


```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```


```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

    Epoch 1
    -------------------------------
    loss: 2.298313  [    0/60000]
    loss: 2.290324  [ 6400/60000]
    loss: 2.279126  [12800/60000]
    loss: 2.283089  [19200/60000]
    loss: 2.272794  [25600/60000]
    loss: 2.268349  [32000/60000]
    loss: 2.258117  [38400/60000]
    loss: 2.251492  [44800/60000]
    loss: 2.264173  [51200/60000]
    loss: 2.238919  [57600/60000]
    Test Error: 
     Accuracy: 40.1%, Avg loss: 2.239110 
    
    Epoch 2
    -------------------------------
    loss: 2.239787  [    0/60000]
    loss: 2.216787  [ 6400/60000]
    loss: 2.192778  [12800/60000]
    loss: 2.218873  [19200/60000]
    loss: 2.179190  [25600/60000]
    loss: 2.184922  [32000/60000]
    loss: 2.164321  [38400/60000]
    loss: 2.145076  [44800/60000]
    loss: 2.173346  [51200/60000]
    loss: 2.120572  [57600/60000]
    Test Error: 
     Accuracy: 44.4%, Avg loss: 2.121306 
    
    Epoch 3
    -------------------------------
    loss: 2.123722  [    0/60000]
    loss: 2.072671  [ 6400/60000]
    loss: 2.040230  [12800/60000]
    loss: 2.098472  [19200/60000]
    loss: 2.008334  [25600/60000]
    loss: 2.002212  [32000/60000]
    loss: 2.020195  [38400/60000]
    loss: 1.976064  [44800/60000]
    loss: 2.025062  [51200/60000]
    loss: 1.927013  [57600/60000]
    Test Error: 
     Accuracy: 50.6%, Avg loss: 1.949477 
    
    Epoch 4
    -------------------------------
    loss: 1.963633  [    0/60000]
    loss: 1.877309  [ 6400/60000]
    loss: 1.846219  [12800/60000]
    loss: 1.938163  [19200/60000]
    loss: 1.826020  [25600/60000]
    loss: 1.823304  [32000/60000]
    loss: 1.863928  [38400/60000]
    loss: 1.823402  [44800/60000]
    loss: 1.884976  [51200/60000]
    loss: 1.745336  [57600/60000]
    Test Error: 
     Accuracy: 52.3%, Avg loss: 1.797091 
    
    Epoch 5
    -------------------------------
    loss: 1.819813  [    0/60000]
    loss: 1.708345  [ 6400/60000]
    loss: 1.686980  [12800/60000]
    loss: 1.801250  [19200/60000]
    loss: 1.703848  [25600/60000]
    loss: 1.694704  [32000/60000]
    loss: 1.742806  [38400/60000]
    loss: 1.716054  [44800/60000]
    loss: 1.782515  [51200/60000]
    loss: 1.616319  [57600/60000]
    Test Error: 
     Accuracy: 53.1%, Avg loss: 1.685995 
    
    Done!

**四、模型保存与加载**


```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

    Saved PyTorch Model State to model.pth

```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
```


```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

    Predicted: "Ankle boot", Actual: "Ankle boot"