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
      - Pytorch数据接口和使用
      - 如何自定义数据集
      - 使用Transform对rawdata做转换
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



## 3. 数据准备

### 3.1 Pytorch数据接口和使用

Pytorch提供了两个原始接口用于处理和数据相关的任务，分别是`torch.utils.data.DataLoader` 和`torch.utils.data.Dataset `。Dataset用于储存样本和对应的标签信息，Dataloader用迭代器的方式封装了Dataset。

在学习pytorch的初级阶段，我们可以直接使用自带的数据素材作为训练样本，其中有三个特定领域的库[TorchText](https://pytorch.org/text/stable/index.html)，[TorchVision](https://pytorch.org/vision/stable/index.html)和 [TorchAudio](https://pytorch.org/audio/stable/index.html)，分别包含了文本数据，图像数据和音频数据，每个库都内置了几十种经典的各自领域的数据样本。

在本节中我们就以torchvision为例，介绍图像数据集的使用。

我们将使用的数据集是[Fashion-MNIST](https://research.zalando.com/project/fashion_mnist/fashion_mnist/)数据集，是最早由Zalando在研究论文中使用的一个分类图像数据集（总共10个类别），包含60000个训练样本和10000个测试样本，每个样本都是一张28*28的灰度图像。

在使用pytorch数据准备时分为两步：

1. datasets API下载数据以及数据变换
2. Dataloader API迭代数据

第一步代码如下

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)
```

datasets api接收以下这些参数作为输入

- `root`用于定义数据下载的路径
- `train` 用于标识训练集（True）还是测试集（False）
- `download`用于标识是否下载数据到定义好的root路径，第二次运行时可以改为False
- `transform` and `target_transform`分别用于进行样本和标签的转换，在这个例子中没有用到target_transform进行标签转换。请注意ToTensor()是一般都会用到的，因为pytorch中所有送入神经网络的数据结构都是张量（tensor），所以在这一步总是应当将原始数据转化为张量，关于transform的其他用法之后再详细解读。

> 知识点：什么是张量？
>
> 关于张量可以用下面这张表格很好地理解其概念。通常我们在描述一个特定的数据结构时会考虑需要用几个维度才能遍历这个数据结构中的每个元素。对这些数据结构我们都有计算机语言和数学语言两套语言来指代。例如0维数据在计算机和数学中分别称为数值和标量，1维数据分别称为数组和向量，2维数据分别称为2维数组和矩阵。张量其实是一个通用性的术语，当把上面提到的n维数据结构的概念衍生到n维时，在数学上就称为n维张量，等价于计算机语言中的n维数组。

| Indexes required | Computer science | Mathematics |
| ---------------- | ---------------- | ----------- |
| 0                | number           | scalar      |
| 1                | array            | vector      |
| 2                | 2d-array         | matrix      |
| n                | nd-array         | nd-tensor   |

第二步代码如下：

```python
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=True)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
```

output

```python
Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
Shape of y:  torch.Size([64]) torch.int64
```

我们将第一步生成的dataset作为参数送入Dataloader中，它以迭代器的方式返回一个个batch的数据。

> 知识点：Epoch, Batch
>
> Epoch: 所有训练样本都训练一轮，称为一个 Epoch
>
> Batch: 一轮训练通常会以多个batch的形式实现，假设数据集共100个样本，分成10个batch，每个batchsize是10，也就是一个epoch会训练100个全部样本，每次训练会将1个batch，也就是10个样本送入模型训练，每训练一个batch，神经网络会进行一次反向传播和梯度更新。

Dataloader API接收以下参数作为输入：

- dataset：Dataset 类，由上一步构建数据集时得到

- batchsize: 批大小

- num_works：是否多进程读取数据，默认是0，表示只使用当前进程。需要注意在windows下对多进程读取数据的效果不是很好，并且很容易出问题，对于小数据集，使用默认参数即可。

- shuffle：是否打乱顺序随机分组

- drop_last: 当样本数不能被 batchsize 整除时，是否舍弃最后一批数据

设置完Dataloader后，试着返回数据集的第一个batch，观察张量的size。

样本的size是torch.Size([64, 1, 28, 28])，四个值分别代表batchsize，channel，height，weight。batchsize是我们设置的64，也就是64张图像，channel为1表示这是一张灰度图，如果是彩色图像channel为3，高度和宽度则是单张图像的大小28*28像素。

至此我们已经学会了如何使用pytorch接口使用自带的数据集进行数据准备。下一小节将进一步学习使用transform进行数据增强。



### 3.2 自定义数据集

上一节我们学习了如何调用pytorch提供的数据集，而在实际生产过程中，我们总是会使用自己的真实数据集，因此这一节就来学习如何自定义数据集。

自定义数据集只需遵循两步：

1. 准备一个路径存放图片数据，以及一个csv文档记录对应图片名称的标签。
2. 自定义一个继承`torch.utils.data.Dataset`类的数据集子类

遵循这两个步骤，本节将自定义一个猫狗数据集。

先看第一步，在某路径下放入四张示例图片（前两张猫，后两张狗），以及一个记录了标签的文档。

![](img\3-1.png)

打开文档里面长这样，第一列记录图片的名称，第二列记录对应的分类，0为猫，1为狗。

![](img\3-2.png)

第一步到此已完成，开始第二步。

写一个继承了Dataset的子类后，初始化后，复写其中的`__len__`和`__getitem__`方法。

完整的类代码如下：

```python
class CatsAndDogsDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        # 读取标签文件
        self.annotations =pd.read_csv(os.path.join(root_dir,csv_file))
        # 定义文件目录
        self.root_dir=root_dir
        # 定义transform
        self.transform=transform
    
    # 返回数据集长度，此处应为4
    def __len__(self):
        return len(self.annotations)
    
    # 获取数据的方法，会和Dataloader连用
    def __getitem__(self,index):
        # 获取图片路径，0表示csv文件的第一列
        img_path=os.path.join(self.root_dir,self.annotations.iloc[index,0])
        # 读取图片
        image = io.imread(img_path)
        # 获取图片对应的标签，1表示csv文件的第二列
        y_label= torch.tensor(int(self.annotations.iloc[index,1]))
        
        # 如果使用时附加了transform参数，则对图片应用转换
        if self.transform:
            image= self.transform(image)
            
        # 返回一个图片和标签的元组
        return (image,y_label)
```

在定义完dataset后就可以实例化了。

```python
# 实例化，输入文件名称，路径和需要的数据转换
dataset = CatsAndDogsDataset(csv_file='label.csv',root_dir='data/cat_dog',transform=transforms.ToTensor())
# 划分训练集和测试集，这里是分为3个训练样本和1个测试样本
training_data,test_data = torch.utils.data.random_split(dataset,[3,1])

# 这一步开始和使用自带数据集的代码完全一致
batch_size=1
train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=True)
```

观察那一个测试样本

```python
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# 以图片的形式查看
to_pil_image = transforms.ToPILImage()
img = to_pil_image(X[0])
img.show()
```

### 3.3 Transforms对图像做转换

在上面的章节中，我们已经见到了在加载数据时用到了torchvision.transforms方法对原始数据进行张量化操作，本节将进一步介绍其他转换方法的使用。

通常对原始数据进行转换的一个主要目的是数据增强（Data Augmentation）。我们都知道，一个深度学习模型的预测准确性在很大程度上是依赖于数据集的数量和多样性的。尤其在复杂结构的神经网络中，随着隐层神经元数量的爆炸式增长，需要训练的参数量也显著增加，比如在CV领域的一个重要模型RESNET就有六千万个参数。伴随着待训练参数的增加，对数据量的要求也相应增加了。但是在实际生产环境中，收集如此庞大的数据量是一个很大的难点。

为了解决这一问题，有两种方法，第一种方式是使用迁移学习，也就是预训练一个复杂的基础模型，在其基础上迁移到实际数据上再次训练。另一种方法则是使用数据增强，数据增强可以满足数据多样性和数据量的需求，还可用于解决分类任务中的类不平衡问题。

那么应当如何进行数据增强就要用到本节提到的transforms。

依然以图像数据为例，[torchvision.transforms](https://pytorch.org/vision/stable/transforms.html#)提供了多种进行数据变换的分类，常用的就以下三类：

- 对image做变换
- 对tensor做变换
- 对类型做转换

#### 3.3.1 对image做变换

对图像做变换是最常见的操作，例如裁剪，旋转，翻转，填充等等，详细的各种变换方法可以查看transforms官方文档。

以resize为例：

定义两个不同参数的resize变换，输入一个参数时，该方法会默认以短边为100，另一边等比例缩放，输入两个参数时，则直接指定了（高度，宽度）。

```python
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
resize1 = transforms.Resize(100)
resize2 = transforms.Resize((100,200))
```

定义一个plot方法，将原图和变换后的图像并排展现

```py
def plot(*imgs):
    fig, axs = plt.subplots(nrows=1,ncols=len(imgs),figsize=(10, 10))
    for i in range(len(imgs)):
        axs[i].imshow(np.asarray(imgs[i]))
```

展示变换效果

```python
plot(img,resize1(img),resize2(img))
```

![](img\3-3.png)



#### 3.3.2 对tensor做变换

Normalize标准化是最常用的对tensor进行变换的方法。

首先计算标准化之前的均值和标准差，

```python
mean = next(iter(test_dataloader))[0].mean()
std=next(iter(test_dataloader))[0].std()
print('mean of input : {}'.format(mean))
print('std of input : {}'.format(std))
```

```
mean of input : 0.43933048844337463
std of input : 0.2347354143857956
```

先将图片转化为tensor之后，再进行标准化，再次观察均值和标准差，发现已经变成标准正态分布

```python
trans= transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
print('mean of input : {}'.format(trans(img).mean()))
print('std of input : {}'.format(trans(img).std()))
```

```
mean of input : -1.1390642384867533e-07
std of input : 0.9999999403953552
```

在这里我们用到了``transforms.Compose``，作用是将多个transforms操作结合成一个序列，从而对原始数据进行多步变化。

#### 3.3.3 对类型做转换

这一类变化的作用是将图片数据转换为张量形式，或将张量数据转换为图片形式，分别由以下这两个方法进行控制，特别注意最终送入训练模型的数据类型一定要是张量的形式。

`torchvision.transforms.ToPILImage`，`torchvision.transforms.ToTensor`