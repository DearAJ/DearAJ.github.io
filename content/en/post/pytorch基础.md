---
date: 2025-07-10T11:00:59-04:00
description: "https://pytorch.ac.cn/tutorials/beginner/basics/quickstart_tutorial.html"
featured_image: "/images/pytorch/jaz.png"
tags: ["pytorch", "AI Infra"]
title: "pytorch 基础"
---

## [数据](https://pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html)

PyTorch 有两个用于处理数据的基元： `torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset` 

`Dataset` 存储样本及其相应的标签，`DataLoader` 则将一个可迭代对象封装在 `Dataset` 周围

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

PyTorch 提供特定于域的库，例如 TorchText、TorchVision 和 TorchAudio，所有这些库都包含数据集。*以 [TorchVision](https://pytorch.ac.cn/vision/stable/datasets.html)) 中的 FashionMNIST 数据集为例：*

每个 TorchVision `Dataset` 都包含两个参数：`transform` 和 `target_transform`，分别用于修改样本和标签：

```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
```

<!--more-->

```python
# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

在我们的数据集上封装一个可迭代对象：将 `Dataset` 作为参数传递给 `DataLoader` —— 支持自动批量处理、采样、洗牌和多进程数据加载。

batch size 定义为 64：数据加载器可迭代对象中的每个元素将返回 a batch of 64 features and labels

```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```

```python
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```

`[N, C, H, W]` 是描述张量（tensor）形状的常用表示方式，具体含义如下：

- **N (Batch Size)**: 当前批次中的样本数量
- **C (Channels)**: 图像的通道数（例如：灰度图为1，RGB彩色图为3）
- **H (Height)**: 图像的高度（像素数）
- **W (Width)**: 图像的宽度（像素数）

&nbsp;

## [模型](https://pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html)

在 PyTorch 中定义神经网络需创建一个继承自 [nn.Module](https://pytorch.ac.cn/docs/stable/generated/torch.nn.Module.html) 的类 —— 在 `__init__` 函数中定义网络的层，并在 `forward` 函数中指定数据如何通过网络。

为了加速神经网络中的运算，可将其移动到 [加速器](https://pytorch.ac.cn/docs/stable/torch.html#accelerators)（如 CUDA、MPS、MTIA 或 XPU）

```python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
```

```python
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()							# 将 [B,C,H,W] 展平为 [B,C*H*W]
        self.linear_relu_stack = nn.Sequential(	# 按顺序堆叠多个层
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)							# 10 维对应 MNIST 的10个类别。
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)			# 输出模型结构
```

```python
Using cuda device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

&nbsp;

## [优化模型参数](https://pytorch.ac.cn/tutorials/beginner/basics/optimization_tutorial.html)

训练模型需要 [损失函数](https://pytorch.ac.cn/docs/stable/nn.html#loss-functions) 和 [优化器](https://pytorch.ac.cn/docs/stable/optim.html)

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

一个训练循环中，模型对训练数据集（以批次形式输入）进行预测，然后通过**反向传播**预测误差来调整模型的参数。

```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)				# 整个数据集的总样本数
    model.train()													# 将模型设置为训练模式
    
    # 每次迭代返回一个batch索引和数据(X, y)（特征和标签）
    for batch, (X, y) in enumerate(dataloader):	
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)		
            # len(X)为 batch 的样本数，current 计算的是当前已经处理了多少个样本
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

**评估模型在测试集上的性能**（计算准确率和平均损失）：

```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()									# 将模型设置为评估模式
    test_loss, correct = 0, 0
    with torch.no_grad():					# 在评估时不计算梯度（不需要反向传播），节约内存
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # pred.argmax(1)：获取预测的类别
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

训练过程会进行多次迭代（*epochs*，即周期）：

```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

```
Epoch 1
-------------------------------
loss: 2.303494  [   64/60000]
loss: 2.294637  [ 6464/60000]
loss: 2.277102  [12864/60000]
loss: 2.269977  [19264/60000]
loss: 2.254234  [25664/60000]
loss: 2.237145  [32064/60000]
loss: 2.231056  [38464/60000]
loss: 2.205036  [44864/60000]
loss: 2.203239  [51264/60000]
loss: 2.170890  [57664/60000]
Test Error:
 Accuracy: 53.9%, Avg loss: 2.168587

Epoch 2
-------------------------------
loss: 2.177784  [   64/60000]
loss: 2.168083  [ 6464/60000]
loss: 2.114908  [12864/60000]
loss: 2.130411  [19264/60000]
loss: 2.087470  [25664/60000]
loss: 2.039667  [32064/60000]
loss: 2.054271  [38464/60000]
loss: 1.985452  [44864/60000]
loss: 1.996019  [51264/60000]
loss: 1.917239  [57664/60000]
Test Error:
 Accuracy: 60.2%, Avg loss: 1.920371

Epoch 3
-------------------------------
loss: 1.951699  [   64/60000]
loss: 1.919513  [ 6464/60000]
loss: 1.808724  [12864/60000]
loss: 1.846544  [19264/60000]
loss: 1.740612  [25664/60000]
loss: 1.698728  [32064/60000]
loss: 1.708887  [38464/60000]
loss: 1.614431  [44864/60000]
loss: 1.646473  [51264/60000]
loss: 1.524302  [57664/60000]
Test Error:
 Accuracy: 61.4%, Avg loss: 1.547089

Epoch 4
-------------------------------
loss: 1.612693  [   64/60000]
loss: 1.570868  [ 6464/60000]
loss: 1.424729  [12864/60000]
loss: 1.489538  [19264/60000]
loss: 1.367247  [25664/60000]
loss: 1.373463  [32064/60000]
loss: 1.376742  [38464/60000]
loss: 1.304958  [44864/60000]
loss: 1.347153  [51264/60000]
loss: 1.230657  [57664/60000]
Test Error:
 Accuracy: 62.7%, Avg loss: 1.260888

Epoch 5
-------------------------------
loss: 1.337799  [   64/60000]
loss: 1.313273  [ 6464/60000]
loss: 1.151835  [12864/60000]
loss: 1.252141  [19264/60000]
loss: 1.123040  [25664/60000]
loss: 1.159529  [32064/60000]
loss: 1.175010  [38464/60000]
loss: 1.115551  [44864/60000]
loss: 1.160972  [51264/60000]
loss: 1.062725  [57664/60000]
Test Error:
 Accuracy: 64.6%, Avg loss: 1.087372

Done!
```

&nbsp;

## [保存模型](https://pytorch.ac.cn/tutorials/beginner/basics/saveloadrun_tutorial.html)

保存模型的常用方法是**序列化内部状态字典**（包含模型参数）:

```python
torch.save(model.state_dict(), "model.pth")
```

## 加载模型

加载模型的过程包括 **重新创建模型结构 **并 **将状态字典加载到其中**：

```python
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

```
<All keys matched successfully>
```

然后该模型就可以用于进行**预测**：

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
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

```
Predicted: "Ankle boot", Actual: "Ankle boot"
```