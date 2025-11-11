---
date: 2025-04-17T05:00:59-07:00
description: ""
featured_image: "/images/pytorch/pia.jpg"
tags: ["data processing", "tool"]
title: "工具链-PyTorch"
---

## 1. 处理数据

PyTorch 有两个用于处理数据的基元： `torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset`。
`Dataset` 存储样本及其相应的标签，`DataLoader` 将 `Dataset` 包装成一个迭代器。

+ 下面以 TorchVision 库模块里的 FashionMNIST 数据集为例：

  每个 TorchVision `Dataset` 都包含两个参数： `transform` 和 `target_transform` 分别修改样本和标签

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

将 `Dataset` 作为参数传递给 `DataLoader` ，将一个可迭代对象包装在数据集上，支持自动批处理、采样、洗牌和多进程数据加载。

定义了一个 batch size 为 64，即 dataloader 迭代器中的每个元素将返回**一个 64 features and labels 的 batch**。

<!--more-->

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

+ 输出：

  ![1](/images/pytorch/1.png)

&nbsp;

## 2. 创建神经网络模型

创建了一个继承自 nn.模块。在 `__init__` 函数中定义网络层，并在 `forward` 函数中指定数据如何通过网络。

为了加速神经网络中，可将其移至加速器，例如 CUDA、MPS 或 XPU。如果当前加速器可用，我们将使用它。否则，我们使用 CPU。

```python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

&nbsp;

## 3. 优化模型参数

要训练模型，需要一个损失函数和一个优化器。

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

在单个训练循环中，模型对训练数据集进行预测（分批提供给它），并反向传播预测误差以调整模型的参数。

```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

