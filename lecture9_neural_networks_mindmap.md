---
title: FRE-GY 7773 复习思维导图 — Lecture 9 Neural Networks & PyTorch
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 380
---

# Lecture 9 — Neural Networks & PyTorch

> Lecture 9 没有 slide PDF，内容来自 9 份 notebook（pytorch_intro / tensor / fundamentals / neural_networks / training）。

## ① 神经网络基本概念

- **从逻辑回归出发**
  - Logistic regression = 单层 NN（线性 + sigmoid 输出）
  - 神经网络 = 堆叠多层 + 加非线性激活 → 表达更复杂函数
- **基本结构**
  - **输入层**（features）
  - **隐藏层**（一或多层 fully connected）
  - **输出层**（回归 → 标量；分类 → softmax / log-softmax）
- **每层做什么**
  - 线性变换：$\mathbf{z} = W\mathbf{x} + \mathbf{b}$
  - 非线性激活：$\mathbf{a} = \phi(\mathbf{z})$
- **关键差异（vs 逻辑回归）**
  - Loss 一般**非凸** → 多个局部最优、初始化敏感
  - 没有闭式解，只能梯度下降
  - 通用近似定理：足够宽的单隐层网络可逼近任意连续函数

## ② Activation Functions

- **Sigmoid**：$\sigma(z) = 1/(1+e^{-z})$，输出 $(0,1)$
  - 深度网络易**梯度消失**
- **Tanh**：$\tanh(z)$，输出 $(-1,1)$
- **ReLU**：$\max(0, z)$，**最常用**
  - 简单、避免梯度消失、计算快
  - 但 $z<0$ 时梯度为 0（dying ReLU）
- **变种**：Leaky ReLU、ELU、GELU
- **Softmax**（输出层多分类）

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

  - 输出概率分布（和为 1）
  - **数值技巧**：$e^{700}$ 直接溢出 → 用 LogSoftmax + NLLLoss 替代 Softmax + CrossEntropyLoss 更稳

## ③ Loss Functions

- **回归**：MSE → `nn.MSELoss`
- **二分类**：Binary cross-entropy = NLL → `nn.BCELoss` / `nn.BCEWithLogitsLoss`
- **多分类**
  - `nn.CrossEntropyLoss`：内部已含 LogSoftmax + NLL，**输入是 logits**（不是概率！）
  - 或者 `nn.LogSoftmax(dim=1)` + `nn.NLLLoss`：手动分两步，等价
- **关键**：CrossEntropyLoss 接受 raw logits，**不要**先 softmax；NLLLoss 接受 log-probabilities

## ④ PyTorch Tensors（基础数据类型）

- **类似 NumPy 但有梯度跟踪**
- **Tensor 维度名词**
  - scalar（0-D）/ vector（1-D）/ matrix（2-D）/ tensor（3-D+）
- **创建**：`torch.tensor`、`torch.zeros`、`torch.ones`、`torch.randn`、`torch.arange`
- **形状操作**：`view` / `reshape` / `unsqueeze` / `squeeze` / `resize_`
- **矩阵乘法**：`@`、`torch.matmul`、`torch.mm`（注意形状要对得上）
- **NumPy ↔ Torch**：`.numpy()`、`torch.from_numpy()`
- **Device**（CPU/GPU）：`.to('cuda')`、`.to(device)`
- **Datatype**：默认 `float32`，可指定 `torch.float16` 等
- **常见错误**：shape mismatch、device mismatch（CPU tensor × GPU tensor）

## ⑤ 前向传播

- **手算**：$\mathbf{a} = \phi(W\mathbf{x} + \mathbf{b})$
- **多层堆叠**：上一层 output → 下一层 input
- **PyTorch 中**
  - 用 `nn.Module` 子类自定义
  - 或用 `nn.Sequential` 串起来

## ⑥ 用 nn.Module 构建网络

- **标准模板（自定义类）**
  - 继承 `nn.Module`
  - `__init__` 里定义所有层
  - `forward(x)` 里写前向计算
- **替代写法（Sequential）**：层简单堆叠时更短

## ⑦ Autograd（自动求导）

- **核心**：tensor 设 `requires_grad=True` 让 PyTorch 跟踪计算图
- **反向传播**：`loss.backward()` 自动计算所有梯度
- **梯度访问**：`param.grad`
- **关闭梯度**：`with torch.no_grad():` 或 `tensor.detach()`（推理时用）
- **重要细节**：每次 backward 前必须 `optimizer.zero_grad()`，否则梯度累积

## ⑧ 训练循环（标准模板）

- **三步循环**
  1. forward：算预测和 loss
  2. backward：`loss.backward()` 算梯度
  3. update：`optimizer.step()` 更新参数
- **每个 batch 都要 `zero_grad()`**
- **关键组件**
  - Optimizer：`torch.optim.SGD`、`torch.optim.Adam` 等
  - Loss：`nn.MSELoss`、`nn.CrossEntropyLoss`、`nn.NLLLoss` 等
  - DataLoader：`torch.utils.data.DataLoader`
- **Epoch vs Iteration**
  - 1 epoch = 数据集完整过一遍
  - 1 iteration = 一个 mini-batch 更新

## ⑨ Dataset / DataLoader

- **torchvision.datasets**：MNIST 等内置数据集
- **transforms.Compose**：链式预处理（ToTensor、Normalize 等）
- **DataLoader**：自动 batching、shuffle、并行加载

## ⑩ 推理 / 预测

- **eval 模式**：`model.eval()` 关闭 dropout / batchnorm 训练行为
- **不算梯度**：`with torch.no_grad():` 节省内存 + 加速
- **MNIST 例子**：拿到 logits → softmax → argmax 得预测类别

## ⑪ 常见坑 & 易错点

- 忘 `optimizer.zero_grad()` → 梯度累积，训练崩
- 训练/测试模式混淆（dropout、batchnorm 行为不同）
- 数据没标准化 → 训练不稳
- 学习率太大 / 太小
- CrossEntropyLoss 喂了 softmax 后的概率（应喂 raw logits）
- shape 不匹配（`(64, 1, 28, 28)` vs `(64, 784)`）

## ⑫ 与之前学过的内容的连接

- **逻辑回归** = 单层 NN（线性 + sigmoid 输出）
- **梯度下降 / SGD / mini-batch** = NN 训练核心算法（L6）
- **convex 不再保证**：NN loss 一般非凸，多个局部最优
- **正则化**：L2 (weight decay)、L1、dropout、early stopping、batch normalization

## ⑬ 💻 代码模板（来自 notebooks）

- **Tensor 基础**

```python
import torch

# 创建
scalar = torch.tensor(7)                     # 0-D
vector = torch.tensor([1, 2, 3])             # 1-D
MATRIX = torch.tensor([[1, 2], [3, 4]])      # 2-D
random = torch.randn(3, 4)                   # 标准正态
zeros = torch.zeros(2, 3)
ones = torch.ones_like(random)

# 信息
random.shape, random.dtype, random.device

# 运算
tensor + 10
torch.matmul(A, B)        # 矩阵乘
A @ B                     # 等价

# Numpy 互转
np_arr = tensor.numpy()
tensor = torch.from_numpy(np_arr)

# 移到 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor.to(device)
```

- **MNIST 数据加载**

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/',
                          download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```

- **构建网络（class 写法）**

```python
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)             # 返回 logits
        return x
```

- **构建网络（Sequential 写法）**

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1),     # 配 NLLLoss 用
)
```

- **配 OrderedDict 给层取名字**

```python
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(784, 128)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(128, 64)),
    ('relu2', nn.ReLU()),
    ('output', nn.Linear(64, 10)),
]))
```

- **Loss 配对（两种等价方案）**

```python
# 方案 A：raw logits + CrossEntropyLoss（推荐）
model = nn.Sequential(..., nn.Linear(64, 10))   # 不加 softmax
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)

# 方案 B：LogSoftmax + NLLLoss（数学等价）
model = nn.Sequential(..., nn.Linear(64, 10), nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
loss = criterion(log_probs, labels)
```

- **Autograd**

```python
# 反向传播
loss.backward()
print(model[0].weight.grad)   # 拿到梯度

# 关闭梯度（推理时）
with torch.no_grad():
    preds = model(x_test)
```

- **完整训练循环**

```python
from torch import optim

model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10), nn.LogSoftmax(dim=1),
)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 20
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)   # flatten

        optimizer.zero_grad()                        # 清梯度
        logits = model(images)                       # forward
        loss = criterion(logits, labels)             # loss
        loss.backward()                              # backward
        optimizer.step()                             # update

        running_loss += loss.item()
    print(f"Epoch {epoch}, loss: {running_loss / len(trainloader)}")
```

- **Softmax 数值实现**

```python
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)

# ⚠️ torch.exp(7e2) → inf，所以不要直接 softmax 大值
# 实际用 nn.LogSoftmax 或 F.log_softmax，内部稳定
```

- **PyTorch 默认 float32**

```python
torch.tensor([3.0, 6.0, 9.0])                # float32
torch.tensor([3.0], dtype=torch.float16)     # float16
```
