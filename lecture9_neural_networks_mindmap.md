---
title: FRE-GY 7773 复习思维导图 — Lecture 9 Neural Networks & PyTorch
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 400
---

# Lecture 9 — Neural Networks & PyTorch

> Lecture 9 没有 slide PDF，内容来自 9 份 notebook（pytorch_intro / tensor / fundamentals / neural_networks / training）。

## ① 神经网络基本概念

- **从逻辑回归出发**
  - Logistic regression = 单层 NN（线性 + sigmoid 输出）
  - 神经网络 = 堆多层 + 加非线性激活 → 表达更复杂的函数
- **基本结构**
  - **输入层 (input)**：特征向量 $\mathbf{x} \in \mathbb{R}^d$
  - **隐藏层 (hidden)**：一层或多层，每层做 $\mathbf{a} = \phi(W\mathbf{x} + \mathbf{b})$
  - **输出层 (output)**
    - 回归：标量 / 向量
    - 二分类：1 个 logit
    - 多分类：$K$ 个 logits → softmax
- **每层的两步**
  - **线性变换**：$\mathbf{z} = W\mathbf{x} + \mathbf{b}$（仿射）
  - **非线性激活**：$\mathbf{a} = \phi(\mathbf{z})$（关键！没它整个网络等价于一个线性模型）
- **Universal Approximation Theorem**
  - 一个足够宽的单隐层网络（用合理的激活函数）可以**任意精度逼近**任何连续函数
  - 实践中"宽不如深"——多层让表达更高效
- **关键差异（vs 逻辑回归）**
  - Loss 一般**非凸** → 多个局部最优、初始化敏感、调参重要
  - 没有闭式解，必须**梯度下降**
  - 参数量大（万到百亿级别），需要 **反向传播 + GPU**

## ② Activation Functions

- **作用**：引入非线性，否则多层网络等价于一层
- **Sigmoid**：$\sigma(z) = 1/(1+e^{-z})$
  - 输出 $(0, 1)$，可解释为概率
  - **梯度消失**：$|z|$ 大时 $\sigma'(z) \to 0$，多层连乘梯度变 0
  - 输出**非零中心**（恒为正），影响优化
  - 现在几乎只用在二分类输出层
- **Tanh**：$\tanh(z) = (e^z - e^{-z})/(e^z + e^{-z})$
  - 输出 $(-1, 1)$，零中心
  - 仍有梯度消失问题
- **ReLU**：$\max(0, z)$
  - **最常用**
  - 简单、计算快
  - $z > 0$ 时梯度恒为 1 → 无梯度消失
  - 缺点：$z < 0$ 时梯度恒 0（**dying ReLU**：神经元"死掉"再也不更新）
- **Leaky ReLU**：$\max(\alpha z, z)$，$\alpha = 0.01$
  - 解决 dying ReLU
- **ELU / GELU**：更平滑的 ReLU 变种，深网络里有时优于 ReLU
- **Softmax**（多分类输出层）

$$\text{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

  - 把 logits 转成概率分布（和为 1）
  - **数值稳定性**：直接 $e^{x_i}$ 可能溢出 → 实现时减最大值；或用 LogSoftmax + NLLLoss 替代 Softmax + CrossEntropyLoss
- **怎么选**
  - 隐藏层默认 **ReLU**
  - 二分类输出 sigmoid（或直接输出 logit + BCEWithLogitsLoss）
  - 多分类输出 softmax（或直接输出 logit + CrossEntropyLoss）

## ③ Loss Functions

- **回归 → MSE**

$$L = \frac{1}{n}\sum_i (\hat y_i - y_i)^2$$

  - PyTorch：`nn.MSELoss`
- **二分类 → Binary Cross-Entropy (BCE) = NLL**

$$L = -\frac{1}{n}\sum_i [y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)]$$

  - `nn.BCELoss`：输入概率
  - `nn.BCEWithLogitsLoss`：输入 raw logits（**数值更稳，推荐**）
- **多分类 → Cross-Entropy**

$$L = -\frac{1}{n}\sum_i \log \hat p_{i, y_i}$$

  - `nn.CrossEntropyLoss`：内部已含 LogSoftmax + NLL，**输入是 raw logits（不是概率！）**
  - 等价方案：`nn.LogSoftmax(dim=1)` + `nn.NLLLoss`
- **常见错误**：CrossEntropyLoss 喂 softmax 后的概率 → 等于做了两次 softmax，loss 偏小且训练慢

## ④ PyTorch Tensors

- **类似 NumPy 但有梯度跟踪**
- **维度名词**
  - scalar (0-D) / vector (1-D) / matrix (2-D) / tensor (3D+)
- **创建**
  - `torch.tensor([1,2,3])`、`torch.zeros(3,4)`、`torch.ones`、`torch.randn`、`torch.arange`
- **形状操作**
  - `view` / `reshape`（reshape 更安全，view 要求连续内存）
  - `unsqueeze(dim)`：加一维 / `squeeze`：去掉长度 1 的维
  - `resize_`：原地改形状（`_` 后缀 = in-place）
- **运算**
  - 元素级：`+ - * /`
  - 矩阵乘：`@`、`torch.matmul`、`torch.mm`
  - 聚合：`.sum()`, `.mean()`, `.max()`, `.argmax()` 等
- **广播 (broadcasting)**：和 numpy 同规则
- **NumPy ↔ Torch**
  - `tensor.numpy()` / `torch.from_numpy(arr)`
  - **共享内存**：改一个另一个也变
- **Device (CPU/GPU)**
  - `tensor.to('cuda')` / `tensor.to(device)`
  - 不同 device 间运算会报错
- **Datatype**
  - 默认 `float32`
  - `torch.float16` / `torch.float64` / `torch.long` 等

## ⑤ 前向传播 (Forward Pass)

- **单层手算**：$\mathbf{a} = \phi(W\mathbf{x} + \mathbf{b})$
- **多层**：上一层 output → 下一层 input
- **完整例子**：MLP for MNIST
  - 输入：784 维（28×28 像素 flatten）
  - 隐藏层 1：784 → 128（ReLU）
  - 隐藏层 2：128 → 64（ReLU）
  - 输出层：64 → 10（10 类）
- **形状检查（关键习惯）**
  - 每层 $W$ 形状：$(\text{in}, \text{out})$
  - PyTorch `nn.Linear(in, out)` 的 weight 形状是 $(\text{out}, \text{in})$（注意倒过来）
  - 输出 = 输入 $@$ $W^\top$ + $b$

## ⑥ 用 nn.Module 构建网络

- **两种写法**
  - **class 写法**（继承 `nn.Module`）：灵活，能加自定义逻辑
  - **`nn.Sequential` 写法**：层简单堆叠时更短
- **class 写法的规矩**
  - 必须 `super().__init__()`
  - 在 `__init__` 里定义所有 layer（`self.fc1 = nn.Linear(...)`）
  - 在 `forward(x)` 里写前向计算
- **OrderedDict 给层命名**：调试时方便看权重 shape
- **`F.relu` vs `nn.ReLU()`**
  - `F.relu(x)` 是函数式调用
  - `nn.ReLU()` 是层（Module）
  - 行为一样，区别仅在用法

## ⑦ Autograd（自动求导）

- **核心思想**：PyTorch 在你做 forward 的同时，**记录计算图**（operation tree），随后 backward 自动算梯度
- **怎么打开梯度跟踪**
  - 模型参数自动 `requires_grad=True`
  - 自定义 tensor：`torch.tensor([1.0], requires_grad=True)`
- **反向传播**：`loss.backward()`
  - 沿计算图从 loss 倒推
  - 把所有需要梯度的 tensor 的 `.grad` 属性填上
- **梯度访问**：`param.grad`
- **关闭梯度（推理时省内存 + 加速）**
  - `with torch.no_grad():`
  - `tensor.detach()` 切断与计算图的连接
- **重要细节**
  - **每次 backward 前必须 `optimizer.zero_grad()`**——PyTorch 默认梯度**累加**
  - `loss.backward()` 默认只能调一次（计算图被释放）；如要多次：`backward(retain_graph=True)`

## ⑧ 反向传播 = 链式法则

- **目标**：算 $\partial L / \partial W$ 的每个分量
- **多层网络层叠的导数**：链式法则展开

$$\frac{\partial L}{\partial W^{(\ell)}} = \frac{\partial L}{\partial \mathbf{a}^{(\ell)}} \cdot \frac{\partial \mathbf{a}^{(\ell)}}{\partial \mathbf{z}^{(\ell)}} \cdot \frac{\partial \mathbf{z}^{(\ell)}}{\partial W^{(\ell)}}$$

- **核心三个量**
  - $\delta^{(\ell)} = \partial L / \partial \mathbf{z}^{(\ell)}$（"误差信号"）
  - 输出层：$\delta^{(L)} = \nabla_{a^{(L)}} L \odot \phi'(\mathbf{z}^{(L)})$
  - 反向递推：$\delta^{(\ell)} = (W^{(\ell+1)})^\top \delta^{(\ell+1)} \odot \phi'(\mathbf{z}^{(\ell)})$
  - 权重梯度：$\partial L / \partial W^{(\ell)} = \delta^{(\ell)}\, (\mathbf{a}^{(\ell-1)})^\top$
- **关键观察**
  - 复杂度和前向相当（每个参数只算一次）
  - 误差从输出层**反向传到**输入层
  - 这就是名字 "back-propagation" 的由来
- **PyTorch 帮你做这一切**：你只写 forward，autograd 自动算 backward

## ⑨ 优化器 (Optimizer)

- **基础：SGD with mini-batch**
  - $\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta \nabla L$
  - PyTorch：`optim.SGD(model.parameters(), lr=0.01)`
- **Momentum**：动量项加速沿稳定方向

$$v \leftarrow \mu v - \eta \nabla L; \quad \boldsymbol\theta \leftarrow \boldsymbol\theta + v$$

  - 像球滚下山有惯性
  - `optim.SGD(..., momentum=0.9)`
- **Adam**（最常用的现代优化器）
  - 结合 momentum + per-parameter 自适应学习率
  - 维护一阶矩 $m$ 和二阶矩 $v$
  - 实战默认首选：`optim.Adam(model.parameters(), lr=1e-3)`
- **AdamW**：Adam 的 weight decay 修正版（深度学习更推荐）
- **RMSProp**、**AdaGrad**：Adam 的祖先，理解原理用
- **Learning Rate Scheduler**
  - StepLR、CosineAnnealing 等
  - 训练后期降学习率，更精细收敛
  - `torch.optim.lr_scheduler`

## ⑩ 训练循环 (Training Loop)

- **三步循环**
  1. **forward**：算预测和 loss
  2. **backward**：`loss.backward()` 算梯度
  3. **update**：`optimizer.step()` 更新参数
- **每个 batch 必须 `optimizer.zero_grad()`**
- **Epoch vs Iteration**
  - 1 epoch = 数据集完整过一遍
  - 1 iteration = 一个 mini-batch 更新
- **训练流程**
  - 循环 epochs（通常 10-100）
    - 循环 mini-batches
      - forward → loss → zero_grad → backward → step
    - epoch 结束：算 train loss、val loss、accuracy 等指标
- **早停 (early stopping)**：val loss 连续若干 epoch 不下降就停

## ⑪ Dataset / DataLoader

- **`torch.utils.data.Dataset`**
  - 自定义类需实现 `__len__` 和 `__getitem__`
- **`torch.utils.data.DataLoader`**
  - 自动 batching、shuffle、并行加载
  - 关键参数：`batch_size`、`shuffle`、`num_workers`
- **torchvision.datasets**
  - 内置 MNIST、CIFAR、ImageNet 等
- **transforms.Compose**
  - 链式预处理：`ToTensor()`、`Normalize(mean, std)`、数据增强等

## ⑫ 推理 / 预测

- **eval 模式**：`model.eval()`
  - 关闭 dropout / batchnorm 的"训练时行为"
- **关闭梯度**：`with torch.no_grad():` 节省内存 + 加速
- **MNIST 例子**
  - logits = model(x)
  - probs = softmax(logits)
  - pred = probs.argmax(dim=1)

## ⑬ 初始化 (Weight Initialization)

- **为什么重要**：初始权重影响梯度幅度和收敛
- **常见方案**
  - **Xavier / Glorot**：用 sigmoid / tanh 时
    - $\text{Var}(W) = 2/(n_\text{in} + n_\text{out})$
  - **He / Kaiming**：用 ReLU 时（最常用）
    - $\text{Var}(W) = 2/n_\text{in}$
  - PyTorch `nn.Linear` 默认是 Kaiming uniform
- **Bias 通常初始化为 0**
- **不能全初始化为 0** → 所有 neuron 一样，对称性永远打不破

## ⑭ 正则化 (Regularization)

- **L2 (Weight Decay)**
  - 在 loss 里加 $\lambda \|\boldsymbol\theta\|^2$
  - PyTorch：`optim.SGD(..., weight_decay=1e-4)`
- **Dropout**
  - 训练时随机把一些激活值设 0（概率 $p$）
  - 防止 co-adaptation（神经元过度依赖）
  - `nn.Dropout(p=0.5)`，eval 模式自动关闭
- **Batch Normalization**
  - 每层归一化输入到 mean 0 / std 1
  - `nn.BatchNorm1d / 2d`
  - 加速训练 + 一定正则化作用
- **Early Stopping**：val loss 不再下降就停
- **Data Augmentation**：图像旋转、翻转、裁剪等

## ⑮ 梯度消失 / 梯度爆炸

- **梯度消失 (vanishing)**
  - 反向传播时梯度连乘多个 < 1 的数 → 趋近 0 → 浅层学不动
  - 经典原因：sigmoid / tanh 在饱和区导数极小
  - 解法：ReLU、合适初始化、BatchNorm、残差连接 (ResNet)
- **梯度爆炸 (exploding)**
  - 梯度连乘多个 > 1 的数 → 爆炸 → NaN
  - 经典场景：RNN 长序列
  - 解法：gradient clipping (`torch.nn.utils.clip_grad_norm_`)

## ⑯ 常见坑 & 易错点

- 忘 `optimizer.zero_grad()` → 梯度累积，训练崩
- 训练 / 评估模式没切换（`model.train()` vs `model.eval()`）
- 数据没标准化 → 训练不稳，loss 卡住
- CrossEntropyLoss 喂了 softmax 后的概率 → 等于做了两次 softmax
- 学习率太大 → loss 飞掉 / NaN；太小 → 训练慢
- shape 不匹配（如 `(64, 1, 28, 28)` 没 flatten 成 `(64, 784)` 直接喂 Linear）
- 模型 / 数据不在同一 device（CPU vs GPU）
- 过拟合：train loss 一路下降，val loss 反弹
- 类别不平衡 → 用 `class_weight` / focal loss

## ⑰ 与之前学过内容的连接

- **逻辑回归** = 单层 NN（线性 + sigmoid）
- **梯度下降 / SGD / mini-batch** = NN 训练核心算法（L6）
- **MLE / NLL** = 分类的 cross-entropy loss（L2 + L5）
- **正则化** L1/L2 思想直接迁移（L4）
- **convex 不再保证**：NN loss 一般非凸，多个局部最优
- **更广义的"机器学习是优化"**：所有模型都在做 $\min L(\boldsymbol\theta)$

## ⑱ 💻 代码模板（来自 notebooks）

- **Tensor 基础**

```python
import torch

# 创建
scalar = torch.tensor(7)                    # 0-D
vector = torch.tensor([1, 2, 3])            # 1-D
M = torch.tensor([[1, 2], [3, 4]])          # 2-D
random = torch.randn(3, 4)                  # 标准正态
zeros = torch.zeros(2, 3)
ones = torch.ones_like(random)

# 信息
random.shape, random.dtype, random.device

# 运算
tensor + 10
torch.matmul(A, B)                           # 矩阵乘
A @ B                                        # 等价

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
    nn.LogSoftmax(dim=1),     # 配 NLLLoss
)
```

- **Loss 配对**

```python
# 方案 A：raw logits + CrossEntropyLoss（推荐）
model = nn.Sequential(..., nn.Linear(64, 10))   # 不加 softmax
criterion = nn.CrossEntropyLoss()

# 方案 B：LogSoftmax + NLLLoss（数学等价）
model = nn.Sequential(..., nn.Linear(64, 10), nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
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

        optimizer.zero_grad()                        # 1. 清梯度
        logits = model(images)                       # 2. forward
        loss = criterion(logits, labels)             # 3. 算 loss
        loss.backward()                              # 4. backward
        optimizer.step()                             # 5. update

        running_loss += loss.item()
    print(f"Epoch {epoch}, loss: {running_loss / len(trainloader)}")
```

- **推理 / 预测**

```python
model.eval()
with torch.no_grad():
    logits = model(x_test)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
```

- **Autograd 演示**

```python
# 反向传播
loss.backward()
print(model[0].weight.grad)   # 拿到梯度

# 关闭梯度
with torch.no_grad():
    preds = model(x_test)
```

- **Adam 优化器**

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

- **学习率调度**

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(epochs):
    train(...)
    scheduler.step()
```
