---
title: FRE-GY 7773 复习思维导图 — Lecture 6 Optimization
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 380
---

# Lecture 6 — Optimization for ML

## ① 为什么需要优化算法

- **逻辑回归 NLL 没有闭式解**（不像 OLS 有 normal equation）
- **必须用数值方法**找

$$\hat{\boldsymbol\beta} \in \arg\min_{\boldsymbol\beta \in \mathbb{R}^d} f(\boldsymbol\beta)$$

- **目标函数**：

$$f(\boldsymbol\beta) = \frac{1}{n}\sum_i \log(1 + \exp(-(2y_i - 1)\mathbf{x}_i^\top \boldsymbol\beta))$$

## ② 朴素策略为什么失败

- **想法**：在 $[0,1]^d$ 上撒网格，逐点求值，挑最小
- **问题：维度灾难**
  - 步长 $\varepsilon$ 在 $d$ 维网格 → $(1/\varepsilon)^d$ 个点
  - $d = 100$，$\varepsilon = 0.1$ → $10^{100}$ 次评估
  - 每次 $0.001$ 秒 → 总共 $10^{97}$ 秒（宇宙年龄都不够）
- **额外问题**：不保证靠近全局最优
- **结论**：必须用更聪明的方法 → 梯度下降

## ③ 凸性 & 平滑性（GD 收敛保证）

- **NLL 关键性质**
- **梯度**

$$\nabla f(\boldsymbol\beta) = \frac{1}{n}\sum_i (\sigma(\mathbf{x}_i^\top \boldsymbol\beta) - y_i)\mathbf{x}_i$$

- **Hessian**

$$\nabla^2 f(\boldsymbol\beta) = \frac{1}{n}\sum_i \sigma(\mathbf{x}_i^\top \boldsymbol\beta)(1-\sigma(\mathbf{x}_i^\top \boldsymbol\beta))\mathbf{x}_i \mathbf{x}_i^\top$$

- **Convexity**：$\nabla^2 f \succeq 0$
  - 标量 $\sigma(1-\sigma) \geq 0$
  - 外积 $\mathbf{x}_i \mathbf{x}_i^\top \succeq 0$
  - 非负标量 × PSD = PSD
  - **关键意义**：任何局部最优 = 全局最优
- **Smoothness ($L$-smooth)**
  - 关键不等式：$\sigma(t)(1-\sigma(t)) \leq 1/4$
  - $\nabla^2 f \preceq \frac{1}{4n}\sum_i \mathbf{x}_i \mathbf{x}_i^\top$
  - $L = \frac{1}{4n}\lambda_\max(\sum \mathbf{x}_i \mathbf{x}_i^\top)$
- **$L$-smooth 的等价定义**
  - $\nabla f$ 是 $L$-Lipschitz：$\|\nabla f(w) - \nabla f(w')\|_2 \leq L \|w - w'\|_2$
  - 二阶可导时等价于 $\lambda_\max(\nabla^2 f(w)) \leq L$

## ④ Descent Lemma（梯度下降的核心）

- **若 $f$ 是 $L$-smooth**，则对任意 $\boldsymbol\beta, \boldsymbol\beta'$

$$f(\boldsymbol\beta') \leq f(\boldsymbol\beta) + \langle \nabla f(\boldsymbol\beta), \boldsymbol\beta' - \boldsymbol\beta \rangle + \frac{L}{2}\|\boldsymbol\beta - \boldsymbol\beta'\|_2^2$$

- **直觉**：用一个**二次上界**把 $f$ 罩住
- **怎么用**：在迭代点 $\boldsymbol\beta^{(k)}$ 处，最小化这个上界 → 得到下一个点
  - 上界关于 $\boldsymbol\beta'$ 的最小点是

$$\boldsymbol\beta^{(k+1)} = \boldsymbol\beta^{(k)} - \frac{1}{L}\nabla f(\boldsymbol\beta^{(k)})$$

  - 这正是梯度下降的更新公式

## ⑤ (Batch) Gradient Descent

- **算法**
  - **输入**：起点 $\boldsymbol\beta^{(0)}$，步长 $\eta > 0$
  - **迭代**：$\boldsymbol\beta^{(k+1)} = \boldsymbol\beta^{(k)} - \eta \nabla f(\boldsymbol\beta^{(k)})$
  - **收敛后输出**最终点
- **步长选择**
  - 安全选择：$\eta = 1/L$
  - $\eta$ 太大 → 不收敛 / 振荡
  - $\eta$ 太小 → 收敛慢
- **代价**：每步计算 $\nabla f$ 需要 $O(nd)$（扫遍所有样本）
- **收敛性**
  - $f$ convex + $L$-smooth：$O(1/k)$ 收敛
  - $f$ 强凸：$O(\rho^k)$ 指数收敛

## ⑥ Stochastic Gradient Descent (SGD)

- **观察**：NLL 是 $n$ 项之和

$$f(\boldsymbol\beta) = \frac{1}{n}\sum_{i=1}^n f_i(\boldsymbol\beta)$$

  - 每个 $f_i$ 只依赖 $(\mathbf{x}_i, y_i)$
- **核心思想**：用**一个随机样本**的梯度近似全梯度
- **算法**
  - **输入**：起点 $\boldsymbol\beta^{(0)}$，步长 $\eta > 0$
  - **每步**：随机选一个 $i \in \{1, \ldots, n\}$
  - **更新**：

$$\boldsymbol\beta^{(k+1)} = \boldsymbol\beta^{(k)} - \frac{\eta}{\sqrt{k+1}}\nabla f_i(\boldsymbol\beta^{(k)})$$

  - **步长衰减** $\eta / \sqrt{k+1}$ 是为了收敛性（噪声不退也行不通）
- **代价**：每步 $O(d)$（vs. GD 的 $O(nd)$）→ 大数据集快得多
- **代价 vs 噪声 trade-off**
  - GD：方向准但慢
  - SGD：方向噪声大但便宜，需要更多步数
  - 实际收敛轨迹**蛇形**摆动

## ⑦ Mini-batch Gradient Descent

- **介于 GD 和 SGD 之间**
- **每步用一个 mini-batch**（大小 $b$，比如 $32, 64, 128$）

$$\boldsymbol\beta^{(k+1)} = \boldsymbol\beta^{(k)} - \eta \cdot \frac{1}{b}\sum_{i \in B_k}\nabla f_i(\boldsymbol\beta^{(k)})$$

- **优势**
  - 比 SGD 噪声小（梯度更稳）
  - 比 GD 便宜（不扫全量）
  - **可以利用矩阵化运算**（GPU 友好）
- **现代深度学习几乎都用 mini-batch SGD（或其变种）**

## ⑧ 三种 GD 路径对比

- **Batch GD**：直接朝最优走，路径平滑
- **SGD**：之字形走，最终也能收敛
- **Mini-batch**：介于两者之间，路径有抖动但比 SGD 稳

## ⑨ 应用到逻辑回归（套用 NLL 梯度）

- **梯度**：$\nabla f = \frac{1}{n}\sum_i (\sigma(\mathbf{x}_i^\top\boldsymbol\beta) - y_i)\mathbf{x}_i$
- **Batch GD**：每步用全部 $n$ 个样本算梯度
- **SGD**：每步随机选一个样本，只算 $(\sigma(\mathbf{x}_i^\top\boldsymbol\beta) - y_i)\mathbf{x}_i$
- **Mini-batch**：每步随机选 $b$ 个样本

## ⑩ 实操要点

- **特征标准化**：让所有 feature 同尺度，否则 Hessian 条件数大、收敛慢
- **学习率调度**：固定 $\eta$ / step decay / cosine decay / Adam 等自适应方法
- **Momentum**：加上动量项，加速沿着稳定方向
- **现代 SGD 变种**
  - Adam (Adaptive Moment Estimation)
  - RMSProp
  - AdaGrad
  - 都基于历史梯度信息自适应调整学习率

## ⑪ 💻 代码模板（来自 notebooks）

- **Batch Gradient Descent（线性回归）**

```python
eta = 1e-1
n_epochs = 1000
m = len(X_b)

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal(2)

for n in range(n_epochs):
    gradients = (1 / m) * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
```

- **Stochastic Gradient Descent**

```python
n_epochs = 50
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)        # 学习率随 t 衰减

theta = rng.standard_normal((2, 1))
for epoch in range(n_epochs):
    for iteration in range(m):
        i = rng.integers(m)
        xi, yi = X_b[i:i+1], y[i:i+1]
        gradients = xi.T @ (xi @ theta - yi)   # 单样本梯度，不除 m
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients
```

- **Mini-batch GD**

```python
n_epochs = 50
batch_size = 20
n_batches = m // batch_size

for epoch in range(n_epochs):
    # 每个 epoch 重新打乱
    shuffled = rng.permutation(m)
    X_shuf, y_shuf = X_b[shuffled], y[shuffled]
    for k in range(n_batches):
        idx = slice(k * batch_size, (k + 1) * batch_size)
        xb, yb = X_shuf[idx], y_shuf[idx]
        gradients = (2 / batch_size) * xb.T @ (xb @ theta - yb)
        eta = learning_schedule(epoch * n_batches + k)
        theta = theta - eta * gradients
```

- **三种 GD 路径可视化（看你给的图）**

```python
fig, ax = plt.subplots()
ax.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], '.-b', label='Batch')   # 平滑直线
ax.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], '*-r', label='SGD')     # 之字形
ax.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], 'x-g', label='Mini-batch')  # 中间
```

- **关键观察**
  - Batch：步数少但每步贵，路径平滑
  - SGD：每步便宜但需要更多步、有抖动
  - Mini-batch：兼顾两者
  - 学习率衰减 schedule（如 $t_0 / (t + t_1)$）对 SGD 收敛重要
