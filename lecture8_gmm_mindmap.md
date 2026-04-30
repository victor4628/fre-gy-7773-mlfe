---
title: FRE-GY 7773 复习思维导图 — Lecture 8 GMM
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 380
---

# Lecture 8 — Gaussian Mixture Models

## ① 动机：单个 Gaussian 不够用

- **Old Faithful 例子**：黄石公园喷泉数据
  - 单个 Gaussian 拟合很差（数据明显双峰）
  - 两个 Gaussian 的组合拟合得多
- **核心思想**：**线性组合多个 Gaussian** 来逼近复杂分布
- **理论保证**：足够多 Gaussian 可以以任意精度逼近**任何连续密度**

## ② GMM 的定义

- **公式**

$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k\, \mathcal{N}(\mathbf{x} \mid \boldsymbol\mu_k, \Sigma_k)$$

- **三组参数**
  - $\boldsymbol\mu_k$：第 $k$ 个 component 的均值
  - $\Sigma_k$：第 $k$ 个 component 的协方差
  - $\pi_k$：**mixing coefficients (混合系数)**
- **约束**（让 $p(\mathbf{x})$ 是合法分布）
  - $\pi_k \geq 0$
  - $\sum_k \pi_k = 1$
- **概率含义**
  - $\pi_k = p(k)$：选第 $k$ 个 component 的**先验概率**
  - $\mathcal{N}(\mathbf{x}\mid\boldsymbol\mu_k, \Sigma_k) = p(\mathbf{x}\mid k)$：选 $k$ 后从 $\mathbf{x}$ 的条件分布

## ③ 采样过程

- **两步采样**
  1. 按 $\{\pi_1, \ldots, \pi_K\}$ 的概率选一个 component $k$
  2. 从 $\mathcal{N}(\boldsymbol\mu_k, \Sigma_k)$ 抽 $\mathbf{x}$
- 对应到生成模型的视角

## ④ Responsibility（后验概率）

- **由 Bayes 定理**

$$\gamma_k(\mathbf{x}) = p(k \mid \mathbf{x}) = \frac{\pi_k \mathcal{N}(\mathbf{x}\mid\boldsymbol\mu_k, \Sigma_k)}{\sum_l \pi_l \mathcal{N}(\mathbf{x}\mid\boldsymbol\mu_l, \Sigma_l)}$$

- **含义**：观测到 $\mathbf{x}$ 后，**component $k$ "解释"该样本的责任比例**
- **特点**
  - $\gamma_k(\mathbf{x}) \in [0, 1]$
  - $\sum_k \gamma_k(\mathbf{x}) = 1$
  - **soft assignment**（软分配）：每个样本部分属于多个 component

## ⑤ 离散隐变量表述

- **1-of-$K$ 编码**：$\mathbf{z} \in \{0, 1\}^K$，恰好一个分量为 1
  - $z_k \in \{0, 1\}$，$\sum_k z_k = 1$
- **联合分布**

$$p(\mathbf{x}, \mathbf{z}) = p(\mathbf{z}) p(\mathbf{x}\mid\mathbf{z})$$

  - $p(\mathbf{z}) = \prod_k \pi_k^{z_k}$
  - $p(\mathbf{x}\mid\mathbf{z}) = \prod_k \mathcal{N}(\mathbf{x}\mid\boldsymbol\mu_k, \Sigma_k)^{z_k}$
- **边缘分布回到 GMM**

$$p(\mathbf{x}) = \sum_\mathbf{z} p(\mathbf{z})p(\mathbf{x}\mid\mathbf{z}) = \sum_k \pi_k \mathcal{N}(\mathbf{x}\mid\boldsymbol\mu_k, \Sigma_k)$$

- **意义**：每个观测 $\mathbf{x}_n$ 都对应一个隐变量 $\mathbf{z}_n$（不可观测，"它属于哪个 cluster"）
- **隐变量公式 → EM 算法的基础**

## ⑥ MLE 难点：为什么需要 EM

- **Log-likelihood**

$$\ln p(\mathbf{X} \mid \boldsymbol\pi, \boldsymbol\mu, \Sigma) = \sum_{n=1}^N \ln\!\left\{\sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k, \Sigma_k)\right\}$$

- **关键问题**：**log 里面有 sum**（不是 sum 里面有 log）
  - 求导后无法解析求解
  - **没有闭式解**
- **可选方案**：梯度法（可以但慢且复杂）
- **更好方案**：**Expectation-Maximization (EM) 算法**

## ⑦ EM 推导：参数的一阶条件

- **对 $\boldsymbol\mu_k$ 求导**得到（用 responsibility 定义）

$$\boldsymbol\mu_k = \frac{1}{N_k}\sum_{n=1}^N \gamma(z_{nk})\mathbf{x}_n,\quad N_k := \sum_n \gamma(z_{nk})$$

  - **直觉**：$\boldsymbol\mu_k$ = **加权样本均值**，权重 = 各样本被 component $k$ "认领"的比例
  - $N_k$ = "有效样本数"（被 component $k$ "拥有"的份额总和）
- **对 $\Sigma_k$ 求导**

$$\Sigma_k = \frac{1}{N_k}\sum_n \gamma(z_{nk})(\mathbf{x}_n - \boldsymbol\mu_k)(\mathbf{x}_n - \boldsymbol\mu_k)^\top$$

  - 同样是**加权样本协方差**
- **对 $\pi_k$ 求导**（约束 $\sum \pi = 1$，用 Lagrangian）

$$\pi_k = \frac{N_k}{N}$$

  - $\pi_k$ = component $k$ 平均承担的责任比例
- **关键观察**：这三个公式**不是闭式解**——因为 $\boldsymbol\mu_k, \Sigma_k, \pi_k$ 出现在 $\gamma$ 里面（循环依赖）

## ⑧ EM 算法（迭代解法）

- **整体思路**：固定一边算另一边，轮流来（类似 Lloyd）
- **Step 1：初始化** $\boldsymbol\mu_k, \Sigma_k, \pi_k$，算初始 log-likelihood
- **Step 2 (E-step)**：用当前参数算 responsibility

$$\gamma(z_{nk}) = \frac{\pi_k \mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_j, \Sigma_j)}$$

- **Step 3 (M-step)**：用当前 responsibility 重估参数

$$\boldsymbol\mu_k^\text{new} = \frac{1}{N_k}\sum_n \gamma(z_{nk})\mathbf{x}_n$$

$$\Sigma_k^\text{new} = \frac{1}{N_k}\sum_n \gamma(z_{nk})(\mathbf{x}_n - \boldsymbol\mu_k^\text{new})(\mathbf{x}_n - \boldsymbol\mu_k^\text{new})^\top$$

$$\pi_k^\text{new} = \frac{N_k}{N}$$

- **Step 4**：算新的 log-likelihood，检查收敛；没收敛回 Step 2

## ⑨ EM 重要性质

- **GMM log-likelihood 非凸**
  - 多个局部最优
  - EM 不保证全局最优
  - 但**单调上升收敛到局部最优**
- **常见技巧**：用 K-means 初始化 EM
  - K-means 算出 cluster → 各 component 的 $\boldsymbol\mu_k$ = K-means 中心
  - $\Sigma_k$ = 各 cluster 的样本协方差
  - $\pi_k$ = 各 cluster 的样本比例
  - 加快收敛，减少坏局部最优概率

## ⑩ K-means vs GMM

- **K-means**
  - hard assignment（每个点属于一个簇）
  - 用欧氏距离
  - 隐含假设：球形等大 cluster
- **GMM**
  - **soft assignment**（每个点按 responsibility 属于多个 component）
  - 用 Mahalanobis 距离（基于 $\Sigma_k$）
  - 各 component 可以有不同形状（椭圆、不同大小）
  - 提供**概率密度**（不只 cluster 标签）
- **K-means 是 GMM 的特例**：若所有 $\Sigma_k = \sigma^2 I$ 且 $\sigma \to 0$，GMM 退化为 K-means
- **代价对比**
  - K-means 一次迭代便宜，GMM 一次更贵
  - GMM 需要更多迭代收敛
  - GMM 对初始化更敏感

## ⑪ 实操要点

- **选 $K$**
  - BIC / AIC（GMM 是概率模型，可以用信息准则）
  - Cross-validation log-likelihood
  - elbow on log-likelihood
- **协方差类型**（sklearn `GaussianMixture(covariance_type=...)`）
  - `'full'`：每 component 一个完整 $\Sigma_k$
  - `'tied'`：所有 component 共享一个 $\Sigma$
  - `'diag'`：对角协方差
  - `'spherical'`：标量乘 $I$（等价 K-means 的几何）
- **退化问题**：某 component 单独"抓住"一个点 → $\Sigma_k \to 0$ → likelihood 爆炸
  - 解决：协方差正则化、限制最小特征值
- **概率应用**
  - 异常检测：低 $p(\mathbf{x})$ 的样本
  - 软聚类：拿 responsibility 作为类别"概率"
  - 密度估计：作为 generative model

## ⑫ 💻 代码模板（来自 notebooks）

- **基础 GMM 拟合**

```python
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=2, random_state=0).fit(x_data)
gm.means_              # 每 component 的 μ_k
gm.covariances_        # 每 component 的 Σ_k
gm.weights_            # 每 component 的 π_k
gm.predict(X_new)      # hard assignment（最大 responsibility 的 k）
gm.predict_proba(X)    # responsibility γ(z_nk) 矩阵
gm.score_samples(X)    # log p(x_n) 每点对数似然
```

- **从 GMM 采样**

```python
samples, labels = gm.sample(n_samples=1000)
```

- **手算 EM 主循环（教学版）**

```python
# E-step: 算 responsibility
for n in range(N):
    for k in range(K):
        gamma[n, k] = pi[k] * multivariate_normal.pdf(X[n], mu[k], Sigma[k])
gamma /= gamma.sum(axis=1, keepdims=True)

# M-step: 重估参数
N_k = gamma.sum(axis=0)
mu = (gamma.T @ X) / N_k[:, None]
for k in range(K):
    diff = X - mu[k]
    Sigma[k] = (gamma[:, k, None] * diff).T @ diff / N_k[k]
pi = N_k / N
```

- **多元 Gaussian 闭式拟合（单 component）**

```python
from scipy.stats import multivariate_normal
mean_estim, cov_estim = multivariate_normal.fit(data)
```

- **协方差类型选择**

```python
GaussianMixture(n_components=K, covariance_type='full')        # 完整 Σ_k
GaussianMixture(n_components=K, covariance_type='tied')        # 共享一个 Σ
GaussianMixture(n_components=K, covariance_type='diag')        # 对角
GaussianMixture(n_components=K, covariance_type='spherical')   # σ²I（≈ K-means）
```

- **选 K：BIC / AIC**

```python
bics = [GaussianMixture(n_components=k, random_state=0).fit(X).bic(X)
        for k in range(1, 10)]
best_k = np.argmin(bics) + 1
```

- **K-means 初始化 GMM（加速收敛）**

```python
GaussianMixture(n_components=K, init_params='kmeans', n_init=5)
```
