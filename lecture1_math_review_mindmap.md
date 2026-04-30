---
title: FRE-GY 7773 复习思维导图 — Lecture 1 Math Review
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 380
---

# Lecture 1 — Intro & Math Review

## ① 课程概览 & ML 类型

- **ML 定义**：算法从数据中学习规律 → 预测 / 决策（无需显式编程）
- **金融场景**：高维 + 噪声大 + 非线性 → ML 优势明显
- **学习类型**
  - **Supervised**：有 (xᵢ, yᵢ) 标签 → 分类（违约/欺诈）+ 回归（收益/波动率/期权价）
  - **Unsupervised**：无标签 → 聚类（市场状态/客户分群）+ 降维（yield curve / 风险因子）+ 异常检测
  - **Semi-supervised**：少量标签 + 大量无标签 → 信用评分、ESG 分类
  - **Reinforcement**：agent 与环境交互拿奖励 → 最优执行、做市、动态调仓
  - **Self-supervised**：从无标签数据自生成标签 → 图像 mask、时间序列预训练、文本预训练
- **训练范式**
  - Batch vs Online：金融数据非平稳 → 倾向 online + regime detection + 持续重训
  - Instance-based (kNN) vs Model-based（回归 / 神经网络）

## ② 矩阵代数基础

- **基本恒等式**
  - $(AB)^\top = B^\top A^\top$
  - $(AB)^{-1} = B^{-1}A^{-1}$
  - $(A^\top)^{-1} = (A^{-1})^\top$
- **正交矩阵 (orthogonal matrix)**
  - 必须方阵
  - $Q^\top Q = QQ^\top = I$
  - $Q^{-1} = Q^\top$（逆 = 转置）
  - 列两两正交 + 单位长度
  - 几何意义：旋转 / 反射，保长度保角度
  - 长方形 + 列正交 → 只 $Q^\top Q = I$，叫 semi-orthogonal
- **Trace（迹）**
  - $\mathrm{tr}(A) = \sum_i A_{ii}$（对角线之和）
  - 循环性：$\mathrm{tr}(ABC) = \mathrm{tr}(CAB) = \mathrm{tr}(BCA)$
- **Determinant（行列式）**
  - $|AB| = |A||B|$
  - $|A^{-1}| = 1/|A|$
  - 2×2：$|A| = a_{11}a_{22} - a_{12}a_{21}$
- **Woodbury identity**（大矩阵求逆的小技巧）
  - $(A + BD^{-1}C)^{-1} = A^{-1} - A^{-1}B(D + CA^{-1}B)^{-1}CA^{-1}$
  - $A$ 大且对角时特别有用

## ③ 矩阵求导

- **标量对向量求导 → 向量**：$(\partial x / \partial \mathbf{a})_i = \partial x / \partial a_i$
- **核心公式**
  - **Affine**：$\dfrac{\partial}{\partial x}(x^\top a + b) = a$
  - **Quadratic**：$\dfrac{\partial}{\partial x}(x^\top A x) = (A + A^\top) x$
  - **对称 $A$**：$\dfrac{\partial}{\partial x}(x^\top A x) = 2Ax$
- ML 里几乎所有梯度推导都靠这两条 + 链式法则

## ④ Eigenvector & 对角化

- **特征值方程**
  - $A u_i = \lambda_i u_i$
  - 求解通过 $\det(A - \lambda I) = 0$（characteristic equation）
  - $M \times M$ 方阵有 $M$ 个特征值（按重数算）
- **直观**：特征向量是被 $A$ 作用后**只伸缩、不转向**的方向
- **对称矩阵的好性质**
  - 特征值全是实数
  - 特征向量两两正交（可选 orthonormal）
  - 一定有 $M$ 个线性无关的特征向量
- **数量与对应**
  - 一个 eigenvector → 一个 eigenvalue ✅
  - 一个 eigenvalue → 可能多个 eigenvector（重特征值时）
- **A 必须是方阵**（否则 $Av$ 形状对不上 $v$）；非方阵用 SVD
- **对角化**
  - $A U = U \Lambda$，$\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_M)$
  - $A = U\Lambda U^\top$，$A^{-1} = U\Lambda^{-1} U^\top$
  - 谱分解：$A = \sum_i \lambda_i u_i u_i^\top$（rank-1 矩阵之和）
- **重要等式**
  - $\det(A) = \prod_i \lambda_i$
  - $\mathrm{tr}(A) = \sum_i \lambda_i$
- **rank-1 矩阵**
  - $u u^\top$ 形式的矩阵
  - 所有列都是 $u$ 的倍数
  - 谱分解 + 协方差矩阵都是 rank-1 之和

## ⑤ SVD（奇异值分解）

- **定义**：任何 $M \times N$ 矩阵都可写成

$$X = U \Sigma V^\top$$

  - $U$（$M \times M$）正交：left singular vectors
  - $V$（$N \times N$）正交：right singular vectors
  - $\Sigma$（$M \times N$）对角：奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- **rank-1 形式**
  - $X = \sum_{i=1}^r \sigma_i u_i v_i^\top$
  - 截断 SVD：只取前 $k$ 项 → 低秩近似
- **和特征分解的连接**

$$X^\top X = V \Sigma^2 V^\top,\quad XX^\top = U \Sigma^2 U^\top$$

  - $V$ 列 = $X^\top X$ 的特征向量
  - $U$ 列 = $XX^\top$ 的特征向量
  - 特征值 = 奇异值的平方 $\sigma_i^2$
- **应用**：PCA、低秩近似、矩阵伪逆、图像压缩

## ⑥ 概率基础

- **概率空间** $(\Omega, \mathcal{F}, p)$
- **乘法公式**：$p(X, Y) = p(Y \mid X) p(X)$
- **全概率公式**（$\{Y_i\}$ 是 $\Omega$ 的划分）

$$p(X) = \sum_i p(X \mid Y_i)\, p(Y_i)$$

- **Bayes 定理**

$$p(Y_i \mid X) = \frac{p(X \mid Y_i) p(Y_i)}{\sum_j p(X \mid Y_j) p(Y_j)}$$

- **PDF 版本**（连续型 RV）

$$p(y \mid x) = \frac{p(x \mid y) p(y)}{\int p(x \mid y) p(y)\, dy}$$

## ⑦ Gaussian 分布

- **单变量**

$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

  - $E[x] = \mu$，$\mathrm{Var}(x) = \sigma^2$
- **多元 (MVG)**

$$\mathcal{N}(\mathbf{x} \mid \boldsymbol\mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol\mu)^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\right)$$

  - 协方差矩阵 $\Sigma$ 对称 + **半正定 (PSD)**
  - 密度有效要求 $\Sigma$ **正定**（特征值严格 > 0）
- **PSD 提醒**：$A$ PSD ⟺ $x^\top A x \geq 0\ \forall x$ ⟺ 特征值都 ≥ 0
- **二次型 = Mahalanobis 距离**
  - $\Delta^2(\mathbf{x}) = (\mathbf{x}-\boldsymbol\mu)^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)$
  - 推广了一维的 z-score $(x-\mu)/\sigma$
  - 等高线是椭球，主轴 = $\Sigma$ 的特征向量
- **MVG 的边缘 & 条件分布**
  - 边缘：$\mathbf{x}_1 \sim \mathcal{N}(\boldsymbol\mu_1, \Sigma_{11})$
  - 条件：$\mathbf{x}_1 \mid \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol\mu_{1\mid 2}, \Sigma_{11 \mid 2})$
    - $\boldsymbol\mu_{1\mid 2} = \boldsymbol\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2 - \boldsymbol\mu_2)$
    - $\Sigma_{11\mid 2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{12}^\top$
  - $\Sigma_{12} = 0$（独立）→ 条件分布 = 边缘分布
- **仿射变换**：$\mathbf{x} \sim \mathcal{N}(\boldsymbol\mu, \Sigma)$，$\mathbf{y} = A\mathbf{x} + \mathbf{b}$ → $\mathbf{y} \sim \mathcal{N}(A\boldsymbol\mu + \mathbf{b}, A\Sigma A^\top)$
- **卷积**：独立 Gaussian 之和仍 Gaussian，参数为均值、协方差之和
- **导出分布**
  - $\chi^2_n$：$n$ 个 i.i.d. 标准正态平方之和
  - $t_n$：$x / \sqrt{y/n}$，其中 $x \sim \mathcal{N}(0,1)$，$y \sim \chi^2_n$
  - 命题：$A$ 对称且幂等 + $\mathbf{x} \sim \mathcal{N}(0, I)$ → $\mathbf{x}^\top A \mathbf{x} \sim \chi^2_{\mathrm{tr}(A)}$

## ⑧ MLE（Maximum Likelihood Estimation）

- **定义**：观测 $\{x^{(i)}\}_{i=1}^n$ i.i.d.，密度 $p(x \mid \theta)$
  - **Likelihood**：$\mathcal{L}(\theta) = \prod_i p(x^{(i)} \mid \theta)$
  - **MLE**：$\hat\theta_\text{MLE} = \arg\max_\theta \mathcal{L}(\theta)$
- **常用对数似然**：$\ell(\theta) = \log \mathcal{L}(\theta)$，把乘积变求和
- **Gaussian MLE 闭式解**
  - $\hat\mu_\text{MLE} = \frac{1}{n}\sum_i x^{(i)}$（样本均值）
  - $\hat\sigma^2_\text{MLE} = \frac{1}{n}\sum_i (x^{(i)} - \hat\mu)^2$（样本方差，**注意分母是 $n$ 不是 $n-1$**）
- **对数恒等式**（推导常用）
  - $\ln(ab) = \ln a + \ln b$
  - $\ln(a^b) = b \ln a$（所以 $\ln \sigma^2 = 2\ln\sigma$）
- **MLE vs OLS**（概念区分）
  - MLE 是**通用原则**：最大化似然，需要假设分布
  - OLS 是**具体方法**：最小化平方误差，不需要分布假设
  - 高斯噪声下两者**等价**

## ⑨ 💻 代码模板（来自 notebooks）

- **Numpy 工具**（`01_tools_numpy.ipynb`）
  - 数组创建：`np.zeros / ones / full / arange / linspace / random`
  - 形状：`reshape / ravel`
  - 广播规则、条件运算、统计函数
  - `import numpy.linalg as linalg`
- **Pandas 工具**（`01_tools_pandas.ipynb`）
  - `Series` / `DataFrame` 基础
  - 时间序列：`time range / resample / timezone / period`
  - 多重索引、stack/unstack、query、sort
- **Matplotlib 工具**（`01_tools_matplotlib.ipynb`）
  - line / scatter / hist / 3D / polar
  - subplots、legend、ticks
- **SVD 实战**（`01_linalg.ipynb`）

```python
U, S, V_T = np.linalg.svd(img_array)  # 图像低秩近似 demo
```

- **概率统计**（`01_probability_statistics.ipynb`）

```python
from scipy import stats
stats.norm.fit(x_samples, method="MLE")   # MLE 估计 mu, sigma
stats.norm.fit(x_samples, method="MM")    # 矩估计
```
