---
title: FRE-GY 7773 复习思维导图 — Lecture 3 Linear Regression
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 380
---

# Lecture 3 — Linear Regression / OLS

## ① 任务定义：回归

- **设定**：$\mathbf{X} \in \mathbb{R}^p$ 是特征向量，$Y$ 是实值目标
- **目标**：找一个函数 $f$ 使 $f(\mathbf{X}) \approx Y$
- **风险 (risk)**：$R(f) = \mathbb{E}(Y - f(\mathbf{X}))^2$（期望平方误差）
- **最优解**：条件期望

$$r(\mathbf{x}) = \mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]$$

  - 这就是**回归函数 (regression function)**
  - 在所有连续函数里，它是 $R(f)$ 的唯一最小化点

## ② 误差分解

$$R(f) = \underbrace{\int [f(\mathbf{x}) - r(\mathbf{x})]^2 p(\mathbf{x})\, d\mathbf{x}}_{\text{近似误差（可控）}} + \underbrace{\int \mathrm{Var}(Y \mid \mathbf{x}) p(\mathbf{x})\, d\mathbf{x}}_{\text{intrinsic 不可降误差}}$$

- 第一项：取 $f = r$ 时归零
- 第二项：$Y$ 的内在不确定性，模型再好都消不掉

## ③ 经验风险最小化 (ERM)

- **训练集**：$\mathcal{D}_n = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ 来自未知 $P(\mathbf{X}, Y)$ 的 i.i.d. 样本
- **真实风险算不出来**（$P$ 未知）→ 用经验风险代替

$$R_n(f) = \frac{1}{n}\sum_{i=1}^n (y_i - f(\mathbf{x}_i))^2$$

- **大数定律**：$R_n(f) \to R(f)$ 当 $n \to \infty$
- **过拟合警告**：函数类 $\mathcal{F}$ 太大时，最小化 $R_n$ 不等于最小化 $R$ → 需要控制复杂度

## ④ 模型选择

- **线性模型**（参数线性）：$f(\mathbf{x}) = \theta_0 + \boldsymbol\theta^\top \mathbf{x}$
- **特征映射后线性**：$f(\mathbf{x}) = \theta_0 + \boldsymbol\theta^\top \varphi(\mathbf{x})$（多项式、核方法...）
- **完全非线性**：树、kernel、神经网络
- 本课聚焦：线性模型 + 最小二乘 = **OLS (Ordinary Least Squares)**

## ⑤ 一维 OLS

- **模型**：$y_i \approx \theta_0 + \theta_1 x_i$
- **概率版**：$y_i = \theta_0 + \theta_1 x_i + \varepsilon_i$，$\varepsilon_i \overset{\text{iid}}{\sim} \varepsilon$，$\mathbb{E}[\varepsilon] = 0$
- **目标函数**：$\sum_i (y_i - \theta_0 - \theta_1 x_i)^2$
- **为什么用平方误差**
  - 计算简单
  - **统计正当性**：高斯噪声下 MLE = 最小二乘
- **闭式解**

$$\hat\theta_1 = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2}, \quad \hat\theta_0 = \bar y - \hat\theta_1 \bar x$$

- **用相关系数表达**

$$\hat\theta_1 = \mathrm{Corr}_n(\mathbf{x}, \mathbf{y}) \cdot \frac{\sqrt{\mathrm{Var}_n(\mathbf{y})}}{\sqrt{\mathrm{Var}_n(\mathbf{x})}}$$

## ⑥ Centering & Standardization

- **Centering**：$x_i' = x_i - \bar x$，$y_i' = y_i - \bar y$
  - 截距自动归 0：$\hat\theta_0' = 0$
  - 斜率：$\hat\theta_1' = \frac{\sum x_i' y_i'}{\sum (x_i')^2}$
  - 几何意义：把数据云重心移到原点
- **Standardization**：再除以标准差
  - 数据归一为 $\|\mathbf{x}''\|_n^2 = 1$，$\|\mathbf{y}''\|_n^2 = 1$
  - 此时 $\hat\theta_1''$ = 经验相关系数

## ⑦ 多元 OLS（向量化）

- **模型**：$y_i \approx \boldsymbol\theta^\top \mathbf{x}_i$（$\mathbf{x}_i$ 末位加 1 吸收截距，**augmented**）
- **设计矩阵**：$\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$，第一列为常数 1
- **目标**

$$\hat{\boldsymbol\theta} \in \arg\min_{\boldsymbol\theta} \frac{1}{2}\|\mathbf{y} - \mathbf{X}\boldsymbol\theta\|_2^2$$

- **梯度**：$\nabla f(\boldsymbol\theta) = \mathbf{X}^\top(\mathbf{X}\boldsymbol\theta - \mathbf{y})$
- **Normal equations**

$$\mathbf{X}^\top \mathbf{X}\, \hat{\boldsymbol\theta} = \mathbf{X}^\top \mathbf{y}$$

  - 形式 $A\theta = b$ 的线性系统
- **闭式解（满秩时）**

$$\hat{\boldsymbol\theta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

- **不唯一性**：$X$ 列线性相关 → $\ker(\mathbf{X}) \neq \{0\}$ → 解集是仿射子空间 $\hat{\boldsymbol\theta} + \ker(\mathbf{X})$
- **数值实践提醒**：**实际不显式计算 $(\mathbf{X}^\top \mathbf{X})^{-1}$**
  - 计算贵
  - 数值不稳
  - 用 QR / SVD 分解或迭代法

## ⑧ Hat Matrix（几何性质）

- **定义**

$$H_X = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top$$

- **作用**：$\hat{\mathbf{y}} = H_X \mathbf{y}$（把 $\mathbf{y}$ 戴上"帽子"）
- **关键性质**
  - **对称**：$H_X^\top = H_X$
  - **幂等**：$H_X^2 = H_X$
- **结论**：$H_X$ 是**正交投影矩阵**
  - 投影到 $\mathbf{X}$ 的**列空间**
  - 投影方向**垂直**于列空间
  - 残差 $\mathbf{e} = (I - H_X)\mathbf{y}$ 与每个 feature 正交：$\mathbf{X}^\top \mathbf{e} = 0$
- **几何直观**
  - $\hat{\mathbf{y}}$ = 列空间里离 $\mathbf{y}$ 最近的点
  - 残差与列空间垂直
  - 这就是 OLS 的几何含义

## ⑨ 统计性质

- **Sampling error**：$\hat{\boldsymbol\theta} - \boldsymbol\theta$（估计与真值的差）
- **Gaussian linear model 假设下**
  - $\mathbf{y} = \mathbf{X}\boldsymbol\theta + \boldsymbol\varepsilon$
  - $\boldsymbol\varepsilon \mid \mathbf{X} \sim \mathcal{N}(0, \sigma^2 I_n)$
  - $\mathbf{X}^\top \mathbf{X}$ 可逆
- **结论**：$\hat{\boldsymbol\theta} - \boldsymbol\theta \mid \mathbf{X} \sim \mathcal{N}(0, \sigma^2(\mathbf{X}^\top \mathbf{X})^{-1})$
- **OLS 是无偏的**：$\mathbb{E}[\hat{\boldsymbol\theta} \mid \mathbf{X}] = \boldsymbol\theta$
- 这是 **t-test、p-value、置信区间**的理论基础（statsmodels 报这些时心里假设了高斯）

## ⑩ Gauss-Markov 定理

- **设定（classical linear model）**
  - $\mathbb{E}[\boldsymbol\varepsilon \mid \mathbf{X}] = 0$（噪声条件均值为 0）
  - $\mathrm{Var}(\boldsymbol\varepsilon \mid \mathbf{X}) = \sigma^2 I_n$（同方差 + 不相关）
  - $\mathbf{X}^\top \mathbf{X}$ 可逆
  - **不需要高斯假设**
- **比较对象**：所有满足以下两条的估计量
  - **线性 in $\mathbf{y}$**：$\tilde{\boldsymbol\theta} = A\mathbf{y}$
  - **无偏**：$\mathbb{E}[\tilde{\boldsymbol\theta} \mid \mathbf{X}] = \boldsymbol\theta$
- **结论**：OLS 估计量在这类里**方差最小**

$$\mathrm{Var}(\hat{\boldsymbol\theta} \mid \mathbf{X}) \preceq \mathrm{Var}(\tilde{\boldsymbol\theta} \mid \mathbf{X})$$

- **OLS is BLUE** (Best Linear Unbiased Estimator)
- **PSD 序 $\preceq$**：右边减左边半正定（在每个方向上 OLS 方差都不更大）
- **关键意义**：噪声**不必是高斯**，只要均值 0、同方差、不相关即可 → 强稳健性
- **BLUE 的局限**
  - 只在"线性 + 无偏"里最好
  - 非线性方法可能更好（神经网络等）
  - Ridge / Lasso 是**有偏**估计，可能总误差更小，但不在 BLUE 比较范围
- **加上高斯假设后升级**
  - OLS 不只是 BLUE，还是 **MVUE**（无偏中方差最小，不限线性）
  - OLS = MLE
  - 可做 t-test、p-value、置信区间

## ⑪ 💻 代码模板（来自 notebooks）

- **手算 OLS 闭式解（4 种写法）**

```python
# 1. 显式求逆（最差，仅做对照）
beta = np.linalg.inv(X.T @ X) @ X.T @ y

# 2. 伪逆（推荐，自动处理奇异情况）
beta = np.linalg.pinv(X) @ y
beta = np.linalg.pinv(X.T @ X) @ X.T @ y

# 3. 用 solve（数值更稳）
beta = np.linalg.solve(X.T @ X, X.T @ y)
```

- **sklearn 写法**

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)        # X: (n, p), y: (n,)
y_hat = model.predict(X_new)
model.coef_            # β（不含截距）
model.intercept_       # β₀
```

- **statsmodels 写法（含 t/p/R² 报告）**

```python
import statsmodels.api as sm
X_sm = sm.add_constant(X)        # 加截距列
model = sm.OLS(y, X_sm).fit()
print(model.summary())           # 自动给 t-test, p-value, R², F-test
```

- **Centering & 标准化**

```python
X_centered = X - X.mean(axis=0)
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```

- **CAPM 例子**：把单个股票收益对市场收益做 OLS，β = market beta，α = 超额收益

- **t-test & F-test**（`03_linear_regression_solution.ipynb`）
  - **t-test**：检验单个 $\beta_j$ 是否 = 0；statsmodels `model.tvalues`、`model.pvalues`
  - **F-test**：检验整体回归是否显著；statsmodels `model.f_pvalue`
  - **$R^2$**：解释方差比例，statsmodels `model.rsquared`
