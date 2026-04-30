---
title: FRE-GY 7773 复习思维导图 — Lecture 2 MLE
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 380
---

# Lecture 2 — Maximum Likelihood Estimation

## ① 任务背景：参数估计

- **场景举例**
  - 偏差硬币：估计正面概率 $p$
  - 随机游走股价模型：$S_t = S_{t-1} + X_t$，估计向上的概率
  - 高斯收益：估均值和波动率
  - Poisson：违约次数计数
  - 指数分布：等待时间
- **统一框架**：假定数据服从某个**参数族** $\mathcal{P}_\theta = \{p(x \mid \theta) : \theta \in \Theta\}$，目标是从数据估计 $\theta$

## ② Bernoulli & Binomial 分布

- **Bernoulli**（单次二元试验）

$$\text{Bern}(x \mid \mu) = \mu^x (1-\mu)^{1-x}, \quad x \in \{0, 1\}$$

- **Binomial**（$n$ 次独立 Bernoulli 中成功次数）

$$\text{Bin}(x \mid n, \mu) = \binom{n}{x} \mu^x (1-\mu)^{n-x}, \quad x = 0,\ldots,n$$

  - $\binom{n}{x} = \frac{n!}{x!(n-x)!}$ 计组合数
- **关系**：Binomial = $n$ 个独立 Bernoulli 之和

## ③ MLE 框架

- **数据**：$\mathcal{D} = \{x_i\}_{i=1}^n$ i.i.d.（统计学叫"样本"，ML 叫"训练集"）
- **似然函数**

$$p(\mathcal{D} \mid \theta) = \prod_{i=1}^n p(x_i \mid \theta)$$

- **MLE**

$$\hat\theta_\text{ML} = \arg\max_{\theta \in \Theta} p(\mathcal{D} \mid \theta)$$

- **实操中用 log-likelihood**

$$\ell(\theta) = \ln p(\mathcal{D} \mid \theta) = \sum_{i=1}^n \ln p(x_i \mid \theta)$$

  - 把乘积变求和 → 优化简单 + 数值稳定（不会下溢到 0）
  - $\log$ 单调不变，最大值位置一样

## ④ 直觉：为什么 MLE 合理

- **核心思想**：选 $\theta$ 让"已观测到的数据"看起来**最自然**
- **错误的 $\theta$**：观测数据看着不太可能
- **对的 $\theta$**：观测数据是"典型样本"
- **一句话**：MLE = 调模型让现实看起来最 typical

## ⑤ Bernoulli MLE（闭式解）

- **设定**：$x_i \overset{\text{iid}}{\sim} \text{Bernoulli}(\theta)$
- **log-likelihood**

$$\ell(\theta) = \left(\sum_i x_i\right)\ln\theta + \left(n - \sum_i x_i\right)\ln(1-\theta)$$

- **凹性**：$\ell(\theta)$ 在 $(0,1)$ 上严格凹 → 最大值唯一
- **一阶条件** $\partial_\theta \ell = 0$ 解得

$$\hat\theta_\text{ML} = \frac{1}{n}\sum_{i=1}^n x_i = \bar x_n$$

- **结论**：**MLE = 样本均值**（经验成功频率）

## ⑥ Gaussian MLE（单变量）

- **设定**：$x_i \overset{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)$
- **log-likelihood**

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2$$

- **一阶条件**
  - $\partial \ell / \partial \mu = 0$ → $\hat\mu_\text{ML} = \bar x_n$
  - $\partial \ell / \partial \sigma^2 = 0$ → $\hat\sigma^2_\text{ML} = \frac{1}{n}\sum_i (x_i - \hat\mu_\text{ML})^2$
- **关键提醒**：MLE 的 $\sigma^2$ 分母是 $n$，**有偏估计**
  - 无偏修正：$\hat\sigma^2_\text{unb} = \frac{1}{n-1}\sum_i (x_i - \bar x)^2$
  - 这就是为什么 numpy `var(ddof=0)` 默认是 MLE，pandas `var()` 默认是 unbiased

## ⑦ Multivariate Gaussian MLE

- **设定**：$\mathbf{x}_i \overset{\text{iid}}{\sim} \mathcal{N}(\boldsymbol\mu, \Sigma)$，每个 $\mathbf{x}_i \in \mathbb{R}^d$
- **闭式解**
  - $\hat{\boldsymbol\mu}_\text{ML} = \frac{1}{n}\sum_i \mathbf{x}_i$（样本均值向量）
  - $\hat\Sigma_\text{ML} = \frac{1}{n}\sum_i (\mathbf{x}_i - \hat{\boldsymbol\mu}_\text{ML})(\mathbf{x}_i - \hat{\boldsymbol\mu}_\text{ML})^\top$（**有偏**样本协方差矩阵）

## ⑧ MLE 的扩展应用

- **OLS = MLE under Gaussian noise**
  - 假设 $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$
  - 最大化 log-likelihood ⟺ 最小化 $\sum (y_i - x_i^\top\beta)^2$
  - 这就是 OLS
- **逻辑回归 = MLE under Bernoulli**
  - $y_i \mid x_i \sim \text{Bernoulli}(\sigma(x_i^\top\beta))$
  - NLL（负对数似然）就是 logistic 的 loss function
- **Poisson 回归 = MLE under Poisson**
- **MLE 是统一框架**：换分布得到不同模型，这就是 GLM 的来源（见 L5）

## ⑨ 💻 代码模板（来自 notebooks）

- **Bernoulli MLE 数值验证**（`02_mle_solution.ipynb`）

```python
rng = np.random.default_rng(seed=42)
p_true = 0.8
tosses = rng.choice([0, 1], size=10**3, p=[1 - p_true, p_true])
p_mle = np.sum(tosses) / len(tosses)   # = 样本均值
```

- **Random Walk 模拟**

```python
n_steps, n_sims = 500, 1000
stock = np.zeros((n_steps, n_sims))
stock[0, :] = 100
for i in range(n_steps - 1):
    step = np.random.choice([-1, 1], p=[0.5, 0.5], size=n_sims)
    stock[i + 1, :] = stock[i, :] + step
```

- **Gaussian MLE**

```python
x_samples = rng.normal(loc=1.0, scale=0.8, size=1000)
mean_mle = x_samples.mean()
var_mle = np.var(x_samples, ddof=0)   # 注意 ddof=0 是 MLE，ddof=1 是无偏
```

- **Multivariate Gaussian MLE**

```python
mean = np.array([-1.0, 1.0])
cov = np.array([[1.0, -0.9], [-0.9, 1.0]])
x = rng.multivariate_normal(mean, cov, size=400)
mean_mle = x.mean(axis=0)
# 两种等价写法：
cov_mle_1 = (x - mean_mle).T @ (x - mean_mle) / len(x)
cov_mle_2 = np.cov(x, rowvar=False, ddof=0)
```

- **金融数据准备**（`02_data_finance.ipynb`）

```python
import yfinance as yf
import pandas_datareader.data as web

# Yahoo: 股价
spx = yf.download(["^GSPC"], start="1950-01-01", end="2026-01-26")

# FRED: 利率
yields = pd.concat([web.DataReader(t, "fred", start, end) for t in tickers], axis=1)

# 收益率
spx["simple_return"] = spx["Close"].pct_change()
spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
```
