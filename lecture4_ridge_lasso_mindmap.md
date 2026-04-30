---
title: FRE-GY 7773 复习思维导图 — Lecture 4 Ridge & Lasso
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 380
---

# Lecture 4 — Ridge & Lasso

## ① 病态问题动机（OLS via SVD）

- **线性模型回顾**：$\mathbf{y} = \mathbf{X}\boldsymbol\theta + \boldsymbol\varepsilon$
- **OLS via SVD**：$\hat{\boldsymbol\theta}_\text{OLS} = \sum_{i=1}^r \frac{1}{s_i} \mathbf{v}_i (\mathbf{u}_i^\top \mathbf{y})$
- **关键问题**：小奇异值 $s_i$ → $\frac{1}{s_i}$ 爆炸 → **方差爆炸**
- **金融场景**：高度相关的因子 → 小 $s_i$ → 估计不稳、out-of-sample 表现差
- **本质**：这是 ill-posed inverse problem 的固有问题，不是数值实现问题
- **SVD 与特征分解的连接**

$$X^\top X = V\Sigma^2 V^\top, \quad XX^\top = U\Sigma^2 U^\top$$

  - $\mathbf{v}_i$ 是 $X^\top X$ 的特征向量，$\mathbf{u}_i$ 是 $XX^\top$ 的，特征值都是 $s_i^2$

## ② Ridge Regression（L2 罚）

- **惩罚定义**

$$\hat{\boldsymbol\theta}_\lambda^\text{rdg} = \arg\min_{\boldsymbol\theta}\, \|\mathbf{y} - \mathbf{X}\boldsymbol\theta\|_2^2 + \lambda \|\boldsymbol\theta\|_2^2$$

- **闭式解**

$$\hat{\boldsymbol\theta}_\lambda^\text{rdg} = (\mathbf{X}^\top \mathbf{X} + \lambda I_p)^{-1} \mathbf{X}^\top \mathbf{y}$$

- **限制行为**
  - $\lambda \to 0$：退化为 OLS
  - $\lambda \to \infty$：所有系数压成 0
- **约束等价形式**：等价于 $\min \|\mathbf{y} - \mathbf{X}\boldsymbol\theta\|^2$ s.t. $\|\boldsymbol\theta\|_2^2 \leq T$
- **正则化通过 SVD 看**
  - 把 $s_i^2$ 替换为 $s_i^2 + \lambda$
  - 每个方向加阻尼，避免无控制求逆
- **Kernel Trick**

$$\mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top + \lambda I_n)^{-1}\mathbf{y} = (\mathbf{X}^\top\mathbf{X} + \lambda I_p)^{-1}\mathbf{X}^\top\mathbf{y}$$

  - $n > p$ 时反右边的 $p \times p$，$n < p$ 时反左边的 $n \times n$
  - 预测只依赖内积 $\mathbf{X}\mathbf{X}^\top$ → SVM 等核方法的基础
- **正交特例**（$\mathbf{X}^\top \mathbf{X} = I_p$）：$\hat{\boldsymbol\theta}_\lambda^\text{rdg} = \frac{1}{1+\lambda} \hat{\boldsymbol\theta}_\text{OLS}$ → 均匀线性收缩
- **Ridge hat matrix**

$$H_\lambda = \mathbf{X}(\lambda I + \mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top = \sum_j \frac{s_j^2}{s_j^2 + \lambda}\mathbf{u}_j \mathbf{u}_j^\top$$

  - $H_\lambda^2 \neq H_\lambda$ 当 $\lambda > 0$ → **不是正交投影**
  - 对小 $s_j$（不稳方向）做强抑制，对大 $s_j$ 几乎不动
- **Bias-Variance**
  - **Bias**：$-\lambda(\mathbf{X}^\top\mathbf{X} + \lambda I)^{-1}\boldsymbol\theta$，小 $s_i$ 方向严重偏
  - **Variance**：$\sum_i \frac{s_i^2 \sigma^2}{(s_i^2 + \lambda)^2}\mathbf{v}_i \mathbf{v}_i^\top$
  - $\lambda \uparrow$ → bias 增大、variance 减小 → 经典 trade-off

## ③ Lasso（L1 罚）

- **惩罚定义**

$$\hat{\boldsymbol\theta}_\lambda^\text{lasso} = \arg\min_{\boldsymbol\theta}\, \|\mathbf{y} - \mathbf{X}\boldsymbol\theta\|_2^2 + \lambda \|\boldsymbol\theta\|_1$$

  - $\|\boldsymbol\theta\|_1 = \sum_j |\theta_j|$
  - 凸但**不可导**（在 0 处尖角）
- **核心特性**：能把系数**精确设为 0** → 自动 feature selection
- **约束等价**：$\min \|\mathbf{y} - \mathbf{X}\boldsymbol\theta\|^2$ s.t. $\|\boldsymbol\theta\|_1 \leq T$
- **正交特例**（soft-thresholding）

$$\hat\theta_j^\text{lasso} = \text{sign}(\hat\theta_j^\text{OLS})\cdot \max(|\hat\theta_j^\text{OLS}| - \lambda,\ 0)$$

  - "shrink + threshold"：小于 $\lambda$ 的直接归 0
- **没闭式解**（一般情况）
  - SVD 不能对角化
  - 用算法：coordinate descent / proximal gradient / LARS

## ④ Ridge vs Lasso 对比

- **几何直觉（最关键）**
  - Ridge：$\ell_2$ 球（圆/球面，光滑）→ 几乎不会在坐标轴上相切 → 系数都不为 0
  - Lasso：$\ell_1$ 球（菱形，尖角对齐坐标轴）→ 解常落在尖角 → 某些系数 = 0
- **对比表**
  - Penalty：$\ell_2$ vs $\ell_1$
  - Sparsity：✗ vs ✓
  - Closed form：✓ vs ✗
  - Correlated features：稳定（一起压小）vs 不稳定（任选一个）
  - Interpretability：中 vs 高
- **经验法则**
  - **预测**为主 → Ridge
  - **变量选择**为主 → Lasso
  - 两个都要 → **Elastic Net**（$\lambda_1 \|\boldsymbol\theta\|_1 + \lambda_2 \|\boldsymbol\theta\|_2^2$）

## ⑤ 特征预处理 & Intercept 处理

- **必须标准化**：让所有 feature 同尺度，惩罚才公平
- **Intercept 处理**
  - 数据中心化后 intercept 自动为 0
  - 否则 intercept 通常**不被惩罚**

$$\hat{\boldsymbol\theta}_\lambda = \arg\min \|\mathbf{y} - \mathbf{X}\boldsymbol\theta - \theta_0 \mathbf{1}\|^2 + \lambda \sum_{j=1}^p \theta_j^2$$

- **没标准化的 fallback**：用加权惩罚 $\sum_j \alpha_j \theta_j^2$，$\alpha_j = \|\mathbf{x}_j\|^2$

## ⑥ Train / Test Split

- **目的**：用没见过的数据估算**真实泛化性能**
- **铁律**：test 集**完全不能碰**——不调超参、不做 feature engineering、最后只评一次
- **训练误差 vs 测试误差**
  - Train error：随复杂度单调下降（甚至到 0）
  - Test error：U 型曲线
  - 两者差距 = 过拟合程度
- **协议（必背 5 步）**
  1. 随机切 train / test（80% / 20%）
  2. 在 train 上拟合
  3. 用 CV 在 train 内部调超参
  4. 用最优超参在**完整 train** 上重训
  5. 在 test 上**只评一次**
- **常见错误**
  - 反复测试 test
  - 用 test performance 选 $\lambda$

## ⑦ Cross-Validation（K-fold）

- **流程**（$K = 10$ 例）
  - 给定 $\lambda_1, \ldots, \lambda_m$ 候选网格
  - train 切 $K$ 折
  - 对每个 $\lambda_i$：
    - 轮流用 $K-1$ 折训练、剩 1 折当 validation
    - 拿到 $K$ 个 val error，取平均 $\overline{\text{Error}}_i$
  - 选 $\hat i_\text{CV} = \arg\min_i \overline{\text{Error}}_i$
  - 用 $\lambda_{\hat i_\text{CV}}$ 在**完整 train** 上重训
- **$K$ 的选择**
  - $K = 5$ 或 $K = 10$：实践标准
  - $K = n$：**leave-one-out (LOOCV)**，jackknife，方差大且贵
- **实操注意**
  - **打乱再切**（shuffle）：不打乱可能 fold 分布偏；sklearn `KFold(shuffle=True)`
  - 时间序列**别 shuffle**：用 `TimeSeriesSplit` 保持时序，避免泄露未来
  - 每折要有代表性
- **Train/Test vs CV**
  - Single split：方差大但便宜
  - CV：方差小、能调超参，但更贵
  - **最佳实践**：在 train 上做 CV 调超参 + 留一份 test 给最终汇报
- **替代方案**：Monte Carlo CV / repeated hold-out（重复 50–100 次随机切分）
- **Ensemble 思路**：不重训，平均 K 个 fold 模型 $\bar f(x) = \frac{1}{K}\sum_k \hat f_\lambda^{(k)}(x)$

## ⑧ 超参 vs 参数

- **超参 (hyperparameter)**：训练前设定，控制模型结构 / 学习方式
  - Ridge / Lasso 的 $\lambda$、KNN 的 $k$、神经网络层数、学习率
  - 通过 CV 选
- **参数 (parameter / weights)**：训练中由优化算法学出
  - 回归的 $\boldsymbol\beta$、神经网络权重
  - 通过梯度下降 / 闭式解求
- **小技巧**：能被梯度下降"动"的是 parameter，不能动的是 hyperparameter
- **sklearn 实操**
  - 超参：写在构造函数里，`Ridge(alpha=1.0)`
  - 参数：训练后存于 `model.coef_`、`model.intercept_`

## ⑨ 完整模型选择 Pipeline

```
Raw Data
    ↓
Train / Test Split (80% / 20%)
    ↓
Cross-Validation on Training Set
    ↓
Select λ̂ via min CV error
    ↓
Refit on Full Training Set with λ̂
    ↓
Final Test Evaluation (one shot)
```

- sklearn `Pipeline` 把这套串起来，避免数据泄露

## ⑩ 💻 代码模板（来自 notebooks）

- **手算 Ridge / OLS**

```python
from sklearn.preprocessing import add_dummy_feature
X_b = add_dummy_feature(X)   # 加截距列
# OLS 闭式解（用伪逆）
theta_ols = np.linalg.pinv(X_b) @ y
# Ridge 闭式解
mat_I = np.eye(X_b.shape[1])
theta_ridge = np.linalg.solve(X_b.T @ X_b + alpha * mat_I, X_b.T @ y)
```

- **sklearn Ridge / Lasso**

```python
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV

ridge = Ridge(alpha=1.0, fit_intercept=True)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
```

- **Train / Test split + StandardScaler**（**fit only on train**）

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
```

- **K-fold CV + 选 best alpha**

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_alpha, best_mse = None, np.inf
for alpha in alphas:
    mse = []
    for tr, va in kf.split(X_train_s):
        m = Ridge(alpha=alpha).fit(X_train_s[tr], y_train[tr])
        mse.append(mean_squared_error(y_train[va], m.predict(X_train_s[va])))
    if np.mean(mse) < best_mse:
        best_alpha, best_mse = alpha, np.mean(mse)
```

- **更简洁的 RidgeCV / LassoCV**

```python
from sklearn.linear_model import RidgeCV, LassoCV
ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train_s, y_train)
ridge_cv.alpha_   # 最优 α
```

- **Pipeline（推荐，避免泄露）**

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=alphas, cv=5))
])
pipeline.fit(X_train, y_train)              # scaler 只 fit train
mse = mean_squared_error(y_test, pipeline.predict(X_test))
```

- **报告稀疏性（Lasso 系数表）**

```python
print(f"{a:10.1e}  {np.linalg.norm(beta):11.4e}  "
      f"{np.max(np.abs(beta)):11.4e}  ...")   # 看 ||β||₂、最大 |β|、零元个数
```
