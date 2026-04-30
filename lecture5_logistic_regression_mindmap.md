---
title: FRE-GY 7773 复习思维导图 — Lecture 5 Classification & Logistic Regression
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 380
---

# Lecture 5 — Classification & Logistic Regression

## ① 任务设置：分类

- **目标**：预测 $Y \in \{0, 1\}$ 给定 $\mathbf{X} \in \mathbb{R}^d$
- **典型应用**：信用违约、欺诈检测、医疗诊断
- **示例数据**：信用卡违约（LIMIT_BAL、AGE、PAY_*、BILL_AMT_* → default y/n）
- **直接用 linear regression 的问题**
  - 预测可能 $\notin [0, 1]$（线性输出在整个 $\mathbb{R}$）
  - 没有概率解释，对二元数据不合适
- **正确做法**：用 Bernoulli 模型
  - $Y \mid \mathbf{X} \sim \text{Bernoulli}(p(\mathbf{X}))$
  - 只需建模 $p(\mathbf{X}) = P(Y=1 \mid \mathbf{X})$

## ② Logit 链接 & 模型

- **核心问题**：$p(\mathbf{X}) \in (0,1)$，$\beta_0 + \boldsymbol\beta^\top \mathbf{X} \in \mathbb{R}$ → 怎么挂钩？
- **方案**：对 $p$ 套一个 link 函数 $g: (0,1) \to \mathbb{R}$
- **Logit 函数**

$$g(t) = \log\frac{t}{1-t}\quad (\text{logit})$$

  - 把概率（$[0,1]$）拉伸到 $\mathbb{R}$
  - $\frac{t}{1-t}$ 叫 **odds (赔率)**
- **Sigmoid（logit 反函数）**

$$g^{-1}(t) = \sigma(t) = \frac{1}{1+e^{-t}} = \frac{e^t}{1+e^t}$$

  - $\mathbb{R}$ 压成 $(0,1)$
  - S 形曲线，$t=0$ 时 $\sigma = 0.5$
- **Sigmoid 导数性质**：$\sigma'(t) = \sigma(t)(1 - \sigma(t))$
- **逻辑回归模型（两种等价写法）**

$$\log\frac{P(Y=1\mid\mathbf{X})}{P(Y=0\mid\mathbf{X})} = \beta_0 + \boldsymbol\beta^\top \mathbf{X}$$

$$\Longleftrightarrow\quad P(Y=1\mid\mathbf{X}) = \sigma(\mathbf{X}^\top \boldsymbol\beta)$$

  - 第一种强调"log-odds 是 feature 的线性组合"——**便于解释系数**
  - 第二种强调"概率是线性组合套 sigmoid"——**便于计算**
  - $\mathbf{X}$ 默认增广（前面加 1 吸收截距 $\beta_0$）
- **其他 link 函数**
  - **Probit**：$g^{-1} = \Phi$（标准正态 CDF）
  - **cloglog**：$g^{-1}(t) = 1 - e^{-e^t}$
  - 都把 $\mathbb{R}$ 映到 $(0,1)$，但 logit 数学最干净

## ③ MLE 推导

- **似然函数**

$$\mathcal{L}_n(\boldsymbol\beta) = \prod_{i=1}^n p_{\boldsymbol\beta}(\mathbf{x}_i)^{y_i}(1-p_{\boldsymbol\beta}(\mathbf{x}_i))^{1-y_i}$$

- **若用 $y_i \in \{-1, +1\}$ 编码**：用 $\sigma(-t) = 1 - \sigma(t)$

$$\mathcal{L}_n(\boldsymbol\beta) = \prod_i \sigma(y_i \mathbf{x}_i^\top \boldsymbol\beta)$$

- **NLL（negative log-likelihood）**

$$\ell_n(\boldsymbol\beta) = -\frac{1}{n}\sum_i \big[y_i \mathbf{x}_i^\top\boldsymbol\beta - \log(1 + e^{\mathbf{x}_i^\top\boldsymbol\beta})\big]$$

  - 取负是为了把"max likelihood"变成"min loss"，符合优化习惯
  - 取 log 是为了把乘积变求和（数值稳定）
  - 加 $1/n$ 是工程惯例（每样本平均损失，跨数据集可比）
- **没有闭式解**（不像 OLS），必须用数值优化

## ④ 梯度 & Hessian（凸性的来源）

- **梯度**（用 sigmoid 导数 + 链式法则推）

$$\nabla \ell_n(\boldsymbol\beta) = \frac{1}{n}\sum_i (\sigma(\mathbf{x}_i^\top\boldsymbol\beta) - y_i)\mathbf{x}_i$$

  - 形式：**(预测 − 真实) × 特征 的样本平均**
  - 和 OLS 的梯度形式一致，只是预测从 $\mathbf{x}^\top\boldsymbol\beta$ 变成 $\sigma(\mathbf{x}^\top\boldsymbol\beta)$
- **Hessian**

$$\nabla^2 \ell_n(\boldsymbol\beta) = \frac{1}{n}\sum_i \sigma(\mathbf{x}_i^\top\boldsymbol\beta)(1-\sigma(\mathbf{x}_i^\top\boldsymbol\beta))\mathbf{x}_i \mathbf{x}_i^\top$$

  - $\mathbf{x}_i \mathbf{x}_i^\top$ 是 rank-1 半正定矩阵
- **Convexity 论证**
  - 每项的标量 $\sigma(1-\sigma) \geq 0$
  - $\mathbf{x}_i \mathbf{x}_i^\top \succeq 0$
  - 非负标量 × PSD = PSD；PSD 之和仍 PSD
  - $\Rightarrow \nabla^2 \ell_n \succeq 0 \Rightarrow \ell_n$ 是 convex
  - **关键意义**：任何局部最小 = 全局最小，梯度下降不会陷入坏的局部最优
- **Smoothness 论证**
  - 关键不等式：$\sigma(t)(1-\sigma(t)) \leq 1/4$（在 $t=0$ 取最大）

$$\nabla^2 \ell_n(\boldsymbol\beta) \preceq \frac{1}{4n}\sum_i \mathbf{x}_i \mathbf{x}_i^\top$$

  - Lipschitz 常数 $L = \frac{1}{4n}\lambda_\max(\sum \mathbf{x}_i \mathbf{x}_i^\top)$
  - 用来定梯度下降步长（$\eta \leq 1/L$ 才保证收敛）
- **PSD 序 $\succeq, \preceq$**
  - $A \succeq 0$ 表示 $A$ 半正定（特征值都 $\geq 0$），不是元素 $\geq 0$
  - $A \succeq B \Leftrightarrow A - B \succeq 0$
  - 是矩阵层面"非负"的概念

## ⑤ 求解：梯度下降

- **更新规则**

$$\boldsymbol\beta_{k+1} = \boldsymbol\beta_k - \eta_k \nabla \ell_n(\boldsymbol\beta_k) = \boldsymbol\beta_k + \frac{\eta_k}{n}\sum_i \mathbf{x}_i(y_i - p_{\boldsymbol\beta_k}(\mathbf{x}_i))$$

- **直觉**：模型低估 $y_i$ 时把 $\boldsymbol\beta$ 朝那个方向推；高估时反向调
- **步长 $\eta_k$ 选择**：$\eta_k \leq 1/L$ 收敛保证
- **凸性保证**：因为 $\ell_n$ convex，迭代会收敛到全局最优

## ⑥ 系数解读：Odds Ratio

- **公式**

$$\text{OR}_j = \exp(\beta_j)$$

- **含义**：$X^{(j)}$ 增加 1 单位 → odds 乘 $e^{\beta_j}$
  - $\beta_j > 0$：odds 上升（正向影响）
  - $\beta_j < 0$：odds 下降
  - $\beta_j = 0$：无影响
- **二元 regressor 特例**：$X^{(j)} \in \{0, 1\}$ 时 OR 直接表示两组的 odds 比
- **Default vs Age 例子**
  - $\log\frac{p}{1-p} = -1.80 - 0.04 \times \text{AGE}$
  - $e^{-0.04} \approx 0.96$ → 年龄每加 1 岁，违约 odds 降约 4%
- **⚠️ Odds Ratio ≠ Probability Ratio**
  - $\text{OR} \neq \frac{P(Y=1\mid X=1)}{P(Y=1\mid X=0)}$
  - 仅当概率非常小时两者近似相等
  - 这就是为什么 OR 可能很大但概率变化只是中等

## ⑦ 分类指标 & Confusion Matrix

- **Confusion matrix**：4 个数（TP / TN / FP / FN）
- **核心指标**

| 指标 | 公式 | 含义 |
|---|---|---|
| Accuracy | $(TP+TN)/n$ | 整体正确率 |
| TPR / Sensitivity / Recall | $TP/(TP+FN)$ | 真正例抓到的比例 |
| FPR | $FP/(FP+TN)$ | 真负例被误报的比例 |
| Specificity | $TN/(TN+FP) = 1 - \text{FPR}$ | 真负例正确否定 |
| Precision | $TP/(TP+FP)$ | 预测为正里真为正的 |
| F1-score | $2\frac{P \cdot R}{P + R}$ | precision/recall 调和平均 |

- **多分类**：confusion matrix 是 $K \times K$，对角线是命中

## ⑧ ROC & AUC

- **背景**：分类用阈值 $\tau$ 把概率转 0/1，$\tau$ 变 → TPR/FPR 变
- **ROC 曲线**
  - $\tau$ 从 0 滑到 1，画 $(\text{FPR}, \text{TPR})$ 曲线
  - 左上角越近越好（FPR 小、TPR 大）
  - 对角线 = 随机分类器
- **AUC = Area Under Curve**
  - = 1：完美
  - = 0.5：随机
  - $< 0.5$：比瞎猜还差
- **AUC 的概率解释**
  - 随机抽一个正样本和一个负样本，模型给正样本概率高于负样本的概率
- **AUC vs Accuracy**
  - AUC 不依赖阈值
  - 类不平衡时更可靠（极端情况 99% 负样本，全猜 0 也有 99% accuracy 但 AUC = 0.5）
  - 关注**排序能力**

## ⑨ 预测使用方式

- **新样本** $\mathbf{x}_\text{new}$
  - 算 $\hat p_\text{new} = \sigma(\hat{\boldsymbol\beta}^\top \mathbf{x}_\text{new})$
  - 决策：$\hat y_\text{new} = 1 \text{ if } \hat p_\text{new} \geq \tau$（默认 $\tau = 0.5$）
- **概率而非确定性**：模型输出是 probability 不是 0/1

## ⑩ Generalized Linear Models (GLM)

- **统一框架**（McCullagh & Nelder 1989）
- **三件套**
  - **Random component**：$Y_i \sim$ 指数族分布，$\mathbb{E}[Y_i] = \mu_i$
  - **Linear predictor**：$\eta_i = \mathbf{x}_i^\top \boldsymbol\beta$
  - **Link function**：$g(\mu_i) = \eta_i$（把 $\mu$ 映到 $\mathbb{R}$）
- **三个特例**
  - 线性回归：Gaussian + identity link，$\mu = \mathbf{x}^\top\boldsymbol\beta$
  - 逻辑回归：Bernoulli + logit link，$\mu = \sigma(\mathbf{x}^\top\boldsymbol\beta)$
  - Poisson 回归：Poisson + log link，$\mu = e^{\mathbf{x}^\top\boldsymbol\beta}$
- **意义**：换分布换 link 得到不同模型，统一 MLE 框架 + IRLS 算法

## ⑪ 高维 / 正则化逻辑回归

- **目标**

$$\arg\min_{\boldsymbol\beta}\, \frac{1}{n}\sum_i \log(1 + e^{-y_i \mathbf{x}_i^\top \boldsymbol\beta}) + \text{pen}(\boldsymbol\beta)$$

- **常见 penalty**
  - **Ridge**：$\lambda \|\boldsymbol\beta\|_2^2$（凸、不稀疏，sklearn 默认）
  - **Lasso**：$\lambda \|\boldsymbol\beta\|_1$（凸、产生稀疏解，自动 feature selection）
  - **Elastic Net**：$\lambda_1 \|\boldsymbol\beta\|_1 + \lambda_2 \|\boldsymbol\beta\|_2^2$
- **Convex loss + convex penalty** → 高效优化
- **选 $\lambda$**：CV、AIC/BIC
- **U 型曲线**
  - $\lambda$ 小 → overfitting
  - $\lambda$ 大 → underfitting
  - 中间最优

## ⑫ sklearn 实现

- **关键参数**

```python
LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)
```

  - `C` = $1/\lambda$（**和 Ridge `alpha` 反着**！）
  - `penalty`: 'l2' / 'l1' / 'elasticnet' / 'none'
  - `max_iter`: 默认 100 常常不够
- **Solver 对照**
  - `'lbfgs'`（默认）：拟牛顿法，**只支持 L2**
  - `'liblinear'`：坐标下降，支持 L1/L2，小数据
  - `'saga'`：方差缩减 SGD，支持 L1/L2/elastic net，大数据
  - `'sag'`：随机平均梯度，只支持 L2
- **L1 必须换 solver**：`solver='saga'` 或 `'liblinear'`
- **预测两种**
  - `predict()` → 0/1（默认阈值 0.5）
  - `predict_proba()` → 两列概率，第二列是 $P(Y=1)$
- **指标 sklearn API**
  - `confusion_matrix(y_true, y_pred)`
  - `classification_report` → precision / recall / F1 / support
  - `roc_auc_score(y_true, y_proba)` （注意是 proba 不是 0/1）
  - `roc_curve(y_true, y_proba)` → fpr, tpr, thresholds

## ⑬ 拓展（多分类等）

- Multinomial logistic regression（多类）
- Ordinal regression（有序类别）
- Mixed-effects logistic（层级数据）
- Calibration & ROC analysis
- 关键信息：**逻辑回归 = GLM 的特例**，整套 GLM 框架更广

## ⑭ 💻 代码模板（来自 notebooks）

- **基础流程**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)              # 0/1 硬预测
y_proba = clf.predict_proba(X_test)[:, 1] # 正类概率
```

- **MNIST 二分类（数字 5 vs not 5）**

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
y_train_5 = (y_train == 5)   # 二分类标签
```

- **CV-based 评估**

```python
from sklearn.model_selection import cross_val_score, cross_val_predict

# CV 平均 accuracy
cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")

# CV 预测（每个样本被 OOF 预测一次）
y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
y_scores = cross_val_predict(clf, X_train, y_train, cv=3, method="decision_function")
```

- **基线对比：DummyClassifier**

```python
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier()  # 总是预测多数类
dummy.fit(X_train, y_train)
cross_val_score(dummy, X_train, y_train_5, cv=3, scoring="accuracy")  # 看类不平衡时基线
```

- **混淆矩阵 & 分类报告**

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)         # 2x2 矩阵
print(classification_report(y_test, y_pred))  # precision/recall/F1/support
```

- **ROC & AUC**

```python
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# Precision-Recall 曲线（类不平衡时更有意义）
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
```

- **手动复制 K-fold CV（StratifiedKFold + clone）**

```python
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
for train_idx, test_idx in skf.split(X_train, y_train):
    cloned = clone(clf)
    cloned.fit(X_train[train_idx], y_train[train_idx])
    y_pred = cloned.predict(X_train[test_idx])
```

- **SGDClassifier**（大数据、在线学习）

```python
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log_loss', random_state=42)  # 等价 logistic regression
sgd.fit(X_train, y_train_5)
```
