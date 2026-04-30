---
title: FRE-GY 7773 复习思维导图 — Lecture 7 PCA
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 380
---

# Lecture 7 — PCA

## ① 目标 & 直觉

- 找投影后**方差最大**的方向 → 用更少维度保留更多信息
- 比喻：3D 点云拍 2D 照片，从最"胖"的那一面拍
- 流程
  - 找 PC1：max variance 方向
  - 找 PC2：⊥ PC1 中 max 剩余 variance
  - 一直找下去
- 用途
  - 降维（多 feature → 少 feature）
  - 可视化（高维 → 2D 散点）
  - 去噪（低方差 ≈ 噪声，丢掉）
  - 去相关（PC 之间互不相关）
- 金融例子 — Yield Curve PCA
  - PC1 = Level（水平移动），≈ 70–80%
  - PC2 = Slope（斜率变化），≈ 15%
  - PC3 = Curvature（曲率），≈ 3–5%
  - 3 个 PC 解释 ~99%

## ② 数据准备

- **Centering**（必做）：$Y = X - \mathbf{1}g^\top$，$g = \frac{1}{n}\sum_i x_i$
  - 让数据云中心落在原点
  - 不 center → 协方差公式不成立
- **Standardization**（可选）：$Z = Y\, D_{1/s}$
  - 用途：feature 量纲不同时（年龄 + 收入 + 身高 …）
  - $s_j = \sqrt{\sigma_j^2}$ 就是标准差（开根号又取平方只是符号写法，$s_j = \sigma_j$）
  - **关键性质**：$\frac{1}{n}Z^\top Z$ = 相关系数矩阵
  - 等价于"对相关系数矩阵做 PCA"
- **不标准化 vs 标准化**
  - 不标准化（covariance PCA）
    - 让大波动 feature 主导
    - 适合同单位场景（如 yield curve 各期限利率，单位都是 %）
  - 标准化（correlation PCA）
    - feature 平等对待
    - 适合单位/量纲不同

## ③ 投影 & 范数（基础概念）

- **向量** = 方向 + 长度（一支有限长度的箭）
  - ≠ 射线（射线是无限长的半线，无长度）
  - ≠ 纯方向（方向只指哪儿，无长度）
  - 三种视角：几何（箭）/ 代数（一串数）/ 数据（一个点）
- **投影**：$x$ 在单位向量 $w$ 上的"影子"
  - 投影标量（一个数）：$x \cdot w = x^\top w = \|x\|\cos\theta$
  - 投影向量（投影点位置）：$(x \cdot w)\, w$
  - 必须 $\|w\| = 1$ 公式才干净
  - 投影标量**有正负**：同向为正、反向为负
  - **降维 = 用投影标量替代原 $p$ 维向量**
- **范数 (norm)** = 向量长度
  - $\|v\| = \sqrt{v_1^2 + \cdots + v_p^2} = \sqrt{v^\top v}$
  - L2 / Euclidean norm（最常用，PCA 用的）
  - L1 norm = $\sum |v_i|$（Lasso 用）
  - unit vector = 范数为 1 的向量

## ④ 形式化：优化问题

- **目标**：投影方差最大

$$v_1 = \arg\max_{\|v\|=1}\ \frac{1}{n}\sum_{i=1}^n (y_i^\top v)^2$$

- **约束写 $\|v\|^2 = 1$ 而非 $\|v\| = 1$**
  - 数学上等价（范数 ≥ 0）
  - 平方版本无根号 → 求导干净
  - 拉格朗日法直接给出特征值方程
- **矩阵形式推导**
  - 关键观察：$y_i^\top v = (Yv)_i$
  - $\sum_i (y_i^\top v)^2 = \|Yv\|^2$
  - $\|Yv\|^2 = (Yv)^\top(Yv) = v^\top Y^\top Y\, v$（用 $(AB)^\top = B^\top A^\top$）
  - $\frac{1}{n}\sum (y_i^\top v)^2 = v^\top \Sigma v$
  - **目标简化为**：$\max_{\|v\|=1}\, v^\top \Sigma v$
- **协方差矩阵 $\Sigma = \frac{1}{n}Y^\top Y$**
  - $p \times p$，对称
  - $Y$ 已 center → 元素就是样本协方差
- **符号约定**：$y_i$ 是 $Y$ 的第 $i$ 行
  - 但单独使用时立起来当**列向量**（$p \times 1$）
  - 默认所有向量都是列向量

## ⑤ 求解：拉格朗日法 → 特征值方程

- **拉格朗日量**

$$\mathcal{L}(v, \lambda) = v^\top \Sigma v - \lambda(v^\top v - 1)$$

- **对 $v$ 求偏导用两条公式**
  - $\partial(v^\top A v)/\partial v = 2Av$（$A$ 对称时）
  - $\partial(v^\top v)/\partial v = 2v$
  - $\Sigma$ 对称（$(Y^\top Y)^\top = Y^\top Y$）→ 套用第一条
- **结果**：$\partial \mathcal{L}/\partial v = 2\Sigma v - 2\lambda v = 0$
- **特征值方程**：$\boxed{\Sigma v = \lambda v}$
- **标量挪动规则**（推导用得到）
  - 标量 ↔ 标量 / 标量 ↔ 矩阵：✅ 随便挪（$cA = Ac$）
  - 矩阵 ↔ 矩阵：❌ 一般不可交换（$AB \neq BA$）
  - 矩阵之间的相对顺序绝对不变

## ⑥ Eigenvector / Eigenvalue 解读

- **定义**：$Av = \lambda v$（$v$ 非零，$\lambda$ 标量）
- **直观**：被 $A$ 作用后只拉伸/压缩，不转向
  - $\lambda > 1$ 拉长 / $0 < \lambda < 1$ 压缩
  - $\lambda < 0$ 翻转 / $\lambda = 0$ 压扁到原点
- **A 的要求**
  - 方阵：✅ 必须（$Av$ 形状要和 $v$ 一致；非方阵用 SVD）
  - 对称：不必须，但 PCA 用对称图它的好处
    - 全实特征值
    - 特征向量两两正交
    - 恰好 $p$ 个线性无关
- **数量与对应关系**
  - $p \times p$ 方阵有 $p$ 个特征值（按重数）
  - 一个 eigenvector → 一个 eigenvalue ✅
  - 一个 eigenvalue → 可能多个 eigenvector（重特征值时整个子空间）
  - 例：$A = 2I$，所有非零向量都是特征向量
- **PCA 里 $\lambda$ = 投影方差**
  - 从 $\Sigma v_1 = \lambda v_1$，**两边左乘 $v_1^\top$**
    - $v_1^\top \Sigma v_1 = \lambda v_1^\top v_1 = \lambda$（用 $\|v_1\|=1$）
  - 右乘不行（形状变 $p \times p$ 矩阵；标量必须 $1\!\times\!p$ 撞 $p\!\times\!1$）
  - 又 $Yv_1$ 已 center，$\mathrm{Var}(Yv_1) = \frac{1}{n}\|Yv_1\|^2 = v_1^\top \Sigma v_1 = \lambda$
  - **结论**：选最大 $\lambda$ ⇔ 选方差最大的方向
- **标准化 PCA 特有**
  - $\sum_i \lambda_i = \mathrm{tr}(\Sigma_\text{corr}) = p$（每列方差归一为 1）
  - $\lambda_i / p$ = 该 PC 解释的方差比例（sklearn `explained_variance_ratio_`）

## ⑦ 应用：表示 & 重构

- **主成分坐标（表示）**
  - $c_j = Y v_j$（每个样本在 PC$j$ 上的投影标量构成的列）
  - $C = YV$（$n \times p$）
  - $C_k = YV_k$（$n \times k$，$V_k = [v_1, \ldots, v_k]$）
  - **真正的降维就在这一步**：$n \times p \to n \times k$
- **重构 (Reconstruction)**
  - $\hat Y = C_k V_k^\top = Y V_k V_k^\top$（$n \times p$）
  - $V_k V_k^\top$ 是投影矩阵（投到前 $k$ 个 PC 张成的子空间）
  - 形状回到 $n \times p$，但**秩 = $k$**
  - 推导：$\hat y = \sum_{j=1}^k c_j v_j = V_k(V_k^\top y)$
- **降维 vs 重构（容易搞混）**
  - 降维成果在 $C_k$（$n \times k$，真省空间）
  - 重构 $\hat Y$ 形状回到 $n \times p$ 但本质躺在 $k$ 维子空间
  - $\hat Y$ 不是再次降维，是把 $C_k$ 解码回原空间方便对比
  - 例：$n=1000$, $p=100$, $k=5$ → 真实信息只 $1000\times 5 + 100\times 5 = 5500$ 个数
- **重构的用途**
  - 去噪（丢低方差方向后更"干净"）
  - 重构误差 $\|Y - \hat Y\|^2$（衡量降维损失）
  - 异常检测（$\|y_i - \hat y_i\|$ 大 → 可能异常）
  - 在原坐标系下可视化（如压缩图像复原）

## ⑧ 实现：SVD（替代特征分解）

- **SVD 分解**：$Z = U\Sigma V^\top$
  - $U$（$n \times n$）：left singular vectors，列两两正交
  - $\Sigma$（$n \times p$ 对角）：奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
  - $V$（$p \times p$）：right singular vectors，列两两正交
  - ⚠️ 这里的 $\Sigma$ 是奇异值矩阵，**和协方差矩阵符号撞车**
- **和 PCA 的对应**
  - $V$ 列 = 主成分方向（即 $\frac{1}{n}Z^\top Z$ 的特征向量）
  - $\sigma_j^2 / n = \lambda_j$（投影方差）
  - 主成分坐标 $C = ZV = U\Sigma$
    - 验证：$ZV = U\Sigma V^\top V = U\Sigma$（$V^\top V = I$）
- **为什么用 SVD**
  - 数值更稳定（不必先算 $Z^\top Z$，避免精度损失）
  - sklearn 的 `PCA` 默认实现
- **不标准化也能用**
  - 直接对 centered $Y$ 做 SVD → covariance PCA
  - 对 standardized $Z$ 做 SVD → correlation PCA
  - SVD 工具本身只要求"已 center"，不要求标准化

## ⑨ 选多少个 PC（k 的取法）

- **Elbow criterion**：scree plot 上找特征值衰减的拐点
- **Kaiser criterion**：保留 $\lambda > 1$（仅适用 standardized PCA，因为 $\lambda = 1$ 等于一个原始变量的方差）
- **Task-driven**：用下游模型 cross-validation 性能决定

## ⑩ 💻 代码模板（来自 notebooks）

- **PCA via SVD（手算）**

```python
# 方法 1：对协方差矩阵做 SVD（教学常用，对应数学定义）
u, s, u_t = np.linalg.svd(np.cov(x_samples.T))
# u 列 = 主成分方向，s = 特征值（按降序）

# 方法 2：直接对数据矩阵做 SVD（数值更稳，sklearn 内部实现）
U, S, Vt = np.linalg.svd(X, full_matrices=True)
# Vt.T 列 = 主成分方向，S²/n = 特征值
```

- **sklearn PCA**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=k)
pca.fit(X)                            # X 应已 center
C_k = pca.transform(X)                # 主成分坐标 (n × k)
X_hat = pca.inverse_transform(C_k)    # 重构回原空间 (n × p)

pca.components_                       # V_k.T，k × p 主成分矩阵
pca.explained_variance_               # 各 PC 方差 λ_j
pca.explained_variance_ratio_         # λ_j / Σλ
```

- **完整流程：标准化 + PCA**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=True, with_std=True)
Z = scaler.fit_transform(X)
pca = PCA(n_components=k)
C_k = pca.fit_transform(Z)
```

- **手动重构验证**

```python
recon = (U_x[:, :k] * S_x[:k]) @ Vt_x[:k, :]
# 等价于 pca.inverse_transform(pca.transform(X))
```

- **Yield Curve PCA**（`07_yield_curve_pca_solution.ipynb`）
  - 输入：日度 yield change 矩阵（行 = 日期，列 = 期限）
  - 减均值后 SVD：`U, S, U_t = np.linalg.svd(cov_X)`
  - PC1 ≈ Level、PC2 ≈ Slope、PC3 ≈ Curvature
  - 前 3 个 PC 累计 ~99% 方差

- **置信椭圆**（`07_gaussian_pca_solution.ipynb`）

```python
P, s, Pt = np.linalg.svd(cov)   # cov 是协方差矩阵
# 椭圆主轴方向 = P 的列，半径 ∝ sqrt(s)
```
