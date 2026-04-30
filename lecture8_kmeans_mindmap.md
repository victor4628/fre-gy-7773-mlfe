---
title: FRE-GY 7773 复习思维导图 — Lecture 8 K-Means
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 380
---

# Lecture 8 — K-Means

## ① 目标 & 直觉

- 任务：把 $I$ 个数据点划分成 $Q$ 个群
- 思路：找 $Q$ 个**中心点 (centroids)**，让每个样本到自己最近中心的距离尽量小
- 输出：每个样本的归属簇 + $Q$ 个中心位置
- 是**无监督学习**（没有真实标签）

## ② 损失函数：quantification error

- **公式**

$$\sum_{i=1}^I \min_{q=1,\ldots,Q}\|x_i - c_q\|_2^2$$

- **双 2 的含义（容易混）**
  - 下标 $\|\cdot\|_2$ = **L2 norm**（区别于 L1 等其他范数）
  - 上标 $\|\cdot\|^2$ = **整体取平方**
  - 合起来：欧氏距离的平方，外层平方把根号抵消，**实际算时不开根号**

$$\|x_i - c_q\|_2^2 = \sum_{k=1}^K (x_{ik} - c_{qk})^2$$

- **为什么用距离平方而不是距离本身**
  - 找最近中心的位置一样
  - 求导更干净（避开根号 → 类比 PCA 的 $\|v\|^2 = 1$）
- **NP-hard**
  - 全局最优计算上不可行
  - $Q^I$ 种可能分配（指数爆炸）
  - → 必须用启发式算法（Lloyd）

## ③ 范数基础（L1 / L2）

- **L2 norm = 欧几里得距离**
  - $\|v\|_2 = \sqrt{v_1^2 + \cdots + v_K^2} = \sqrt{v^\top v}$
  - 几何：直线长度
  - K-means、PCA、Ridge 都用它
- **L1 norm = 曼哈顿距离**
  - $\|v\|_1 = |v_1| + \cdots + |v_K|$
  - 几何：网格里沿街道走的步数（不能斜穿）
  - Lasso 用它，能产生稀疏解
- **不带下标的 $\|v\|$ 默认 = L2**（ML 标准约定）
- **ML 里常见角色**
  - L2 → 平滑、可导、产生稠密解
  - L1 → 不平滑（0 处尖角）、产生**精确为 0** 的稀疏解

## ④ Lloyd 算法（求局部最优）

- **核心思想**：两步交替，**固定一边优化另一边**
- **Step 0：初始化**
  - 随机选 $Q$ 个点当起始中心 $c_1, \ldots, c_Q$
- **Step 1：分配 (assignment)**
  - 每个样本找**最近的中心**入队
  - $C_q = \{i : \|x_i - c_q\| \leq \|x_i - c_{q'}\|,\ \forall q' \neq q\}$
- **Step 2：更新中心 (update)**

$$c_q = \frac{1}{|C_q|}\sum_{i \in C_q} x_i$$

  - 把每个队伍里所有样本的坐标取平均
  - $|C_q|$ = 队伍人数
- **Step 3：重复**
  - 回到 Step 1，直到 $C_q$ 不再变化（收敛）
- **重要性质**
  - 每步都让损失**单调下降**
  - 一定收敛（有限步内）
  - **但只能保证局部最优**，不一定全局最优
  - → 常用不同初始化跑多次取最好

## ⑤ 初始化策略

- **普通随机**
  - 简单但风险大：中心扎堆 → 糟糕局部最优
- **Furthest point initialization**
  - 每次选离已有中心**最远**的点
  - 中心分散
  - **缺点**：对 outlier 敏感（异常点会被当中心）
- **k-means++（推荐折中）**
  - **Step 1**：第一个中心随机选
  - **Step 2–5**：后续中心**按概率**抽样

$$P(x_i\ \text{被选}) = \frac{\min_{q'} \|x_i - c_{q'}\|_2^2}{\sum_{i'} \min_{q'} \|x_{i'} - c_{q'}\|_2^2}$$

  - 分子：$x_i$ 到最近已有中心的距离平方
  - 分母：归一化使概率和 = 1
  - 直觉：**离已有中心远的点更可能被选**，但不绝对（对 outlier 鲁棒）
  - 完成 $Q$ 个中心后再跑普通 Lloyd
- **三种方法对比**
  - 随机：可能扎堆
  - Furthest：易选 outlier
  - **k-means++**：随机性 + 倾向远离已有中心，最稳健
- **sklearn**：`KMeans(init='k-means++')` 是默认

## ⑥ 选多少个 Q（待补充）

- **Elbow on inertia**：`kmeans.inertia_` = 每个样本到最近中心距离平方和；画 inertia vs Q，找拐点
- **Silhouette score**：综合考虑簇内紧密度 + 簇间分离度
- **Task-driven**：下游任务表现

## ⑦ K-means vs GMM（待补充）

- 见 lecture8_gmm_mindmap.md 末尾的对比表

## ⑧ 💻 代码模板（来自 notebooks）

- **基础 KMeans**

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs

X, y = make_blobs(n_samples=400, centers=4, random_state=43)

kmeans = KMeans(n_clusters=4, random_state=43, n_init=20)
labels = kmeans.fit_predict(X)
kmeans.cluster_centers_              # 每簇中心坐标
kmeans.inertia_                      # 损失函数值
kmeans.predict(X_new)                # 新样本归类
kmeans.transform(X)[:10]             # 每个样本到所有中心的距离矩阵
```

- **可视化迭代过程**（用 `max_iter=1, init=...` 控制）

```python
kmeans_iter1 = KMeans(n_clusters=4, init=initial_centers, n_init=1, max_iter=1)
kmeans_iter1.fit(X)
# 用 cluster_centers_ 看每轮中心移动到哪
```

- **决策边界（用网格预测）**

```python
xx, yy = np.meshgrid(np.linspace(...), np.linspace(...))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
plt.contourf(xx, yy, Z.reshape(xx.shape))
```

- **PCA + KMeans on stock returns**（`08_pca_k_means_returns_solution.ipynb`）

```python
# 先标准化 → PCA → 在 PC 空间聚类
scaler = StandardScaler(with_mean=True, with_std=True)
Z = scaler.fit_transform(returns)

pca = PCA(n_components=k)
Z_pc = pca.fit_transform(Z)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
labels = kmeans.fit_predict(Z_pc)

# 评估聚类质量
from sklearn.metrics import silhouette_score
silhouette_score(Z_pc, labels)
```
