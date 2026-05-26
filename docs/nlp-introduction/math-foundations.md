# 第三章：数学基础

NLP 涉及大量的数学知识。本章将介绍理解 NLP 算法所必需的数学基础，包括线性代数、概率论和信息论的核心概念。

---

## 线性代数基础

### 标量、向量、矩阵和张量

| 术语 | 符号 | 形状 | 说明 | 示例 |
|------|------|------|------|------|
| 标量（Scalar） | $x$ | () | 单个数值 | $x = 3.14$ |
| 向量（Vector） | $\mathbf{x}$ | $(n,)$ | 一维数组 | $\mathbf{x} = [1, 2, 3]$ |
| 矩阵（Matrix） | $\mathbf{X}$ | $(m, n)$ | 二维数组 | $\mathbf{X} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ |
| 张量（Tensor） | $\mathcal{X}$ | $(d_1, d_2, ..., d_n)$ | 多维数组 | 形状为 $(2, 3, 4)$ 的三维数组 |

::: tip NLP 中的张量
在 NLP 中，我们经常使用张量来表示数据。例如：
- 一个词的词向量：向量，形状为 $(d,)$，如 $(300,)$
- 一个句子的词向量序列：矩阵，形状为 $(n, d)$，如 $(20, 300)$
- 一个批次的句子：三维张量，形状为 $(b, n, d)$，如 $(32, 20, 300)$
:::

### 向量运算

#### 向量加法

$$\mathbf{a} + \mathbf{b} = [a_1 + b_1, a_2 + b_2, ..., a_n + b_n]$$

示例：$[1, 2, 3] + [4, 5, 6] = [5, 7, 9]$

#### 标量乘法

$$c \cdot \mathbf{a} = [c \cdot a_1, c \cdot a_2, ..., c \cdot a_n]$$

示例：$3 \cdot [1, 2, 3] = [3, 6, 9]$

#### 点积（Dot Product）

点积是 NLP 中最常用的操作之一：

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + ... + a_n b_n$$

示例：$[1, 2, 3] \cdot [4, 5, 6] = 1 \times 4 + 2 \times 5 + 3 \times 6 = 32$

::: info 点积的几何意义
点积衡量两个向量的**相似程度**：
- $\mathbf{a} \cdot \mathbf{b} > 0$：两个向量方向大致相同
- $\mathbf{a} \cdot \mathbf{b} = 0$：两个向量正交（垂直）
- $\mathbf{a} \cdot \mathbf{b} < 0$：两个向量方向大致相反
:::

### 向量范数

#### L1 范数（曼哈顿距离）

$$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$$

#### L2 范数（欧几里得范数）

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$$

#### 余弦相似度

余弦相似度是 NLP 中衡量两个向量相似程度的重要指标：

$$\text{sim}(\mathbf{a}, \mathbf{b}) = \cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \cdot \|\mathbf{b}\|_2}$$

其中：
- $\mathbf{a} \cdot \mathbf{b}$ 是两个向量的点积
- $\|\mathbf{a}\|_2$ 和 $\|\mathbf{b}\|_2$ 分别是两个向量的 L2 范数
- 返回值范围为 $[-1, 1]$
  - 值为 1 表示方向完全相同（最相似）
  - 值为 0 表示正交（无相关性）
  - 值为 -1 表示方向完全相反（最不相似）

**示例**：

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 点积
dot = np.dot(a, b)  # 32

# 余弦相似度
cos_sim = dot / (np.linalg.norm(a) * np.linalg.norm(b))
print(cos_sim)  # 0.9746
```

### 矩阵运算

#### 矩阵乘法

矩阵乘法是深度学习中最核心的运算。对于矩阵 $\mathbf{A} \in \mathbb{R}^{m \times k}$ 和 $\mathbf{B} \in \mathbb{R}^{k \times n}$：

$$\mathbf{C} = \mathbf{A} \mathbf{B}, \quad C_{ij} = \sum_{l=1}^{k} A_{il} B_{lj}$$

其中 $\mathbf{C} \in \mathbb{R}^{m \times n}$。

**示例**：

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \times 5 + 2 \times 7 & 1 \times 6 + 2 \times 8 \\ 3 \times 5 + 4 \times 7 & 3 \times 6 + 4 \times 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

::: warning 维度要求
矩阵乘法要求第一个矩阵的列数等于第二个矩阵的行数：$(m, \color{red}{k}) \times (\color{red}{k}, n) = (m, n)$
:::

#### 矩阵转置

矩阵转置是将矩阵的行和列互换：

$$\mathbf{A}^T_{ij} = \mathbf{A}_{ji}$$

$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

### 广播机制

在深度学习框架（如 PyTorch）中，**广播**（Broadcasting）允许不同形状的张量进行运算：

```python
# 矩形 + 向量（广播）
A = torch.randn(3, 4)  # 形状 (3, 4)
b = torch.randn(4)      # 形状 (4,)
C = A + b               # 形状 (3, 4)，b 被广播到每一行
```

---

## 概率论基础

### 随机变量与概率分布

**随机变量**是一个取值不确定的变量，用大写字母表示（如 $X$），其具体取值用小写字母表示（如 $x$）。

#### 离散随机变量

对于离散随机变量 $X$，其**概率质量函数**（PMF）为：

$$P(X = x_i) = p_i, \quad \sum_{i} p_i = 1, \quad p_i \geq 0$$

示例：掷骰子，$P(X = i) = \frac{1}{6}$，$i = 1, 2, 3, 4, 5, 6$

#### 连续随机变量

对于连续随机变量 $X$，其**概率密度函数**（PDF）为 $f(x)$，满足：

$$P(a \leq X \leq b) = \int_a^b f(x) dx, \quad \int_{-\infty}^{+\infty} f(x) dx = 1$$

### 期望与方差

#### 期望（Expectation）

期望是随机变量的"平均值"：

**离散情况**：
$$E[X] = \sum_{i} x_i P(X = x_i)$$

**连续情况**：
$$E[X] = \int_{-\infty}^{+\infty} x f(x) dx$$

#### 方差（Variance）

方差衡量随机变量的离散程度：

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

其中：
- $E[X]$ 是期望
- $\text{Var}(X)$ 越大，表示数据越分散

### 条件概率与贝叶斯定理

#### 条件概率

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

含义：在事件 $B$ 已经发生的条件下，事件 $A$ 发生的概率。

#### 贝叶斯定理

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

其中：
- $P(A)$ 是**先验概率**（在观测到 $B$ 之前对 $A$ 的信念）
- $P(A|B)$ 是**后验概率**（在观测到 $B$ 之后对 $A$ 的更新信念）
- $P(B|A)$ 是**似然**（在 $A$ 为真的条件下观测到 $B$ 的概率）
- $P(B)$ 是**证据**（归一化常数）

::: info 贝叶斯定理在 NLP 中的应用
在垃圾邮件过滤中：
- $A$ = 邮件是垃圾邮件
- $B$ = 邮件中包含"免费"这个词
- $P(A|B)$ = 包含"免费"的邮件是垃圾邮件的概率

通过贝叶斯定理，我们可以用已知的统计数据来计算这个概率。
:::

### Softmax 函数

Softmax 函数将一个实数向量转换为概率分布，是 NLP 中最常用的函数之一：

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

其中：
- $z_i$ 是输入向量的第 $i$ 个元素
- $K$ 是类别数
- 输出满足 $\sum_{i} \text{Softmax}(z_i) = 1$，且每个值都在 $(0, 1)$ 之间

**示例**：

```
输入：z = [2.0, 1.0, 0.1]
计算：
  e^2.0 = 7.389
  e^1.0 = 2.718
  e^0.1 = 1.105
  总和 = 11.212

输出：[7.389/11.212, 2.718/11.212, 1.105/11.212]
     = [0.659, 0.242, 0.099]
```

```python
import torch

z = torch.tensor([2.0, 1.0, 0.1])
probs = torch.softmax(z, dim=0)
print(probs)  # tensor([0.6590, 0.2424, 0.0986])
```

---

## 信息论基础

### 信息量

一个事件的**信息量**与其发生的概率成反比——越不可能发生的事件，携带的信息越多：

$$I(x) = -\log_2 P(x)$$

其中：
- $P(x)$ 是事件 $x$ 发生的概率
- $I(x)$ 的单位是**比特**（bit）

**示例**：
- 抛硬币正面朝上：$P = 0.5$，$I = -\log_2 0.5 = 1$ bit
- 掷骰子掷出6：$P = 1/6$，$I = -\log_2 (1/6) \approx 2.58$ bit

### 熵（Entropy）

**熵**是随机变量不确定性的度量，即信息量的期望值：

$$H(X) = -\sum_{i} P(x_i) \log_2 P(x_i)$$

其中：
- $P(x_i)$ 是随机变量 $X$ 取值为 $x_i$ 的概率
- $H(X) \geq 0$，熵越大表示不确定性越高

**示例**：

公平硬币（$P(\text{正}) = P(\text{反}) = 0.5$）：
$$H = -(0.5 \times \log_2 0.5 + 0.5 \times \log_2 0.5) = 1 \text{ bit}$$

不公平硬币（$P(\text{正}) = 0.9, P(\text{反}) = 0.1$）：
$$H = -(0.9 \times \log_2 0.9 + 0.1 \times \log_2 0.1) \approx 0.469 \text{ bits}$$

不公平硬币的熵更低，因为结果更确定。

### 交叉熵（Cross-Entropy）

**交叉熵**衡量用分布 $q$ 来编码来自分布 $p$ 的数据时的平均编码长度：

$$H(p, q) = -\sum_{i} p(x_i) \log q(x_i)$$

其中：
- $p$ 是真实分布（ground truth）
- $q$ 是预测分布（模型输出）
- 当 $p = q$ 时，交叉熵等于熵 $H(p)$，此时编码最优

::: info 交叉熵在 NLP 中的应用
交叉熵是 NLP 中最常用的**损失函数**。在语言模型中：
- $p$ 是真实的下一个词的分布（one-hot 向量）
- $q$ 是模型预测的概率分布
- 最小化交叉熵 = 让模型的预测尽可能接近真实分布
:::

### KL 散度（Kullback-Leibler Divergence）

**KL 散度**衡量两个概率分布之间的差异：

$$D_{KL}(p \| q) = \sum_{i} p(x_i) \log \frac{p(x_i)}{q(x_i)} = H(p, q) - H(p)$$

其中：
- $D_{KL} \geq 0$，当且仅当 $p = q$ 时等于 0
- KL 散度**不对称**：$D_{KL}(p \| q) \neq D_{KL}(q \| p)$

**与交叉熵的关系**：

$$H(p, q) = H(p) + D_{KL}(p \| q)$$

由于 $H(p)$ 是常数（真实分布的熵不变），最小化交叉熵等价于最小化 KL 散度。

---

## 激活函数

激活函数为神经网络引入**非线性**，使其能够学习复杂的模式。

### Sigmoid 函数

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- 输出范围：$(0, 1)$
- 常用于二分类的输出层和门控机制（如 LSTM 的门）
- 缺点：存在梯度消失问题

### Tanh 函数

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- 输出范围：$(-1, 1)$
- 零中心化，比 Sigmoid 更好
- 常用于 RNN 的隐藏状态

### ReLU 函数

$$\text{ReLU}(x) = \max(0, x)$$

- 计算简单，收敛快
- 解决了梯度消失问题（正区间）
- 缺点：负区间梯度为 0（"死亡 ReLU"）

### GELU 函数

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。

- GELU 是 Transformer 和 BERT 中使用的激活函数
- 比 ReLU 更平滑，效果通常更好

---

## 小结

本章介绍了理解 NLP 算法所必需的数学基础：

| 领域 | 核心概念 | NLP 中的应用 |
|------|---------|-------------|
| 线性代数 | 向量、矩阵、点积、余弦相似度 | 词向量、注意力计算 |
| 概率论 | 条件概率、贝叶斯、Softmax | 语言模型、分类 |
| 信息论 | 熵、交叉熵、KL 散度 | 损失函数、模型评估 |
| 激活函数 | Sigmoid、ReLU、GELU | 神经网络非线性变换 |

这些数学工具将在后续章节中反复使用。如果某些概念不够熟悉，建议在学习后续章节时回头参考本章。
