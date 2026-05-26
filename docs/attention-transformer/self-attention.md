# 第十六章：自注意力与多头注意力

**自注意力**（Self-Attention）是 Transformer 最核心的创新。本章将从直觉到数学，详细讲解自注意力的工作原理。

---

## 自注意力的直觉

### 什么是自注意力？

自注意力让序列中的**每个位置**都能关注到序列中的**所有其他位置**，从而捕捉全局的依赖关系。

**示例**：理解句子 "小明 在 北京 大学 学习 人工智能"

当处理"学习"这个词时，自注意力会计算它与其他所有词的相关程度：

```
小明(0.3)  在(0.05)  北京(0.05)  大学(0.1)  学习(0.4)  人工智能(0.1)
```

- "学习"对"小明"的注意力权重很高（因为"小明"是主语）
- "学习"对"人工智能"的注意力权重较高（因为"人工智能"是宾语）
- "学习"对"在"的注意力权重较低

### 自注意力 vs RNN

| 特性 | RNN | 自注意力 |
|------|-----|---------|
| 路径长度 | $O(n)$（最远距离） | $O(1)$（直接连接） |
| 并行化 | 无法并行 | 完全并行 |
| 长距离依赖 | 受限于梯度消失 | 直接建模 |

---

## Query-Key-Value 框架

### 核心思想

自注意力使用 **QKV**（Query-Key-Value）框架：

- **Query（查询）**：当前位置想要查找什么信息
- **Key（键）**：每个位置能提供什么信息的"标签"
- **Value（值）**：每个位置的实际内容

**类比**：在图书馆查书时：
- Query = 你的查询关键词
- Key = 每本书的索引标签
- Value = 书的实际内容

### 计算流程

给定输入序列 $X \in \mathbb{R}^{n \times d_{model}}$：

**第 1 步：线性投影**

将输入分别投影为 Q、K、V：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中：
- $W^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $W^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W^V \in \mathbb{R}^{d_{model} \times d_v}$

**第 2 步：计算注意力分数**

$$\text{scores} = QK^T \in \mathbb{R}^{n \times n}$$

$\text{scores}_{ij}$ 表示位置 $i$ 的 Query 与位置 $j$ 的 Key 的点积，衡量它们的"匹配程度"。

**第 3 步：缩放**

$$\text{scaled\_scores} = \frac{QK^T}{\sqrt{d_k}}$$

**第 4 步：Softmax 归一化**

$$\text{attention\_weights} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}$$

**第 5 步：加权求和**

$$\text{output} = \text{attention\_weights} \cdot V \in \mathbb{R}^{n \times d_v}$$

### 完整公式

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

---

## 详细计算示例

假设输入序列有 3 个词，$d_{model} = 4$，$d_k = d_v = 2$：

**输入**：

$$X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix}$$

**投影矩阵**（简化示例）：

$$W^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{bmatrix}, \quad W^K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad W^V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 1 & 1 \end{bmatrix}$$

**计算 Q、K、V**：

$$Q = XW^Q = \begin{bmatrix} 2 & 1 \\ 0 & 2 \\ 1 & 1 \end{bmatrix}$$

$$K = XW^K = \begin{bmatrix} 0 & 2 \\ 2 & 0 \\ 1 & 1 \end{bmatrix}$$

$$V = XW^V = \begin{bmatrix} 1 & 0 \\ 1 & 2 \\ 1 & 1 \end{bmatrix}$$

**计算 $QK^T$**：

$$QK^T = \begin{bmatrix} 2 & 1 \\ 0 & 2 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 0 & 2 & 1 \\ 2 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 4 & 3 \\ 4 & 0 & 2 \\ 2 & 2 & 2 \end{bmatrix}$$

**缩放**（$\sqrt{d_k} = \sqrt{2} \approx 1.41$）：

$$\frac{QK^T}{\sqrt{2}} = \begin{bmatrix} 1.41 & 2.83 & 2.12 \\ 2.83 & 0 & 1.41 \\ 1.41 & 1.41 & 1.41 \end{bmatrix}$$

**Softmax**（逐行）：

$$\text{attention\_weights} = \begin{bmatrix} 0.12 & 0.52 & 0.26 \\ 0.65 & 0.04 & 0.16 \\ 0.33 & 0.33 & 0.33 \end{bmatrix}$$

**加权求和**：

$$\text{output} = \text{attention\_weights} \cdot V = \begin{bmatrix} 0.12 & 0.52 & 0.26 \\ 0.65 & 0.04 & 0.16 \\ 0.33 & 0.33 & 0.33 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 2 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 0.90 & 1.30 \\ 0.85 & 0.24 \\ 1.00 & 1.00 \end{bmatrix}$$

---

## 多头注意力（Multi-Head Attention）

### 为什么要多头？

单个注意力头只能捕捉一种类型的关系。多头注意力允许模型**同时关注不同类型的信息**：

```
头1：关注语法关系（主语-谓语-宾语）
头2：关注语义相似性
头3：关注位置邻近性
头4：关注指代关系（代词-先行词）
...
```

### 计算流程

**第 1 步：线性投影**

将 Q、K、V 分别投影 $h$ 次：

$$Q_i = XW_i^Q, \quad K_i = XW_i^K, \quad V_i = XW_i^V$$

其中 $i = 1, ..., h$。

**第 2 步：并行计算注意力**

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

**第 3 步：拼接**

$$\text{MultiHead} = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

其中 $W^O \in \mathbb{R}^{hd_v \times d_{model}}$ 是输出投影矩阵。

### 维度分析

假设 $d_{model} = 512$，$h = 8$：

```
每个头的维度：d_k = d_v = d_model / h = 512 / 8 = 64

输入 X: (n, 512)
Q_i, K_i: (n, 64)
V_i: (n, 64)
head_i: (n, 64)

拼接后：(n, 64 × 8) = (n, 512)
输出投影后：(n, 512)
```

### 多头注意力的优势

| 优势 | 说明 |
|------|------|
| 多种关系建模 | 不同的头可以关注不同类型的关系 |
| 表达能力更强 | 多个子空间的表示比单个更丰富 |
| 计算效率 | 每个头的维度更小，总计算量与单头相同 |

---

## PyTorch 实现

### 手动实现自注意力

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        Q = self.W_q(x)  # (batch, seq_len, d_k)
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_v)
        
        # 计算注意力分数
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)
        # scores: (batch, seq_len, seq_len)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.bmm(attn_weights, V)
        # output: (batch, seq_len, d_v)
        
        return output, attn_weights
```

### 手动实现多头注意力

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 线性投影
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 拆分为多头
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # (batch, n_heads, seq_len, d_k)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        # output: (batch, n_heads, seq_len, d_k)
        
        # 拼接
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.W_o(output)
        # output: (batch, seq_len, d_model)
        
        return output, attn_weights
```

### 使用 PyTorch 内置模块

```python
import torch.nn as nn

mha = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    batch_first=True
)

x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
output, attn_weights = mha(x, x, x)  # self-attention
print(output.shape)  # (2, 10, 512)
```

---

## 注意力模式可视化

通过可视化注意力权重，我们可以理解模型学到了什么模式：

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, ax in enumerate(axes.flat):
        sns.heatmap(attention_weights[i], xticklabels=tokens, 
                   yticklabels=tokens, ax=ax, cmap='Blues')
        ax.set_title(f'Head {i+1}')
    plt.tight_layout()
    plt.show()
```

常见的注意力模式：
- **对角线模式**：关注自身和相邻位置
- **垂直模式**：某些位置被所有位置关注（如 [CLS] token）
- **块状模式**：同一短语内的词相互关注

---

## 小结

| 概念 | 说明 |
|------|------|
| 自注意力 | 序列内部每个位置关注所有其他位置 |
| QKV 框架 | Query=查询, Key=索引, Value=内容 |
| 缩放因子 | $\sqrt{d_k}$ 防止点积过大导致梯度消失 |
| 多头注意力 | 多个注意力头并行计算，捕捉不同类型的关系 |
| 注意力权重 | 表示位置间的关注程度，可可视化 |

自注意力是 Transformer 的灵魂。理解了它，就理解了 Transformer 为什么如此强大。
