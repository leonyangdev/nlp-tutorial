# 第十七章：位置编码与 Layer Normalization

本章详细讲解 Transformer 中两个重要的技术细节：**位置编码**（Positional Encoding）和**层归一化**（Layer Normalization）。

---

## 位置编码

### 为什么需要位置编码？

Transformer 的自注意力机制是**排列不变的**（permutation invariant）——如果打乱输入序列的顺序，自注意力的输出只是相应地打乱，不会改变每个位置的值。

```
输入1: ["我", "爱", "你"] → 自注意力 → [a, b, c]
输入2: ["你", "爱", "我"] → 自注意力 → [c, b, a]（只是打乱了顺序）
```

但语言是有顺序的！"我爱你"和"你爱我"含义完全不同。因此需要位置编码来注入位置信息。

### 正弦位置编码

原论文使用正弦和余弦函数：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $pos$ 是位置索引（0, 1, 2, ...）
- $i$ 是维度索引（0, 1, ..., $d_{model}/2 - 1$）
- $d_{model}$ 是模型维度

### 为什么选择正弦函数？

#### 1. 每个位置都有唯一编码

不同位置的正弦/余弦值组合是唯一的，可以区分不同位置。

#### 2. 相对位置可以通过线性变换表示

对于任意固定偏移 $k$，存在一个线性变换 $M_k$，使得：

$$PE_{pos+k} = M_k \cdot PE_{pos}$$

这意味着模型可以通过学习来捕捉**相对位置关系**。

**数学证明**：

$$\sin(pos + k) = \sin(pos)\cos(k) + \cos(pos)\sin(k)$$
$$\cos(pos + k) = \cos(pos)\cos(k) - \sin(pos)\sin(k)$$

因此：

$$\begin{bmatrix} \sin(pos+k) \\ \cos(pos+k) \end{bmatrix} = \begin{bmatrix} \cos(k) & \sin(k) \\ -\sin(k) & \cos(k) \end{bmatrix} \begin{bmatrix} \sin(pos) \\ \cos(pos) \end{bmatrix}$$

#### 3. 值有界

正弦和余弦函数的值都在 $[-1, 1]$ 之间，不会随位置增大而爆炸。

### 代码实现

```python
import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

### 可视化位置编码

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_positional_encoding(d_model=128, max_len=100):
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(pe.T, aspect='auto', cmap='RdBu')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding')
    plt.colorbar()
    plt.show()

visualize_positional_encoding()
```

### 可学习的位置编码

除了正弦位置编码，还可以使用**可学习的位置编码**：

```python
class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.position_embeddings = torch.nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        position_emb = self.position_embeddings(positions)
        return x + position_emb
```

**对比**：

| 类型 | 优点 | 缺点 | 使用模型 |
|------|------|------|---------|
| 正弦编码 | 无需训练，可外推到更长序列 | 固定不变 | 原始 Transformer |
| 可学习编码 | 可以适应数据 | 无法外推到训练时未见过的长度 | BERT, GPT |

---

## Layer Normalization

### 为什么需要归一化？

深度神经网络在训练时会遇到**内部协变量偏移**（Internal Covariate Shift）问题——每一层的输入分布会随着前面层参数的更新而不断变化，导致训练不稳定。

归一化可以：
- 稳定训练过程
- 加速收敛
- 允许使用更大的学习率

### Batch Normalization vs Layer Normalization

| 特性 | Batch Norm | Layer Norm |
|------|-----------|------------|
| 归一化维度 | 沿 batch 维度 | 沿特征维度 |
| 依赖 batch size | 是 | 否 |
| 适用于变长序列 | 困难 | 容易 |
| 训练/推理一致性 | 不一致 | 一致 |

**为什么 Transformer 用 Layer Norm？**

1. NLP 任务中序列长度是变化的，Batch Norm 不适用
2. Layer Norm 不依赖 batch size，更稳定
3. 推理时行为与训练时一致

### Layer Norm 的数学定义

对于输入向量 $\mathbf{x} \in \mathbb{R}^{d}$：

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：
- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$ 是均值
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$ 是方差
- $\gamma, \beta \in \mathbb{R}^{d}$ 是可学习的缩放和偏移参数
- $\epsilon$ 是一个很小的常数（如 $10^{-5}$），防止除以零

### 代码实现

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

# 使用 PyTorch 内置版本
layer_norm = nn.LayerNorm(d_model=512)
x = torch.randn(2, 10, 512)
output = layer_norm(x)  # (2, 10, 512)
```

---

## 残差连接

### 什么是残差连接？

残差连接（Residual Connection）是将子层的输入直接加到子层的输出上：

$$\text{output} = x + \text{SubLayer}(x)$$

### 为什么需要残差连接？

#### 1. 缓解梯度消失

在反向传播时，梯度可以通过残差连接直接传递：

$$\frac{\partial \text{output}}{\partial x} = 1 + \frac{\partial \text{SubLayer}(x)}{\partial x}$$

即使 $\frac{\partial \text{SubLayer}(x)}{\partial x}$ 很小，梯度也不会消失。

#### 2. 恒等映射

如果某个子层不需要做任何变换，残差连接可以让它学习恒等映射 $f(x) = x$，这比学习零映射更容易。

### Pre-Norm vs Post-Norm

Transformer 中 Layer Norm 的位置有两种变体：

**Post-Norm**（原始 Transformer）：

$$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

**Pre-Norm**（后续改进）：

$$\text{output} = x + \text{SubLayer}(\text{LayerNorm}(x))$$

| 变体 | 优点 | 缺点 |
|------|------|------|
| Post-Norm | 理论上更优 | 训练不稳定，需要 warmup |
| Pre-Norm | 训练更稳定 | 可能略逊于 Post-Norm |

现代大模型（如 GPT、LLaMA）通常使用 **Pre-Norm**。

---

## Transformer Block 完整实现

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm 变体
        # 子层1：多头自注意力
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = residual + self.dropout1(attn_output)
        
        # 子层2：前馈网络
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout2(ffn_output)
        
        return x

# 使用
block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048)
x = torch.randn(2, 10, 512)
output = block(x)  # (2, 10, 512)
```

---

## 小结

| 组件 | 作用 | 关键公式 |
|------|------|---------|
| 位置编码 | 注入位置信息 | $PE_{pos} = \sin/\cos(pos/10000^{2i/d})$ |
| Layer Norm | 稳定训练 | $\gamma \cdot \frac{x-\mu}{\sigma} + \beta$ |
| 残差连接 | 缓解梯度消失 | $x + \text{SubLayer}(x)$ |
| Pre-Norm | 更稳定的训练方式 | $x + \text{SubLayer}(\text{LN}(x))$ |

这些技术细节虽然看起来"不那么性感"，但它们是 Transformer 能够成功训练的关键。没有位置编码，模型无法感知顺序；没有 Layer Norm 和残差连接，深层 Transformer 难以训练。
