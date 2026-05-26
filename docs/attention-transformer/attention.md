# 第十四章：注意力机制

**注意力机制**（Attention Mechanism）是现代 NLP 最重要的技术突破之一。它解决了 Seq2Seq 模型的信息瓶颈问题，并成为 Transformer 架构的核心组件。

---

## 从 Seq2Seq 到注意力

### 信息瓶颈问题

回顾标准 Seq2Seq 模型的问题：编码器将整个输入序列压缩为一个固定长度的上下文向量 $\mathbf{c} = \mathbf{h}_T$，这在输入序列很长时会导致信息丢失。

### 注意力的核心思想

注意力机制的核心思想是：**解码器在生成每个输出词时，应该"关注"输入序列中最相关的部分**。

```
生成 "我" 时 → 主要关注 "I"
生成 "爱" 时 → 主要关注 "love"
生成 "自然语言处理" 时 → 主要关注 "NLP"
```

而不是像标准 Seq2Seq 那样，所有输出都依赖同一个固定的上下文向量。

---

## Bahdanau 注意力

### 历史背景

Bahdanau 注意力由 Bahdanau 等人于 2014 年提出（"Neural Machine Translation by Jointly Learning to Align and Translate"），是第一个成功的注意力机制。

### 工作原理

**第 1 步：计算注意力分数**

对于解码器时间步 $t$ 和编码器时间步 $j$，计算它们之间的"相关程度"：

$$e_{t,j} = \mathbf{v}^T \tanh(\mathbf{W}_s \mathbf{s}_{t-1} + \mathbf{W}_h \mathbf{h}_j)$$

其中：
- $\mathbf{s}_{t-1}$ 是解码器上一个时间步的隐藏状态
- $\mathbf{h}_j$ 是编码器第 $j$ 个时间步的隐藏状态
- $\mathbf{W}_s, \mathbf{W}_h, \mathbf{v}$ 是可学习的参数

**第 2 步：Softmax 归一化**

将注意力分数转换为概率分布：

$$\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{k=1}^{T_x} \exp(e_{t,k})}$$

其中 $\alpha_{t,j}$ 是注意力权重，表示解码器时间步 $t$ 对编码器时间步 $j$ 的关注程度。

**第 3 步：计算上下文向量**

使用注意力权重对编码器的隐藏状态进行加权求和：

$$\mathbf{c}_t = \sum_{j=1}^{T_x} \alpha_{t,j} \mathbf{h}_j$$

注意：这里的 $\mathbf{c}_t$ 是**动态的**——每个解码器时间步都有不同的上下文向量！

**第 4 步：更新解码器状态**

将上下文向量与当前输入一起送入解码器：

$$\mathbf{s}_t = \text{Decoder}(\mathbf{y}_{t-1}, \mathbf{s}_{t-1}, \mathbf{c}_t)$$

---

## Luong 注意力

### 与 Bahdanau 的区别

Luong 注意力（2015）提出了几种更简单的注意力分数计算方式：

| 方法 | 公式 | 说明 |
|------|------|------|
| **点积**（Dot） | $e_{t,j} = \mathbf{s}_t^T \mathbf{h}_j$ | 最简单，要求维度相同 |
| **通用**（General） | $e_{t,j} = \mathbf{s}_t^T \mathbf{W} \mathbf{h}_j$ | 引入可学习的权重矩阵 |
| **拼接**（Concat） | $e_{t,j} = \mathbf{v}^T \tanh(\mathbf{W}[\mathbf{s}_t; \mathbf{h}_j])$ | 类似 Bahdanau |

### 计算流程

```
编码器隐藏状态：h₁, h₂, ..., hₙ
解码器当前状态：sₜ

1. 计算分数：eₜ = [sₜᵀh₁, sₜᵀh₂, ..., sₜᵀhₙ]  (点积注意力)
2. Softmax：αₜ = softmax(eₜ)
3. 上下文向量：cₜ = Σ αₜⱼ hⱼ
4. 输出：ŷₜ = tanh(Wc[cₜ; sₜ])
```

---

## 注意力的直觉理解

### 注意力作为"软对齐"

在机器翻译中，注意力可以被理解为源语言和目标语言之间的**软对齐**（soft alignment）：

```
英文：  I    love   NLP
        ↓     ↓     ↓
中文：  我    爱    自然语言处理

注意力权重（热力图）：
         I    love   NLP
我      0.8   0.1   0.1
爱      0.1   0.7   0.2
NLP     0.0   0.1   0.9
```

每个中文词对英文词有不同的关注程度，形成一个"软对齐"矩阵。

### 注意力权重可视化

```python
import matplotlib.pyplot as plt
import numpy as np

# 注意力权重矩阵
attention_weights = np.array([
    [0.8, 0.1, 0.1],  # "我" 主要关注 "I"
    [0.1, 0.7, 0.2],  # "爱" 主要关注 "love"
    [0.0, 0.1, 0.9],  # "自然语言处理" 主要关注 "NLP"
])

# 可视化
plt.figure(figsize=(8, 6))
plt.imshow(attention_weights, cmap='YlOrRd')
plt.xticks(range(3), ['I', 'love', 'NLP'])
plt.yticks(range(3), ['我', '爱', '自然语言处理'])
plt.colorbar()
plt.title('Attention Weights')
plt.show()
```

---

## 注意力的数学本质

注意力机制本质上是一种**加权平均**操作。让我们用更抽象的方式来理解：

### Query-Key-Value 框架

注意力可以统一表示为 **Query-Key-Value**（QKV）框架：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中：
- $Q$（Query）：查询矩阵，代表"我在找什么"
- $K$（Key）：键矩阵，代表"我能提供什么"
- $V$（Value）：值矩阵，代表"我的实际内容"
- $d_k$ 是 Key 的维度（缩放因子，防止点积过大）

在 Bahdanau 注意力中：
- $Q = \mathbf{s}_{t-1}$（解码器状态）
- $K = V = [\mathbf{h}_1, ..., \mathbf{h}_{T_x}]$（编码器隐藏状态）

---

## 自注意力（Self-Attention）预览

在标准注意力中，Query 来自解码器，Key/Value 来自编码器。**自注意力**（Self-Attention）则是序列自身内部的注意力：

$$\text{Self-Attention}(X) = \text{Softmax}\left(\frac{XW_Q (XW_K)^T}{\sqrt{d_k}}\right) XW_V$$

其中 $X$ 是输入序列的表示矩阵。

**自注意力的优势**：
- 可以直接建模序列中任意两个位置之间的关系
- 不受距离限制（不像 RNN 有长期依赖问题）
- 可以并行计算（不像 RNN 需要逐步处理）

自注意力是 Transformer 的核心组件，我们将在后续章节详细讲解。

---

## PyTorch 实现：带注意力的 Seq2Seq

### 注意力层

```python
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, dec_hidden, enc_outputs):
        # dec_hidden: (batch, hidden) — 解码器当前隐藏状态
        # enc_outputs: (batch, src_len, hidden) — 编码器所有隐藏状态
        
        # 计算注意力分数
        score = self.v(torch.tanh(
            self.W_s(dec_hidden).unsqueeze(1) + self.W_h(enc_outputs)
        ))  # (batch, src_len, 1)
        
        # Softmax
        attn_weights = torch.softmax(score, dim=1)  # (batch, src_len, 1)
        
        # 上下文向量
        context = torch.bmm(attn_weights.transpose(1, 2), enc_outputs)
        # context: (batch, 1, hidden)
        
        return context.squeeze(1), attn_weights.squeeze(2)
```

### 带注意力的解码器

```python
class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, h_n, c_n, enc_outputs):
        # x: (batch, 1)
        embedded = self.embedding(x)  # (batch, 1, embed_dim)
        
        # 计算注意力
        context, attn_weights = self.attention(h_n[-1], enc_outputs)
        # context: (batch, hidden)
        
        # 拼接输入和上下文
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        
        # LSTM
        output, (h_n, c_n) = self.lstm(lstm_input, (h_n, c_n))
        
        # 输出
        logits = self.fc(output.squeeze(1))
        return logits, h_n, c_n, attn_weights
```

---

## 小结

| 概念 | 说明 |
|------|------|
| 注意力机制 | 允许模型动态关注输入的不同部分 |
| Bahdanau 注意力 | 加性注意力，使用前馈网络计算分数 |
| Luong 注意力 | 乘性注意力，使用点积计算分数 |
| 注意力权重 | 表示关注程度的概率分布 |
| QKV 框架 | Query-Key-Value 统一表示 |
| 自注意力 | 序列内部的注意力，Transformer 的核心 |

注意力机制解决了 Seq2Seq 的信息瓶颈问题，并为 Transformer 的诞生奠定了基础。下一章我们将详细讲解 Transformer 架构。
