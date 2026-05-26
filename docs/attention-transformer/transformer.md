# 第十五章：Transformer 架构

2017 年，Google 团队发表了划时代的论文 **"Attention Is All You Need"**，提出了 **Transformer** 架构。它完全抛弃了循环结构，仅使用注意力机制来建模序列中的依赖关系，彻底改变了 NLP 的研究范式。

---

## 为什么需要 Transformer？

### RNN/LSTM 的局限

| 问题 | 说明 |
|------|------|
| 无法并行计算 | 必须按时间步逐步处理，训练慢 |
| 长距离依赖 | 虽然 LSTM 缓解了梯度消失，但仍然有限 |
| 信息瓶颈 | 即使有注意力，编码器仍然需要逐步处理 |

### Transformer 的优势

| 优势 | 说明 |
|------|------|
| 完全并行 | 所有位置可以同时计算，训练速度快 |
| 长距离依赖 | 通过自注意力直接建模任意两个位置的关系 |
| 可扩展性 | 可以通过增加层数和参数量来提升性能 |

---

## 整体架构

Transformer 采用**编码器-解码器**（Encoder-Decoder）架构：

```
┌─────────────────────────────────────────────────────────────┐
│                      Transformer                            │
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐       │
│  │      编码器          │    │      解码器          │       │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │       │
│  │  │  多头自注意力   │  │    │  │  掩码多头自注意力│  │       │
│  │  └───────┬───────┘  │    │  └───────┬───────┘  │       │
│  │          ↓          │    │          ↓          │       │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │       │
│  │  │  前馈神经网络   │  │    │  │  多头交叉注意力  │  │       │
│  │  └───────┬───────┘  │    │  └───────┬───────┘  │       │
│  │          ↓          │    │          ↓          │       │
│  │    × N 层堆叠       │    │  ┌───────────────┐  │       │
│  │                     │    │  │  前馈神经网络   │  │       │
│  └─────────────────────┘    │  └───────┬───────┘  │       │
│           │                 │          ↓          │       │
│           │                 │    × N 层堆叠       │       │
│           │                 └─────────────────────┘       │
│           ↓                          ↓                     │
│  ┌─────────────┐            ┌─────────────┐              │
│  │  输入嵌入    │            │  输出嵌入    │              │
│  │ + 位置编码   │            │ + 位置编码   │              │
│  └─────────────┘            └─────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

---

## 编码器（Encoder）

编码器由 $N$ 个相同的层堆叠而成（原论文中 $N=6$）。每一层包含两个子层：

### 子层 1：多头自注意力（Multi-Head Self-Attention）

自注意力让序列中的每个位置都能关注到序列中的所有其他位置：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中：
- $Q = K = V = X$（自注意力，输入序列自身）
- $h$ 是注意力头数（原论文中 $h=8$）
- $d_k$ 是每个头的 Key 维度
- $W_i^Q, W_i^K, W_i^V$ 是每个头的投影矩阵
- $W^O$ 是输出投影矩阵

### 子层 2：前馈神经网络（Feed-Forward Network）

每个位置独立地通过一个两层的前馈网络：

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

其中：
- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$
- $d_{ff}$ 是前馈网络的隐藏层维度（原论文中 $d_{ff} = 4 \times d_{model} = 2048$）

### 残差连接与层归一化

每个子层都使用**残差连接**（Residual Connection）和**层归一化**（Layer Normalization）：

$$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

其中：
- $x + \text{SubLayer}(x)$ 是残差连接，帮助梯度流动
- LayerNorm 稳定训练过程

---

## 解码器（Decoder）

解码器同样由 $N$ 个相同的层堆叠而成。每一层包含三个子层：

### 子层 1：掩码多头自注意力（Masked Multi-Head Self-Attention）

与编码器的自注意力类似，但增加了**掩码**（Mask），防止解码器"看到"未来的词：

$$\text{MaskedAttention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

其中掩码矩阵 $M$ 的定义为：

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

**为什么需要掩码？**

在自回归生成中，解码器在预测第 $t$ 个词时只能看到前 $t-1$ 个词，不能"偷看"未来的词：

```
预测 "爱" 时：
输入：  <SOS> 我
掩码：  可见  可见  不可见  不可见
              爱    自然语言处理
```

### 子层 2：多头交叉注意力（Multi-Head Cross-Attention）

交叉注意力让解码器关注编码器的输出：

- $Q$ 来自解码器
- $K, V$ 来自编码器的输出

这类似于 Seq2Seq 中的注意力机制，但使用了多头注意力。

### 子层 3：前馈神经网络

与编码器相同。

---

## 位置编码（Positional Encoding）

### 为什么需要位置编码？

Transformer 没有循环结构，无法感知词语的顺序。例如：

```
"我 爱 你" 和 "你 爱 我" 在没有位置编码时，自注意力的输出是相同的！
```

因此需要**位置编码**来注入位置信息。

### 正弦位置编码

原论文使用正弦和余弦函数生成位置编码：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $pos$ 是位置索引（0, 1, 2, ...）
- $i$ 是维度索引（0, 1, ..., $d_{model}/2 - 1$）
- $d_{model}$ 是模型维度

### 位置编码的直觉

```
位置 0: [sin(0/10000⁰), cos(0/10000⁰), sin(0/10000²), ...]
位置 1: [sin(1/10000⁰), cos(1/10000⁰), sin(1/10000²), ...]
位置 2: [sin(2/10000⁰), cos(2/10000⁰), sin(2/10000²), ...]
```

每个位置都有唯一的位置编码，且不同位置之间的编码具有可区分的模式。

### 最终输入

输入嵌入与位置编码相加：

$$\text{input} = \text{TokenEmbedding}(x) + \text{PositionalEncoding}(pos)$$

---

## 缩放点积注意力

### 为什么需要缩放？

当 $d_k$ 较大时，点积 $QK^T$ 的值会很大，导致 Softmax 的梯度很小（进入饱和区）。

缩放因子 $\sqrt{d_k}$ 确保点积的方差保持在合理范围内：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**数学解释**：

假设 $Q$ 和 $K$ 的每个元素都是独立的均值为 0、方差为 1 的随机变量，那么 $Q \cdot K$ 的方差为 $d_k$。除以 $\sqrt{d_k}$ 后，方差恢复为 1。

---

## 超参数汇总

| 超参数 | 符号 | 原论文值 | 说明 |
|--------|------|---------|------|
| 模型维度 | $d_{model}$ | 512 | 输入/输出的向量维度 |
| 前馈维度 | $d_{ff}$ | 2048 | FFN 的隐藏层维度 |
| 注意力头数 | $h$ | 8 | 多头注意力的头数 |
| 每头维度 | $d_k = d_v$ | 64 | $d_{model} / h$ |
| 编码器层数 | $N$ | 6 | 编码器的层数 |
| 解码器层数 | $N$ | 6 | 解码器的层数 |
| Dropout | - | 0.1 | Dropout 比率 |

---

## Transformer 的计算复杂度

自注意力的时间复杂度为 $O(n^2 \cdot d)$，其中 $n$ 是序列长度，$d$ 是模型维度：

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 自注意力 | $O(n^2 d)$ | 每个位置关注所有位置 |
| FFN | $O(n d^2)$ | 每个位置独立计算 |
| 总计 | $O(n^2 d + n d^2)$ | 当 $n < d$ 时，FFN 主导 |

::: warning 长序列的挑战
当序列长度 $n$ 很大时，$O(n^2)$ 的复杂度会成为瓶颈。这也是后续各种高效 Transformer（如 Linformer、Longformer）的研究动机。
:::

---

## PyTorch 实现

```python
import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 嵌入
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Transformer
        output = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # 输出
        logits = self.fc_out(output)
        return logits

# 使用
model = TransformerModel(vocab_size=30000)
src = torch.randint(0, 30000, (2, 10))  # (batch, src_len)
tgt = torch.randint(0, 30000, (2, 8))   # (batch, tgt_len)
output = model(src, tgt)  # (2, 8, 30000)
```

---

## 小结

| 组件 | 作用 |
|------|------|
| 多头自注意力 | 建模序列内部的依赖关系 |
| 前馈神经网络 | 对每个位置进行非线性变换 |
| 位置编码 | 注入位置信息 |
| 残差连接 | 帮助梯度流动 |
| 层归一化 | 稳定训练 |
| 掩码 | 防止解码器看到未来信息 |

Transformer 的设计思想——**完全基于注意力、抛弃循环结构**——开创了 NLP 的新纪元。接下来两章我们将深入讲解自注意力和位置编码的细节。
