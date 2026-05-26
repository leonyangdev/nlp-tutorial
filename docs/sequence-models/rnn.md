# 第十一章：RNN 循环神经网络

在自然语言中，词语的顺序对于理解句子的含义至关重要。虽然词向量能够表示词语的语义，但它本身并不包含词语之间的顺序信息。**循环神经网络**（Recurrent Neural Network，RNN）正是为了解决这一问题而设计的。

---

## 为什么需要 RNN？

### 前馈神经网络的局限

传统的前馈神经网络（如 MLP）将输入视为独立的样本，无法建模序列中的**时序依赖关系**：

```
句子："我 爱 自然 语言 处理"

前馈网络的处理方式：
输入 [vec("我"), vec("爱"), vec("自然"), vec("语言"), vec("处理")]
→ 各个词的处理是独立的，无法知道 "我" 在 "爱" 之前
```

### RNN 的核心思想

RNN 通过**循环连接**来维护一个**隐藏状态**（hidden state），使得模型在处理当前输入时能够参考之前的信息：

```
时间步1: 输入 "我"   → 结合初始状态 h₀ → 输出 h₁
时间步2: 输入 "爱"   → 结合 h₁         → 输出 h₂
时间步3: 输入 "自然" → 结合 h₂         → 输出 h₃
时间步4: 输入 "语言" → 结合 h₃         → 输出 h₄
时间步5: 输入 "处理" → 结合 h₄         → 输出 h₅
```

每个时间步的隐藏状态 $h_t$ 都包含了从序列开始到当前时间步的所有信息。

---

## RNN 的数学定义

### 隐藏状态计算

在每个时间步 $t$，RNN 的隐藏状态通过以下公式计算：

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

其中：
- $\mathbf{x}_t \in \mathbb{R}^{d}$ 是时间步 $t$ 的输入向量（词向量）
- $\mathbf{h}_{t-1} \in \mathbb{R}^{h}$ 是上一个时间步的隐藏状态
- $\mathbf{h}_t \in \mathbb{R}^{h}$ 是当前时间步的隐藏状态
- $\mathbf{W}_{xh} \in \mathbb{R}^{h \times d}$ 是输入到隐藏层的权重矩阵
- $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ 是隐藏层到隐藏层的权重矩阵
- $\mathbf{b}_h \in \mathbb{R}^{h}$ 是偏置项
- $\tanh$ 是激活函数，将输出映射到 $(-1, 1)$

### 输出计算

RNN 的输出通过以下公式计算：

$$\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$

其中：
- $\mathbf{W}_{hy} \in \mathbb{R}^{V \times h}$ 是隐藏层到输出层的权重矩阵
- $\mathbf{b}_y \in \mathbb{R}^{V}$ 是输出偏置
- $V$ 是词表大小

::: info 参数共享
关键点：**所有时间步共享相同的参数** $\mathbf{W}_{xh}$、$\mathbf{W}_{hh}$、$\mathbf{W}_{hy}$。这意味着 RNN 对序列中的每个位置使用相同的处理逻辑，从而能够处理任意长度的序列。
:::

---

## RNN 的展开表示

虽然 RNN 的计算图看起来有循环，但我们可以将其**展开**（unroll）为一个前向的计算图：

```
     x₁        x₂        x₃        x₄
      │         │         │         │
      ▼         ▼         ▼         ▼
  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
  │  RNN  │→│  RNN  │→│  RNN  │→│  RNN  │
  └───────┘ └───────┘ └───────┘ └───────┘
      │         │         │         │
      ▼         ▼         ▼         ▼
     y₁        y₂        y₃        y₄
      │         │         │         │
     h₁  ───→  h₂  ───→  h₃  ───→  h₄
```

展开后，RNN 就像一个很深的前馈网络，其中：
- 每一层对应一个时间步
- 层与层之间通过隐藏状态传递信息
- 所有层共享相同的参数

---

## RNN 的变体结构

### 单层单向 RNN

最基本的 RNN 结构：

```
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃
```

### 多层 RNN

将多个 RNN 层堆叠起来，使不同层学习不同层次的语义信息：

```
输入层:    x₁ → [RNN₁] → h₁¹ → [RNN₁] → h₂¹ → [RNN₁] → h₃¹
                ↓              ↓              ↓
隐藏层1:      [RNN₂] → h₁² → [RNN₂] → h₂² → [RNN₂] → h₃²
                ↓              ↓              ↓
隐藏层2:      [RNN₃] → h₁³ → [RNN₃] → h₂³ → [RNN₃] → h₃³
```

**核心假设**：
- 底层网络捕捉局部模式（如词组、短语）
- 高层网络学习更抽象的语义信息（如句子主题）

### 双向 RNN

基础 RNN 只能利用前文信息，但很多任务需要同时利用前文和后文：

```
正向 RNN:  x₁ → h₁ → x₂ → h₂ → x₃ → h₃
反向 RNN:  x₁ ← h₁ ← x₂ ← h₂ ← x₃ ← h₃

组合:      [h₁; h₁] , [h₂; h₂] , [h₃; h₃]
```

双向 RNN 同时使用两个 RNN：
- **正向 RNN**：按时间顺序处理序列
- **反向 RNN**：按逆时间顺序处理序列

每个时间步的输出是正向和反向隐藏状态的**拼接**：

$$\mathbf{h}_t = [\mathbf{h}_t^{\rightarrow}; \mathbf{h}_t^{\leftarrow}]$$

---

## PyTorch 实现

### 基本用法

```python
import torch
import torch.nn as nn

# 创建 RNN 层
rnn = nn.RNN(
    input_size=100,      # 输入维度（词向量维度）
    hidden_size=256,     # 隐藏层维度
    num_layers=2,        # RNN 层数
    batch_first=True,    # 输入格式为 (batch, seq, feature)
    bidirectional=True,  # 双向 RNN
    dropout=0.3          # Dropout 比率
)

# 输入：batch_size=32, seq_len=20, input_size=100
x = torch.randn(32, 20, 100)

# 前向传播
output, h_n = rnn(x)

# output: (32, 20, 512)  — 每个时间步的输出（双向拼接，所以是 256*2=512）
# h_n: (4, 32, 256)      — 最后一个时间步的隐藏状态（2层*2方向=4）
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `input_size` | int | 输入特征的维度 |
| `hidden_size` | int | 隐藏状态的维度 |
| `num_layers` | int | RNN 的层数 |
| `batch_first` | bool | 若为 True，输入输出格式为 (batch, seq, feature) |
| `bidirectional` | bool | 若为 True，使用双向 RNN |
| `dropout` | float | 除最后一层外的 Dropout 比率 |

### 输入输出形状

**单层单向 RNN**（`batch_first=True`）：

| 张量 | 形状 | 说明 |
|------|------|------|
| 输入 | $(B, T, d_{in})$ | $B$=batch size, $T$=序列长度, $d_{in}$=输入维度 |
| 输出 | $(B, T, h)$ | 每个时间步的隐藏状态 |
| $h_n$ | $(1, B, h)$ | 最后一个时间步的隐藏状态 |

**多层双向 RNN**（`num_layers=L`, `bidirectional=True`）：

| 张量 | 形状 | 说明 |
|------|------|------|
| 输入 | $(B, T, d_{in})$ | |
| 输出 | $(B, T, 2h)$ | 双向拼接 |
| $h_n$ | $(2L, B, h)$ | 每层每个方向的最终隐藏状态 |

---

## 完整示例：文本分类

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向所以 *2
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        output, h_n = self.rnn(embedded)       # output: (batch, seq_len, 2*hidden)
        
        # 取最后一个时间步的输出
        last_output = output[:, -1, :]         # (batch, 2*hidden)
        
        # 分类
        logits = self.fc(last_output)          # (batch, num_classes)
        return logits

# 使用
model = TextClassifier(vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=2)
x = torch.randint(0, 10000, (32, 20))  # batch_size=32, seq_len=20
logits = model(x)  # (32, 2)
```

---

## RNN 的问题：梯度消失

### 什么是梯度消失？

在反向传播时，梯度需要从最后一个时间步传递到第一个时间步。由于 RNN 展开后相当于一个很深的网络，梯度在传递过程中会逐渐**缩小**（消失）或**放大**（爆炸）。

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$$

如果 $\|\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}\| < 1$，连乘后梯度会趋近于 0（梯度消失）。

### 梯度消失的影响

```
句子："我 今天 早上 在 北京 大学 的 图书馆 里 看了 一本 关于 人工智能 的 书"

当处理到 "书" 时，模型很难记住开头的 "我"。
因为从 "书" 到 "我" 的梯度在传递过程中已经消失了。
```

这就是 RNN 的**长期依赖问题**——它难以捕捉序列中相距较远的词之间的关系。

### 梯度裁剪

虽然梯度消失难以完全解决，但**梯度爆炸**可以通过**梯度裁剪**来缓解：

```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 小结

| 概念 | 说明 |
|------|------|
| 隐藏状态 | RNN 的"记忆"，包含序列的历史信息 |
| 参数共享 | 所有时间步共享相同的权重矩阵 |
| 多层 RNN | 堆叠多个 RNN 层，学习不同层次的特征 |
| 双向 RNN | 同时利用前文和后文信息 |
| 梯度消失 | RNN 的核心问题，导致难以捕捉长期依赖 |

RNN 的梯度消失问题催生了 LSTM 和 GRU 等改进模型，我们将在下一章详细讲解。
