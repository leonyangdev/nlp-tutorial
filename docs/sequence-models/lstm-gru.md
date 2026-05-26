# 第十二章：LSTM 与 GRU

标准 RNN 存在严重的**梯度消失**问题，导致它难以捕捉长距离依赖。**LSTM**（Long Short-Term Memory）和 **GRU**（Gated Recurrent Unit）通过引入**门控机制**来解决这一问题。

---

## LSTM（Long Short-Term Memory）

### 核心思想

LSTM 由 Hochreiter & Schmidhuber 于 1997 年提出。它的核心创新是引入了**细胞状态**（Cell State）和三个**门**（Gate）来控制信息的流动。

**直觉**：把 LSTM 想象成一条**传送带**（细胞状态），信息可以沿着传送带无损地传递很远的距离。三个门负责控制什么信息可以写入传送带、什么信息需要从传送带中删除。

### LSTM 的结构

```
         ┌─────────────────────────────────────────┐
         │              细胞状态 cₜ                  │
         │  ─────────────────────────────────────→  │
         │    ×         +           ×               │
         │   遗忘门    输入门       输出门            │
         │    │         │           │               │
    ┌────┴────┐   ┌────┴────┐  ┌───┴────┐          │
    │  遗忘门  │   │  输入门  │  │  输出门 │          │
    │  fₜ     │   │  iₜ     │  │  oₜ    │          │
    └─────────┘   └─────────┘  └────────┘          │
         │         │     │         │               │
         │         │   c̃ₜ         │               │
         │         │     │         │               │
         └─────────┴─────┴─────────┘               │
                   │                               │
                   ▼                               │
                  hₜ ←─────────────────────────────┘
```

### 三个门和细胞状态

#### 1. 遗忘门（Forget Gate）

决定从细胞状态中**丢弃**哪些信息：

$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

其中：
- $\mathbf{f}_t \in (0, 1)^h$ 是遗忘门的输出
- $\sigma$ 是 Sigmoid 函数
- $\mathbf{W}_f$ 是权重矩阵
- $[\mathbf{h}_{t-1}, \mathbf{x}_t]$ 是将上一个隐藏状态和当前输入拼接

$\mathbf{f}_t$ 的每个元素在 0 到 1 之间：
- 0 表示"完全遗忘"
- 1 表示"完全保留"

**示例**：当模型遇到新的主语时，遗忘门可能会"遗忘"之前的主语信息。

#### 2. 输入门（Input Gate）

决定将哪些**新信息**写入细胞状态：

$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

其中：
- $\mathbf{i}_t \in (0, 1)^h$ 控制哪些位置需要更新
- $\tilde{\mathbf{c}}_t \in (-1, 1)^h$ 是候选的新信息

#### 3. 细胞状态更新

结合遗忘门和输入门，更新细胞状态：

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

其中 $\odot$ 表示**逐元素乘法**（Hadamard 积）。

**直觉**：
- $\mathbf{f}_t \odot \mathbf{c}_{t-1}$：保留旧信息中应该记住的部分
- $\mathbf{i}_t \odot \tilde{\mathbf{c}}_t$：添加新信息中应该记住的部分

#### 4. 输出门（Output Gate）

决定从细胞状态中**输出**哪些信息到隐藏状态：

$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

其中：
- $\mathbf{o}_t \in (0, 1)^h$ 控制哪些信息输出
- $\tanh(\mathbf{c}_t)$ 将细胞状态映射到 $(-1, 1)$
- $\mathbf{h}_t$ 是最终的隐藏状态

### LSTM 为什么能解决梯度消失？

关键在于细胞状态的更新方式：

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

这是一个**线性递推关系**（而不是像 RNN 那样经过非线性变换）。当遗忘门 $\mathbf{f}_t$ 接近 1 时，梯度可以沿着细胞状态无损地传递很远的距离：

$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} \approx \mathbf{f}_t \approx \mathbf{1}$$

这就像一条"梯度高速公路"，让梯度能够顺畅地流动。

---

## GRU（Gated Recurrent Unit）

### 核心思想

GRU 由 Cho 等人于 2014 年提出，是 LSTM 的简化版本。它将 LSTM 的三个门简化为**两个门**，并且取消了独立的细胞状态。

### GRU 的结构

#### 1. 重置门（Reset Gate）

决定如何将新的输入与之前的记忆结合：

$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$$

#### 2. 更新门（Update Gate）

同时扮演 LSTM 中遗忘门和输入门的角色：

$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$$

#### 3. 候选隐藏状态

$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)$$

#### 4. 隐藏状态更新

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

**直觉**：
- $\mathbf{z}_t$ 接近 0 时：保留旧状态 $\mathbf{h}_{t-1}$（类似 LSTM 的遗忘门打开、输入门关闭）
- $\mathbf{z}_t$ 接近 1 时：使用新状态 $\tilde{\mathbf{h}}_t$（类似 LSTM 的遗忘门关闭、输入门打开）

---

## LSTM vs GRU

| 特性 | LSTM | GRU |
|------|------|-----|
| 门的数量 | 3 个（遗忘门、输入门、输出门） | 2 个（重置门、更新门） |
| 细胞状态 | 有（独立的 $\mathbf{c}_t$） | 无 |
| 参数量 | 较多 | 较少（约 LSTM 的 75%） |
| 训练速度 | 较慢 | 较快 |
| 效果 | 在大数据集上通常更好 | 在小数据集上可能更好 |

**选择建议**：
- 数据量大、任务复杂 → LSTM
- 数据量小、需要快速训练 → GRU
- 不确定时 → 先试 LSTM

---

## PyTorch 实现

### LSTM

```python
import torch
import torch.nn as nn

# 创建 LSTM
lstm = nn.LSTM(
    input_size=100,      # 输入维度
    hidden_size=256,     # 隐藏层维度
    num_layers=2,        # 层数
    batch_first=True,
    bidirectional=True,
    dropout=0.3
)

# 输入
x = torch.randn(32, 20, 100)  # (batch, seq_len, input_size)

# 前向传播
output, (h_n, c_n) = lstm(x)

# output: (32, 20, 512)  — 每个时间步的输出
# h_n: (4, 32, 256)      — 最终隐藏状态
# c_n: (4, 32, 256)      — 最终细胞状态
```

### GRU

```python
# 创建 GRU
gru = nn.GRU(
    input_size=100,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.3
)

# 前向传播
output, h_n = gru(x)

# output: (32, 20, 512)
# h_n: (4, 32, 256)
```

### 文本分类示例

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (h_n, c_n) = self.lstm(embedded)
        
        # 取最后一层的正向和反向隐藏状态拼接
        forward_h = h_n[-2]   # 正向最后一层
        backward_h = h_n[-1]  # 反向最后一层
        combined = torch.cat([forward_h, backward_h], dim=1)
        
        logits = self.fc(self.dropout(combined))
        return logits

# 使用
model = LSTMClassifier(vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=2)
x = torch.randint(0, 10000, (32, 20))
logits = model(x)  # (32, 2)
```

---

## LSTM vs RNN 对比实验

在实际应用中，LSTM 通常比标准 RNN 表现更好，尤其是在需要捕捉长距离依赖的任务中：

```python
# 标准 RNN
rnn_model = nn.RNN(input_size=100, hidden_size=256, num_layers=2, 
                   batch_first=True, dropout=0.3)

# LSTM
lstm_model = nn.LSTM(input_size=100, hidden_size=256, num_layers=2, 
                     batch_first=True, dropout=0.3)

# GRU
gru_model = nn.GRU(input_size=100, hidden_size=256, num_layers=2, 
                   batch_first=True, dropout=0.3)

# 参数量对比
def count_params(model):
    return sum(p.numel() for p in model.parameters())

print(f"RNN 参数量: {count_params(rnn_model):,}")   # ~197K
print(f"LSTM 参数量: {count_params(lstm_model):,}")  # ~789K
print(f"GRU 参数量: {count_params(gru_model):,}")    # ~592K
```

---

## 小结

| 模型 | 核心创新 | 解决的问题 |
|------|---------|-----------|
| RNN | 循环连接、隐藏状态 | 建模序列依赖 |
| LSTM | 细胞状态 + 三个门 | 梯度消失、长期依赖 |
| GRU | 两个门、简化结构 | 参数效率、训练速度 |

LSTM 和 GRU 是序列建模的重要里程碑。虽然它们已被 Transformer 超越，但理解门控机制的思想对于理解后续的 Transformer 架构非常有帮助。
