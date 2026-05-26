# 第十三章：Seq2Seq 模型

**Seq2Seq**（Sequence-to-Sequence）模型是处理序列到序列任务的基础架构，广泛应用于机器翻译、文本摘要、对话系统等场景。它是理解 Transformer 的重要前置知识。

---

## 什么是 Seq2Seq？

Seq2Seq 模型处理的是**输入序列和输出序列长度不同**的任务：

| 任务 | 输入 | 输出 |
|------|------|------|
| 机器翻译 | "I love NLP" | "我爱自然语言处理" |
| 文本摘要 | 一篇长文章 | 一段摘要 |
| 对话系统 | "你好吗？" | "我很好，谢谢！" |
| 语音识别 | 音频信号 | 文字 |

---

## 编码器-解码器架构

Seq2Seq 的核心是**编码器-解码器**（Encoder-Decoder）架构：

```
编码器（Encoder）              解码器（Decoder）
    ┌─────┐                    ┌─────┐
    │ RNN │                    │ RNN │
    └──┬──┘                    └──┬──┘
       │                          │
   x₁→h₁                      <s>→s₁→y₁
   x₂→h₂                      y₁→s₂→y₂
   x₃→h₃                      y₂→s₃→y₃
       │                          │
       ▼                          ▼
   上下文向量 c ──────────────→  初始化解码器
```

### 编码器（Encoder）

编码器将输入序列编码为一个**上下文向量**（Context Vector）$\mathbf{c}$：

$$\mathbf{h}_t = \text{Encoder}(\mathbf{x}_t, \mathbf{h}_{t-1})$$

$$\mathbf{c} = \mathbf{h}_T$$

其中 $\mathbf{h}_T$ 是编码器最后一个时间步的隐藏状态，它包含了整个输入序列的信息。

### 解码器（Decoder）

解码器以上下文向量 $\mathbf{c}$ 为初始状态，逐步生成输出序列：

$$\mathbf{s}_t = \text{Decoder}(\mathbf{y}_{t-1}, \mathbf{s}_{t-1}, \mathbf{c})$$

$$P(\mathbf{y}_t | \mathbf{y}_{<t}, \mathbf{x}) = \text{Softmax}(\mathbf{W}_o \mathbf{s}_t)$$

其中：
- $\mathbf{s}_t$ 是解码器在时间步 $t$ 的隐藏状态
- $\mathbf{y}_{t-1}$ 是上一个时间步的输出（训练时用真实标签，推理时用模型预测）
- $\mathbf{W}_o$ 是输出层的权重矩阵

---

## 详细示例：机器翻译

以 "I love NLP" → "我 爱 自然语言处理" 为例：

### 编码阶段

```
输入序列：I, love, NLP
特殊标记：<EOS> (序列结束)

编码器处理：
t=1: x₁ = vec("I")    → h₁
t=2: x₂ = vec("love") → h₂
t=3: x₃ = vec("NLP")  → h₃

上下文向量：c = h₃
```

### 解码阶段

```
解码器初始状态：s₀ = c = h₃

t=1: 输入 <SOS> (序列开始) → s₁ → P(y₁) → "我"
t=2: 输入 "我"             → s₂ → P(y₂) → "爱"
t=3: 输入 "爱"             → s₃ → P(y₃) → "自然语言处理"
t=4: 输入 "自然语言处理"    → s₄ → P(y₄) → <EOS> (结束)
```

---

## 训练与推理

### 训练阶段：Teacher Forcing

训练时使用 **Teacher Forcing** 策略：解码器的输入使用**真实的标签**，而不是模型的预测。

```
解码器输入：  <SOS>  我    爱    自然语言处理
解码器目标：  我    爱    自然语言处理  <EOS>
```

**损失函数**：交叉熵损失

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(\mathbf{y}_t^* | \mathbf{y}_{<t}^*, \mathbf{x})$$

其中 $\mathbf{y}_t^*$ 是时间步 $t$ 的真实标签。

### 推理阶段：自回归生成

推理时，解码器的输入是**上一个时间步的预测结果**：

```
t=1: 输入 <SOS> → 预测 "我"
t=2: 输入 "我"  → 预测 "爱"
t=3: 输入 "爱"  → 预测 "自然语言处理"
t=4: 输入 "自然语言处理" → 预测 <EOS> → 停止
```

### 推理解码策略

#### 贪心解码（Greedy Decoding）

每一步选择概率最大的词：

$$\mathbf{y}_t = \arg\max P(\mathbf{y}_t | \mathbf{y}_{<t}, \mathbf{x})$$

简单高效，但可能不是全局最优。

#### Beam Search

维护 $k$ 个最优候选序列（beam），每一步扩展所有可能的下一个词，保留得分最高的 $k$ 个：

```
beam_size = 3

t=1: 候选 → "我"(0.6), "我们"(0.2), "你"(0.1)
t=2: 扩展 → "我 爱"(0.4), "我 喜欢"(0.15), "我们 一起"(0.12), ...
t=3: 扩展 → "我 爱 自然语言处理"(0.3), "我 爱 你"(0.1), ...
```

---

## Seq2Seq 的问题

### 信息瓶颈

标准 Seq2Seq 将整个输入序列压缩为一个固定长度的上下文向量 $\mathbf{c}$，这会导致**信息瓶颈**：

```
输入："我 今天 早上 在 北京 大学 的 图书馆 里 看了 一本 关于 人工智能 的 书"
                                                              ↓
所有信息压缩为一个向量 c ←──────────────────────────────────────┘
                                                              ↓
解码器基于 c 生成摘要 ←────────────────────────────────────────┘
```

当输入序列很长时，一个固定长度的向量无法承载所有信息。

### 解决方案：注意力机制

**注意力机制**（Attention Mechanism）是解决信息瓶颈的关键创新。它允许解码器在每一步**动态地关注**输入序列的不同部分，而不是只依赖一个固定的上下文向量。

我们将在下一章详细讲解注意力机制。

---

## PyTorch 实现

### 编码器

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        # x: (batch, src_len)
        embedded = self.embedding(x)            # (batch, src_len, embed_dim)
        outputs, (h_n, c_n) = self.lstm(embedded)  # outputs: (batch, src_len, hidden)
        return outputs, h_n, c_n
```

### 解码器

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, h_n, c_n):
        # x: (batch, 1) — 一个时间步的输入
        embedded = self.embedding(x)                    # (batch, 1, embed_dim)
        output, (h_n, c_n) = self.lstm(embedded, (h_n, c_n))
        logits = self.fc(output.squeeze(1))             # (batch, vocab_size)
        return logits, h_n, c_n
```

### 完整的 Seq2Seq 模型

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch, src_len)
        # trg: (batch, trg_len)
        
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features
        
        # 存储输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 编码
        enc_outputs, h_n, c_n = self.encoder(src)
        
        # 解码
        input = trg[:, 0:1]  # <SOS>
        for t in range(1, trg_len):
            logits, h_n, c_n = self.decoder(input, h_n, c_n)
            outputs[:, t] = logits
            
            # Teacher Forcing
            if random.random() < teacher_forcing_ratio:
                input = trg[:, t:t+1]  # 使用真实标签
            else:
                input = logits.argmax(dim=1, keepdim=True)  # 使用模型预测
        
        return outputs
```

---

## 小结

| 概念 | 说明 |
|------|------|
| 编码器 | 将输入序列编码为上下文向量 |
| 解码器 | 基于上下文向量逐步生成输出序列 |
| Teacher Forcing | 训练时使用真实标签作为解码器输入 |
| 信息瓶颈 | 固定长度的上下文向量无法承载所有信息 |
| 注意力机制 | 解决信息瓶颈的关键创新 |

Seq2Seq 模型是理解 Transformer 的重要基础。下一章我们将讲解注意力机制，它是 Transformer 的核心组件。
