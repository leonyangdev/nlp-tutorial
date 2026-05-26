# 第二十章：BERT

**BERT**（Bidirectional Encoder Representations from Transformers）由 Google 于 2018 年 10 月提出，是 NLP 历史上最具影响力的预训练模型之一。它首次在 11 个 NLP 任务上取得了最先进的结果。

---

## BERT 的核心创新

### 与 GPT 的区别

| 特性 | GPT | BERT |
|------|-----|------|
| 架构 | Transformer 解码器（单向） | Transformer 编码器（双向） |
| 预训练任务 | 自回归语言模型 | 掩码语言模型 + 下一句预测 |
| 注意力方向 | 只能看到前文 | 可以看到前后文 |
| 适用任务 | 生成任务 | 理解任务 |

### 为什么双向很重要？

考虑句子："我去银行存了钱" vs "我在河边的银行散步"

GPT（单向）处理"银行"时：
- "我去银行" → 只能看到"我去"，无法确定"银行"的含义

BERT（双向）处理"银行"时：
- "我去**银行**存了钱" → 可以同时看到"存钱"，确定是金融机构
- "我在河边的**银行**散步" → 可以同时看到"河边"，确定是河岸

---

## BERT 的架构

### 模型结构

BERT 使用 Transformer 的**编码器**部分：

```
输入：[CLS] 我 爱 自然 语言 处理 [SEP]
      ↓
Token 嵌入 + 段落嵌入 + 位置嵌入
      ↓
┌─────────────────────────────┐
│  多头自注意力（双向）          │
│  前馈网络                     │
│  残差连接 + LayerNorm         │
│         × 12 层（BERT-Base） │
└─────────────────────────────┘
      ↓
输出：每个位置的上下文相关表示
```

### 三种嵌入

BERT 的输入由三种嵌入相加而成：

**1. Token 嵌入**：词的语义表示

**2. 段落嵌入**（Segment Embedding）：区分句子对中的两个句子

```
[CLS] 今天天气很好 [SEP] 我们去公园吧 [SEP]
  A    A   A  A  A   A    B  B  B  B  B   B
```

**3. 位置嵌入**：可学习的位置编码（不是正弦编码）

### 模型规模

| 版本 | 层数 | 隐藏维度 | 注意力头数 | 参数量 |
|------|------|---------|-----------|--------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

---

## 预训练任务

### 任务 1：掩码语言模型（Masked Language Model, MLM）

**核心思想**：随机遮盖输入中的一些词，让模型预测被遮盖的词。

```
输入：我 爱 [MASK] 语言 处理
标签：        自然
```

**掩码策略**：

对于被选中的 15% 的 token：
- 80% 的概率替换为 `[MASK]`
- 10% 的概率替换为随机词
- 10% 的概率保持不变

**为什么不是全部替换为 [MASK]？**

如果总是用 `[MASK]` 替换，模型会学到"看到 [MASK] 就预测"的模式，但在实际使用时没有 `[MASK]`。通过随机替换和保持不变，模型被迫学习每个位置的真实语义。

### 任务 2：下一句预测（Next Sentence Prediction, NSP）

**核心思想**：预测两个句子是否是连续的。

```
正例（IsNext）：
  [CLS] 今天天气很好 [SEP] 我们去公园吧 [SEP]
  标签：1

负例（NotNext）：
  [CLS] 今天天气很好 [SEP] 我喜欢吃苹果 [SEP]
  标签：0
```

::: info NSP 的争议
后续研究（如 RoBERTa）发现 NSP 任务对模型性能的贡献不大，甚至可能有害。因此很多后续模型取消了 NSP。
:::

---

## BERT 的输入输出

### 输入格式

```
单句输入：  [CLS] 句子 [SEP]
句子对输入：[CLS] 句子A [SEP] 句子B [SEP]
```

### 输出

BERT 为每个输入 token 输出一个向量：

```
输入：  [CLS] 我  爱  自然 语言 处理 [SEP]
输出：  h_CLS h₁  h₂  h₃   h₄   h₅   h_SEP
维度：  768  768 768 768  768  768   768
```

- `h_CLS`：整个序列的聚合表示，用于分类任务
- `h₁, ..., h₅`：每个 token 的上下文相关表示，用于序列标注

---

## 微调 BERT

BERT 可以通过添加一个简单的输出层来微调各种下游任务：

### 文本分类

```
[CLS] 这个电影很好看 [SEP]
      ↓
BERT
      ↓
h_CLS → 线性层 → Softmax → 正面/负面
```

### 命名实体识别（NER）

```
[CLS] 张三在北京大学读书 [SEP]
      ↓
BERT
      ↓
h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈
 ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
线性层 → Softmax → B-PER I-PER O B-ORG I-ORG O O O
```

### 问答

```
[CLS] 问题 [SEP] 文章 [SEP]
      ↓
BERT
      ↓
预测答案的起始和结束位置
```

---

## 代码实现

### 加载 BERT

```python
from transformers import BertTokenizer, BertModel

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

# 编码
text = "我爱自然语言处理"
inputs = tokenizer(text, return_tensors="pt")

# 前向传播
outputs = model(**inputs)

# 输出
last_hidden_state = outputs.last_hidden_state  # (1, seq_len, 768)
pooler_output = outputs.pooler_output            # (1, 768)
```

### 文本分类微调

```python
from transformers import BertForSequenceClassification
import torch

# 加载分类模型
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2
)

# 准备数据
texts = ["这个电影很好看", "这个电影很难看"]
labels = torch.tensor([1, 0])  # 1=正面, 0=负面

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 前向传播
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits  # (2, 2)

# 预测
predictions = torch.argmax(logits, dim=1)
```

---

## BERT 的影响

BERT 的出现带来了深远的影响：

1. **范式转变**：确立了"预训练 + 微调"的标准范式
2. **性能突破**：在 11 个 NLP 任务上取得最先进结果
3. **开源影响**：推动了预训练模型的开源文化
4. **后续工作**：催生了 RoBERTa、ALBERT、DistilBERT 等大量变体

---

## 小结

| 概念 | 说明 |
|------|------|
| 双向编码器 | 同时利用前文和后文信息 |
| 掩码语言模型 | 随机遮盖词并预测，实现双向训练 |
| 下一句预测 | 预测两个句子是否连续 |
| 三种嵌入 | Token + 段落 + 位置 |
| 微调 | 添加简单输出层适配下游任务 |

BERT 的成功证明了：**双向上下文信息对于语言理解至关重要**。它与 GPT 代表了两种不同的预训练范式，共同推动了 NLP 的发展。
