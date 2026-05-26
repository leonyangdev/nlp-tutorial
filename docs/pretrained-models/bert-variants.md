# 第二十一章：BERT 变体

BERT 的成功催生了大量改进版本。本章介绍几个最重要的 BERT 变体。

---

## RoBERTa（2019）

### 核心改进

RoBERTa（Robustly Optimized BERT Approach）由 Facebook 提出，通过更充分的训练来提升 BERT 的性能。

| 改进项 | BERT | RoBERTa |
|--------|------|---------|
| 训练数据 | 16GB | 160GB |
| 训练步数 | 1M | 500K |
| 批大小 | 256 | 8K |
| 掩码策略 | 静态（训练前确定） | 动态（每个 epoch 重新生成） |
| NSP 任务 | 有 | 移除 |
| 分词方式 | WordPiece | BPE（更大词表） |

### 关键发现

1. **NSP 无用**：移除 NSP 任务后性能反而更好
2. **动态掩码**：每次训练时重新生成掩码，比静态掩码效果更好
3. **更多数据**：使用更多数据训练能持续提升性能
4. **更长训练**：训练更久能获得更好的结果

---

## ALBERT（2019）

### 核心改进

ALBERT（A Lite BERT）由 Google 提出，旨在减少 BERT 的参数量。

| 改进项 | 说明 |
|--------|------|
| 嵌入矩阵分解 | 将 $V \times H$ 分解为 $V \times E$ 和 $E \times H$ |
| 跨层参数共享 | 所有层共享相同的参数 |
| 句子顺序预测 | 用 SOP 替代 NSP |

### 嵌入矩阵分解

原始 BERT：词表大小 $V = 30000$，隐藏维度 $H = 768$

$$\text{嵌入参数} = V \times H = 30000 \times 768 = 23M$$

ALBERT：嵌入维度 $E = 128$

$$\text{嵌入参数} = V \times E + E \times H = 30000 \times 128 + 128 \times 768 = 3.9M$$

参数量减少了约 80%！

### 跨层参数共享

ALBERT 的所有 Transformer 层共享相同的参数：

```
BERT：层1参数, 层2参数, ..., 层12参数（各不相同）
ALBERT：层1参数 = 层2参数 = ... = 层12参数（共享）
```

这大大减少了参数量，但可能损失一些表达能力。

---

## DistilBERT（2019）

### 核心思想

DistilBERT 通过**知识蒸馏**（Knowledge Distillation）将 BERT 压缩为一个更小的模型。

### 蒸馏过程

```
教师模型（BERT-Base, 110M 参数）
      ↓
蒸馏（软标签 + 硬标签）
      ↓
学生模型（DistilBERT, 66M 参数）
```

### 蒸馏损失

$$\mathcal{L} = \alpha \mathcal{L}_{\text{CE}}(y, \hat{y}) + (1-\alpha) T^2 D_{KL}(\sigma(z_t/T), \sigma(z_s/T))$$

其中：
- $\mathcal{L}_{\text{CE}}$ 是交叉熵损失（硬标签）
- $D_{KL}$ 是 KL 散度（软标签）
- $T$ 是温度参数，控制软标签的平滑程度
- $\alpha$ 是权重系数

### 性能对比

| 模型 | 参数量 | 层数 | 性能（vs BERT） |
|------|--------|------|----------------|
| BERT-Base | 110M | 12 | 基准 |
| DistilBERT | 66M | 6 | ~97% |
| TinyBERT | 14.5M | 4 | ~96% |

---

## DeBERTa（2020）

### 核心改进

DeBERTa（Decoding-enhanced BERT with disentangled attention）由微软提出。

| 改进项 | 说明 |
|--------|------|
| 解耦注意力 | 将内容和位置分开计算注意力 |
| 增强的掩码解码器 | 在解码层使用绝对位置信息 |

### 解耦注意力

传统 BERT 的注意力：

$$\text{Attention}(H_i, H_j) = H_i W_q (H_j W_k)^T$$

DeBERTa 的解耦注意力：

$$A_{i,j} = H_i W_q (H_j W_k)^T + H_i W_q (P_{i|j} W_k^p)^T + P_{j|i} W_q^p (H_j W_k)^T$$

其中 $P_{i|j}$ 是位置 $i$ 相对于位置 $j$ 的相对位置编码。

---

## 中文 BERT 变体

| 模型 | 特点 |
|------|------|
| BERT-wwm-ext | 全词掩码 + 扩展训练数据 |
| MacBERT | 使用同义词替换而非 [MASK] |
| ERNIE | 融合知识图谱信息 |
| RoFormer | 使用旋转位置编码 |

### 全词掩码（Whole Word Masking）

标准 BERT 按 token 掩码，可能只遮盖词的一部分：

```
标准 BERT：[CLS] 张 三 在 北 [MASK] 大 [MASK] 读 书
全词掩码：  [CLS] 张 三 在 [MASK] [MASK] [MASK] 读 书
```

全词掩码确保整个词被一起遮盖，有助于模型学习完整的词义。

---

## 变体对比

| 模型 | 核心创新 | 参数量 | 适用场景 |
|------|---------|--------|---------|
| BERT | 双向编码器 | 110M/340M | 通用 NLP |
| RoBERTa | 更充分的训练 | 123M/355M | 追求最佳性能 |
| ALBERT | 参数压缩 | 12M/18M | 资源受限场景 |
| DistilBERT | 知识蒸馏 | 66M | 快速推理 |
| DeBERTa | 解耦注意力 | 134M/350M | 最新 SOTA |

---

## 代码示例：使用 RoBERTa

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 加载 RoBERTa
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")

# 掩码预测
text = "我爱[MASK]语言处理"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# 获取 [MASK] 位置的预测
mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]
predictions = outputs.logits[0, mask_idx].topk(5)

for token_id, score in zip(predictions.indices, predictions.values):
    token = tokenizer.decode(token_id)
    print(f"{token}: {score.item():.3f}")
# 自然: 8.234
# 计算: 5.123
# ...
```

---

## 小结

| 变体 | 核心改进 | 解决的问题 |
|------|---------|-----------|
| RoBERTa | 更充分训练、移除 NSP | BERT 训练不充分 |
| ALBERT | 参数共享、嵌入分解 | 参数量过大 |
| DistilBERT | 知识蒸馏 | 推理速度慢 |
| DeBERTa | 解耦注意力 | 位置信息建模不足 |

这些变体从不同角度改进了 BERT，推动了预训练模型的发展。
