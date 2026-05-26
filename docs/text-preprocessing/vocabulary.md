# 第六章：词表构建与文本编码

分词完成后，需要构建**词表**（Vocabulary）并将 token 转换为模型可以处理的数值 ID。本章详细讲解词表构建、特殊 Token、以及完整的文本编码流程。

---

## 词表（Vocabulary）

### 什么是词表？

词表是由语料库构建出的、包含模型可识别 token 的集合。词表中每个 token 都分配有唯一的 ID，支持 token 与 ID 之间的双向映射。

```
词表示例：
Token → ID:
"<PAD>"   → 0
"<UNK>"   → 1
"<CLS>"   → 2
"<SEP>"   → 3
"<MASK>"  → 4
"我"      → 5
"爱"      → 6
"自然"    → 7
"语言"    → 8
"处理"    → 9
...

ID → Token:
0 → "<PAD>"
1 → "<UNK>"
2 → "<CLS>"
...
```

### 词表构建流程

```
1. 收集语料
   ↓
2. 分词（Tokenization）
   ↓
3. 统计 token 频率
   ↓
4. 按频率排序，保留 Top-K
   ↓
5. 添加特殊 Token
   ↓
6. 建立双向映射
```

### 词表大小的选择

| 词表大小 | 优点 | 缺点 | 典型应用 |
|---------|------|------|---------|
| 小（~10K） | 嵌入层参数少 | 序列长，OOV 多 | 小型模型 |
| 中（~30K） | 平衡 | - | BERT (30,522) |
| 大（~50K） | 序列短，OOV 少 | 嵌入层参数多 | GPT-2 (50,257) |
| 超大（~100K+） | 覆盖广 | 参数多，稀疏 | 多语言模型 |

---

## 特殊 Token

特殊 Token 是词表中具有特殊含义的标记，不对应任何实际的文本内容。

### 常见特殊 Token

| Token | 含义 | 用途 |
|-------|------|------|
| `[PAD]` / `<pad>` | 填充标记 | 将不同长度的序列补齐到相同长度 |
| `[UNK]` / `<unk>` | 未知标记 | 表示词表中不存在的 token |
| `[CLS]` | 分类标记 | BERT 中用于表示整个序列的聚合表示 |
| `[SEP]` | 分隔标记 | 分隔两个句子（如句子对任务） |
| `[MASK]` | 掩码标记 | BERT 预训练时遮盖的 token |
| `<s>` / `</s>` | 序列开始/结束 | GPT/T5 中标记序列边界 |
| `<bos>` / `<eos>` | 句子开始/结束 | 标记生成文本的开始和结束 |

### 特殊 Token 在不同模型中的使用

**BERT 的输入格式**：
```
[CLS] 今天 天气 很 好 [SEP] 我们 去 公园 吧 [SEP]
```

**GPT 的输入格式**：
```
<s> 今天 天气 很 好，我们 去 公园 吧 </s>
```

**T5 的输入格式**：
```
<s> 翻译成英文：今天天气很好 </s>
```

---

## 文本编码完整流程

文本编码是将原始文本转换为模型输入张量的完整过程。

### 流程图

```
原始文本
  ↓
分词（Tokenization）
  ↓
Token 序列
  ↓
添加特殊 Token（[CLS], [SEP] 等）
  ↓
Token → ID 映射
  ↓
ID 序列
  ↓
截断（Truncation）/ 填充（Padding）
  ↓
等长 ID 序列
  ↓
生成 Attention Mask
  ↓
模型输入张量
```

### 详细示例

假设词表为：

```
[PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
我=5, 爱=6, 自=7, 然=8, 语=9, 言=10, 处=11, 理=12
```

**输入文本**："我爱自然语言处理"

**第 1 步：分词**
```
["我", "爱", "自", "然", "语", "言", "处", "理"]
```

**第 2 步：添加特殊 Token**
```
["[CLS]", "我", "爱", "自", "然", "语", "言", "处", "理", "[SEP]"]
```

**第 3 步：Token → ID**
```
[2, 5, 6, 7, 8, 9, 10, 11, 12, 3]
```

**第 4 步：Padding（假设 max_length=12）**
```
[2, 5, 6, 7, 8, 9, 10, 11, 12, 3, 0, 0]
```

**第 5 步：生成 Attention Mask**
```
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
```

Attention Mask 告诉模型哪些位置是真实 token（1），哪些是 padding（0）。

---

## 截断与填充

### 为什么需要等长序列？

深度学习模型通常使用**批量处理**（batch processing），要求同一批次内的所有序列长度相同。

### 填充（Padding）

将短序列用 `[PAD]` token 补齐到目标长度：

```
原始序列：  [CLS] 我 爱 NLP SEP
填充后：    [CLS] 我 爱 NLP SEP PAD PAD PAD
长度：5  →  8
```

### 截断（Truncation）

当序列超过最大长度时，需要截断：

```
原始序列：  [CLS] 今天 天气 很 好 我们 去 公园 玩 得 很 开心 SEP
截断后：    [CLS] 今天 天气 很 好 我们 去 公园 玩 SEP
max_length: 10
```

### 截断策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| 从末尾截断 | 保留开头，截掉结尾 | BERT（重要信息常在开头） |
| 从开头截断 | 保留结尾，截掉开头 | 生成任务（重要信息常在结尾） |
| 从两端截断 | 保留首尾，截掉中间 | 句子对任务 |

---

## 批量编码

在实际应用中，我们通常需要同时处理多个文本：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

texts = [
    "我爱自然语言处理",
    "我爱人工智能",
    "我们一起学习"
]

# 批量编码
inputs = tokenizer(
    texts,
    padding="max_length",   # 填充到最大长度
    truncation=True,        # 超长则截断
    max_length=10,          # 最大长度
    return_tensors="pt"     # 返回 PyTorch 张量
)

print(inputs)
```

**输出**：

```python
{
    'input_ids': tensor([
        [ 101, 2769, 4263, 5632, 4197, 6427, 6241, 1905, 4415,  102],
        [ 101, 2769, 4263,  782, 2339, 3255, 5543,  102,    0,    0],
        [ 101, 2769,  812,  671, 6629, 2110,  739,  102,    0,    0]
    ]),
    'token_type_ids': tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    ])
}
```

三个字段的含义：

| 字段 | 形状 | 含义 |
|------|------|------|
| `input_ids` | `(batch_size, seq_len)` | token 的 ID 序列 |
| `token_type_ids` | `(batch_size, seq_len)` | 句子类型标识（0=第一句，1=第二句） |
| `attention_mask` | `(batch_size, seq_len)` | 注意力掩码（1=真实token，0=padding） |

---

## Embedding 层

编码完成后，ID 序列需要通过**嵌入层**（Embedding Layer）转换为稠密的向量表示。

### 嵌入层的本质

嵌入层本质上是一个**查找表**（Lookup Table）：

```
嵌入矩阵 W（形状：词表大小 × 向量维度）：
         dim_0  dim_1  dim_2  ...  dim_d
token_0 [ 0.12, -0.34, 0.56, ..., 0.78]
token_1 [-0.11,  0.45, 0.23, ..., -0.67]
token_2 [ 0.33,  0.12, -0.89, ..., 0.44]
...

输入 ID = 2
输出 = W[2] = [0.33, 0.12, -0.89, ..., 0.44]
```

### PyTorch 实现

```python
import torch
import torch.nn as nn

# 创建嵌入层
# num_embeddings: 词表大小
# embedding_dim: 向量维度
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=768)

# 输入 ID 序列
input_ids = torch.tensor([[2, 5, 6, 7, 8]])  # 形状: (1, 5)

# 查找词向量
output = embedding(input_ids)  # 形状: (1, 5, 768)
print(output.shape)  # torch.Size([1, 5, 768])
```

### 两种初始化方式

| 方式 | 说明 | 适用场景 |
|------|------|---------|
| 随机初始化 | 随机生成向量，通过训练学习 | 从头训练模型 |
| 预训练词向量 | 使用 Word2Vec 等预训练向量初始化 | 低资源任务、迁移学习 |

---

## 完整编码流程代码示例

```python
from transformers import AutoTokenizer, AutoModel
import torch

# 1. 加载分词器和模型
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. 准备文本
texts = ["我爱自然语言处理", "我爱人工智能", "我们一起学习"]

# 3. 编码
encoded = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=10,
    return_tensors="pt"
)

# 4. 模型推理
with torch.no_grad():
    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        token_type_ids=encoded["token_type_ids"]
    )

# 5. 查看输出
print("last_hidden_state:", outputs.last_hidden_state.shape)
# torch.Size([3, 10, 768])

print("pooler_output:", outputs.pooler_output.shape)
# torch.Size([3, 768])
```

---

## 小结

本章介绍了从原始文本到模型输入的完整编码流程：

```
原始文本 → 分词 → 添加特殊Token → Token→ID → Padding/Truncation → Attention Mask → 模型输入
```

关键概念：

| 概念 | 说明 |
|------|------|
| 词表 | token 与 ID 的双向映射 |
| 特殊 Token | [PAD], [UNK], [CLS], [SEP], [MASK] 等 |
| Padding | 将短序列补齐到相同长度 |
| Truncation | 将长序列截断到最大长度 |
| Attention Mask | 标记哪些位置是真实 token |
| Embedding | 将 ID 转换为稠密向量 |

掌握这些基础后，我们将在下一章开始讲解各种文本表示方法。
