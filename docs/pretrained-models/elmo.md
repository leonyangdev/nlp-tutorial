# 第十八章：ELMo

**ELMo**（Embeddings from Language Models）是第一个成功应用于下游任务的**上下文相关词向量**模型，标志着从静态词向量到动态词向量的重要转变。

---

## 静态词向量的局限

回顾 Word2Vec、GloVe 等静态词向量的核心问题：**每个词只有一个固定的向量表示**。

```
"苹果很好吃" 中的 "苹果" → vec("苹果") = [0.12, -0.34, ...]
"苹果发布新手机" 中的 "苹果" → vec("苹果") = [0.12, -0.34, ...]
```

同一个词在不同语境中有相同的表示，无法区分多义词。

---

## ELMo 的核心思想

ELMo 由 Allen AI 的 Peters 等人于 2018 年 2 月提出，核心思想是：

> **一个词的表示应该是其上下文的函数。**

ELMo 使用**双向 LSTM 语言模型**来为每个词生成上下文相关的向量表示。

---

## 双向语言模型

### 前向语言模型

前向语言模型从左到右处理序列，预测下一个词：

$$P(x_1, x_2, ..., x_N) = \prod_{k=1}^{N} P(x_k | x_1, ..., x_{k-1})$$

使用 LSTM 编码：

$$\overrightarrow{\mathbf{h}}_k = \text{LSTM}(\overrightarrow{\mathbf{h}}_{k-1}, \mathbf{x}_k)$$

### 后向语言模型

后向语言模型从右到左处理序列，预测前一个词：

$$P(x_1, x_2, ..., x_N) = \prod_{k=1}^{N} P(x_k | x_{k+1}, ..., x_N)$$

$$\overleftarrow{\mathbf{h}}_k = \text{LSTM}(\overleftarrow{\mathbf{h}}_{k+1}, \mathbf{x}_k)$$

### 双向语言模型

ELMo 同时训练前向和后向语言模型，最大化两者的联合对数似然：

$$\mathcal{L} = \sum_{k=1}^{N} \left[\log P(x_k | x_1, ..., x_{k-1}) + \log P(x_k | x_{k+1}, ..., x_N)\right]$$

---

## ELMo 的表示

### 多层表示

ELMo 不是只使用最后一层的输出，而是使用**所有层的表示**的加权组合：

对于词 $x_k$，ELMo 有 $L+1$ 层表示（$L$ 层 LSTM + 1 层输入嵌入）：

$$\mathbf{R}_k = \{\mathbf{x}_k, \overrightarrow{\mathbf{h}}_k^{(1)}, \overleftarrow{\mathbf{h}}_k^{(1)}, ..., \overrightarrow{\mathbf{h}}_k^{(L)}, \overleftarrow{\mathbf{h}}_k^{(L)}\}$$

其中 $\mathbf{x}_k$ 是输入嵌入，$\mathbf{h}_k^{(l)}$ 是第 $l$ 层 LSTM 的隐藏状态。

### 加权组合

ELMo 的最终表示是所有层的加权组合：

$$\text{ELMo}_k = \gamma \sum_{l=0}^{L} s_l \mathbf{h}_k^{(l)}$$

其中：
- $s_l$ 是每层的权重（通过 Softmax 归一化）
- $\mathbf{h}_k^{(l)} = [\overrightarrow{\mathbf{h}}_k^{(l)}; \overleftarrow{\mathbf{h}}_k^{(l)}]$ 是第 $l$ 层的拼接表示
- $\gamma$ 是一个缩放因子

### 不同层捕捉不同信息

研究发现，ELMo 的不同层捕捉了不同类型的语言信息：

| 层 | 捕捉的信息 |
|---|----------|
| 底层 | 句法信息（词性、句法结构） |
| 中层 | 语义信息（词义消歧） |
| 高层 | 任务相关的高级特征 |

---

## ELMo 的使用方式

### 作为特征

将 ELMo 的输出作为额外特征拼接到现有的词向量上：

$$\mathbf{h}_k = [\mathbf{x}_k; \text{ELMo}_k]$$

### 下游任务微调

ELMo 的参数可以在下游任务中进行微调，让模型更好地适应特定任务。

---

## ELMo 的优缺点

| 优点 | 缺点 |
|------|------|
| 第一个成功的上下文词向量 | 基于 LSTM，并行化能力有限 |
| 可以区分多义词 | 单向 LSTM，无法同时利用前后文 |
| 可以作为即插即用的特征 | 效果不如后来的 BERT |
| 开创了预训练的范式 | 训练成本较高 |

---

## ELMo vs Word2Vec vs BERT

| 特性 | Word2Vec | ELMo | BERT |
|------|----------|------|------|
| 上下文相关 | 否 | 是 | 是 |
| 架构 | 浅层神经网络 | 双向 LSTM | Transformer |
| 预训练任务 | 预测上下文 | 语言模型 | 掩码语言模型 |
| 双向性 | 无 | 伪双向（拼接） | 真双向 |
| 可微调 | 否 | 是 | 是 |

---

## 代码示例

```python
# 使用 AllenNLP 加载 ELMo
from allennlp.modules.elmo import Elmo, batch_to_ids

# 加载预训练 ELMo
elmo = Elmo(
    options_file="https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
    weight_file="https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
    num_output_representations=1,
)

# 输入句子
sentences = [["I", "love", "NLP"], ["ELMo", "is", "great"]]
character_ids = batch_to_ids(sentences)

# 获取 ELMo 表示
output = elmo(character_ids)
elmo_embeddings = output['elmo_representations'][0]  # (batch, seq_len, 1024)
```

---

## 小结

| 概念 | 说明 |
|------|------|
| 上下文词向量 | 同一个词在不同上下文中有不同的表示 |
| 双向语言模型 | 前向 + 后向 LSTM，捕捉双向上下文 |
| 多层表示加权 | 不同层捕捉不同层次的语言信息 |
| 预训练 + 微调 | 先在大规模语料上预训练，再在下游任务上微调 |

ELMo 开创了"预训练 + 微调"的范式，为后续的 GPT 和 BERT 奠定了基础。
