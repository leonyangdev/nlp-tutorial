# 第五章：子词分词算法

子词分词（Subword Tokenization）是现代 NLP 的核心技术之一。本章将详细讲解三种主流的子词分词算法：**BPE**、**WordPiece** 和 **Unigram Language Model**。

---

## 为什么需要子词分词？

回顾上一章的内容，词级分词和字符级分词各有明显的缺陷：

| 方法 | 问题 |
|------|------|
| 词级分词 | OOV 严重，词表膨胀 |
| 字符级分词 | 语义信息弱，序列过长 |

子词分词的核心思想是：**高频词保持完整，低频词拆分为有意义的子词片段**。

```
高频词："the" → ["the"]          （保持完整）
中频词："unhappy" → ["un", "happy"] （拆分为有意义的子词）
低频词："tokenization" → ["token", "ization"] （拆分为常见片段）
罕见词："ChatGPT" → ["Chat", "G", "PT"] （拆分为字符组合）
```

---

## BPE（Byte Pair Encoding）

### 历史背景

BPE 最初是一种**数据压缩算法**，由 Philip Gage 于 1994 年提出。2016 年，Rico Sennrich 等人将其应用于 NLP 的子词分词，发表在论文 *"Neural Machine Translation of Rare Words with Subword Units"* 中。

### 核心思想

BPE 的训练过程是一个**迭代合并**的过程：

1. 将语料中的所有词拆分为单个字符
2. 统计所有相邻字符对的出现频率
3. 将频率最高的字符对合并为一个新的子词
4. 重复步骤 2-3，直到词表大小达到预设上限

### 详细示例

假设我们有以下语料（已统计词频）：

```
语料词频：
"low"    : 5 次
"lower"  : 2 次
"newest" : 6 次
"widest": 3 次
```

**第 0 步：初始化**

将每个词拆分为字符，并在末尾添加特殊符号 `</w>` 表示词结束：

```
"low"    → l o w </w>     (出现 5 次)
"lower"  → l o w e r </w> (出现 2 次)
"newest" → n e w e s t </w> (出现 6 次)
"widest" → w i d e s t </w> (出现 3 次)
```

初始词表：`{l, o, w, e, r, n, s, t, i, d, </w>}`

**第 1 步：统计所有相邻字符对的频率**

```
字符对       频率
(e, s)       6 + 3 = 9  ← 最高！
(s, t)       6 + 3 = 9  ← 并列最高
(l, o)       5 + 2 = 7
(o, w)       5 + 2 = 7
(n, e)       6
(w, e)       6 + 2 = 8
(e, r)       2
...
```

选择频率最高的 `(e, s)` 合并为 `es`：

**第 1 步后**：

```
"low"    → l o w </w>         (5 次)
"lower"  → l o w e r </w>     (2 次)
"newest" → n e w es t </w>    (6 次)
"widest" → w i d es t </w>    (3 次)
```

词表更新：`{..., es}`

**第 2 步：重新统计频率**

```
字符对       频率
(es, t)      6 + 3 = 9  ← 最高！
(l, o)       5 + 2 = 7
(o, w)       5 + 2 = 7
(w, e)       6 + 2 = 8
...
```

合并 `(es, t)` → `est`：

```
"newest" → n e w est </w>    (6 次)
"widest" → w i d est </w>    (3 次)
```

**第 3 步**：

```
合并 (est, </w>) → est</w>：
"newest" → n e w est</w>    (6 次)
"widest" → w i d est</w>    (3 次)
```

**继续迭代...**

合并 `(l, o)` → `lo`，合并 `(lo, w)` → `low`，依此类推。

最终，高频的词（如 "newest"）会被保留为完整单元，而低频词会被拆分为更小的片段。

### BPE 分词过程

训练完成后，对新文本进行分词时：

1. 将文本拆分为字符
2. 按照训练时学到的合并规则，从左到右依次应用
3. 重复直到无法继续合并

```
输入："lowest"
拆分：l o w e s t </w>
应用规则：lo → lo, low → low, es → es, est → est
结果：low est</w>
```

### BPE 的优缺点

| 优点 | 缺点 |
|------|------|
| 简单高效，易于实现 | 合并顺序固定，贪心策略可能非最优 |
| 有效解决 OOV 问题 | 训练过程较慢（需要多次遍历语料） |
| 词表大小可控 | 对于形态学丰富的语言效果一般 |
| 被 GPT、LLaMA 等模型广泛采用 | |

---

## WordPiece

### 历史背景

WordPiece 由 Google 的 Schuster & Nakajima 于 2012 年提出，被 BERT、DistilBERT 等模型采用。

### 与 BPE 的区别

WordPiece 与 BPE 的核心区别在于**合并策略**：

| 算法 | 合并标准 |
|------|---------|
| BPE | 选择**频率最高**的字符对 |
| WordPiece | 选择能使**语言模型似然最大化**的字符对 |

具体来说，WordPiece 选择合并后能最大程度降低语言模型困惑度（perplexity）的字符对。

### 合并公式

WordPiece 使用以下公式来评估合并的收益：

$$\text{score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \times \text{freq}(y)}$$

其中：
- $\text{freq}(xy)$ 是字符对 $(x, y)$ 连续出现的频率
- $\text{freq}(x)$ 和 $\text{freq}(y)$ 分别是 $x$ 和 $y$ 单独出现的频率
- 分数越高，说明 $x$ 和 $y$ 的共现越"不独立"，合并更有价值

**直觉**：如果两个字符经常一起出现，但各自单独出现的频率不高，那么它们很可能是一个有意义的子词单元。

### WordPiece 分词过程

WordPiece 分词时使用**最长匹配优先**策略：

```
输入："unaffable"
词表中存在：un, ##aff, ##able, ##affable, ...

分词过程：
1. 尝试匹配 "unaffable" → 不在词表中
2. 尝试匹配 "unaffabl" → 不在词表中
3. ...逐步缩短...
4. 匹配 "un" → 在词表中！→ 记录 "un"
5. 剩余 "affable"
6. 尝试匹配 "affable" → 不在词表中
7. ...逐步缩短...
8. 匹配 "##aff" → 在词表中！→ 记录 "##aff"
9. 剩余 "able"
10. 匹配 "##able" → 在词表中！→ 记录 "##able"

结果：["un", "##aff", "##able"]
```

::: info `##` 前缀
WordPiece 使用 `##` 前缀来标记**非词首**的子词。例如：
- `"un"`：词首子词
- `"##able"`：非词首子词（前面还有其他子词）

这帮助模型区分词的边界。
:::

### WordPiece 的优缺点

| 优点 | 缺点 |
|------|------|
| 合并策略更"聪明"，考虑了语言模型的似然 | 训练比 BPE 更复杂 |
| 被 BERT 等经典模型采用 | 分词时需要回溯，效率略低 |
| `##` 前缀清晰标记词边界 | |

---

## Unigram Language Model

### 历史背景

Unigram Language Model 由 Kudo (2018) 提出，发表在论文 *"Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates"* 中。它与 BPE 和 WordPiece 的思路**完全不同**。

### 核心思想

BPE 和 WordPiece 是**自底向上**的：从字符开始，逐步合并。

Unigram 是**自顶向下**的：从一个大词表开始，逐步删除不重要的子词。

### 算法流程

**第 1 步：初始化**

构建一个包含大量候选子词的初始词表（通常包含所有字符、所有常见子串等）。

**第 2 步：计算每个子词的概率**

对于当前词表中的每个子词 $x_i$，计算其在语料中的出现概率：

$$P(x_i) = \frac{\text{count}(x_i)}{\sum_j \text{count}(x_j)}$$

**第 3 步：计算每个词的最优分词**

对于语料中的每个词 $w$，使用 **Viterbi 算法**找到概率最大的分词方式：

$$\mathbf{x}^* = \arg\max_{\mathbf{x} \in S(w)} \sum_{i} \log P(x_i)$$

其中：
- $S(w)$ 是词 $w$ 所有可能的分词方式的集合
- $P(x_i)$ 是子词 $x_i$ 的概率
- 我们选择使子词概率之积（对数概率之和）最大的分词

**第 4 步：计算每个子词的"损失"**

对于每个子词 $x_i$，计算如果删除它，语料的总对数似然会下降多少：

$$\text{loss}(x_i) = \sum_{w \in \text{corpus}} \left[\log P_{\text{with}}(w) - \log P_{\text{without}}(w)\right]$$

**第 5 步：删除低损失的子词**

删除损失最小的一批子词（通常是 10%-20%），缩小词表。

**第 6 步：重复步骤 2-5**

直到词表大小达到预设上限。

### Unigram 的优势

| 优点 | 说明 |
|------|------|
| 概率化分词 | 每个子词都有概率，可以输出多种分词结果 |
| 正则化效果 | 训练时可以随机采样不同的分词方式，增强模型鲁棒性 |
| 全局最优 | 基于语言模型似然，理论上更优 |

### Unigram 分词示例

```
词表及概率：
"自然" : 0.15
"语言" : 0.12
"处理" : 0.10
"自然语言" : 0.08
"自" : 0.05
"然" : 0.03
"语" : 0.04
"言" : 0.04
...

输入："自然语言处理"

可能的分词方式：
A: ["自然", "语言", "处理"] → log(0.15) + log(0.12) + log(0.10) = -6.37
B: ["自然语言", "处理"]     → log(0.08) + log(0.10) = -4.83  ← 概率更高！
C: ["自", "然", "语", "言", "处理"] → log(0.05) + log(0.03) + ... = -12.1

最优分词：B
```

---

## Byte-Level BPE

### 动机

标准 BPE 在**字符级别**操作，这意味着：
- 需要预先定义所有可能的字符
- 对于 Unicode 字符（如中文、日文、emoji），字符集非常大

**Byte-Level BPE** 的思路是：在**字节级别**（而非字符级别）应用 BPE。

### 工作原理

1. 将文本编码为 UTF-8 字节序列
2. 在字节级别应用 BPE

```
"你好" → UTF-8 字节：[0xE4, 0xBD, 0xA0, 0xE5, 0xA5, 0xBD]
→ BPE 合并字节对
```

**优势**：
- 基础词表只需 256 个字节，覆盖所有语言
- 不存在 OOV 问题（任何文本都可以用字节表示）
- 被 GPT-2、GPT-3、LLaMA 等模型采用

### GPT-2 的 BPE 实现

GPT-2 使用 Byte-Level BPE，其词表大小为 50,257：

```
256 个基础字节 token
+ 通过 BPE 合并产生的 ~50,000 个子词 token
= 50,257 个 token
```

---

## 三种算法对比

| 特性 | BPE | WordPiece | Unigram |
|------|-----|-----------|---------|
| 方向 | 自底向上（合并） | 自底向上（合并） | 自顶向下（删除） |
| 合并标准 | 频率最高 | 似然最大化 | - |
| 分词确定性 | 确定性 | 确定性 | 概率性 |
| 代表模型 | GPT, LLaMA | BERT, DistilBERT | T5, ALBERT |
| 实现库 | HuggingFace Tokenizers | HuggingFace Tokenizers | SentencePiece |

---

## 实际应用：HuggingFace Tokenizers

HuggingFace 提供了统一的分词器接口，支持上述所有算法：

```python
from transformers import AutoTokenizer

# BERT 使用 WordPiece
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
tokens = tokenizer.tokenize("我爱自然语言处理")
print(tokens)  # ['我', '爱', '自', '然', '语', '言', '处', '理']

# GPT-2 使用 Byte-Level BPE
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("I love NLP")
print(tokens)  # ['I', 'Ġlove', 'ĠN', 'LP']
```

::: info `Ġ` 前缀
GPT-2 的分词器使用 `Ġ`（U+0120）来标记**空格**。例如 `Ġlove` 表示前面有一个空格的 "love"。
:::

---

## 小结

| 算法 | 核心思想 | 关键区别 |
|------|---------|---------|
| BPE | 从字符开始，反复合并频率最高的字符对 | 简单高效 |
| WordPiece | 类似 BPE，但用似然最大化代替频率 | 更"聪明"的合并策略 |
| Unigram | 从大词表开始，逐步删除不重要的子词 | 概率化分词，支持多种分词结果 |
| Byte-Level BPE | 在字节级别应用 BPE | 语言无关，无 OOV |

这些算法是现代 NLP 模型的基石。理解它们的工作原理，有助于你理解为什么 BERT、GPT 等模型的分词行为是这样的。
