# 第八章：TF-IDF 与 N-gram

本章介绍两种对词袋模型的重要改进：**TF-IDF**（考虑词的重要性）和 **N-gram**（保留局部词序）。

---

## TF-IDF

### 动机

在词袋模型中，每个词的权重是其出现的次数。但这存在一个问题：**高频常见词**（如"的"、"是"、"在"）的权重很高，但它们对区分文本主题的贡献很小。

TF-IDF 的核心思想是：**一个词的重要性与它在当前文本中出现的频率成正比，与它在整个语料库中出现的频率成反比**。

### TF（词频）

**词频**（Term Frequency）衡量一个词在当前文本中出现的频率：

$$\text{TF}(t, d) = \frac{f(t, d)}{\sum_{t' \in d} f(t', d)}$$

其中：
- $f(t, d)$ 是词 $t$ 在文档 $d$ 中出现的次数
- 分母是文档 $d$ 中所有词的出现次数之和

有时也直接使用词频 $f(t, d)$ 或对数词频 $1 + \log f(t, d)$。

### IDF（逆文档频率）

**逆文档频率**（Inverse Document Frequency）衡量一个词的普遍重要性：

$$\text{IDF}(t) = \log \frac{N}{\text{df}(t)}$$

其中：
- $N$ 是语料库中的文档总数
- $\text{df}(t)$ 是包含词 $t$ 的文档数
- 如果一个词在很多文档中都出现（如"的"），则 $\text{df}(t)$ 大，$\text{IDF}$ 小
- 如果一个词只在少数文档中出现（如"量子"），则 $\text{df}(t)$ 小，$\text{IDF}$ 大

### TF-IDF 公式

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

**直觉**：
- 如果一个词在当前文档中频繁出现（TF 高），且在整个语料库中不常见（IDF 高），那么它对当前文档很重要
- 如果一个词在所有文档中都频繁出现（如"的"），即使 TF 高，IDF 也会很低，最终权重不高

### 详细示例

假设语料库包含 3 个文档：

```
文档1："自然 语言 处理 是 人工 智能 的 分支"
文档2："深度 学习 是 人工 智能 的 核心 技术"
文档3："自然 语言 处理 使用 深度 学习 技术"
```

**计算 IDF**（$N = 3$）：

| 词 | df(t) | IDF = log(N/df) |
|---|-------|----------------|
| 自然 | 2 | log(3/2) = 0.405 |
| 语言 | 2 | log(3/2) = 0.405 |
| 处理 | 2 | log(3/2) = 0.405 |
| 是 | 2 | log(3/2) = 0.405 |
| 人工 | 2 | log(3/2) = 0.405 |
| 智能 | 2 | log(3/2) = 0.405 |
| 的 | 2 | log(3/2) = 0.405 |
| 深度 | 2 | log(3/2) = 0.405 |
| 学习 | 2 | log(3/2) = 0.405 |
| 核心 | 1 | log(3/1) = 1.099 |
| 技术 | 2 | log(3/2) = 0.405 |
| 分支 | 1 | log(3/1) = 1.099 |
| 使用 | 1 | log(3/1) = 1.099 |

注意："核心"、"分支"、"使用"只在一个文档中出现，所以 IDF 值更高。

### 代码实现

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "自然 语言 处理 是 人工 智能 的 分支",
    "深度 学习 是 人工 智能 的 核心 技术",
    "自然 语言 处理 使用 深度 学习 技术"
]

# 创建 TF-IDF 向量器
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# 查看词表
print("词表:", vectorizer.get_feature_names_out())

# 查看 TF-IDF 矩阵
import numpy as np
print("TF-IDF 矩阵:")
print(np.round(X.toarray(), 3))
```

### TF-IDF 的优缺点

| 优点 | 缺点 |
|------|------|
| 考虑了词的全局重要性 | 仍然忽略词序 |
| 简单高效 | 无法捕捉语义关系 |
| 广泛用于信息检索 | 稀疏表示 |
| 无需训练 | 对短文本效果一般 |

---

## N-gram

### 动机

词袋模型完全忽略了词序，而**N-gram** 通过将相邻的 $n$ 个词作为一个整体来保留局部的词序信息。

### 什么是 N-gram？

**N-gram** 是由 $n$ 个连续的词组成的序列：

| N | 名称 | 示例（"我爱自然语言处理"） |
|---|------|------------------------|
| 1 | Unigram | ["我", "爱", "自然", "语言", "处理"] |
| 2 | Bigram | ["我爱", "爱自然", "自然语言", "语言处理"] |
| 3 | Trigram | ["我爱自然", "爱自然语言", "自然语言处理"] |

### N-gram 解决词序问题

回顾之前的问题：

```
评论A："服务很好但味道差劲"
评论B："味道很好但服务差劲"

Bigram：
A: ["服务很好", "很好但", "好但味道", "但味道差劲"]
B: ["味道很好", "很好但", "好但服务", "但服务差劲"]

词袋表示（Bigram 特征）：
       服务很好  很好但  好但味道  但味道差劲  味道很好  好但服务  但服务差劲
A:      1       1       1         1         0        0         0
B:      0       1       0         0         1        1         1
```

现在两条评论的特征向量不同了！

### N-gram 语言模型

N-gram 不仅可以用于文本表示，还可以用于**语言建模**——预测下一个词的概率。

**Unigram 模型**：

$$P(w_i) = \frac{\text{count}(w_i)}{\sum_w \text{count}(w)}$$

**Bigram 模型**：

$$P(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})}$$

**Trigram 模型**：

$$P(w_i | w_{i-2}, w_{i-1}) = \frac{\text{count}(w_{i-2}, w_{i-1}, w_i)}{\text{count}(w_{i-2}, w_{i-1})}$$

**通用公式**（N-gram）：

$$P(w_i | w_{i-n+1}, ..., w_{i-1}) = \frac{\text{count}(w_{i-n+1}, ..., w_{i-1}, w_i)}{\text{count}(w_{i-n+1}, ..., w_{i-1})}$$

### N-gram 语言模型示例

假设语料库包含以下句子：

```
我 坐 地铁 上班
我 坐 公交 上班
我 坐 地铁 回家
他 坐 公交 上班
```

**Bigram 概率**：

```
P(坐|我) = count(我, 坐) / count(我) = 3/3 = 1.0
P(地铁|坐) = count(坐, 地铁) / count(坐) = 2/3 ≈ 0.67
P(公交|坐) = count(坐, 公交) / count(坐) = 2/3 ≈ 0.67
P(上班|地铁) = count(地铁, 上班) / count(地铁) = 1/2 = 0.5
P(回家|地铁) = count(地铁, 回家) / count(地铁) = 1/2 = 0.5
P(上班|公交) = count(公交, 上班) / count(公交) = 2/2 = 1.0
```

**计算句子概率**（Bigram）：

$$P(\text{我 坐 地铁 上班}) = P(\text{坐}|\text{我}) \times P(\text{地铁}|\text{坐}) \times P(\text{上班}|\text{地铁})$$
$$= 1.0 \times 0.67 \times 0.5 = 0.335$$

### N-gram 的问题：数据稀疏

当 $N$ 增大时，很多 N-gram 组合在语料中从未出现，导致概率为 0。

**解决方案——平滑技术**：

#### 加一平滑（Laplace Smoothing）

$$P(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i) + 1}{\text{count}(w_{i-1}) + |V|}$$

其中 $|V|$ 是词表大小。加一平滑确保每个 N-gram 的概率都不为 0。

#### 回退（Backoff）

当高阶 N-gram 未出现时，回退到低阶 N-gram：

$$P(w_i | w_{i-2}, w_{i-1}) = \begin{cases} P(w_i | w_{i-2}, w_{i-1}) & \text{if count}(w_{i-2}, w_{i-1}, w_i) > 0 \\ \lambda_1 P(w_i | w_{i-1}) & \text{otherwise} \end{cases}$$

### N-gram 的代码实现

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "服务很好但味道差劲",
    "味道很好但服务差劲",
    "环境优美菜品精致"
]

# Bigram 词袋模型
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
X = bigram_vectorizer.fit_transform(corpus)

print("Bigram 词表:", bigram_vectorizer.get_feature_names_out())
print("Bigram 矩阵:\n", X.toarray())

# Bigram TF-IDF
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf = tfidf_vectorizer.fit_transform(corpus)
```

---

## N-gram vs 词袋模型

| 特性 | 词袋（Unigram） | N-gram |
|------|----------------|--------|
| 词序 | 完全忽略 | 保留局部词序 |
| 特征数量 | $|V|$ | $|V|^n$（指数增长） |
| 稀疏性 | 中等 | 高 |
| 语义理解 | 无 | 有限 |

::: warning 维度爆炸
当 $N$ 增大时，N-gram 特征空间呈指数增长。例如词表大小为 10,000：
- Unigram: 10,000 个特征
- Bigram: 100,000,000 个特征
- Trigram: 1,000,000,000,000 个特征

因此在实际应用中，通常只使用 Bigram 或 Trigram。
:::

---

## 小结

| 方法 | 核心改进 | 优势 | 局限 |
|------|---------|------|------|
| TF-IDF | 考虑词的全局重要性 | 降低常见词权重 | 仍忽略词序 |
| N-gram | 保留局部词序 | 能区分词序不同的文本 | 维度爆炸，数据稀疏 |

TF-IDF 和 N-gram 是传统 NLP 中非常重要的特征工程方法。虽然它们已被深度学习方法超越，但在某些场景下（如小数据集、可解释性要求高的场景）仍然有实用价值。
