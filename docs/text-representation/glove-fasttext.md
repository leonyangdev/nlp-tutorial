# 第十章：GloVe 与 FastText

Word2Vec 虽然开创了词向量时代，但它并非完美。本章介绍两种重要的改进方法：**GloVe**（利用全局统计信息）和 **FastText**（利用子词信息）。

---

## GloVe（Global Vectors）

### 动机

Word2Vec 通过**局部上下文窗口**学习词向量，但它没有充分利用语料库的**全局统计信息**。

GloVe 的作者 Pennington 等人（2014）提出：应该同时利用**局部上下文**和**全局共现统计**来学习词向量。

### 共现矩阵

首先，我们需要构建**共现矩阵**（Co-occurrence Matrix）$\mathbf{X}$，其中 $X_{ij}$ 表示词 $i$ 和词 $j$ 在指定窗口大小内共同出现的次数。

**示例**：

语料：
```
"我 坐 地铁 上班"
"我 坐 公交 上班"
"地铁 是 交通工具"
```

窗口大小 = 1 的共现矩阵（部分）：

```
         我  坐  地铁  上班  公交  是  交通工具
我       0   2    0    0    0   0    0
坐       2   0    1    0    1   0    0
地铁     0   1    0    1    0   1    0
上班     0   0    1    0    1   0    0
公交     0   1    0    1    0   0    0
...
```

### GloVe 的目标函数

GloVe 的核心思想是：**词向量应该能够重建共现矩阵中的信息**。

具体来说，GloVe 优化以下目标函数：

$$\mathcal{L} = \sum_{i,j=1}^{V} f(X_{ij}) \left(\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$

其中：
- $V$ 是词表大小
- $X_{ij}$ 是词 $i$ 和词 $j$ 的共现次数
- $\mathbf{w}_i$ 是词 $i$ 的词向量（"主"向量）
- $\tilde{\mathbf{w}}_j$ 是词 $j$ 的上下文向量（"辅助"向量）
- $b_i, \tilde{b}_j$ 是偏置项
- $f(x)$ 是**加权函数**，用于控制高频共现对的影响

### 加权函数

$$f(x) = \begin{cases} (x / x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

其中通常取 $\alpha = 0.75$，$x_{\max} = 100$。

**直觉**：
- 当 $X_{ij}$ 较小时，$f(X_{ij})$ 也较小，减少稀有共现的噪声影响
- 当 $X_{ij}$ 较大时，$f(X_{ij})$ 趋近于 1，正常学习
- 当 $X_{ij}$ 过大时，$f(X_{ij}) = 1$，避免高频共现主导训练

### GloVe 的优势

| 优势 | 说明 |
|------|------|
| 利用全局信息 | 共现矩阵包含了整个语料库的统计信息 |
| 训练高效 | 只需遍历共现矩阵中的非零元素 |
| 理论优雅 | 目标函数有明确的数学解释 |
| 效果优秀 | 在多种评测任务上表现优异 |

### GloVe vs Word2Vec

| 特性 | Word2Vec | GloVe |
|------|----------|-------|
| 信息来源 | 局部上下文窗口 | 全局共现矩阵 |
| 训练方式 | 预测式（predictive） | 计数式（count-based） |
| 优化目标 | 最大化似然 | 最小化共现重建误差 |
| 实现复杂度 | 简单 | 需要先构建共现矩阵 |

### 使用预训练 GloVe 词向量

```python
from gensim.models import KeyedVectors

# 加载 GloVe 词向量（需要先转换格式）
# 原始 GloVe 格式：word1 0.123 -0.456 ...
# Word2Vec 格式：第一行是 "词数 维度"

# 方法1：使用 gensim 的 API
import gensim.downloader as api
model = api.load("glove-wiki-gigaword-300")

# 查找相似词
model.most_similar("king", topn=5)
# [('prince', 0.76), ('queen', 0.75), ...]

# 类比推理
model.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
# [('queen', 0.75), ('princess', 0.70), ...]
```

---

## FastText

### 动机

Word2Vec 和 GloVe 有两个共同的局限：

1. **无法处理 OOV**：不在词表中的词没有词向量
2. **忽略形态学**：没有利用词的内部结构信息

FastText 由 Facebook AI Research（2016）提出，通过引入**子词信息**来解决这些问题。

### 核心思想

FastText 的核心改进是：**将每个词表示为其字符 n-gram 的向量之和**。

例如，对于词 "where"（假设 $n=3$）：

```
字符 n-gram："<wh", "whe", "her", "ere", "re>"
特殊 n-gram："<where>"（完整词）

词向量 = vec("<wh") + vec("whe") + vec("her") + vec("ere") + vec("re>") + vec("<where>")
```

其中 `<` 和 `>` 是边界符号，标记词的开始和结束。

### 子词信息的优势

#### 1. 解决 OOV 问题

对于一个新词（不在词表中），只要它的字符 n-gram 在词表中，就能计算出词向量：

```
训练词表中没有 "unhappiness"
但存在子词："un", "happ", "appy", "ppin", "piness", ...
→ 可以计算 "unhappiness" 的词向量
```

#### 2. 捕捉形态学信息

具有相似形态的词会有相似的向量：

```
"happy" 和 "unhappy" 共享子词 "happ"
"running" 和 "swimming" 共享子词 "ing"
"national" 和 "international" 共享子词 "national"
```

#### 3. 对拼写错误更鲁棒

```
"happiness" → ["happ", "appi", "ppin", "piness"]
"happyness" → ["happ", "appy", "ppyn", "pyness"]

虽然不完全相同，但共享了 "happ" 等子词，因此向量仍然相似
```

### FastText 的模型结构

FastText 的模型结构与 Word2Vec 的 Skip-gram 基本相同，唯一的区别是输入：

**Word2Vec Skip-gram**：

```
输入：中心词 "地铁" 的 One-hot 向量
      ↓
查找词向量矩阵 W
      ↓
词向量 h
```

**FastText Skip-gram**：

```
输入：中心词 "地铁" 的字符 n-gram
      "地", "铁", "<地", "地铁", "铁>", "<地铁>"
      ↓
查找每个 n-gram 的向量
      ↓
求和得到词向量 h = vec("地") + vec("铁") + vec("<地铁") + ...
```

### FastText 的损失函数

与 Word2Vec 类似，FastText 使用负采样的 Skip-gram 损失：

$$\mathcal{L} = -\log \sigma(\mathbf{u}_o^T \mathbf{v}_w) - \sum_{i=1}^{K} \mathbb{E}_{w_i \sim P_n} [\log \sigma(-\mathbf{u}_{w_i}^T \mathbf{v}_w)]$$

其中 $\mathbf{v}_w$ 是中心词 $w$ 的向量（由其子词向量求和得到），$\mathbf{u}_o$ 是目标上下文词的向量。

---

## 使用 FastText

### 安装

```bash
pip install fasttext
```

### 训练 FastText 模型

```python
import fasttext

# 训练 Skip-gram 模型
model = fasttext.train_unsupervised(
    "corpus.txt",        # 训练语料文件（一行一个句子）
    model="skipgram",    # 模型类型：skipgram 或 cbow
    dim=100,             # 词向量维度
    ws=5,                # 窗口大小
    epoch=10,            # 训练轮数
    minCount=2           # 最小词频
)

# 获取词向量
vector = model.get_word_vector("地铁")
print(vector.shape)  # (100,)

# 查找相似词
similar = model.get_nearest_neighbors("地铁", k=5)
print(similar)

# 保存模型
model.save_model("fasttext_model.bin")
```

### 处理 OOV

```python
# 获取 OOV 词的向量（通过子词组合）
oov_vector = model.get_word_vector("一个从未见过的新词")
print(oov_vector.shape)  # (100,)
```

### 使用 Gensim 的 FastText

```python
from gensim.models import FastText

sentences = [
    ['我', '每天', '乘坐', '地铁', '上班'],
    ['我', '每天', '乘坐', '公交', '上班'],
    # ...更多句子
]

model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,          # 1=Skip-gram, 0=CBOW
    workers=4
)

# 获取词向量（包括 OOV 词）
print(model.wv['地铁'])
print(model.wv['一个新词'])  # OOV 也能获得向量！
```

---

## 三种词向量方法对比

| 特性 | Word2Vec | GloVe | FastText |
|------|----------|-------|----------|
| 子词信息 | 无 | 无 | 有 |
| OOV 处理 | 无法 | 无法 | 可以 |
| 全局信息 | 无 | 有 | 无 |
| 训练方式 | 预测式 | 计数式 | 预测式 |
| 形态学建模 | 无 | 无 | 有 |
| 训练效率 | 高 | 中 | 中 |
| 代表实现 | Gensim | 官方代码 | fasttext/Gensim |

---

## 词向量的局限性

尽管 Word2Vec、GloVe 和 FastText 取得了巨大成功，但它们都有一个根本性的局限：**静态词向量**。

```
"苹果很好吃" 中的 "苹果" → 同一个向量
"苹果发布新手机" 中的 "苹果" → 同一个向量
```

这意味着同一个词在不同上下文中永远有相同的表示，无法区分多义词。

这一局限推动了**上下文相关词表示**的发展，如 ELMo 和 BERT。我们将在后续章节中详细讲解。

---

## 小结

| 方法 | 核心创新 | 解决的问题 |
|------|---------|-----------|
| GloVe | 利用全局共现矩阵 | 更好地利用语料统计信息 |
| FastText | 引入字符 n-gram | OOV 问题、形态学建模 |

这三种方法共同奠定了词向量时代的基础。虽然它们已被 Transformer 模型超越，但理解它们对于理解后续的预训练模型至关重要。
