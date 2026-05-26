# 第九章：Word2Vec

**Word2Vec** 是 NLP 发展史上的里程碑式工作。它将词语从稀疏的 One-hot 表示转换为**稠密的、低维的、具有语义含义**的向量表示，开启了词向量时代。

---

## 从 One-hot 到分布式表示

### One-hot 的问题

回顾 One-hot 编码的局限：

```
"国王" → [1, 0, 0, 0, ...]
"女王" → [0, 1, 0, 0, ...]
"香蕉" → [0, 0, 1, 0, ...]

cos(国王, 女王) = 0
cos(国王, 香蕉) = 0
```

One-hot 向量之间完全正交，无法表达语义关系。

### 分布式表示

Word2Vec 的核心思想基于**分布式假设**（Distributional Hypothesis）：

> **一个词的含义由它周围的词决定。**

> "You shall know a word by the company it keeps." — J.R. Firth (1957)

基于这个假设，Word2Vec 为每个词生成一个**低维稠密向量**（如 100 维、300 维），使得语义相近的词在向量空间中距离更近。

```
vec("国王") = [0.23, -0.45, 0.67, ...]
vec("女王") = [0.21, -0.43, 0.65, ...]
vec("香蕉") = [-0.78, 0.12, -0.34, ...]

cos(国王, 女王) ≈ 0.95  ← 很相似！
cos(国王, 香蕉) ≈ 0.12  ← 不相似
```

---

## Word2Vec 的两种模型

Word2Vec 提供了两种模型结构来学习词向量：

### CBOW（Continuous Bag-of-Words）

**CBOW** 的目标是：**根据上下文词预测中心词**。

```
上下文：["我", "坐", "上班"]
目标词："地铁"

模型学习：给定上下文 ["我", "坐", "上班"]，预测中间词是 "地铁"
```

### Skip-gram

**Skip-gram** 的目标是：**根据中心词预测上下文词**。

```
中心词："地铁"
上下文：["我", "坐", "上班"]

模型学习：给定中心词 "地铁"，预测上下文是 ["我", "坐", "上班"]
```

---

## Skip-gram 详解

### 训练数据构造

给定句子 `"我 每天 坐 地铁 上班"` 和窗口大小 `window=2`：

```
中心词    上下文词
"我"   → ["每天", "坐"]
"每天" → ["我", "坐", "地铁"]
"坐"   → ["我", "每天", "地铁", "上班"]
"地铁" → ["每天", "坐", "上班"]
"上班" → ["坐", "地铁"]
```

每一对 `(中心词, 上下文词)` 构成一个训练样本：

```
(我, 每天), (我, 坐), (每天, 我), (每天, 坐), (每天, 地铁), ...
```

### 模型结构

Skip-gram 的模型结构非常简单，只有一个隐藏层：

```
输入层          隐藏层          输出层
(One-hot)       (词向量)       (Softmax)
   ↓               ↓               ↓
[0,0,1,0,...] → [0.23,-0.45,...] → [0.01, 0.02, ..., 0.15, ...]
    ↑                ↑                    ↑
  V × 1            d × 1              V × 1
```

其中：
- $V$ 是词表大小
- $d$ 是词向量维度（如 100、300）

### 前向传播过程

以中心词"地铁"、上下文词"乘坐"为例：

**第 1 步：输入中心词**

"地铁" 用 One-hot 向量表示：$\mathbf{x} \in \mathbb{R}^{V}$

**第 2 步：查找词向量**

将 One-hot 向量与输入权重矩阵 $\mathbf{W}_V \in \mathbb{R}^{V \times d}$ 相乘：

$$\mathbf{h} = \mathbf{W}_V^T \mathbf{x} = \mathbf{W}_V[\text{地铁}] \in \mathbb{R}^{d}$$

实际上就是查表：取出"地铁"对应的那行向量。$\mathbf{h}$ 就是"地铁"的词向量。

**第 3 步：预测上下文**

将词向量与输出权重矩阵 $\mathbf{W}_U \in \mathbb{R}^{d \times V}$ 相乘，得到对整个词表的预测得分：

$$\mathbf{z} = \mathbf{W}_U^T \mathbf{h} \in \mathbb{R}^{V}$$

**第 4 步：Softmax 输出**

通过 Softmax 将得分转换为概率分布：

$$P(w_j | w_i) = \frac{\exp(\mathbf{u}_j^T \mathbf{h})}{\sum_{k=1}^{V} \exp(\mathbf{u}_k^T \mathbf{h})}$$

其中：
- $\mathbf{u}_j$ 是词 $w_j$ 的输出向量（$\mathbf{W}_U$ 的第 $j$ 列）
- $\mathbf{h}$ 是中心词的词向量

**第 5 步：计算损失**

使用交叉熵损失：

$$\mathcal{L} = -\log P(\text{乘坐} | \text{地铁})$$

对于多个上下文词，损失取平均：

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

其中：
- $T$ 是训练样本总数
- $c$ 是窗口大小
- $w_t$ 是中心词
- $w_{t+j}$ 是上下文词

**第 6 步：反向传播更新**

通过反向传播更新权重矩阵 $\mathbf{W}_V$ 和 $\mathbf{W}_U$。训练完成后，$\mathbf{W}_V$ 的每一行就是对应词的词向量。

---

## CBOW 详解

### 与 Skip-gram 的区别

| | Skip-gram | CBOW |
|---|-----------|------|
| 输入 | 中心词 | 上下文词 |
| 预测目标 | 上下文词 | 中心词 |
| 训练速度 | 较慢 | 较快 |
| 适用场景 | 小数据集、罕见词 | 大数据集、常见词 |

### CBOW 的前向传播

**第 1 步：输入上下文词**

上下文词 ["乘坐", "上班"] 各自用 One-hot 表示。

**第 2 步：查找词向量**

分别查出"乘坐"和"上班"的词向量：$\mathbf{h}_1, \mathbf{h}_2$

**第 3 步：平均上下文向量**

$$\mathbf{h} = \frac{1}{2}(\mathbf{h}_1 + \mathbf{h}_2)$$

**第 4 步：预测中心词**

$$\mathbf{z} = \mathbf{W}_U^T \mathbf{h}$$

**第 5 步：Softmax + 交叉熵损失**

$$\mathcal{L} = -\log P(\text{地铁} | \mathbf{h})$$

---

## 训练优化：负采样

### Softmax 的计算瓶颈

标准 Softmax 需要对词表中**所有词**计算概率：

$$P(w_j | w_i) = \frac{\exp(\mathbf{u}_j^T \mathbf{h})}{\sum_{k=1}^{V} \exp(\mathbf{u}_k^T \mathbf{h})}$$

当词表大小 $V$ 很大（如 100 万）时，分母的计算非常昂贵。

### 负采样（Negative Sampling）

负采样的核心思想：**不计算完整的 Softmax，而是将问题转化为二分类**。

对于一个正样本 $(w_i, w_j)$（$w_j$ 确实是 $w_i$ 的上下文词），我们：

1. 保留这个正样本
2. 随机采样 $K$ 个"负样本"（$w_j'$ 不是 $w_i$ 的上下文词）
3. 训练模型区分正样本和负样本

**损失函数**：

$$\mathcal{L} = -\log \sigma(\mathbf{u}_j^T \mathbf{h}) - \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} [\log \sigma(-\mathbf{u}_k^T \mathbf{h})]$$

其中：
- $\sigma$ 是 Sigmoid 函数
- $P_n(w)$ 是负采样分布，通常取词频的 3/4 次方：$P_n(w) \propto \text{freq}(w)^{3/4}$
- $K$ 是负样本数量（通常 5-20）

**直觉**：
- 第一项：让正样本 $(w_i, w_j)$ 的得分尽量高
- 第二项：让负样本 $(w_i, w_k)$ 的得分尽量低

---

## 词向量的神奇性质

### 语义相似性

```python
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("word_vectors.kv")

# 查找相似词
model.most_similar("国王", topn=5)
# [('女王', 0.89), ('王子', 0.85), ('君主', 0.83), ...]

# 计算相似度
model.similarity("国王", "女王")  # 0.89
model.similarity("国王", "香蕉")  # 0.12
```

### 类比关系

Word2Vec 最令人惊叹的性质是：**词向量可以捕捉类比关系**！

$$\vec{\text{国王}} - \vec{\text{男人}} + \vec{\text{女人}} \approx \vec{\text{女王}}$$

这意味着词向量空间中存在有意义的**方向**：

```
方向 "男性→女性"：
国王 → 女王
男人 → 女人
叔叔 → 阿姨
王子 → 公主

方向 "过去→现在"：
走 → 走了
说 → 说了
写 → 写了
```

```python
# 类比推理
result = model.most_similar(
    positive=["国王", "女人"],  # 加上这些
    negative=["男人"],           # 减去这些
    topn=3
)
print(result)  # [('女王', 0.89), ('公主', 0.82), ...]
```

---

## 使用 Gensim 训练 Word2Vec

### 安装

```bash
pip install gensim
```

### 训练代码

```python
from gensim.models import Word2Vec

# 准备语料（已分词的句子列表）
sentences = [
    ['我', '每天', '乘坐', '地铁', '上班'],
    ['我', '每天', '乘坐', '公交', '上班'],
    ['他', '喜欢', '乘坐', '地铁', '出行'],
    ['地铁', '是', '城市', '交通', '的', '骨干'],
    ['公交', '线路', '覆盖', '城市', '各个', '角落'],
]

# 训练 Word2Vec 模型
model = Word2Vec(
    sentences,         # 已分词的句子序列
    vector_size=100,   # 词向量维度
    window=5,          # 上下文窗口大小
    min_count=2,       # 最小词频（低于此值的词被忽略）
    sg=1,              # 1=Skip-gram, 0=CBOW
    workers=4,         # 并行训练线程数
    epochs=10          # 训练轮数
)

# 查看词向量
print(model.wv['地铁'])  # 输出 100 维的词向量

# 查找相似词
similar_words = model.wv.most_similar("地铁", topn=5)
print(similar_words)

# 计算相似度
similarity = model.wv.similarity("地铁", "公交")
print("地铁 vs 公交 相似度:", similarity)

# 保存词向量
model.wv.save_word2vec_format("my_vectors.kv")
```

### 使用预训练词向量

```python
from gensim.models import KeyedVectors

# 加载预训练词向量
model = KeyedVectors.load_word2vec_format("sgns.weibo.word.bz2")

# 查看词向量维度
print("向量维度:", model.vector_size)  # 300

# 查看某个词的向量
print("地铁的向量:", model['地铁'])

# 相似度查询
similarity = model.similarity('地铁', '公交')
print('地铁 vs 公交 相似度:', similarity)

# 找出最相似的词
similar_words = model.most_similar(positive=["上班"], topn=5)
print(similar_words)

# 类比推理
result = model.most_similar(positive=["爸爸", "女性"], negative=["男性"], topn=3)
print(result)  # [('妈妈', 0.85), ...]
```

---

## 词向量作为嵌入层初始化

训练好的词向量可以用来初始化深度学习模型的嵌入层：

```python
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

# 1. 加载预训练词向量
word_vectors = KeyedVectors.load_word2vec_format("my_vectors.kv")

# 2. 构建词向量矩阵
word2index = word_vectors.key_to_index
embedding_dim = word_vectors.vector_size
num_embeddings = len(word2index)

embedding_matrix = torch.zeros(num_embeddings, embedding_dim)
for word, idx in word2index.items():
    embedding_matrix[idx] = torch.tensor(word_vectors[word])

# 3. 创建嵌入层
embedding_layer = nn.Embedding.from_pretrained(
    embedding_matrix,
    freeze=False  # True=冻结词向量不更新, False=继续微调
)

# 4. 使用
input_words = ["我", "喜欢", "乘坐", "地铁"]
input_indices = [word2index[word] for word in input_words]
input_tensor = torch.tensor([input_indices])

output = embedding_layer(input_tensor)
print(output.shape)  # torch.Size([1, 4, 100])
```

---

## Word2Vec 的局限性

| 局限 | 说明 |
|------|------|
| 静态词向量 | 每个词只有一个固定的向量，无法处理多义词 |
| 忽略形态学 | "happy" 和 "unhappy" 的向量没有直接关系 |
| 无法处理 OOV | 不在词表中的词没有词向量 |
| 局部上下文窗口 | 只能捕捉窗口内的共现信息 |

这些局限推动了后续方法的发展：
- **FastText**：解决形态学和 OOV 问题
- **GloVe**：利用全局共现信息
- **ELMo/BERT**：生成上下文相关的词向量

---

## 小结

| 概念 | 说明 |
|------|------|
| 分布式假设 | 一个词的含义由其上下文决定 |
| Skip-gram | 中心词 → 预测上下文 |
| CBOW | 上下文 → 预测中心词 |
| 负采样 | 将 Softmax 转化为二分类，加速训练 |
| 语义类比 | 词向量可以捕捉 "国王-男人+女人=女王" 这样的关系 |

Word2Vec 的核心贡献是证明了：**通过简单的神经网络训练，可以从大量文本中自动学习到有意义的词向量表示**。这一思想深刻影响了后续所有的 NLP 研究。
