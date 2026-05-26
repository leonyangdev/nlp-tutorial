# 第二十六章：HuggingFace 生态

**HuggingFace** 是当前最重要的 NLP 开源生态，提供了从数据处理到模型部署的完整工具链。本章全面介绍 HuggingFace 的核心组件和使用方法。

---

## HuggingFace 生态概览

### 核心组件

| 组件 | 功能 |
|------|------|
| **Hub** | 模型、数据集的托管平台 |
| **Transformers** | 预训练模型库 |
| **Tokenizers** | 高效分词器 |
| **Datasets** | 数据处理库 |
| **PEFT** | 参数高效微调 |
| **TRL** | 强化学习训练 |
| **Accelerate** | 分布式训练 |

---

## Transformers 库

### 安装

```bash
pip install transformers
```

### 加载模型

```python
from transformers import AutoModel, AutoTokenizer

# 加载模型和分词器
model_name = "bert-base-chinese"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### AutoModel vs AutoModelForXXX

| 类型 | 用途 | 输出 |
|------|------|------|
| `AutoModel` | 特征提取 | 隐藏状态 |
| `AutoModelForSequenceClassification` | 文本分类 | 类别概率 |
| `AutoModelForTokenClassification` | 序列标注 | 每个 token 的标签 |
| `AutoModelForQuestionAnswering` | 问答 | 答案的起止位置 |
| `AutoModelForCausalLM` | 文本生成 | 下一个词的概率 |
| `AutoModelForMaskedLM` | 掩码预测 | 被遮盖词的概率 |

### 使用分词器

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 编码
text = "我爱自然语言处理"
encoded = tokenizer(text, return_tensors="pt")
print(encoded)
# {'input_ids': tensor([[101, 2769, 4263, ...]]),
#  'token_type_ids': tensor([[0, 0, 0, ...]]),
#  'attention_mask': tensor([[1, 1, 1, ...]])}

# 批量编码
texts = ["我爱NLP", "我爱AI"]
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 解码
decoded = tokenizer.decode(encoded["input_ids"][0])
print(decoded)  # [CLS] 我 爱 自 然 语 言 处 理 [SEP]
```

### 完整推理流程

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

# 编码
text = "这个电影很好看"
inputs = tokenizer(text, return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1)
    print(f"预测类别: {prediction.item()}")
```

---

## Datasets 库

### 安装

```bash
pip install datasets
```

### 加载数据集

```python
from datasets import load_dataset

# 加载本地 CSV
dataset = load_dataset("csv", data_files="data.csv")

# 加载在线数据集
dataset = load_dataset("imdb")

# 查看数据
print(dataset)
print(dataset["train"][0])
```

### 数据预处理

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("csv", data_files="data.csv")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 使用 map 方法批量处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 删除原始文本列
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# 设置格式
tokenized_dataset.set_format("torch")
```

### 数据集操作

```python
# 划分数据集
split_dataset = dataset["train"].train_test_split(test_size=0.2)

# 过滤
filtered = dataset.filter(lambda x: x["label"] in [0, 1])

# 删除列
cleaned = dataset.remove_columns(["id", "timestamp"])
```

---

## 使用 Pipeline

Transformers 提供了高级的 `pipeline` 接口，一行代码完成常见任务：

```python
from transformers import pipeline

# 情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# 文本生成
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)
print(result)

# 问答
qa = pipeline("question-answering")
result = qa(question="Who is the president?", context="The president is Biden.")
print(result)  # {'answer': 'Biden', 'score': 0.99, ...}

# 命名实体识别
ner = pipeline("ner", grouped_entities=True)
result = ner("张三在北京大学工作")
print(result)

# 掩码语言模型
filler = pipeline("fill-mask", model="bert-base-chinese")
result = filler("我爱[MASK]语言处理")
print(result)
```

---

## Hub 使用

### 搜索模型

```python
from huggingface_hub import list_models

# 搜索中文 BERT 模型
models = list(list_models(search="chinese-bert", limit=5))
for m in models:
    print(m.modelId, m.downloads)
```

### 上传模型

```python
from huggingface_hub import login

# 登录
login(token="your_token")

# 上传模型
model.push_to_hub("your-username/your-model-name")
tokenizer.push_to_hub("your-username/your-model-name")
```

### 下载文件

```python
from huggingface_hub import hf_hub_download

# 下载单个文件
path = hf_hub_download(
    repo_id="bert-base-chinese",
    filename="config.json"
)
```

---

## 环境变量配置

### 使用国内镜像

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### 缓存目录

```python
# 设置缓存目录
os.environ["HF_HOME"] = "/path/to/cache"
```

---

## 最佳实践

### 1. 选择合适的模型

```python
# 根据任务选择模型
from transformers import AutoModelForSequenceClassification

# 文本分类
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2
)

# 文本生成
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

### 2. 使用 Pipeline 快速验证

```python
# 快速验证模型效果
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="your-model")
results = classifier(["好", "差", "一般"])
```

### 3. 批量处理

```python
# 使用 batched=True 加速
dataset = dataset.map(tokenize_function, batched=True, batch_size=1000)
```

---

## 小结

| 组件 | 功能 | 常用 API |
|------|------|---------|
| Transformers | 模型加载与推理 | `AutoModel.from_pretrained()` |
| Tokenizers | 文本编码 | `AutoTokenizer.from_pretrained()` |
| Datasets | 数据处理 | `load_dataset()`, `.map()` |
| Pipeline | 快速推理 | `pipeline("task")` |
| Hub | 模型托管 | `push_to_hub()`, `list_models()` |

HuggingFace 生态大大降低了 NLP 的门槛，让你可以专注于任务本身而不是底层实现。
