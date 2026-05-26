# 第二十七章：模型微调与训练

**微调**（Fine-tuning）是将预训练模型适配到特定任务的关键步骤。本章介绍完整的微调流程和最佳实践。

---

## 微调的基本流程

```
1. 选择预训练模型
      ↓
2. 准备数据集
      ↓
3. 数据预处理（分词、编码）
      ↓
4. 配置训练参数
      ↓
5. 训练模型
      ↓
6. 评估与保存
```

---

## 完整示例：情感分析

### 1. 准备数据

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# 加载数据
dataset = load_dataset("csv", data_files={
    "train": "train.csv",
    "test": "test.csv"
})

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 数据预处理
def preprocess(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized = dataset.map(preprocess, batched=True)
```

### 2. 配置模型

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2
)
```

### 3. 配置训练参数

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
)
```

### 4. 定义评估指标

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }
```

### 5. 训练

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
```

### 6. 评估与保存

```python
# 评估
results = trainer.evaluate()
print(results)

# 保存模型
trainer.save_model("./best_model")
tokenizer.save_pretrained("./best_model")
```

---

## 训练参数详解

### 常用参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `num_train_epochs` | 训练轮数 | 3-5 |
| `per_device_train_batch_size` | 每设备批大小 | 16-32 |
| `learning_rate` | 学习率 | 1e-5 ~ 5e-5 |
| `warmup_steps` | 预热步数 | 总步数的 10% |
| `weight_decay` | 权重衰减 | 0.01 |
| `gradient_accumulation_steps` | 梯度累积步数 | 1-8 |

### 学习率调度

```python
training_args = TrainingArguments(
    learning_rate=2e-5,
    lr_scheduler_type="cosine",  # 余弦退火
    warmup_ratio=0.1,            # 前 10% 步数预热
)
```

### 混合精度训练

```python
training_args = TrainingArguments(
    fp16=True,  # 使用 FP16 混合精度
)
```

---

## 参数高效微调（PEFT）

### 全量微调的问题

全量微调需要更新模型的所有参数，对于大模型（如 LLaMA-7B）来说：
- 需要大量 GPU 显存
- 训练时间长
- 容易过拟合

### LoRA（Low-Rank Adaptation）

LoRA 的核心思想：在原始权重矩阵旁边添加一个**低秩分解**的增量矩阵。

$$W' = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$，$r \ll d$。

```
原始参数量：d × d = 768 × 768 = 589,824
LoRA 参数量：d × r + r × d = 768 × 8 + 8 × 768 = 12,288
参数减少：97.9%！
```

### 使用 PEFT 库

```bash
pip install peft
```

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # 秩
    lora_alpha=32,          # 缩放因子
    lora_dropout=0.1,       # Dropout
    target_modules=["q_proj", "v_proj"],  # 应用 LoRA 的模块
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 查看可训练参数量
model.print_trainable_parameters()
# trainable params: 1,310,720 || all params: 6,739,013,632 || trainable%: 0.0194
```

### QLoRA

QLoRA 结合了量化和 LoRA：

```python
from transformers import BitsAndBytesConfig

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
)

# 然后应用 LoRA
model = get_peft_model(model, lora_config)
```

---

## 训练监控

### TensorBoard

```python
training_args = TrainingArguments(
    logging_dir="./logs",
    logging_steps=10,
    report_to="tensorboard",
)

# 启动 TensorBoard
# tensorboard --logdir ./logs
```

### Wandb

```python
training_args = TrainingArguments(
    report_to="wandb",
    run_name="my-experiment",
)
```

---

## 常见问题与解决方案

### 1. 过拟合

| 方法 | 说明 |
|------|------|
| 增加数据量 | 收集更多训练数据 |
| 数据增强 | 回译、同义词替换 |
| 增加 Dropout | 设置更大的 dropout |
| 早停 | 验证集性能不再提升时停止 |
| 权重衰减 | 增加 weight_decay |

### 2. 欠拟合

| 方法 | 说明 |
|------|------|
| 增加训练轮数 | 更多的 epoch |
| 增大模型 | 使用更大的预训练模型 |
| 增大学习率 | 尝试更大的学习率 |
| 减少正则化 | 降低 dropout、weight_decay |

### 3. 显存不足

| 方法 | 说明 |
|------|------|
| 减小批大小 | 减少 per_device_train_batch_size |
| 梯度累积 | 使用 gradient_accumulation_steps |
| 混合精度 | 使用 fp16=True |
| 梯度检查点 | 使用 gradient_checkpointing=True |
| 使用 LoPE | 使用 LoRA 等参数高效方法 |

---

## 小结

| 方法 | 适用场景 | 参数量 |
|------|---------|--------|
| 全量微调 | 数据充足、资源充足 | 100% |
| LoRA | 资源受限 | ~0.1-1% |
| QLoRA | 显存极受限 | ~0.1-1% |
| Prompt Tuning | 极端资源受限 | ~0.01% |

微调是将预训练模型适配到特定任务的关键步骤。选择合适的微调方法可以事半功倍。
