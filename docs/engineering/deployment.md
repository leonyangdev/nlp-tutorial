# 第二十八章：模型部署与推理优化

训练好的模型需要部署到生产环境才能产生价值。本章介绍模型部署和推理优化的关键技术。

---

## 模型导出

### 保存模型

```python
# 保存完整模型
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# 仅保存权重
torch.save(model.state_dict(), "model.pt")
```

### ONNX 导出

```python
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese")
model.eval()

# 创建示例输入
dummy_input = tokenizer("示例文本", return_tensors="pt")

# 导出 ONNX
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch"}
    }
)
```

---

## 推理优化技术

### 1. 模型量化

将模型权重从高精度压缩到低精度：

| 精度 | 大小 | 速度 | 精度损失 |
|------|------|------|---------|
| FP32 | 100% | 基准 | 无 |
| FP16 | 50% | ~2x | 几乎无 |
| INT8 | 25% | ~3x | 很小 |
| INT4 | 12.5% | ~4x | 较小 |

```python
# 动态量化
import torch
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 使用 GPTQ 量化
from transformers import AutoModelForCausalLM, GPTQConfig

quantization_config = GPTQConfig(bits=4, dataset="c4")
model = AutoModelForCausalLM.from_pretrained(
    "your-model",
    quantization_config=quantization_config
)
```

### 2. 模型蒸馏

用大模型（教师）训练小模型（学生）：

```python
from transformers import Trainer

# 教师模型
teacher = AutoModelForSequenceClassification.from_pretrained("bert-large")

# 学生模型
student = AutoModelForSequenceClassification.from_pretrained("bert-base")

# 蒸馏训练
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        with torch.no_grad():
            teacher_outputs = teacher(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # 软标签损失
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean'
        ) * T * T
        
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, inputs["labels"])
        
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        return (loss, outputs) if return_outputs else loss
```

### 3. KV Cache

在自回归生成时缓存计算结果：

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")

# 使用 KV Cache
input_ids = tokenizer("Hello", return_tensors="pt").input_ids

# 第一步：生成并缓存
outputs = model(input_ids, use_cache=True)
past_key_values = outputs.past_key_values

# 后续步骤：只输入新 token
new_token = torch.tensor([[model.config.eos_token_id]])
outputs = model(new_token, past_key_values=past_key_values)
```

---

## 服务部署

### 使用 FastAPI

```python
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("./my_model")
model = AutoModelForSequenceClassification.from_pretrained("./my_model")
model.eval()

@app.post("/predict")
async def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return {"prediction": prediction}

# 启动：uvicorn app:app --host 0.0.0.0 --port 8000
```

### 使用 Text Generation Inference (TGI)

```bash
# Docker 部署
docker run --gpus all -p 8080:80 \
    -v ./data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id your-model-name
```

### 使用 vLLM

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 生成
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

---

## 批量推理

```python
from torch.utils.data import DataLoader

# 批量处理
def batch_predict(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        results.extend(predictions.tolist())
    return results
```

---

## 性能监控

### 延迟监控

```python
import time

def measure_latency(text, n_runs=100):
    times = []
    for _ in range(n_runs):
        start = time.time()
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            model(**inputs)
        times.append(time.time() - start)
    
    print(f"平均延迟: {sum(times)/len(times)*1000:.2f}ms")
    print(f"P99 延迟: {sorted(times)[int(0.99*len(times))]*1000:.2f}ms")
```

### 吞吐量测试

```python
def measure_throughput(texts, batch_size=32):
    start = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model(**inputs)
    elapsed = time.time() - start
    print(f"吞吐量: {len(texts)/elapsed:.2f} samples/sec")
```

---

## 部署清单

| 检查项 | 说明 |
|--------|------|
| 模型量化 | 是否使用 INT8/INT4 量化 |
| 批量推理 | 是否支持批量处理 |
| 缓存机制 | 是否使用 KV Cache |
| 错误处理 | 是否有异常捕获和重试机制 |
| 监控告警 | 是否有延迟和错误率监控 |
| 自动扩缩容 | 是否支持水平扩展 |
| A/B 测试 | 是否支持模型版本切换 |

---

## 小结

| 技术 | 效果 | 复杂度 |
|------|------|--------|
| FP16 | 2x 速度，几乎无精度损失 | 低 |
| INT8 量化 | 3x 速度，很小精度损失 | 中 |
| 模型蒸馏 | 更小模型，接近原模型效果 | 高 |
| KV Cache | 减少重复计算 | 低 |
| ONNX Runtime | 通用推理加速 | 中 |
| vLLM | LLM 专用加速 | 低 |

模型部署是将研究转化为产品的关键环节。选择合适的优化技术可以在保证效果的同时大幅提升推理速度。
