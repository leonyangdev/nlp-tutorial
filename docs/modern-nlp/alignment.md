# 第二十五章：人类对齐：RLHF 与 DPO

**人类对齐**（Alignment）是指让大语言模型的输出符合人类的价值观和偏好。本章介绍两种主流的对齐方法：**RLHF** 和 **DPO**。

---

## 为什么需要对齐？

预训练模型的目标是**预测下一个词**，但这不等于**有帮助地回答问题**。

```
用户：请帮我写一首关于春天的诗。

未对齐的模型：
"春天春天春天春天春天春天春天..."（重复）

对齐后的模型：
"春风拂面暖如酥，
万物复苏绿满途。
桃花笑迎归燕至，
细雨润物细无声。"
```

对齐的目标是让模型：
- **有帮助**：真正回答用户的问题
- **无害**：不生成有害、偏见或危险的内容
- **诚实**：不编造虚假信息

---

## RLHF（Reinforcement Learning from Human Feedback）

### 三阶段流程

```
阶段1：监督微调（SFT）
  数据：人工标注的高质量指令-回复对
  目标：让模型学会遵循指令

阶段2：训练奖励模型（Reward Model）
  数据：人工对模型输出的排序
  目标：学习人类偏好

阶段3：强化学习优化（PPO）
  方法：使用奖励模型作为奖励信号
  目标：优化模型使其输出更高奖励
```

### 阶段 1：监督微调

收集高质量的指令-回复对：

```
指令："解释什么是机器学习"
回复："机器学习是人工智能的一个分支，它使计算机能够从数据中学习..."
```

使用这些数据对预训练模型进行微调。

### 阶段 2：训练奖励模型

**数据收集**：

对于同一个指令，让模型生成多个回复，然后由人工标注员对这些回复进行排序：

```
指令："给我讲一个笑话"

回复 A："为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 = Dec 25。"
回复 B："哈哈哈哈哈"
回复 C："有一天，一个程序员去面试..."

人工排序：A > C > B
```

**奖励模型训练**：

奖励模型学习预测人类的偏好排序：

$$\mathcal{L}_{\text{RM}} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

其中：
- $r_\theta(x, y)$ 是奖励模型对输入 $x$ 和输出 $y$ 的评分
- $y_w$ 是人类偏好的回复（winner）
- $y_l$ 是人类不偏好的回复（loser）
- $\sigma$ 是 Sigmoid 函数

### 阶段 3：PPO 优化

使用**近端策略优化**（Proximal Policy Optimization, PPO）算法来优化语言模型：

**目标函数**：

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_{(x,y) \sim \pi_\theta} \left[r_\theta(x, y) - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

其中：
- $\pi_\theta$ 是当前策略（语言模型）
- $\pi_{\text{ref}}$ 是参考策略（SFT 模型）
- $r_\theta(x, y)$ 是奖励模型的评分
- $\beta$ 是 KL 散度的权重，防止模型偏离太远

**直觉**：
- 最大化奖励模型的评分（让输出更符合人类偏好）
- 同时不要偏离原始模型太远（保持语言能力）

---

## DPO（Direct Preference Optimization）

### 动机

RLHF 的实现复杂，需要训练奖励模型、使用 PPO 等。DPO 提出了一种更简单的方法：**直接从偏好数据中优化策略**。

### 核心思想

DPO 推导出了一个闭式解，将奖励模型隐式地包含在策略中：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

其中：
- $\pi_\theta$ 是当前策略
- $\pi_{\text{ref}}$ 是参考策略
- $y_w$ 是偏好的回复
- $y_l$ 是不偏好的回复
- $\beta$ 是温度参数

### DPO vs RLHF

| 特性 | RLHF | DPO |
|------|------|-----|
| 需要奖励模型 | 是 | 否 |
| 需要 PPO | 是 | 否 |
| 实现复杂度 | 高 | 低 |
| 训练稳定性 | 较难调参 | 更稳定 |
| 效果 | 好 | 相当或更好 |

### DPO 的直觉

DPO 的目标是：
- 增加偏好的回复 $y_w$ 的概率
- 减少不偏好的回复 $y_l$ 的概率
- 同时不要偏离参考模型太远

```
偏好的回复概率 ↑
不偏好的回复概率 ↓
偏离参考模型的程度 → 最小化
```

---

## 其他对齐方法

### RLAIF（RL from AI Feedback）

使用 AI 模型代替人类进行偏好标注：

```
人类标注：A > B > C（昂贵、慢）
AI 标注：  A > B > C（便宜、快）
```

### Constitutional AI

让 AI 根据一组原则（宪法）自我改进：

```
原则：
1. 不要帮助用户做有害的事
2. 不要编造信息
3. 保持客观中立

AI 根据这些原则自我批评和改进。
```

### ORPO（Odds Ratio Preference Optimization）

不需要参考模型的简化版本：

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} - \lambda \log \sigma\left(\log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)$$

---

## 对齐的挑战

| 挑战 | 说明 |
|------|------|
| 标注成本 | 人工标注偏好数据昂贵且耗时 |
| 主观性 | 不同人对"好"的定义不同 |
| 过度对齐 | 模型可能变得过于保守 |
| 能力损失 | 对齐可能损害模型的某些能力 |

---

## 代码示例：DPO 训练

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# 加载模型
model = AutoModelForCausalLM.from_pretrained("your-sft-model")
ref_model = AutoModelForCausalLM.from_pretrained("your-sft-model")
tokenizer = AutoTokenizer.from_pretrained("your-sft-model")

# 准备偏好数据
# 每条数据包含：prompt, chosen (偏好的回复), rejected (不偏好的回复)
train_dataset = [
    {
        "prompt": "请解释什么是机器学习",
        "chosen": "机器学习是人工智能的一个分支...",
        "rejected": "机器学习就是学习机器..."
    },
    # ...更多数据
]

# 配置
config = DPOConfig(
    output_dir="./dpo-output",
    beta=0.1,
    learning_rate=5e-7,
    batch_size=4,
    num_train_epochs=3,
)

# 训练
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    config=config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

---

## 小结

| 方法 | 核心思想 | 优缺点 |
|------|---------|--------|
| RLHF | 奖励模型 + PPO | 效果好，但复杂 |
| DPO | 直接从偏好数据优化 | 简单，效果相当 |
| RLAIF | AI 代替人类标注 | 便宜，但质量不确定 |

人类对齐是让大语言模型变得安全、有用的关键技术。DPO 的出现大大降低了对齐的门槛。
