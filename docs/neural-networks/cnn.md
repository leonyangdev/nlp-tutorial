# CNN（卷积神经网络）

> CNN（Convolutional Neural Network）是专为处理**网格结构数据**（图像、音频）设计的神经网络。其核心创新是**权值共享（weight sharing）**和**局部感受野（local receptive field）**，使参数量大幅减少，同时保持对平移的不变性。

## 1. 为什么需要 CNN？

### 全连接网络的瓶颈

假设输入一张 224×224 的彩色图像：

- 像素数：224 × 224 × 3 = **150,528**
- 第一个全连接层（1024 个神经元）的参数量：150,528 × 1024 ≈ **1.5 亿**

这导致两个严重问题：
1. **参数爆炸**：内存占用巨大，极易过拟合
2. **忽略空间结构**：将图像拉平为向量后，像素间的空间关系（上下左右邻域）完全丢失

### CNN 的两个核心假设

| 假设 | 含义 | 好处 |
|---|---|---|
| **局部连接** | 每个神经元只连接输入的局部区域（感受野） | 参数量大幅减少 |
| **权值共享** | 同一个卷积核在全图滑动，参数被所有位置复用 | 进一步减少参数，学习平移不变特征 |

---

## 2. 卷积层（Convolutional Layer）

### 卷积操作原理

```
输入图像（5×5）      卷积核（3×3）        输出特征图（3×3）

┌─────────────────┐   ┌─────────┐        ┌─────────┐
│ 1  0  1  0  0   │   │ 1  0  1 │        │ 4  3  4 │
│ 0  1  1  0  0   │ × │ 0  1  0 │   →    │ 2  4  3 │
│ 1  0  1  1  0   │   │ 1  0  1 │        │ 2  3  4 │
│ 0  0  1  1  0   │   └─────────┘        └─────────┘
│ 0  1  1  0  0   │
└─────────────────┘
```

卷积核在输入图像上**从左到右、从上到下**滑动，每次计算对应位置的**元素乘积之和**：

$$Z[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} W[m,n] \cdot X[i+m, j+n] + b$$

### 输出尺寸计算

$$H_{out} = \left\lfloor \frac{H_{in} - k + 2p}{s} \right\rfloor + 1$$

| 参数 | 含义 | 常用值 |
|---|---|---|
| $k$ | 卷积核大小（kernel size） | 3, 5, 7 |
| $p$ | 填充（padding） | 0, 1, 2 |
| $s$ | 步长（stride） | 1, 2 |

**例子**：输入 32×32，卷积核 3×3，padding=1，stride=1：

$$H_{out} = \frac{32 - 3 + 2 \times 1}{1} + 1 = 32$$

加 padding=1 后输出与输入同尺寸（"same padding"）。

### 多通道卷积

RGB 图像有 3 个通道，卷积核也需要是 3 通道的：

```
输入：[3, 32, 32]（3通道 × 32×32）
卷积核：[64, 3, 3, 3]（64个卷积核，每个 3×3×3）

      ↓

输出：[64, 32, 32]（64个特征图）
```

参数量 = 64 × 3 × 3 × 3 + 64 = **1,792**（远少于全连接的1.5亿！）

---

## 3. 池化层（Pooling Layer）

池化层对特征图进行下采样，减少空间尺寸，增强平移不变性。

### 最大池化（Max Pooling）

```
特征图（4×4）         最大池化（2×2，stride=2）     输出（2×2）

┌────────────────┐                                ┌─────────┐
│  1   2   3   4 │   Max of [1,2,5,6] = 6          │ 6   8   │
│  5   6   7   8 │   Max of [3,4,7,8] = 8    →     │ 14  16  │
│  9  10  11  12 │   Max of [9,10,13,14] = 14       └─────────┘
│ 13  14  15  16 │   Max of [11,12,15,16] = 16
└────────────────┘
```

| 类型 | 操作 | 特点 |
|---|---|---|
| **Max Pooling** | 取窗口内最大值 | 保留最显著特征，最常用 |
| **Average Pooling** | 取窗口内平均值 | 更平滑，用于全局压缩 |
| **Global Average Pooling** | 整个特征图取均值→标量 | 现代网络替代全连接层 |

---

## 4. 完整 CNN 架构

### 经典卷积块（Conv Block）

```
输入
 ↓
Conv2d（提取特征）
 ↓
BatchNorm2d（稳定训练）
 ↓
ReLU（非线性激活）
 ↓
MaxPool2d（下采样）
 ↓
输出
```

### PyTorch 实现：LeNet 风格分类器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # Block 1: [3, 32, 32] → [32, 16, 16]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # same padding
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 尺寸减半
            
            # Block 2: [32, 16, 16] → [64, 8, 8]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: [64, 8, 8] → [128, 4, 4]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # 分类部分
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling → [128, 1, 1]
            nn.Flatten(),                  # → [128]
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 测试
model = CNN(num_classes=10)
x = torch.randn(32, 3, 32, 32)  # CIFAR-10 输入
output = model(x)
print(f"输出形状: {output.shape}")  # [32, 10]
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 5. 感受野（Receptive Field）

感受野指某个特征图上的一个像素点，对应原始输入图像的区域大小。

```
输入（5×5）          第1层 Conv3×3      第2层 Conv3×3
                    感受野 = 3×3        感受野 = 5×5

┌───────────────┐   ┌─────────┐        ┌───┐
│               │   │█████    │        │███│
│  原始像素      │ → │█████    │   →    │███│ ← 实际上覆盖了
│               │   │█████    │        │███│   原始 5×5 区域
└───────────────┘   └─────────┘        └───┘
```

**结论**：堆叠多个小卷积核（如 3×3）比使用一个大卷积核（如 7×7）更高效：
- 两个 3×3 卷积的感受野 = 一个 5×5 卷积（但参数量更少）
- 三个 3×3 卷积的感受野 = 一个 7×7 卷积（参数量减少 44%）

---

## 6. 重要 CNN 架构演进

### 6.1 AlexNet（2012）— 深度学习实用时代开始

```
输入(224×224×3) → Conv11 → Pool → Conv5 → Pool → 
Conv3 → Conv3 → Conv3 → Pool → FC4096 → FC4096 → FC1000
```

关键创新：
- **ReLU 激活**（取代 sigmoid）：解决梯度消失
- **Dropout**：防止全连接层过拟合
- **数据增强**：随机裁剪、翻转
- **GPU 训练**：双 GTX 580 并行

ImageNet Top-5 错误率：**15.3%**（第二名 26.2%，降低了近 11%）

### 6.2 VGGNet（2014）— 深度的力量

```python
# VGG 核心思想：统一用 3×3 卷积，靠深度取胜
vgg_block = nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1),  # 3×3 conv
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),  # 再来一个
    nn.ReLU(),
    nn.MaxPool2d(2, 2),               # 下采样
)
```

- VGG-16：16 层，138M 参数
- Top-5 错误率：**7.3%**
- 结构简洁，至今仍用于特征提取

### 6.3 ResNet（2015）— 残差连接突破深度瓶颈

深度网络训练时存在**退化问题**（Degradation Problem）：网络越深，训练误差反而上升（不是过拟合，是优化困难）。

ResNet 的解决方案：**残差连接（Skip Connection）**

```
普通卷积块：              残差块（Residual Block）：

x → Conv → BN → ReLU     x ────────────────┐
  → Conv → BN → ReLU →    ↓                  │
输出 H(x)                Conv → BN → ReLU   │
                          ↓                  │
                         Conv → BN           │
                          ↓              (恒等映射)
                          +  ←────────────────┘
                          ↓
                        ReLU → 输出 F(x) + x
```

**关键洞察**：学习残差 $F(x) = H(x) - x$ 比直接学习 $H(x)$ 更容易，极端情况下 $F(x) \approx 0$（恒等映射）也不会让网络退化。

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x                      # 保存输入
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out = out + residual              # 残差连接（关键！）
        out = F.relu(out)
        return out
```

ResNet-50：50 层，**Top-5 错误率 3.57%**（超过人类水平 5.1%！）

### 6.4 架构演进总览

| 模型 | 年份 | 层数 | 参数量 | Top-5 错误率 | 核心创新 |
|---|---|---|---|---|---|
| AlexNet | 2012 | 8 | 60M | 15.3% | ReLU + Dropout + GPU |
| VGGNet | 2014 | 16/19 | 138M | 7.3% | 深度 3×3 卷积 |
| GoogLeNet | 2014 | 22 | 6.8M | 6.7% | Inception 模块 |
| **ResNet** | **2015** | **50/101/152** | **25M** | **3.57%** | **残差连接** |
| DenseNet | 2016 | - | 8M | 3.46% | 密集连接 |
| EfficientNet | 2019 | - | 5M | 2.9% | 复合缩放 |
| Vision Transformer | 2020 | - | 86M | 1.5% | 纯 Attention |

---

## 7. 完整 CIFAR-10 训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),     # 数据增强
    transforms.RandomHorizontalFlip(),         # 随机翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),   # CIFAR-10 均值
                         (0.2023, 0.1994, 0.2010)),   # CIFAR-10 标准差
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# 数据集
train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
test_dataset  = datasets.CIFAR10('./data', train=False, transform=transform_test)
train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader   = DataLoader(test_dataset,  batch_size=128, shuffle=False)

# 模型、优化器、损失函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = CNN(num_classes=10).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)

# 训练循环
for epoch in range(50):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    test_acc = evaluate(model, test_loader)
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: loss={train_loss:.4f}, "
              f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")
```

---

## 8. 迁移学习（Transfer Learning）

在实际应用中，几乎不需要从头训练 CNN，而是使用**预训练模型（ImageNet 权重）**进行微调：

```python
import torchvision.models as models

# 加载预训练 ResNet-50
model = models.resnet50(pretrained=True)

# 冻结所有层（只微调最后的分类头）
for param in model.parameters():
    param.requires_grad = False

# 替换最后的全连接层（适配自己的类别数）
num_classes = 5   # 自定义数据集的类别数
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 只有 fc 层的参数会被更新
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
```

### 迁移学习策略

| 策略 | 做法 | 适用场景 |
|---|---|---|
| **特征提取** | 冻结全部预训练层，只训练分类头 | 数据量极少（<1000张） |
| **部分微调** | 冻结前几层，微调后几层+分类头 | 中等数据量 |
| **全量微调** | 以很小 LR（1e-5）微调全部层 | 数据量较多，与预训练任务差异大 |

**经验法则**：
- 数据集与 ImageNet 相似 → 冻结前层，微调后层
- 数据集很小且相似 → 只微调分类头
- 数据集很大且不同 → 全量微调，或从头训练

---

## 9. 可视化卷积核学到了什么

深度学习的一个直觉：**不同层学习不同抽象级别的特征**：

```
第1层：学习方向性边缘（横线、竖线、斜线）
       ┌──┐ ┌──┐ ┌──┐
       │──│ │  │ │ /│
       └──┘ └──┘ └──┘

第2层：学习纹理（网格、条纹、角落）
       ┌──┐ ┌──┐
       │▦ │ │≡ │
       └──┘ └──┘

第3-4层：学习物体部件（眼睛、轮子、纹理）
          ◎  ⊙  O

第5层+：学习高级语义特征（"狗脸"、"车轮"）
```

---

## 10. CNN vs Transformer（ViT）

2020 年，Vision Transformer（ViT）将 Transformer 引入图像领域：

| 对比项 | CNN | Vision Transformer (ViT) |
|---|---|---|
| **归纳偏置** | 局部性 + 平移不变性（内置） | 无归纳偏置（纯数据驱动） |
| **计算复杂度** | $O(n)$（n=像素数） | $O(n^2)$（注意力全局计算） |
| **小数据集** | ✅ 更好（内置先验） | ❌ 需要大量数据 |
| **大数据集** | ✅ 较好 | ✅ **更好**（可扩展性强） |
| **全局建模** | ❌ 有限（受感受野限制） | ✅ 天然全局 |
| **部署友好** | ✅ 成熟生态 | ✅ 逐渐成熟 |

**结论**：数据有限时优先选 CNN（ResNet/EfficientNet），大规模数据时 ViT/混合架构表现更优。

---

## 总结

| 特性 | CNN |
|---|---|
| **核心操作** | 卷积（局部连接 + 权值共享） |
| **主要组件** | Conv → BN → ReLU → Pool |
| **参数优势** | 比全连接少 1-2 个数量级 |
| **最强应用** | 图像分类、目标检测、图像生成 |
| **代表模型** | ResNet、EfficientNet、YOLO |
| **关键改进** | 残差连接（ResNet）解决深度退化 |
| **当前地位** | 视觉领域仍是基础，与 ViT 并驾齐驱 |
