# AI Atlas 🗺️

> 从数学基础到大语言模型的完整 AI 学习知识库

一份系统性的中文 AI 学习笔记，覆盖从线性代数、机器学习、深度学习、神经网络架构，到 NLP 与大语言模型的完整知识体系。每个主题均配有数学推导与 PyTorch 代码示例，循序渐进，适合有一定 Python 基础的学习者。

📖 **在线阅读**：[https://leonyangdev.github.io/ai-atlas/](https://leonyangdev.github.io/ai-atlas/)

---

## 知识体系

| 模块 | 内容 |
|------|------|
| 🧮 **数学基础** | 线性代数 · 微积分与梯度 · 向量化编程 · 归一化与标准化 |
| 🤖 **机器学习** | 特征工程 · 模型评估 · 监督学习全算法 · 集成学习 · 无监督学习 |
| 🔬 **监督学习** | 线性回归 · 逻辑回归 · KNN · 决策树 · SVM · 朴素贝叶斯 · 感知机 |
| 🌲 **集成学习** | 随机森林 · GBDT · AdaBoost · XGBoost · LightGBM |
| 📊 **无监督学习** | K-Means · DBSCAN · 层次聚类 · GMM · PCA · t-SNE |
| 🧠 **深度学习基础** | 激活函数 · 损失函数 · 反向传播 · 优化器 · 正则化与初始化 |
| 🏗️ **神经网络架构** | FNN · CNN · RNN · LSTM · GRU · Seq2Seq |
| 💬 **NLP** | 分词 · 词向量 · Word2Vec · GloVe · 注意力机制 · Transformer |
| 🚀 **预训练模型** | ELMo · BERT · GPT · 大语言模型 · 提示工程 · RLHF |
| ⚙️ **工程实践** | HuggingFace · 模型微调 · 模型部署 |

---

## 技术栈

- **文档框架**：[VitePress](https://vitepress.dev/)
- **数学公式**：MathJax 3（`markdown.math: true`）
- **代码示例**：Python · PyTorch · HuggingFace Transformers
- **部署**：GitHub Pages（GitHub Actions 自动构建）

---

## 本地运行

```bash
# 克隆仓库
git clone https://github.com/leonyangdev/ai-atlas.git
cd ai-atlas

# 安装依赖
npm install

# 启动本地开发服务器（支持热更新）
npm run dev
```

启动后访问 `http://localhost:5173/ai-atlas/` 即可预览。

```bash
# 构建静态文件
npm run build

# 本地预览构建产物
npm run preview
```

---

## License

[MIT](./LICENSE)
