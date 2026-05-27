# NLP 体系教程

> 从零到一，全面掌握自然语言处理

一份系统性的中文 NLP 学习教程，涵盖从基础文本表示到 Transformer 架构，再到现代大语言模型的完整知识体系。每章均配有数学推导与 PyTorch / HuggingFace 代码示例，循序渐进，适合有一定 Python 基础的学习者。

📖 **在线阅读**：[https://leonyangdev.github.io/nlp-tutorial/](https://leonyangdev.github.io/nlp-tutorial/)

---

## 目录

| 部分 | 章节 |
|------|------|
| **一、NLP 基础与历史** | 1. NLP 导论 · 2. NLP 发展史 · 3. 数学基础 |
| **二、文本预处理** | 4. 分词技术 · 5. 子词分词算法 · 6. 词表与编码 |
| **三、文本表示** | 7. One-hot 与词袋 · 8. TF-IDF 与 N-gram · 9. Word2Vec · 10. GloVe 与 FastText |
| **四、传统序列模型** | 11. RNN · 12. LSTM 与 GRU · 13. Seq2Seq |
| **五、注意力机制与 Transformer** | 14. 注意力机制 · 15. Transformer 架构 · 16. 自注意力与多头注意力 · 17. 位置编码与 Layer Norm |
| **六、预训练语言模型** | 18. ELMo · 19. GPT 系列 · 20. BERT · 21. BERT 变体 · 22. 其他预训练模型 |
| **七、现代 NLP 与大语言模型** | 23. 大语言模型原理 · 24. 提示工程 · 25. 人类对齐：RLHF 与 DPO |
| **八、工程实践** | 26. HuggingFace 生态 · 27. 模型微调与训练 · 28. 模型部署与推理优化 |

---

## 技术栈

- **文档框架**：[VitePress](https://vitepress.dev/)
- **数学公式**：[MathJax 3](https://www.mathjax.org/)（通过 `markdown-it-mathjax3`）
- **代码示例**：Python · PyTorch · HuggingFace Transformers
- **部署**：GitHub Pages

---

## 本地运行

```bash
# 克隆仓库
git clone https://github.com/leonyangdev/nlp-tutorial.git
cd nlp-tutorial

# 安装依赖
npm install

# 启动本地开发服务器（支持热更新）
npm run dev
```

启动后访问 `http://localhost:5173/nlp-tutorial/` 即可预览。

```bash
# 构建静态文件
npm run build

# 本地预览构建产物
npm run preview
```

---

## License

[MIT](./LICENSE)
