import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'NLP 体系教程',
  description: '从零到一，全面掌握自然语言处理',
  lang: 'zh-CN',
  base: '/nlp-tutorial/',
  lastUpdated: true,
  cleanUrls: true,

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
  ],

  markdown: {
    math: true,
  },

  vue: {
    template: {
      compilerOptions: {
        // 将 MathJax 输出的 mjx-* 自定义元素告知 Vue 编译器，避免 SSR 报错
        isCustomElement: (tag) => tag.startsWith('mjx-'),
      },
    },
  },

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'NLP 教程',
    nav: [
      { text: '首页', link: '/' },
      {
        text: '教程',
        items: [
          {
            text: 'NLP 基础与历史',
            items: [
              { text: '1. NLP 导论', link: '/nlp-introduction/what-is-nlp' },
              { text: '2. NLP 发展史', link: '/nlp-introduction/history' },
              { text: '3. 数学基础', link: '/nlp-introduction/math-foundations' },
            ]
          },
          {
            text: '文本预处理',
            items: [
              { text: '4. 分词技术', link: '/text-preprocessing/tokenization' },
              { text: '5. 子词分词算法', link: '/text-preprocessing/subword' },
              { text: '6. 词表与编码', link: '/text-preprocessing/vocabulary' },
            ]
          },
          {
            text: '文本表示',
            items: [
              { text: '7. One-hot 与词袋', link: '/text-representation/one-hot-bow' },
              { text: '8. TF-IDF 与 N-gram', link: '/text-representation/tfidf-ngram' },
              { text: '9. Word2Vec', link: '/text-representation/word2vec' },
              { text: '10. GloVe 与 FastText', link: '/text-representation/glove-fasttext' },
            ]
          },
          {
            text: '传统序列模型',
            items: [
              { text: '11. RNN', link: '/sequence-models/rnn' },
              { text: '12. LSTM 与 GRU', link: '/sequence-models/lstm-gru' },
              { text: '13. Seq2Seq', link: '/sequence-models/seq2seq' },
            ]
          },
          {
            text: '注意力与 Transformer',
            items: [
              { text: '14. 注意力机制', link: '/attention-transformer/attention' },
              { text: '15. Transformer', link: '/attention-transformer/transformer' },
              { text: '16. 自注意力详解', link: '/attention-transformer/self-attention' },
              { text: '17. 位置编码', link: '/attention-transformer/positional-encoding' },
            ]
          },
          {
            text: '预训练语言模型',
            items: [
              { text: '18. ELMo', link: '/pretrained-models/elmo' },
              { text: '19. GPT 系列', link: '/pretrained-models/gpt' },
              { text: '20. BERT', link: '/pretrained-models/bert' },
              { text: '21. BERT 变体', link: '/pretrained-models/bert-variants' },
              { text: '22. 其他预训练模型', link: '/pretrained-models/other-models' },
            ]
          },
          {
            text: '现代 NLP',
            items: [
              { text: '23. 大语言模型', link: '/modern-nlp/llm' },
              { text: '24. 提示工程', link: '/modern-nlp/prompt-engineering' },
              { text: '25. 人类对齐', link: '/modern-nlp/alignment' },
            ]
          },
          {
            text: '工程实践',
            items: [
              { text: '26. HuggingFace 生态', link: '/engineering/huggingface' },
              { text: '27. 模型微调', link: '/engineering/finetuning' },
              { text: '28. 模型部署', link: '/engineering/deployment' },
            ]
          }
        ]
      }
    ],

    sidebar: [
      {
        text: '第一部分：NLP 基础与历史',
        collapsed: false,
        items: [
          { text: '1. NLP 导论', link: '/nlp-introduction/what-is-nlp' },
          { text: '2. NLP 发展史', link: '/nlp-introduction/history' },
          { text: '3. 数学基础', link: '/nlp-introduction/math-foundations' },
        ]
      },
      {
        text: '第二部分：文本预处理',
        collapsed: false,
        items: [
          { text: '4. 分词技术总览', link: '/text-preprocessing/tokenization' },
          { text: '5. 子词分词算法', link: '/text-preprocessing/subword' },
          { text: '6. 词表构建与文本编码', link: '/text-preprocessing/vocabulary' },
        ]
      },
      {
        text: '第三部分：文本表示',
        collapsed: false,
        items: [
          { text: '7. One-hot 与词袋模型', link: '/text-representation/one-hot-bow' },
          { text: '8. TF-IDF 与 N-gram', link: '/text-representation/tfidf-ngram' },
          { text: '9. Word2Vec', link: '/text-representation/word2vec' },
          { text: '10. GloVe 与 FastText', link: '/text-representation/glove-fasttext' },
        ]
      },
      {
        text: '第四部分：传统序列模型',
        collapsed: false,
        items: [
          { text: '11. RNN 循环神经网络', link: '/sequence-models/rnn' },
          { text: '12. LSTM 与 GRU', link: '/sequence-models/lstm-gru' },
          { text: '13. Seq2Seq 模型', link: '/sequence-models/seq2seq' },
        ]
      },
      {
        text: '第五部分：注意力机制与 Transformer',
        collapsed: false,
        items: [
          { text: '14. 注意力机制', link: '/attention-transformer/attention' },
          { text: '15. Transformer 架构', link: '/attention-transformer/transformer' },
          { text: '16. 自注意力与多头注意力', link: '/attention-transformer/self-attention' },
          { text: '17. 位置编码与 Layer Norm', link: '/attention-transformer/positional-encoding' },
        ]
      },
      {
        text: '第六部分：预训练语言模型',
        collapsed: false,
        items: [
          { text: '18. ELMo', link: '/pretrained-models/elmo' },
          { text: '19. GPT 系列', link: '/pretrained-models/gpt' },
          { text: '20. BERT', link: '/pretrained-models/bert' },
          { text: '21. BERT 变体', link: '/pretrained-models/bert-variants' },
          { text: '22. 其他预训练模型', link: '/pretrained-models/other-models' },
        ]
      },
      {
        text: '第七部分：现代 NLP 与大语言模型',
        collapsed: false,
        items: [
          { text: '23. 大语言模型原理', link: '/modern-nlp/llm' },
          { text: '24. 提示工程', link: '/modern-nlp/prompt-engineering' },
          { text: '25. 人类对齐：RLHF 与 DPO', link: '/modern-nlp/alignment' },
        ]
      },
      {
        text: '第八部分：工程实践',
        collapsed: false,
        items: [
          { text: '26. HuggingFace 生态', link: '/engineering/huggingface' },
          { text: '27. 模型微调与训练', link: '/engineering/finetuning' },
          { text: '28. 模型部署与推理优化', link: '/engineering/deployment' },
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/leonyangdev/nlp-tutorial' }
    ],

    footer: {
      message: 'NLP 体系教程 - 从零到一全面掌握自然语言处理',
      copyright: 'Copyright 2024'
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        }
      }
    },

    outline: {
      level: [2, 3],
      label: '页面导航'
    },

    lastUpdated: {
      text: '最后更新于'
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    }
  }
})
