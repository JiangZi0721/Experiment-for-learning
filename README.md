# Experiment-for-learning

这是一个用于学习 AI 相关底层技术的实验仓库。每个子项目都尽量保持自包含，提供源码、实验记录、数学推导和可复现实验入口，避免只调用高层框架而看不到内部机制。

## 实验目录

| 编号 | 项目 | 内容 | 状态 |
| --- | --- | --- | --- |
| 01 | [`01_word2vec_acceleration/`](./01_word2vec_acceleration) | Word2Vec、负采样、向量化实现与可视化 | 已完成 |
| 02 | [`02_whitebox_rag_system/`](./02_whitebox_rag_system) | 白盒 RAG：切片、BM25、Dense、RRF、重排、生成与评估 | 已完成 |
| 03 | [`03_RNN-LM/`](./03_RNN-LM) | 白盒 RNN、GRU、BPTT、RNNLM 与梯度检验 | 已完成 |
| 04 | [`04_PPO-GRPO-DPO/`](./04_PPO-GRPO-DPO) | 白盒 PPO、DPO、GRPO 与大模型对齐实验 | 已完成 |

## 通用规范

1. **白盒透明**：关键算法尽量使用直接、可阅读的实现，展示中间结果和排名变化。
2. **数学可追溯**：代码对应的核心公式、假设和局限写入 Markdown 笔记。
3. **实验可复现**：每个项目提供依赖说明、运行命令和结果记录。
4. **避免提交生成物**：模型权重、虚拟环境、缓存和编译产物不进入 Git；需要时根据 README 下载或重新生成。

## 许可证

本仓库采用 [MIT License](./LICENSE)。
