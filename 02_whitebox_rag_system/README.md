# White-Box RAG System

这是一个用于学习和验证 RAG（Retrieval-Augmented Generation）底层机制的实验项目。项目不依赖 LangChain 等高层 RAG 封装，而是把文档切分、稀疏检索、稠密检索、RRF 融合、重排序、上下文组装和生成过程拆开实现，方便观察每个阶段的输入、输出和排名变化。

## 学习目标

- 理解 Markdown 结构化切片如何保留标题层级和上下文。
- 对比 BM25 的关键词匹配与 Dense Embedding 的语义匹配。
- 理解 RRF 如何融合不同量纲的检索结果。
- 观察 Cross-Encoder 对候选片段进行二次排序的作用。
- 用可解释的评估指标定位检索噪声和上下文质量问题。
- 通过小规模对比学习实验理解领域 Embedding 微调。

## 技术栈

- Python 3.10+
- `jieba`：中文分词
- `rank-bm25`：Okapi BM25 检索
- `numpy`：向量计算和轻量级缓存
- `rich`：终端诊断面板
- `openai`、`requests`：兼容 OpenAI 协议的生成、Embedding 和 Reranker API
- `python-dotenv`：读取本地环境变量
- 可选 `torch`、`transformers`：本地 BGE 模型和对比学习微调

## 目录结构

```text
02_whitebox_rag_system/
├── main.py                         # 单 Query 和 Benchmark 入口
├── requirements.txt                # 基础依赖
├── .env.example                    # API 配置模板
├── benchmarks/test_queries.json    # 基准 Query
├── data/corpus/                    # 检索语料，同时也是学习笔记
├── data/cache/                     # 自动生成的缓存，不提交
├── experiments/                    # 分模块对比实验
├── scripts/                        # 语料构建、评估和训练脚本
├── src/                            # RAG 各阶段的直接实现
├── notes/                          # 项目外层 RAG 学习笔记
├── RAG_REPORT.md                   # 全流程实验报告
└── EVALUATION_METRICS_EXPERIMENT.md
```

## 快速开始

```powershell
cd 02_whitebox_rag_system
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

按需填写 `.env` 中的 DeepSeek、Embedding 和 Reranker 配置。未配置可选 API 时，程序会使用项目内的轻量级降级实现，但结果不能代表生产级模型效果。

运行单个 Query：

```powershell
python main.py --query "LSM-Tree 和 B+ Tree 在写放大与点查性能上有什么权衡？"
```

运行基准测试：

```powershell
python main.py --benchmark
```

运行自动化实验：

```powershell
python scripts/run_automated_experiments.py
python experiments/run_all_experiments.py
```

## 核心流程

1. **结构化切片**：解析 Markdown 标题层级，为正文附加完整标题路径。
2. **双路召回**：BM25 捕获专有名词和精确关键词，Dense 检索捕获同义表达和隐含语义。
3. **RRF 融合**：使用排名倒数融合两路结果，避免直接相加不同量纲的分数。
4. **Cross-Encoder 重排**：只对有限候选集做 Query-Chunk 交互打分，提升精度并控制成本。
5. **上下文生成**：将重排后的片段编号注入 Prompt，要求生成结果基于引用片段回答。
6. **指标评估**：从召回、上下文、生成和端到端四个层面观察系统瓶颈。

## 实验模块

| 模块 | 关注问题 | 入口 |
| --- | --- | --- |
| `01_reranker_ablation` | 有无重排序的排名差异 | `compare_ablation.py` |
| `02_embedding_finetune` | 领域微调是否扩大正负样本间隔 | `train_bge_contrastive.py` |
| `03_evaluation_metrics` | 优化前后的指标变化 | `compare_metrics_dashboard.py` |
| `04_hybrid_retrieval_duel` | BM25 与 Dense 的互补边界 | `compare_duel_synergy.py` |

## 复现实验数据

`data/corpus/`、`benchmarks/` 和 Markdown 报告会提交到仓库，因为它们是学习内容和实验输入。`data/cache/`、`models/` 和虚拟环境不会提交：

- 缓存可通过 `scripts/generate_corpus.py`、`scripts/run_automated_experiments.py` 等脚本重新生成。
- BGE 权重体积较大，应通过 `scripts/download_bge.py` 下载，或自行配置 API。
- 微调模型可运行 `scripts/train_bge_live.py` 重新生成。

## 已知限制

- 当前评估数据规模较小，实验结论用于理解机制，不能直接当作生产系统基准。
- 本地模型和 API 的结果受模型版本、硬件和服务端配置影响。
- 部分脚本会生成缓存文件，重复运行时应确认缓存来自同一份语料和配置。
- 没有 API Key 时只能验证检索链路，不能验证真实生成效果。

## 相关笔记

- [`notes/# RAG 评估指标体系.md`](./notes/%23%20RAG%20评估指标体系.md)
- [`notes/Hybrid_Retrieval_Reranking.md`](./notes/Hybrid_Retrieval_Reranking.md)
- [`notes/RAG_Complete_Architecture_Master.md`](./notes/RAG_Complete_Architecture_Master.md)
- [`RAG_REPORT.md`](./RAG_REPORT.md)
