# White-Box RAG Lab (白盒可透视 RAG 实战与算法印证实验室)

> 这是一个专为深入理解 RAG (Retrieval-Augmented Generation) 底层技术原理而构建的**白盒可透视教学级实战工程**。
> 本项目摒弃任何第三方框架的黑盒封装（拒绝 `chain.run()` 一笔带过），将文档切分、稀疏检索、稠密检索、倒数排名融合 (RRF)、交叉编码器 (Cross-Encoder) 重排序、Prompt 组装至大模型流式生成的**每一个物理阶段完全解构，并提供高信息密度的终端看板与排位轨迹透视**。

---

## 📖 目录 (Table of Contents)

1. [项目设计哲学与“白盒透视”体系](#-项目设计哲学与白盒透视体系)
2. [系统核心架构流程](#-系统核心架构流程)
3. [核心数学原理与物理机制](#-核心数学原理与物理机制)
4. [预设 Query 基准矩阵 (Query Benchmark Suite)](#-预设-query-基准矩阵-query-benchmark-suite)
5. [多领域语料库设计 (Corpus Structure)](#-多领域语料库设计-corpus-structure)
6. [目录组织规范](#-目录组织规范)
7. [快速开始 (Quick Start)](#-快速开始-quick-start)
8. [透视面板解读指南 (Trace Interpretation)](#-透视面板解读指南-trace-interpretation)

---

## 💡 项目设计哲学与“白盒透视”体系

在学习和研发 RAG 系统时，最常见的问题是：**当系统回答出现幻觉或缺失要点时，你无法快速定位究竟是切片丢了主谓宾、初排没召回、RRF 权重失衡，还是重排序误杀了黄金切片。**

本项目通过以下五层**全透明探针 (Inspection Probes)**，让你实时观察数据的每一次蜕变：

```
[原始文档]
    │
    ▼ 【探针 1：切块透视】打印字符区间、Token 计数、AST 父子标题元数据
[结构化切片 Chunks]
    │
    ├───────────────┬───────────────┐
    ▼               ▼               ▼
[Jieba 分词]    [向量化 API]    [原味切片]
    │               │               │
    ▼               ▼               │
[BM25 词频打分]  [Cosine 相似度]     │
    │               │               │
    └───────┬───────┘               │
            ▼ 【探针 2：初排双路对抗看板】打印 BM25 分数 vs Dense 分数及各自 Top-K
    [双路名单合并]
            │
            ▼ 【探针 3：RRF 融合演算过程】展示 1/(k+rank) 算分过程与名次洗牌
    [Top-N 粗排候选集]
            │
            ▼ 【探针 4：Cross-Encoder 交叉重排】显示一对一打分与排位颠覆幅度
    [Top-K 精选黄金切片]
            │
            ▼ 【探针 5：真实 Prompt 组装透视】展示送给 DeepSeek 的完整上下文 payload
    [DeepSeek 流式生成带引用回答]
```

---

## ⚙️ 系统核心架构流程

1. **AST 结构化切分 (Structural Markdown Chunking)**：
   - 不采用暴力滑动窗口截断，基于 Markdown 标题树解析。
   - 自动为每个子块头部附加其所属的完整父级标题路径（Breadcrumbs），彻底杜绝切片脱离上下文导致的指代不清。
2. **第一阶段：双路并发召回 (First-Stage Retrieval)**：
   - **稀疏路 (BM25)**：针对中文采用 Jieba 精确分词，构建倒排索引并依据 TF-IDF/BM25 算法严格计算词频与逆文档频率得分。
   - **稠密路 (Dense Embedding)**：将切片送入向量模型提取高维语义嵌入向量，使用余弦相似度计算与 Query 的几何对齐得分。
3. **第二阶段：倒数排名融合 (RRF Fusion)**：
   - 抹平两路检索引擎的分数量纲差异，依据排名名次倒数计算综合评分并去重合并。
4. **第三阶段：二次精排 (Cross-Encoder Reranking)**：
   - 调用交叉编码器 API 或轻量接口对 Query 与每个候选切片执行一对一全注意力交互打分，执行严酷淘汰。
5. **第四阶段：上下文注入与生成 (DeepSeek Generation)**：
   - 将精排后的黄金切片按置信度编号注入带有抗幻觉约束的 System Prompt，流式调用 DeepSeek API 生成直击问题的严谨回答并完成溯源引用。

---

## 📐 核心数学原理与物理机制

### 1. BM25 (Best Matching 25) 稀疏评分公式
$$Score_{BM25}(D, Q) = \sum_{i=1}^{N} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{avgdl}\right)}$$

- $f(q_i, D)$：查询词 $q_i$ 在文档 $D$ 中的词频。
- $|D| / avgdl$：当前文档长度与知识库平均文档长度之比。
- $k_1$（通常取 1.5）：控制词频饱和度的上限。
- $b$（通常取 0.75）：文档长度惩罚系数。

### 2. 向量余弦相似度 (Cosine Similarity)
$$Sim_{Dense}(q, d) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|} = \frac{\sum_{i=1}^{n} q_i d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \sqrt{\sum_{i=1}^{n} d_i^2}}$$

- 将文本压缩至连续低维空间，评估语义方向的一致性。

### 3. RRF (Reciprocal Rank Fusion) 倒数排名融合算法
$$RRF\_Score(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

- $M$：参与召回的检索系统集合（如 `[BM25, Dense]`）。
- $r_m(d)$：文档 $d$ 在检索系统 $m$ 中的名次排名（1-indexed，如第 1 名取 1）。
- $k$：平滑常数（工业标准常取 $k = 60$），防止排在第 1 名的文档拥有过大的主导权重。

### 4. 交叉重排序 (Cross-Encoder Attention)
$$\text{Input} = [\text{CLS}] \circ Q \circ [\text{SEP}] \circ D \circ [\text{SEP}]$$
$$\text{Score} = \sigma\left(W \cdot \text{Transformer}(\text{Input})_{[\text{CLS}]}\right)$$

- Query 与 Chunk 中的每一个 Token 互相计算全注意力权重，复杂度为 $O((L_Q + L_D)^2)$。
- 彻底突破双塔模型（Bi-Encoder）仅在最终向量点乘的粗糙交互，实现深层语义对齐。

---

## 🎯 预设 Query 基准矩阵 (Query Benchmark Suite)

系统预设了 5 类攻防维度的测试用例，位于 `benchmarks/test_queries.json`：

| 类别 | 典型测试 Query | 侧重点与算法验证目标 |
| :--- | :--- | :--- |
| **Q1: 精确符号与配置 (Exact Match)** | `"ZeRO-3 是如何通过状态切片优化显存占用的？"` | 验证 BM25 对冷门技术缩写的敏锐捕获力，对比 Dense 是否发生语义漂移。 |
| **Q2: 零关键词纯语义 (Zero-Keyword)** | `"数据库在断电崩溃后如何保证已提交的事务不丢失数据？"` | 故意隐去 `WAL`、`Redo Log` 专有词，验证 Dense 跨越词汇鸿沟的能力，观察 BM25 的失灵。 |
| **Q3: 跨领域多义词 (Ambiguity)** | `"WAL 在刷盘时如何处理检查点 (Checkpoint)？"` | 语料库中同时存在 DB 的 WAL 和 Raft Log。观察两路召回的冲突，以及 Cross-Encoder 如何根据语境去噪。 |
| **Q4: 架构权衡对比 (Trade-off)** | `"对比 LSM-Tree 与 B+ Tree 在写放大与点查性能上的权衡。"` | 跨多篇切片的信息拼装，检验 RRF 融合是否能同时覆盖两套存储引擎的切片。 |
| **Q5: 对抗诱导与拒答 (Hallucination)** | `"Kubernetes 调度器采用了哪种量子退火优化算法？"` | 检验知识库为空/无关时的拒答机制，严格评测 Faithfulness 指标。 |

---

## 📚 多领域语料库设计 (Corpus Structure)

语料库按四个高壁垒的技术领域分类存放在 `data/corpus/` 下，共约 100 篇高质量技术拆解文档：

- `01_distributed_systems/`：分布式共识协议、事务与容灾调度机制（约 25 篇）
- `02_storage_engines/`：存储引擎底层、LSM/B+Tree、WAL、MVCC 机制（约 25 篇）
- `03_llm_infra/`：大模型底层架构、注意力变体、显存优化与量化系统（约 25 篇）
- `04_os_kernel_network/`：操作系统虚拟内存、零拷贝、epoll/io_uring 与网络栈（约 25 篇）

---

## 📁 目录组织规范

```text
whitebox_rag/
├── README.md               # 本手册
├── requirements.txt        # 纯轻量级依赖项 (rich, openai, rank_bm25, jieba 等)
├── .env.example            # 环境变量配置模板
├── benchmarks/
│   └── test_queries.json   # 5 大维度的基准评测 Query
├── data/
│   └── corpus/             # 4 大领域约 100 篇精选 Markdown 语料
├── src/
│   ├── __init__.py
│   ├── config.py           # 环境变量与运行时参数加载
│   ├── chunker.py          # AST 结构化切分器 (Markdown 标题树与元数据拼装)
│   ├── visualizer.py       # 基于 Rich 的中间态多维表格透视看板
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py    # 稀疏检索实现 (Jieba 分词 + 词频命中透视)
│   │   ├── dense_retriever.py   # 稠密检索实现 (API 嵌入向量提取 + 余弦相似度)
│   │   └── rrf_fusion.py        # RRF 倒数排名融合算法与名次跃迁追踪
│   ├── reranker.py         # 交叉编码器精排与排位颠覆透视
│   └── generator.py        # DeepSeek API 流式生成与溯源引用验证
└── main.py                 # 一键运行入口（支持单 Query 诊断与 Benchmark 批量跑测）
```

---

## 🚀 快速开始 (Quick Start)

### 1. 安装纯轻量依赖（无需 GPU，无需大型 PyTorch）
```bash
cd whitebox_rag
python -m venv venv
# Windows 激活
.\venv\Scripts\activate
# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key
复制 `.env.example` 为 `.env`：
```ini
# DeepSeek API Key (必填，用于回答生成)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Embedding API Key (推荐 SiliconFlow 或兼容 OpenAI 协议的接口)
EMBEDDING_API_KEY=your_embedding_api_key_here
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3

# Reranker API Key (支持 SiliconFlow 或备用模拟打分模式)
RERANKER_API_KEY=your_reranker_api_key_here
RERANKER_BASE_URL=https://api.siliconflow.cn/v1
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

*(提示：若暂未配置 Embedding/Reranker 的 API Key，系统会自动平滑降级至内置的高性能轻量计算引擎，确保单凭 DeepSeek Key 也能完整跑通全链路！)*

### 3. 运行单个 Query 进行端到端透视
```bash
python main.py --query "ZeRO-3 是如何通过状态切片优化显存占用的？"
```

### 4. 批量运行 5 维基准测试矩阵
```bash
python main.py --benchmark
```

---

## 📊 透视面板解读指南 (Trace Interpretation)

运行程序后，终端将依次打印出以下几张核心诊断表：

### 面板 1：初排双路对抗表 (First-Stage Retrieval Duel)
```text
┌──────┬────────────────────────────────┬────────────┬─────────────┬───────────┐
│ Rank │ Chunk ID & 标题路径             │ BM25 得分  │ Dense 相似度│ 命中特征  │
├──────┼────────────────────────────────┼────────────┼─────────────┼───────────┤
│ 1    │ [llm_infra#03] ZeRO显存切片机制 │ 18.420     │ 0.865       │ 语义+专名 │
│ 2    │ [llm_infra#08] 梯度与优化器状态 │ 12.150     │ 0.812       │ 词频高    │
│ 3    │ [os_kernel#02] 虚拟内存页表分配 │ 0.000      │ 0.690       │ 语义发散  │
└──────┴────────────────────────────────┴────────────┴─────────────┴───────────┘
```
- **分析技巧**：观察 BM25 得分是否在精准匹配时显著拉开差距；观察 Dense 是否把“内存”相关但领域错误的操作系统切片拉进了前 5。

### 面板 2：RRF 融合名次跃迁表 (RRF Rank Shifts)
展示每个切片在 BM25 排名 $r_1$、Dense 排名 $r_2$ 以及计算 $\frac{1}{60+r_1} + \frac{1}{60+r_2}$ 后的最终排名，直观显示**“双路互保”**如何捞出被单一算法忽视的候选块。

### 面板 3：Cross-Encoder 重排颠覆榜 (Rerank Shake-up)
高亮展示哪些切片在初排中凭借关键词名列前茅，但经全注意力打分后因“答非所问”被直接淘汰；哪些切片在初排边缘（如第 15 名）却被 Cross-Encoder 一举提拔至前 3 名。

### 面板 4：DeepSeek 最终 Prompt 与流式生成
打印真实注入的上下文窗口，以及 DeepSeek 如何依据编号标注 `[1]`、`[2]` 精确输出事实论据。
