# 模块导读：混合专家模型 (MoE) 与前沿自主智能体 (AI Agents)

> **模块路径**：`f:\LearningNotes\MoE_and_AI_Agents`  
> **核心定位**：前沿大模型基座架构（MoE/长推理）、慢思考强化学习（GRPO/Test-Time Compute）与下一代自主智能体系统工程

---

## 模块全景介绍

本模块系统解构当前人工智能从“暴力预训练”迈向“高效模型架构、推理慢思考与具身自治智能体”的核心技术变迁，由两份万字级深度笔记构成：

### 1. [01_MoE_Architecture_and_Expert_Mechanisms.md](./01_MoE_Architecture_and_Expert_Mechanisms.md)
* **核心内容**：
  * **专家到底是什么**：彻底打破“专家是外挂 Agent”的误区，从代码与数学上证明专家本质上就是前馈神经网络（FFN），并揭示 FFN 作为“键值联想记忆库（Key-Value Associative Memory）”的物理本质。
  * **MoE 的核心作用**：参数容量与计算开销的彻底解耦，打破预训练 Scaling Law 瓶颈。
  * **门控路由机制**：Top-K 稀疏打分与动态加权求和数学推导。
  * **传统 MoE 的致命痛点**：路由崩溃、专家饿死与传统辅助损失对语言建模表征的污染。
  * **DeepSeekMoE 的突破性革命**：细粒度专家切分（Fine-Grained Experts）、共享专家隔离（Shared Experts）、无辅助损失动态偏置平衡（Auxiliary-Loss-Free）与 DualPipe 双向流水线重叠通信。
  * **实战测评题库与极简复习闪卡**。

### 2. [02_Frontier_LLM_and_Agent_Technical_Reports.md](./02_Frontier_LLM_and_Agent_Technical_Reports.md)
* **核心内容**：
  * **顶级基座大模型标杆报告**：
    * **DeepSeek-V3**：MLA + DeepSeekMoE + FP8 混合精度 + MTP 多 Token 预测。
    * **DeepSeek-R1**：零 SFT 纯强化学习、顿悟时刻（Aha Moment）、GRPO 算法（彻底移除 Critic 网络）、四阶段管线及小模型逻辑蒸馏。
    * **推理时计算扩展（Test-Time Compute Scaling）**：过程奖励模型（PRM）与自适应算力分配（OpenAI o1/o3 体系）。
  * **顶尖自治智能体（Agent）实战架构**：
    * **SWE-agent**：专为大模型量身定制的 ACI（Agent-Computer Interface）理念，受限浏览与静态语法拦截。
    * **Agent Q**：MCTS 蒙特卡洛树搜索 + Self-Critique 价值网络 + 离线 DPO 飞轮，破解 Web 长程误差累积。
    * **Agentless**：反思多智能体虚火，极简三阶段（分层定位 $\to$ 补丁生成 $\to$ 测试验证），以 \$0.70 超低成本登顶 SWE-bench Lite。
    * **OSWorld & Anthropic Computer Use**：操作系统级全模态 GUI 智能体，视觉坐标映射与键鼠动态闭环。
  * **实战测评题库与极简复习闪卡**。

---

## 关联模块指引
* **长上下文注意力与 KV Cache 压缩**：参见 [`../LLM_Serving/02_KV_Cache_Compression_MQA_GQA_MLA_and_Beyond.md`](../LLM_Serving/02_KV_Cache_Compression_MQA_GQA_MLA_and_Beyond.md) 中的 5.5 节（DeepSeek CSA 与 HCA 混合架构）。
* **强化学习算法底层推导**：参见 [`../PPO-GRPO-DPO/PPO-DPO-GRPO强化学习笔记.md`](../PPO-GRPO-DPO/PPO-DPO-GRPO强化学习笔记.md)。
* **标准 Transformer 结构基础**：参见 [`../Transformer/Transformer_Architecture_Master.md`](../Transformer/Transformer_Architecture_Master.md)。

