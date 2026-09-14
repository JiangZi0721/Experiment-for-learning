# Experiment-for-learning (AI 底层技术全流程透视实验库)

> 这是我在学习人工智能、自然语言处理与大模型相关底层技术时创建的开源实验仓库。
> 每一个子项目都坚持**“白盒透视”与“全流程可复现”**设计哲学：拒绝黑盒调用框架直接一笔带过，提供纯手推底层源码、不同方法的横向对比消融、不同评估准则的对比、可视化看板以及严密的数学物理推导笔记。

---

## 🧭 全景实验导航目录 (Experiment Index)

| 编号 | 核心实验项目 | 核心机制与全流程透视重点 | 交付成果与可视化 | 状态 |
| :---: | :--- | :--- | :--- | :---: |
| **01** | [`01_word2vec_acceleration/`](./01_word2vec_acceleration) | **词表征与负采样加速**：共现矩阵、PMI、CBOW、Skip-Gram 与 Negative Sampling 高速化 | 词向量 2D/3D 降维流形投影、交互式 HTML、CLI 词类比计算 | `已完成` |
| **02** | [`02_whitebox_rag_system/`](./02_whitebox_rag_system) | **白盒端到端 RAG 检索生成**：切片策略、BM25 稀疏检索、Dense 稠密语义检索、RRF 倒数秩融合、交叉重排器、生成与评估 | 检索分阶段 Hit@K、MRR、NDCG 评测大盘、BGE 微调前后对比 | `已完成` |
| **03** | [`03_RNN-LM/`](./03_RNN-LM) | **循环神经网络与自回归语言模型**：NumPy 底层手推 RNNCell、TimeRNN、Gated RNN (GRU)、截断反向传播 (Truncated BPTT)、梯度检验 | 梯度暴冲与裁剪动画、隐状态记忆流动探针、自回归文本生成 | `已完成` |
| **04** | [`04_PPO-GRPO-DPO/`](./04_PPO-GRPO-DPO) | **大语言模型后训练强化对齐**：PPO (广义优势估计 GAE/Critic)、GRPO (群组相对优势估计免 Critic)、DPO (直接偏好优化隐式奖励) | 奖励建模、KL 散度漂移惩罚、三对齐算法训练动力学曲线 | `已完成` |
| **05** | [`05_Seq2seq_and_Attention/`](./05_Seq2seq_and_Attention) | **序列建模与注意力机制演进**：Classic Seq2Seq、Reverse 逆序输入、Peeky 解码器直连、Bahdanau 软注意力机制 | 算术加法 Exact Match 评测、英法翻译 Corpus-BLEU 曲线、注意力热力图 | `已完成` |
| **06** | [`06_whitebox_transformer/`](./06_whitebox_transformer) | **标准 Transformer 架构全景白盒透视**：Scaled Dot-Product、Multi-Head Attention、Sinusoidal PE、Causal Mask 与自回归解码器 | Web 交互式微观透视透镜、微观矩阵数值推演、公式级主动探针 | `已完成` |
| **07** | [`07_RL_Foundations/`](./07_RL_Foundations) | **强化学习理论基石与算法全景实证**：MDP 动态规划 (策略/价值迭代)、无模型 TD(0)/SARSA/Q-Learning、Robbins-Monro 收敛、DQN 与 Rainbow、分布 vs 样本模型、深度世界模型 MPC | 4×4/5×5 网格极限博弈、剧变时刻单步算术回溯、Gym 摆杆做梦渲染 GIF | `已完成` |
| **08** | [`08_CRF_Structured_Prediction/`](./08_CRF_Structured_Prediction) | **条件随机场与结构化预测**：线性链 CRF 原生手推、对数空间 Log-Sum-Exp 配分函数 $Z(X)$、维特比 (Viterbi) 动态规划解码、标注偏置 (Label Bias) 破解、NER 序列标注实战 | HMM vs BiLSTM vs BiLSTM-CRF 横向基准、转移矩阵硬约束热力图、0% 非法跳变证明 | `已完成` |
| **09** | [`09_MoE_Sparse_Routing/`](./09_MoE_Sparse_Routing) | **混合专家模型与稀疏门控路由**：Top-K 门控、Noisy Router、Switch/GShard 辅助负载均衡损失、DeepSeekMoE (细粒度专家 + 隔离共享专家)、DeepSeek-V3 免辅助损失动态偏置 | 路由塌缩垄断 vs 均衡消融对比图、算力对齐架构收敛曲线、领域专家分工热力图 | `已完成` |
| **10** | [`10_LLM_Foundations_and_Kernels/`](./10_LLM_Foundations_and_Kernels) | **大模型核心算子与系统底座基准**：RoPE 旋转位置编码、KV Cache 压缩演化 (MHA/GQA/MQA/DeepSeek MLA)、FlashAttention 在线分块 Softmax、RMSNorm、LoRA | RoPE 相对位置不变性严格证明、128K 超长文本显存节省 96.5% 建模、FlashAttention 零误差检验 | `已完成` |

---

## 🏛️ 项目设计哲学与通用规范

1. **白盒透明 (White-box Transparency)**：
   - 核心算法摒弃直接调用封装深不见底的黑盒库，尽量使用直接、可阅读、可断点调试的实现；
   - 暴露中间张量、梯度流动、隐藏状态和概率矩阵变化。
2. **数学可追溯 (Mathematical Rigor)**：
   - 每个代码实现均在 Markdown 笔记中配备严密的数学推导、矩阵计算图与物理直觉解读；
   - 标注所有推导的前提假设、逼近误差与工程边界。
3. **全流程横向对比消融 (Ablation & Duel)**：
   - 绝不针对单一“好结果”编程，所有项目均配备 Baseline 基线、消融对照组与极端压力测试；
   - 采用统一测试集与自回归测试标准（严禁在推断测试中使用 Teacher Forcing 等作弊手段）。
4. **统一指标与可视化大盘 (Visual Dashboards)**：
   - 每次实验自动生成并持久化出版级高清图表（收敛曲线、分布柱状图、对齐热力图等）；
   - 配备交互式终端 CLI（通过 Rich 终端看板输出高密度实测汇总）。
5. **轻量纯净与防生成物污染 (Clean Engineering)**：
   - 严格过滤编译缓存、虚拟环境、视频录屏与大体积权重；
   - 所有实验均提供一键复现入口：`python main.py --exp all`。

---

## 🛠️ 全局环境配置

本项目各个子工程均保持自闭环与独立运行能力，推荐使用 Python 3.10+ 环境：

```bash
# 核心基础科学计算与终端美化库
pip install torch numpy matplotlib rich tqdm
```

对于特定子工程，可以直接进入对应目录并安装该模块专属的 `requirements.txt`。

---

## 许可证 (License)

本仓库所有源码与技术文档采用 [MIT License](./LICENSE) 开源。欢迎 Star、Fork 并一起交流学习！
