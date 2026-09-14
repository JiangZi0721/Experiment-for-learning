# 前沿大模型与自主智能体 (Agent) 技术报告精要与架构演进

> **归属模块**：`MoE_and_AI_Agents`  
> **更新策略**：增量追加（严格遵循 Zero-Shrinkage 规范）  
> **面向对象**：大模型架构演进、推理时计算 (Test-Time Compute)、慢思考强化学习与工程级自治智能体研发

---

## 目录 (Table of Contents)
- [1. 学习记录流水线 (Changelog)](#1-学习记录流水线-changelog)
- [2. 行业宏观范式跃迁：从“预训练 Scaling”到“推理强化与具身智能”](#2-行业宏观范式跃迁从预训练-scaling到推理强化与具身智能)
- [3. 第一部分：顶级基座大模型技术报告深度解析](#3-第一部分顶级基座大模型技术报告深度解析)
  - [3.1 DeepSeek-V3 技术报告 (arXiv: 2412.19437)](#31-deepseek-v3-技术报告-arxiv-241219437)
    - [3.1.1 架构三叉戟：MLA + DeepSeekMoE + 无辅助损失平衡](#311-架构三叉戟mla--deepseekmoe--无辅助损失平衡)
    - [3.1.2 极致工程：FP8 混合精度与 DualPipe 零通信气泡并行](#312-极致工程fp8-混合精度与-dualpipe-零通信气泡并行)
    - [3.1.3 多 Token 预测 (MTP) 与投机解码加速](#313-多-token-预测-mtp-与投机解码加速)
  - [3.2 DeepSeek-R1 技术报告：纯强化学习激发自我推理涌现 (arXiv: 2501.12948)](#32-deepseek-r1-技术报告纯强化学习激发自我推理涌现-arxiv-250112948)
    - [3.2.1 DeepSeek-R1-Zero：零 SFT 纯强化学习与“顿悟时刻 (Aha Moment)”](#321-deepseek-r1-zero零-sft-纯强化学习与顿悟时刻-aha-moment)
    - [3.2.2 GRPO 算法：彻底移除 Critic 网络的群组相对策略优化](#322-grpo-算法彻底移除-critic-网络的群组相对策略优化)
    - [3.2.3 工业级四阶段管线 (Cold-Start -> RL -> SFT -> RL)](#323-工业级四阶段管线-cold-start---rl---sft---rl)
    - [3.2.4 核心认知突破：推理能力直接蒸馏至致密小模型](#324-核心认知突破推理能力直接蒸馏至致密小模型)
  - [3.3 OpenAI o1/o3 体系与推理时计算扩展定律 (Test-Time Compute Scaling)](#33-openai-o1o3-体系与推理时计算扩展定律-test-time-compute-scaling)
    - [3.3.1 过程奖励模型 (PRM) 与分步验证器](#331-过程奖励模型-prm-与分步验证器)
    - [3.3.2 动态算力自适应分配 (Adaptive Compute Allocation)](#332-动态算力自适应分配-adaptive-compute-allocation)
- [4. 第二部分：前沿顶尖自治智能体 (Agent) 架构与实战突破](#4-第二部分前沿顶尖自治智能体-agent-架构与实战突破)
  - [4.1 SWE-agent：为大模型定制 Agent-Computer Interface (ACI) (NeurIPS 2024)](#41-swe-agent为大模型定制-agent-computer-interface-aci-neurips-2024)
    - [4.1.1 痛点：为什么 LLM 直接操作 Bash 终端容易崩溃？](#411-痛点为什么-llm-直接操作-bash-终端容易崩溃)
    - [4.1.2 ACI 核心武器：受限查看、精准补丁与语法守卫](#412-aci-核心武器受限查看精准补丁与语法守卫)
  - [4.2 Agent Q：MCTS 树搜索与自我批判驱动的自进化 Web Agent (arXiv: 2408.07199)](#42-agent-qmcts-树搜索与自我批判驱动的自进化-web-agent-arxiv-240807199)
    - [4.2.1 破解误差累积 (Compounding Errors) 困局](#421-破解误差累积-compounding-errors-困局)
    - [4.2.2 MCTS 在线推演 + Self-Critique + 离线 DPO 飞轮机制](#422-mcts-在线推演--self-critique--离线-dpo-飞轮机制)
  - [4.3 Agentless：反思多智能体虚火，极简主义的胜利 (arXiv: 2407.01489)](#43-agentless反思多智能体虚火极简主义的胜利-arxiv-240701489)
    - [4.3.1 对复杂 Multi-Agent 架构“幻觉级联与天价 Token”的学术批判](#431-对复杂-multi-agent-架构幻觉级联与天价-token的学术批判)
    - [4.3.2 极简确定性三阶段：分层定位 -> 补丁生成 -> 测试用例验证](#432-极简确定性三阶段分层定位---补丁生成---测试用例验证)
  - [4.4 OSWorld 与 Anthropic Computer Use：操作系统级全模态 GUI 智能体](#44-osworld-与-anthropic-computer-use操作系统级全模态-gui-智能体)
    - [4.4.1 像素级视觉目标定位 (Visual Grounding & Coordinate Mapping)](#441-像素级视觉目标定位-visual-grounding--coordinate-mapping)
    - [4.4.2 观察-思考-执行-重绘自愈闭环 (Sense-Plan-Act-Verify Loop)](#442-观察-思考-执行-重绘自愈闭环-sense-plan-act-verify-loop)
- [5. 专业实战测评题库 (含采分点与硬核解析)](#5-专业实战测评题库-含采分点与硬核解析)
- [6. 极简复习闪卡 (CheatSheet)](#6-极简复习闪卡-cheatsheet)

---

## 1. 学习记录流水线 (Changelog)
- **2026-09-11**：系统编写前沿大模型技术报告与自主智能体研究全景笔记。深度拆解 DeepSeek-V3（MLA、DeepSeekMoE、MTP）、DeepSeek-R1（纯强化学习冷启动、GRPO 去 Critic、顿悟现象、四阶段管线及小模型逻辑蒸馏）；透视推理时计算扩展定律（PRM 与 MCTS 引导的思维链推演）；全面对比四大顶尖 Agent 研究架构（SWE-agent 的 ACI 人机接口、Agent Q 的 MCTS 树搜索与自进化飞轮、Agentless 极简三阶段定位验证流、OSWorld 桌面级多模态 GUI 智能体）。

---

## 2. 行业宏观范式跃迁：从“预训练 Scaling”到“推理强化与具身智能”

在 2024~2026 年间，人工智能行业发生了两大根本性范式转移：
1. **基座模型层**：
   - 依赖海量网页清洗数据的“暴力预训练 Scaling”逼近收益边际曲线（Data Wall & Power Wall）；
   - 行业全面转向**后训练强化学习（Post-Training RL）**与**推理时计算（Test-Time Compute）**，通过拉长推理阶段的自回归思维链（System 2 慢思考），实现逻辑、数学与代码的智力爆发。
2. **智能体应用层**：
   - 从盲目堆叠“角色扮演的多智能体对话（Multi-Agent Chat）”虚火中降温；
   - 转向**专为模型设计的操作接口（ACI）**、**可回溯的树搜索规划（MCTS）**以及**操作系统原生键鼠/视觉交互（GUI Computer Use）**。

---

## 3. 第一部分：顶级基座大模型技术报告深度解析

### 3.1 DeepSeek-V3 技术报告 (arXiv: 2412.19437)

DeepSeek-V3 是一个总参数量 671B、每个 Token 激活仅 37B 的极高效 MoE 语言模型，仅使用 2.788M H800 GPU 小时（约 600 万美元成本）即在 14.8T Tokens 上完成了训练，其能力匹敌全球闭源最顶尖模型。

```
                    ┌──────────────────────────────┐
                    │    DeepSeek-V3 核心技术基石    │
                    └──────────────┬───────────────┘
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   MLA 潜在注意力  │     │   DeepSeekMoE    │     │   工程体系创新   │
│ • 联合低秩隐变量  │     │ • 细粒度专家切分 │     │ • FP8 混合精度   │
│ • 解耦 RoPE 机制 │     │ • 共享专家常识池 │     │ • DualPipe通信重叠│
│ • KV 显存减少 93%│     │ • 无辅助损失平衡 │     │ • MTP 多Token预测│
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

#### 3.1.1 架构三叉戟：MLA + DeepSeekMoE + 无辅助损失平衡
* **MLA（Multi-head Latent Attention）**：
  将原本需要缓存在显存中的高维 Key 和 Value 投影压缩到一个仅 512 维的联合潜在隐变量中，配合矩阵吸收技术，**在保留 128 个全多头表达力的前提下，单 Token KV 缓存逼近极限的 576 字节**。
* **DeepSeekMoE**：
  将传统 MoE 的 8 个大专家细拆为 256 个微型路由专家（每个 Token 激活 8 个），并设立 1 个全时激活的**共享专家（Shared Expert）**隔离底层常识，大幅提升专家专业化纯度与知识组合自由度。
* **无辅助损失负载均衡（Auxiliary-Loss-Free Load Balancing）**：
  在门控打分中引入动态偏置 $b_i$，根据专家物理吞吐实时自动反馈微调，彻底甩掉损伤语言建模表征质量的辅助平衡损失项。

#### 3.1.2 极致工程：FP8 混合精度与 DualPipe 零通信气泡并行
* **FP8 混合精度训练**：细粒度 Tile-wise 和 Block-wise 量化策略，在 GEMM 乘法中将内存与算力带宽压到极限，且全程无溢出或损失突增。
* **DualPipe 双向调度**：计算流水线的前向计算和反向梯度更新与跨卡跨节点的 MoE All-to-All 巨量网络通信**时序完全交错重叠**，算力利用率（MFU）逼近理论极限。

#### 3.1.3 多 Token 预测 (MTP) 与投机解码加速
在训练阶段要求模型不仅预测 $t+1$ Token，还顺带推测 $t+2$ 等后续 Token；在推理阶段，该额外预测分支充当原生的投机解码器（Speculative Decoder），实现生成速度提升 1.8 倍以上。

---

### 3.2 DeepSeek-R1 技术报告：纯强化学习激发自我推理涌现 (arXiv: 2501.12948)

DeepSeek-R1 震撼了全球学术界与工业界，它用详实的实验证明：**无需昂贵的人类标注思维链，仅靠纯强化学习，基座模型即可自主学会 System 2 深度推理。**

#### 3.2.1 DeepSeek-R1-Zero：零 SFT 纯强化学习与“顿悟时刻 (Aha Moment)”
* **实验设计**：完全不给任何人工写的推理样例（Zero SFT），直接在 Base 模型上运行大规模强化学习。
* **奖励函数极其纯粹（Rule-based Reward）**：坚决不用容易被模型欺骗的神经奖励模型，只用**基于规则的客观判定**（数学答案是否正确、代码测试用例是否全过、格式标签是否合规）。
* **顿悟时刻涌现**：
  随着训练 Step 推进，模型自发学会了生成超长思考轨迹（数十万字）：
  在遇到难题卡壳时，模型自发输出：“*Wait, let me double check this... This doesn't seem right... Let's rethink from another angle...*”（等等，让我重新验算一下……这好像不对……让我换个思路重推）。模型自主掌握了**回溯验证、多分支探索与反思纠错**能力！

#### 3.2.2 GRPO 算法：彻底移除 Critic 网络的群组相对策略优化
传统 PPO 算法中，需要维护一个与 Policy 模型体积相当的 Critic（价值评估模型），在千亿参数下不仅显存直接翻倍，且 Value 网络极难收敛。
DeepSeek-R1 全面采用 **GRPO（Group Relative Policy Optimization）**：
1. 对同一个 Prompt 采样生成一组（如 $G$ 个）候选解答 $o_1, o_2, \dots, o_G$；
2. 计算每个解答的规则奖励得分 $r_1, r_2, \dots, r_G$；
3. 以该组得分的均值和方差对奖励进行组内归一化，直接得到优势函数（Advantage）：
   $$
   A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}
   $$
4. **彻底砍掉了 Critic 网络的显存占用与计算开销**，支持极大规模并发强化学习。

#### 3.2.3 工业级四阶段管线 (Cold-Start -> RL -> SFT -> RL)
为了解决 R1-Zero 存在的“语言混杂（中英乱窜）、格式难以阅读、无限死循环”的缺陷，DeepSeek 构筑了成熟的四阶段工业级管线：
```
1. 冷启动 SFT (数千条高质量长思考数据注入，规范语言与格式)
       │
       ▼
2. 大规模推理 RL (GRPO 针对数学/代码题目暴力强化，涌现反思与验证)
       │
       ▼
3. 拒绝采样 + 通识微调 (生成 80 万条推理数据，并混入 20 万条通识安全数据 SFT)
       │
       ▼
4. 全场景二次 RL (结合规则奖励与人类偏好对齐，兼顾深度思考与通识易读性)
```

#### 3.2.4 核心认知突破：推理能力直接蒸馏至致密小模型
报告证明：**直接用 R1 生成的高质量长推理轨迹，去微调（SFT）普通的小型 Dense 模型（如 Qwen-2.5-Coder、Llama-3），其效果碾压在小模型上自己从头做 RL！**
由此衍生的 DeepSeek-R1-Distill-Qwen-32B 等开源模型，在多项推理榜单上直接跨级击败此前数百亿规模的闭源大模型。

---

### 3.3 OpenAI o1/o3 体系与推理时计算扩展定律 (Test-Time Compute Scaling)

* **代表论文**：《Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters》（Snell et al., DeepMind / UC Berkeley, arXiv: 2408.03314）

#### 核心命题
当预训练数据耗尽后，**在推理阶段分配更多算力（Test-Time Compute），比单纯把模型参数翻倍更加有效**。

#### 3.3.1 过程奖励模型 (PRM) 与分步验证器
* **ORM（Outcome Reward Model，结果奖励）**：只在整段解答结束时看对错。在步骤繁复的长程证明中，一旦第 2 步有瑕疵，后续 50 步全成废话。
* **PRM（Process Reward Model，过程奖励）**：对思考链条中的**每一个单独推演步骤（Step）**逐一评估打分。配合树搜索（MCTS）或束搜索（Beam Search），在发现某一步得分暴跌时，立即砍掉该分支并回溯重推。

#### 3.3.2 动态算力自适应分配 (Adaptive Compute Allocation)
* 传统模型对所有问题一视同仁，固定输出；
* 新范式根据问题内在熵与难度自适应调整思考长度：
  * 面试问候语：消耗 50 个 Token 直出；
  * IMO 国际奥数或复杂安全漏洞挖掘：允许模型在后台静默展开 30,000 个 Token 的反复验算与自我推导，耗费数分钟计算，最终换取确定性的正确答案。

---

## 4. 第二部分：前沿顶尖自治智能体 (Agent) 架构与实战突破

如果说推理模型（R1 / o1）为 Agent 提供了强大的大脑（System 2 思考力），那么智能体架构则决定了大模型能否在真实复杂的软件工程、操作系统和网页交互中稳定落地。

---

### 4.1 SWE-agent：为大模型定制 Agent-Computer Interface (ACI) (NeurIPS 2024)

* **论文**：《SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering》（arXiv: 2405.15793，普林斯顿大学团队）
* **行业基准**：SWE-bench（在真实 GitHub 代码仓库中全自动修复真实 Bug 的黄金评测基准）。

#### 4.1.1 痛点：为什么 LLM 直接操作 Bash 终端容易崩溃？
早期开发者让模型直接调用 Linux 命令行，结果频频翻车：
1. **上下文爆炸**：模型无脑输入 `cat main.py`，几千行代码瞬间灌满上下文窗口，丢失前序关键记忆；
2. **死循环与报错蔓延**：终端报错刷屏数十页，模型在同一处语法错误上无休止打转；
3. **行号错位**：模型用 sed 替换代码时频繁因行号偏差改错位置，导致整个文件被破坏。

#### 4.1.2 ACI 核心武器：受限查看、精准补丁与语法守卫
SWE-agent 明确指出：**人类需要 IDE（代码高亮、跳转、静态检查），模型同样需要专属于它的交互界面（ACI）！**

```
┌────────────────────────────────────────────────────────┐
│             Agent-Computer Interface (ACI)             │
├──────────────────────────┬─────────────────────────────┤
│ 专用分页查看器 (Viewer)   │ 限制每次最多翻阅 100 行，   │
│                          │ 提供 open_file、scroll 语义  │
├──────────────────────────┼─────────────────────────────┤
│ 精确块替换工具 (Edit)     │ 类似统一 Diff 结构，强制匹配 │
│                          │ 唯一上下文代码段，杜绝误替换 │
├──────────────────────────┼─────────────────────────────┤
│ 静态语法拦截器 (Linter)   │ 文件保存前自动执行 AST 校验，│
│                          │ 语法不合法直接就地拦截纠正   │
└──────────────────────────┴─────────────────────────────┘
```

**成效**：通过合理设计 ACI 工具链，甚至不需要重新训练模型权重，就能将同等基座模型在真实软件仓库 Bug 修复上的成功率提升数倍。

---

### 4.2 Agent Q：MCTS 树搜索与自我批判驱动的自进化 Web Agent (arXiv: 2408.07199)

* **论文**：《Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents》（MultiOn & Stanford）
* **代表战绩**：真实网页预订任务零样本成功率从 **18.6% 暴增至 81.7%**，结合在线搜索更达到 **95.4%**。

#### 4.2.1 破解误差累积 (Compounding Errors) 困局
在网页交互（如预订多段机票、复杂税务申报表单）中，决策是**长程且多步不可逆的**。模型第 3 步点击了一个错误链接或输错了一个日期，后续 10 步就算再聪明也永远无法到达目标。

#### 4.2.2 MCTS 在线推演 + Self-Critique + 离线 DPO 飞轮机制
Agent Q 创造性地将 AlphaGo 的制胜法宝移植到了 Web 智能体上：
1. **蒙特卡洛树搜索（MCTS）前瞻推演**：
   在真正向外部环境提交动作前，Agent 在模拟副本中向前推演多步可能的分支路径；
2. **Self-Critique（自我评估价值网络）**：
   充当价值评估器，评估当前页面状态距离用户意图的胜率。若推演发现进入了死胡同或表单报错，**立即触发回溯（Backtracking）并修剪该搜索枝丫**；
3. **数据飞轮（Off-Policy DPO 持续学习）**：
   把所有搜索过程中踩过的坑（失败路径）和探索成功的终局（成功路径）记录为偏好对（Preference Pairs），离线执行 DPO 强化微调，让智能体在常态化使用中实现**自演进与持续增强**。

---

### 4.3 Agentless：反思多智能体虚火，极简主义的胜利 (arXiv: 2407.01489)

* **论文**：《Agentless: Demystifying LLM-based Software Engineering Agents》（UIUC 团队）
* **代表战绩**：在 SWE-bench Lite 上以仅仅 **$0.70 美元/题** 的超低成本，斩获 **32.0%** 成功率，正面击溃众多极其繁复的商业级多智能体系统。

#### 4.3.1 对复杂 Multi-Agent 架构“幻觉级联与天价 Token”的学术批判
当时业界充斥着盲目推崇“多角色辩论（PM、Coder、Reviewer、Tester 互相开会）”的风潮。Agentless 团队通过实证直指皇帝的新衣：
* 多个 Agent 之间长篇累牍的自然语言闲聊消耗了天文数字的 Token；
* 一个 Agent 的轻微幻觉会被其他 Agent 当作事实进一步演进，形成致命的**“幻觉级联（Hallucination Cascade）”**；
* 自主循环难以终止，缺乏确定性保障。

#### 4.3.2 极简确定性三阶段：分层定位 -> 补丁生成 -> 测试用例验证
Agentless 彻底砍掉复杂的自主循环，设计了一条高度确定性的工业级流水线：
```
1. 分层定位 (Localization):
   - 项目级：关键词搜索 + 骨干类树过滤出候选文件 (Top-3 文件)
   - 文件级：AST 解析定位关键函数和类
   - 行级别：精确圈定 20~50 行疑点代码区间
          │
          ▼
2. 修复生成 (Repair):
   - 仅将精准定位出的微小代码片段提交给 LLM，并行采样 5~10 个候选补丁 Patch
          │
          ▼
3. 验证与重排 (Validation):
   - 在 Docker 沙箱中运行现有测试集，拦截编译报错与回归失败
   - 选出最终唯一有效补丁提交
```
**工业界启示**：在可验证的高精工程任务中，**“精准的信息检索降噪 + 确定性流水线 + 沙箱用例验证”远比失控的“自由智能体漫游”更稳定、更便宜、更高效**。

---

### 4.4 OSWorld 与 Anthropic Computer Use：操作系统级全模态 GUI 智能体

* **代表研究**：
  * 《OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real-World Computer Environments》（NeurIPS 2024，arXiv: [2404.07972](https://arxiv.org/abs/2404.07972)）
  * Anthropic Claude 3.5/3.7 Sonnet Computer Use 架构

#### 4.4.1 像素级视觉目标定位 (Visual Grounding & Coordinate Mapping)
过去 Agent 只能依靠 Web DOM 树或文本 API，面对没有 API 的专业桌面软件（Excel、Photoshop、VS Code、操作系统设置）束手无策。
**Computer Use 打破了这一界限**：
* 模型直接接收桌面全分辨率截屏（RGB 图像）；
* 具备强悍的空间视觉对齐能力（Spatial Grounding），能精准输出屏幕上任意微小按钮、输入框的绝对像素坐标 $(x, y)$；
* 配合虚拟键鼠驱动直接下发 `mouse_move(x, y)`、`mouse_click(left)`、`key_press("ctrl+s")`。

#### 4.4.2 观察-思考-执行-重绘自愈闭环 (Sense-Plan-Act-Verify Loop)
1. **Sense（视觉截屏）**：抓取当前最新帧；
2. **Plan（思维推导）**：分析上一步操作是否生效（例如下拉菜单是否已经弹开）；
3. **Act（键鼠触发）**：执行下一步微操；
4. **Verify（重绘校验）**：若发现点击无效或出现意外错误弹窗，触发自愈分支进行纠偏。

---

## 5. 专业实战测评题库 (含采分点与硬核解析)

### 试题 1：GRPO 算法相较于 PPO 算法的显存与计算机制对比分析 (8分)
> **题目**：
> 在训练 DeepSeek-R1 这类长思考推理模型时：
> 1. （4分）写出标准 PPO 算法与 GRPO 算法在优势函数（Advantage）估计上的本质区别；说明为什么 GRPO 可以完全丢弃 Critic 网络；
> 2. （4分）假设基座模型参数量为 671B（按 FP8 存储，权重占用 671 GB）。若采用标准 PPO（Critic 采用同等尺寸模型），分析其相比 GRPO 在静态权重、优化器状态（使用 AdamW）以及训练并发度上的显存开销劣势。

#### 采分点与硬核标准解答：
1. **第 1 小问（4分）**：
   * PPO 的优势函数依赖广义优势估计（GAE），需要一个独立的 Critic（价值网络 $V(s)$）来估计每个状态的期望折扣回报（得分点：2分）；
   * GRPO 不对单点状态拟合绝对价值，而是对同一个输入 Prompt 采样生成一组（$G$ 个）候选解答，直接通过组内规则打分的**相对偏离度（$A_i = \frac{r_i - \mu_r}{\sigma_r}$）**作为优势（得分点：1分）；
   * 由于组内均值 $\mu_r$ 天然充当了基线（Baseline），无需显式参数化建模 $V(s)$，从而在数学上严谨地彻底移除了 Critic 网络（得分点：1分）。

2. **第 2 小问（4分）**：
   * 在 PPO 下，若单独维护一个 671B 的 Critic 网络：
     仅 Critic 自身的模型权重就需要增加 671 GB（FP8），其 AdamW 优化器状态（FP32 动量与二阶方差）更需占用 $671 \times 8 = 5368 \text{ GB}$ 显存（得分点：2分）；
   * GRPO 彻底砍掉了 Critic 的权重、梯度与优化器状态，直接为整套集群省去了超过 6000 GB 的静态显存开销（得分点：1分）；
   * **并发度收益**：省下的巨量显存可以全部分配给上下文激活值（Activation Memory）与 KV 缓存，允许模型展开长达数万 Token 的超长思维链推理采样，并支撑极高的并发 Prompt 吞吐量（得分点：1分）。

---

## 6. 极简复习闪卡 (CheatSheet)

| 标杆项目 / 研究 | 核心首创技术 | 核心优势 | 工业级最佳实践场景 |
| :--- | :--- | :--- | :--- |
| **DeepSeek-V3** | MLA + DeepSeekMoE + 无辅助损失平衡 + DualPipe | 671B 参数仅 37B 激活，极低训练/推理成本 | 高并发基座模型生产部署 |
| **DeepSeek-R1** | 纯强化学习激发推理 (GRPO) + 四阶段管线 + 小模型逻辑蒸馏 | 突破标注瓶颈，自发涌现 System 2 顿悟与自我反思 | 深度数学、代码算法与复杂逻辑推理 |
| **Test-Time Compute** | PRM 分步验证器 + 自适应算力分配 | 推理阶段用思考 Token 换智力，突破预训练上限 | 难题攻坚、形式化定理证明 |
| **SWE-agent** | ACI（Agent-Computer Interface）理念 | 专用受限浏览 + 统一补丁替换 + 静态语法守卫 | 自主软件工程编码与 GitHub Issue 修复 |
| **Agent Q** | MCTS 蒙特卡洛树搜索 + Self-Critique + 离线 DPO 飞轮 | 前瞻推演与错误回溯，破解多步不可逆 Web 误差雪崩 | 长流程复杂网页自动化与表单预订 |
| **Agentless** | 极简三阶段：分层定位 $\to$ 补丁生成 $\to$ 测试用例验证 | 破除 Multi-Agent 虚火，成本仅 $0.70/题，确定性极高 | 企业级代码自动修复与工业落地 |
| **OSWorld / Computer Use** | 像素级视觉坐标对齐 (Visual Grounding) + 动态闭环 | 跳出文本 API，像人类一样操作任意桌面软件与跨应用流 | 跨桌面系统复杂 GUI 自动化交互 |

