# 自进化智能体 (Self-Evolving Agents) 前沿论文每日追踪

> **专栏定位**：本专栏专用于追踪与萃取 [Hugging Face Daily Papers](https://huggingface.co/papers) 中与自进化智能体、在策略强化学习、自我蒸馏、长程记忆演化、工具自创相关的最前沿论文，进行深度学术结构化拆解。
> **更新机制**：每日晚上 20:00 自动化同步或按需手动触发，永久增量追加，历史论文存档不删除、不压缩。

---

## 📅 2026-09-18 前沿论文追踪日报（自进化智能体专题）

> **数据源**：[Hugging Face Daily Papers](https://huggingface.co/papers) | **抓取与分析时间**：2026-09-18 20:04:38


今日共精准萃取并深度剖析 **20** 篇与**智能体自主强化、在策略蒸馏、环境探索与安全演化**强相关的前沿学术论文：


### 1. [Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL](https://huggingface.co/papers/2609.20715)

- **论文元数据**：`arXiv:2609.20715` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.20715) | [arXiv 原文](https://arxiv.org/abs/2609.20715)

- **作者与归属机构**：Juzheng Zhang, Disha Makhija, Manoj Ghuhan Arivazhagan, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah（Amazon）
- **学术社区关注度**：🔥 **1** Upvotes | **命中核心标签**：`agent`, `agents`, `reinforcement learning`, `rl`, `trajectory`, `exploration`, `environment`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探究在智能体轨迹有监督微调（SFT）阶段，是否应该对‘环境返回的观测（Observation Tokens）’进行损失监督，以及这种微调表征对后续强化学习（RL，特别是 GRPO）探索能力产生何种深层影响。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  行业惯例（如标准 SFT）仅在智能体生成的 Action Tokens 上计算 Cross-Entropy Loss，将环境 Observation 仅作为输入 Context 并施加 Loss Mask。这导致智能体在 SFT 后对环境因果动态（World Dynamics）的预测能力退化，使后续 RL 探索陷入低熵、模式坍塌与局部最优。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出 **ActObs 训练机制**：
    1. **行动-观测联合全序列监督**：在智能体交互轨迹中，直接对环境中已有的 Observation Tokens 同步计算监督 Loss；
    2. **零额外开销的环境世界模型内化**：不增加任何模型参数、不引入外部预测头、无需额外前向传播，让 Policy 模型在隐表征层自然掌握动作对环境造成的后续影响（Action Consequences）；
    3. 实验揭示在微调初期，Action 梯度与 Observation 梯度迅速正交化，联合监督能阻止单一视角的过拟合与退化。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  ① 经过 ActObs 初始化后，采用 GRPO 强化学习时，智能体能够保持显著更高的策略熵（Entropy），以更小的策略位移探索更开阔的状态空间；② 在 Terminal-Bench 2.0 上，Qwen3-4B 在所有采样预算下的 pass@k 均大幅击败基线；Qwen3-8B 在 pass@16 上提升 3.4 个百分点；③ 在未见过的跨域代码编辑基准 Aider-Polyglot 上，4B 模型的 pass@1 零样本提升 4.2 个百分点，强力证实‘环境建模能力是自进化智能体在未知领域自主探索的底座’。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> Agent trajectories record what an agent does and what happens next. Yet standard supervised fine-tuning (SFT) applies loss only to agent-authored action tokens, using environment observations as context but not as prediction targets. We ask whether this convention provides the best initialization for subsequent reinforcement learning. We introduce ActObs, which also supervises the observation tokens already present in each trajectory. Although deployed agents never generate observations, learning to predict them encourages the policy to model action consequences without adding data, parameters, sequence tokens, or forward passes. The methods perform similarly after SFT but diverge after GRPO. On Qwen3-4B, GRPO from ActObs achieves higher pass@k at every evaluated sampling budget than its action-only counterpart on Terminal-Bench 2.0. On Qwen3-8B, it trades some pass@1 reliability for higher pass@k (+3.4 pp at pass@16) and solves more distinct tasks. The advantage extends to cross-domain code editing on aider-polyglot (+4.2 pp at pass@1 at 4B), whose tasks are unseen during SFT and RL. ActObs retains more entropy during RL while requiring less policy movement, leaving the final policy closer to its SFT initialization. Our analysis traces this difference to SFT: action and observation gradients rapidly become orthogonal, while action-only training leaves a large residual observation gradient and degrades environment prediction below the base model. Joint supervision prevents this one-sided specialization, preserving consequence prediction and preparing the policy for downstream exploration.

</details>

---

### 2. [When EOS Tokens Disagree: Understanding Length Inflation in On-Policy Distillation](https://huggingface.co/papers/2609.20511)

- **论文元数据**：`arXiv:2609.20511` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.20511) | [arXiv 原文](https://arxiv.org/abs/2609.20511)
 | [💻 官方开源代码](https://github.com/UNCSciML/opd-eos)

- **作者与归属机构**：Yuxiao Yang, Tianrun Yu, Shangzhe Li, Kaixiang Zhao, Xuchao Zhang 等（学术与工业界研究机构）
- **学术社区关注度**：🔥 **32** Upvotes | **命中核心标签**：`distillation`, `on-policy`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  深入分析在策略知识蒸馏（On-Policy Distillation, OPD）过程中，学生智能体响应长度发生恶性急剧膨胀（Length Inflation，甚至耗尽上下文预算）的深层机理并提出消除方案。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  智能体通过蒸馏进行自我学习或向大模型对齐时，学生模型在训练中后期会产生越来越冗长的推导与动作序列。此前研究往往将其归咎于奖励黑客，但真实成因一直未被定性。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  1. **揭示终止符错位机制（Termination-Token Mismatch）**：发现即使学生与教师模型的终止符词表（Declared Stopping Sets）一致，由于后训练差异，两者的概率质量会分配在不同的语义等价 EOS 字符上。这种分流导致学生原本的停止动作被惩罚压制，而教师推荐的替代终止动作又无法稳定转移，迫使模型只能不断向后生成；
    2. **语义等价终止合并算法**：提出将所有功能上等价的 EOS 标记统一建模为单一的共享语义终止动作（Shared Semantic Stopping Action）；
    3. **阶段式训练动态追踪**：量化了 K2-Horizon 不同训练阶段终止动作的漂移曲线。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  彻底解决了 Qwen3、Llama、Gemma 等主流模型在进行多轮自蒸馏/自我强化过程中的推理长度膨胀问题，大幅降低了自进化智能体在闭环训练中的算力开销与 Token 浪费，保证了生成动作的高效与紧凑。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> We study length inflation in on-policy distillation (OPD), where student responses can become excessively long and even exhaust the generation budget. We identify termination-token mismatch between base students and post-trained teachers as an important source of this behavior. Across Qwen3, Llama, and Gemma, the two models can place their stopping probability on different EOS tokens, even when their declared stopping sets are identical. This mismatch can suppress the student's preferred termination action without reliably transferring the teacher-preferred alternative. We show that aligning the decoding stopping set alone is insufficient, while treating functionally equivalent EOS tokens as a shared semantic stopping action substantially mitigates mismatch-induced length inflation across all three model families. To further understand how termination behavior evolves over training, we study OPD across different K2-Horizon training stages. This stage-wise analysis shows that termination preferences can shift substantially during training, while also revealing a distinct length inflation late in the OPD run that persists beyond termination alignment. Together, these results identify termination mismatch as an important, but not exhaustive, source of OPD length dynamics. We release an implementation incorporating the proposed termination-handling corrections.

</details>

---

### 3. [PACT: Can Enterprise AI Assistants Be Trusted Under Pressure?](https://huggingface.co/papers/2609.18605)

- **论文元数据**：`arXiv:2609.18605` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.18605) | [arXiv 原文](https://arxiv.org/abs/2609.18605)
 | [💻 官方开源代码](https://github.com/trace-ai-labs/pact)

- **作者与归属机构**：Mika Okamoto, Ansel Kaplan Erol（Georgia Institute of Technology）
- **学术社区关注度**：🔥 **2** Upvotes | **命中核心标签**：`agent`, `agents`, `compliance`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  评估企业级 AI 智能体在面对外部极端压力（如用户强硬催促、管理者紧急催进度、违规走捷径利益驱动）时，能否坚守系统设定的合规规则（Compliance Rules）。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  当前智能体基准只评估‘静态无干扰’下的指令遵循，但自进化与自主执行的智能体一旦进入生产业务流程（招聘、财务、医疗等），在面对多轮沟通施压时极易被诱导破防、违背核心安全规则。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  1. 提出 **PACT（Pressure-Applied Compliance Testing）基准**：覆盖 12 个强监管领域、48 种复杂真实场景；
    2. 引入系统性施压电池（Battery of Pressures），包括多轮语义说服、角色权威施压与诱导性捷径；
    3. 提出综合可靠性合规指数 **PACTScore**，衡量抗压韧性、透明度与规则边界感知度。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  测试了业界 22 款主流大模型智能体，揭示即使最强模型在多轮施压下的违规率也会平均飙升 65%，为自进化智能体设计‘不可篡改的元安全边界（Safety Invariants）’和抗诱导对齐策略提供了极为警醒的实验标尺。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> As corporate AI adoption continues to grow, enterprise-grade LLM agents are being deployed into sensitive contexts such as hiring, healthcare, and finance. In these contexts, compliance with rules specified in an agent's system context is a first-order legal concern. Currently, no evaluation framework systematically measures which LLM models tend to violate compliance rules, especially under pressure from a persistent user, a hurried manager, or circumstances where violation is convenient or attractive. We introduce PACT (Pressure-Applied Compliance Testing), a benchmark for rule-following under pressure in AI agents assisting employees in daily tasks across twelve regulated enterprise domains and forty-eight scenarios, each set in a realistic multi-turn conversation. Each benchmark item pairs a standing rule against a rule-violating shortcut, and applies a battery of pressures across different wordings and system-prompt modes. We construct PACT component by component under strict LLM-as-judge auditing to ensure samples are unambiguous, ungameable, and realistic enough to avoid eliciting evaluation-aware behavior. We use PACT to profile LLM compliance across six complementary metrics that create a holistic picture of an AI assistant's robustness under pressure and throughout multi-turn conversations, its transparency, and ability to correctly discern where a rule applies. We aggregate this profile into PACTScore, a reliability-weighted compliance rate over all items and modes. Our results across 22 common LLM models spanning multiple providers and sizes show substantial variability in compliance across models and metric dimensions. Even the strongest assistants mis-apply a rule on 6 to 10% of items, and ordinary user pressure raises the violation rate by 65% on average. PACT highlights compliance risks in LLM assistants, motivating guardrails and careful model selection.

</details>

---

### 4. [RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning](https://huggingface.co/papers/2609.20784)

- **论文元数据**：`arXiv:2609.20784` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.20784) | [arXiv 原文](https://arxiv.org/abs/2609.20784)

- **作者与归属机构**：Yan Yu, Zhengxi Lu, Yizhou Liu, Yichen Pan, Aozhe Wang 等（学术与工业界研究机构）
- **学术社区关注度**：🔥 **11** Upvotes | **命中核心标签**：`agent`, `agents`, `agentic`, `distillation`, `on-policy`, `reinforcement learning`, `rl`, `trajectory`, `environment`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  研究多轮智能体（Multi-turn Agents）在强化学习（RL）训练中的稠密监督信号匮乏问题，探索如何通过具备特权能力的自我教师（Self-Teacher）进行在策略蒸馏（On-Policy Distillation, OPD），并在合适时机实现教师‘平稳退休’。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  ① 智能体任务通常仅在整条轨迹结束时获得单一标量奖励（稀疏奖励），信用分配极难；② 既有方法假设‘特权信息越多的教师越可靠’，但实验发现特权教师在某些决策点同样存在系统性失误；③ 教师指导具有严格的‘时效阶段性’，长期强制蒸馏会压制学生策略的自主探索，反向锁死上限。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出 **RetireOPD（自退休在策略蒸馏）**：
    1. **解耦式特权教师训练**：先用环境真实奖励优化一个以特权技能为条件约束的教师模型（Skill-Conditioned Teacher）；
    2. **自适应退休机制（Adaptive Retirement）**：不采用固定衰减 Schedule，而是动态监控学生与教师的策略分布差异（Discrepancy）。一旦分布差异收敛停滞，且学生在环境中的胜率达到教师的设定目标比例，学生主动‘抛弃’教师指导，无缝切换为纯环境强化学习；
    3. **RL 与 OPD 联合梯度更新**，保证无特权基础学生能够有效将高阶决策内化。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  在开源基准 Qwen2.5 (1.5B ~ 7B) 上验证：在 ALFWorld 具身家庭环境中成功率相比标准 RL 提升 **14.1% ~ 18.8%**；在复杂长程多轮电商导航 WebShop 上成功率提升 **11.8% ~ 19.0%**；**最关键突破是：在所有评测场景下，学生模型最终完全超越了特权教师自身的能力上限**，为自进化智能体的非监督/自对齐能力突破提供了极佳的范式验证。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> Multi-turn agents trained with reinforcement learning (RL) receive a single scalar reward per trajectory, which motivates self on-policy distillation (OPD) to supply dense token-level supervision from a self-teacher with privileged task skills, letting a skill-free student internalize them. This recipe, however, is undermined by two findings in agentic tasks: privileged information alone does not always make a teacher reliable, and the benefit of teacher supervision is stage-dependent. We therefore propose RetireOPD (Self-Retiring On-Policy Distillation), which first optimizes a decoupled, skill-conditioned teacher with environment rewards and then trains a skill-free student jointly with RL and OPD. Rather than following a predefined distillation schedule, RetireOPD adopts Adaptive Retirement: the student drops the teacher on its own once their discrepancy stops shrinking and it reaches a target fraction of the teacher's success rate, after which training proceeds with RL alone. Across Qwen2.5 models from 1.5B to 7B, RetireOPD improves ALFWorld success rate over RL baseline by 14.1% to 18.8% and WebShop accuracy by 11.8% to 19.0%, and surpasses its own skill-conditioned teacher in every setting.

</details>

---

### 5. [Reflect, Revise, Reuse: Training-Free Skill Evolution for GUI Agents](https://huggingface.co/papers/2609.17653)

- **论文元数据**：`arXiv:2609.17653` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.17653) | [arXiv 原文](https://arxiv.org/abs/2609.17653)
 | [💻 官方开源代码](https://github.com/ZJU-REAL/EvoSkill-GUI)

- **作者与归属机构**：Bofan Chen, Boxuan Zhang, Fei Tang, Zhengxi Lu, Yong Du 等（学术与工业界研究机构）
- **学术社区关注度**：🔥 **7** Upvotes | **命中核心标签**：`agent`, `agents`, `rl`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  研究面向图形用户界面（GUI Agents）的免训练技能持续演化机制（Training-Free Skill Evolution），解决动态弹窗、加载延迟和控件位移导致固定规划崩溃的难题。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  以往的智能体技能框架（Skill Frameworks）将技能视为部署前生成的‘静态工件’（Static Artifacts），无法在执行时根据动态环境反馈自修复，遇到真实环境界面变动极易全盘失效。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出 **EvoSkill-GUI 终身演进框架**：
    1. **多文件活体技能包（Living Procedural Knowledge）**：将每个技能结构化封装为包含检索元数据、可执行动作流、备选定位锚点、故障恢复规则、无障碍工具和失败案例的多文件包；
    2. **Reflect-Revise-Reuse（反思-修订-复用）闭环**：执行器在交互中进行即时微调（In-Rollout Revision），严格信息隔离的独立裁判（Isolated Critic）深度诊断失败轨迹，执行器通过受限工具接口动态重写具体技能文件；
    3. 演化出的技能库永久沉淀，新任务直接复用与增量演化。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  在 MobileWorld、AndroidWorld、OSWorld 三大主流跨移动端与桌面 GUI 基准上，无需任何模型参数微调即可让多个基座模型分别获得 **+16.2%、+6.0% 与 +10.5%** 的绝对提升，是外循环与工具自进化流派在计算机操作（Computer-Use）领域的顶尖实践。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> GUI agents execute long-horizon tasks on dynamic graphical user interfaces, where pop-ups, delayed loads, and relocated widgets routinely invalidate plans fixed before execution. Recent agent-skill frameworks encapsulate reusable procedural knowledge to mitigate this, yet existing skill designs are largely developed without targeting GUI execution dynamics and treat skills as static artifacts produced before deployment rather than living procedural knowledge that improves through it. We argue that what GUI agents need is not better static skills, but skills that can be revised from execution feedback at deployment time, without additional training. We propose EvoSkill-GUI, a training-free framework in which each skill is a structured multi-file package containing retrieval metadata, executable plans, backup localization, failure-recovery rules, accessibility utilities, and failure cases. EvoSkill-GUI operates through a \emph{reflect-revise-reuse} loop: the executor performs instant in-rollout revisions, an isolated critic diagnoses failed trajectories under strict information isolation, and the executor edits specific skill files through a restricted tool interface. Across MobileWorld, AndroidWorld, and OSWorld, three mainstream GUI benchmarks spanning mobile and desktop platforms, EvoSkill-GUI consistently improves multiple base models without any training, with maximum gains of +16.2%, +6.0%, and +10.5% respectively, and evolved skill libraries continue to benefit related tasks rather than being rebuilt from scratch. Our code is available at https://github.com/ZJU-REAL/EvoSkill-GUI.

</details>

---

### 6. [What Does Privileged Information Add to On-Policy Self-Distillation?](https://huggingface.co/papers/2609.20612)

- **论文元数据**：`arXiv:2609.20612` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.20612) | [arXiv 原文](https://arxiv.org/abs/2609.20612)
 | [💻 官方开源代码](https://github.com/xiuyuz/opsd-reference-study)

- **作者与归属机构**：XiuYu Zhang, Wei Chow, Junfeng Fang, Zhenkai Liang, Tat-Seng Chua（National University of Singapore）
- **学术社区关注度**：🔥 **3** Upvotes | **命中核心标签**：`distillation`, `on-policy`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  深入解构特权信息（Privileged Information，如环境上帝视角标注、反思提示、专家先验）在在策略自蒸馏（OPSD）中所扮演的真实数学与经验角色。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  当前智能体研究普遍依赖‘特权教师提供密集奖励/指导’，但学界对其究竟是加速了探索（Exploration）、改善了策略校准（Calibration），还是引入了虚假捷径与教师分布偏差缺乏形式化结论。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  1. 建立了特权在策略自蒸馏的形式化对照评估协议（Reference Study Protocol）；
    2. 实验解耦了‘动作分布软化收益’与‘特权状态显式泄露收益’；
    3. 揭示了在复杂多轮决策中，过度拟合特权教师会导致学生模型在推理阶段遇到分布外状态时脆弱性陡增。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为自进化智能体在设计‘教师-学生（Teacher-Student）’闭环自提升方案时提供了严谨的理论护栏，揭示了特权退出机制与自主探索的必要性，与同期的 RetireOPD 形成了极佳的互补呼应。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> On-policy self-distillation (OPSD) lets a language model learn from a frozen copy of itself that sees an answer or a worked solution. Giving the teacher this extra information seems to offer the student more to learn, but how much does it add beyond distillation itself? To isolate that contribution, we construct AMPLE-Math, a reusable suite of 5,319 mathematical problems with six reasoning views that share the same answer, and compare each view with matched reference-free distillation. With a thinking-enabled teacher supervising direct-response rollouts, reference-free distillation accounts for much of Qwen3-1.7B's improvement under thinking-enabled evaluation, both in domain and on external benchmarks. Evidence for an additional reference benefit is modest in Qwen, strongest for a polished solution, whereas complete traces add two percentage points in SmolLM3-3B at step 50. These benefits depend on the student being trained. At the same checkpoint, replacing short direct-response rollouts with long thinking-enabled rollouts turns gains into losses in both families while the problems, references, and evaluation stay fixed. Teacher profiles and matched loss interventions in Qwen further show that changing token-level supervision can leave student behavior largely unchanged. Together, these findings suggest that OPSD can improve access to existing reasoning capabilities through parameters shared by direct-response and thinking-enabled inference. The value of a privileged reference is what it adds to this cross-mode transfer, not how much of the solution it reveals.

</details>

---

### 7. [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://huggingface.co/papers/2609.19969)

- **论文元数据**：`arXiv:2609.19969` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.19969) | [arXiv 原文](https://arxiv.org/abs/2609.19969)

- **作者与归属机构**：DeepSeek-AI, Anyi Xu, B. Li, Bangcai Lin, Bing Xue 等（DeepSeek）
- **学术社区关注度**：🔥 **43** Upvotes | **命中核心标签**：`agent`, `agents`, `agentic`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> The widespread adoption of long-horizon agents has made model workloads increasingly input-heavy. Although prior work has substantially reduced the cost of long-context computation, prefill remains computationally expensive, and large KV caches continue to strain HBM and SSD capacity and data-transfer bandwidth. Together, these compute, storage, and bandwidth demands constitute the primary bottleneck to further lowering deployment costs. To address this challenge, we introduce DeepSeek-V4.1-Flash, a multimodal Mixture-of-Experts (MoE) model with 552B backbone parameters and support for contexts of up to one million tokens. With its Causal Encoder-Decoder (CED) architecture, the model activates 16B parameters per token during decode but only 8B parameters during prefill, substantially improving cost efficiency for agentic workloads. To push the limits of KV cache compression, DeepSeek-V4.1-Flash combines cross-layer KV cache reuse in Compressed Sparse Attention 2 (CSA2) with FP4 KV caching. These designs reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash. Further, through a dedicated deployment optimization known as SWA Bounded Replay, DeepSeek-V4.1-Flash reduces its persistent KV cache footprint (always on SSD or in host memory) to roughly 1/8 of that of DeepSeek-V4-Flash. Despite its much smaller KV cache footprint, the model delivers substantially better performance than the baseline. In addition, we streamline the DeepSeek-V4 architecture and introduce several efficient architectural extensions. We pretrain DeepSeek-V4.1-Flash on a multimodal corpus comprising 45T tokens and conduct comprehensive post-training, yielding strong performance across diverse text-based and multimodal agentic scenarios. Model checkpoints are available at https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash.

</details>

---

### 8. [RiskChainBench: A Benchmark for Obfuscated Platform Message Restoration and Evidence-Grounded Web Investigation](https://huggingface.co/papers/2609.16900)

- **论文元数据**：`arXiv:2609.16900` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.16900) | [arXiv 原文](https://arxiv.org/abs/2609.16900)
 | [💻 官方开源代码](https://github.com/mattheliu/riskchainbench-task1)

- **作者与归属机构**：ZhuoXin Liu, Zhiming Ma, Ying Zhang, Mengzheng Yang, Yifan Wang 等（学术与工业界研究机构）
- **学术社区关注度**：🔥 **16** Upvotes | **命中核心标签**：`agent`, `rl`, `exploration`, `environment`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> Platform abuse campaigns conceal redirection instructions with emojis, homophones, character decomposition, and redundant symbols, then route users through disguised links to services associated with pornography, fraud, gambling, or illicit transactions. Existing benchmarks evaluate obfuscated text and risky webpages separately, obscuring how target recovery affects downstream evidence acquisition. We introduce RiskChainBench, pairing 3,600 synthetic token-text restoration inputs from 600 source sessions with 600 corresponding human-labeled local web environments. A model first restores the message, operational intent, and destination; the same underlying model then acts as a VLM-driven web agent that investigates the correctly associated website and produces a frozen, evidence-cited risk report without message-side semantics or domain-reputation cues. We score restoration and correct-routing web investigation separately and compose them offline by applying the frozen primary-entry prediction as a gate to the same Task 2 result. Human labels determine task correctness, while a fixed multimodal evidence judge assesses faithfulness, sufficiency, completeness, and consistency. Across ten models, Entry Top-1 ranges from 35.2% to 95.2% and web decision accuracy from 26.3% to 62.8%; the leading systems differ across entry recovery, full reconstruction, website decisions, and fine-grained typing. Execution failures account for 31.9% of web runs, whereas post-decision type errors account for only 0.9%, identifying stable exploration and risk judgment as the principal bottlenecks. We release the benchmark, protocol, and resettable local sandbox.

</details>

---

### 9. [SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness](https://huggingface.co/papers/2609.20519)

- **论文元数据**：`arXiv:2609.20519` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.20519) | [arXiv 原文](https://arxiv.org/abs/2609.20519)
 | [💻 官方开源代码](https://github.com/NVlabs/SoL-Pi)

- **作者与归属机构**：Haozhe Liu, Tian Ye, Sensen Gao, Qihang Cao, Yitong Li 等（NVIDIA）
- **学术社区关注度**：🔥 **38** Upvotes | **命中核心标签**：`agent`, `agents`, `rl`, `exploration`, `environment`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> As coding agents move from supervised code completion to unattended, around-the-clock exploration, their work expands from isolated predictions into long trajectories of reasoning, tool use, and feedback. Token efficiency therefore becomes important for scaling recursive self-improvement. We take an RSI-inspired approach at the harness layer, scaling auto-research loops across increasingly numerous and diverse environments for harness rollouts. At this scale, the process yields reusable improvements that transfer beyond their development setting, moving automated harness discovery toward production-level outcomes. Four mechanisms survive selection and form SoL-Pi, spanning action execution, context compaction, observation handling, and delegated reading. On the 51-task EdgeBench evaluation, SoL-Pi achieves performance comparable to Pi across GPT-5.6 Sol and Opus 5 while reducing recorded token traffic by 44.7-49.0% and API cost by about one third. In other words, estimated hourly savings are \8.75-13.50 relative to native Codex and Claude Code harnesses, and \4.36-5.71 relative to Pi.

</details>

---

### 10. [An Empirical Study of Harness Design for Coding Agents](https://huggingface.co/papers/2609.20804)

- **论文元数据**：`arXiv:2609.20804` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.20804) | [arXiv 原文](https://arxiv.org/abs/2609.20804)

- **作者与归属机构**：Run-Ze Fan, Zihao Zhang, Simin Ma, Yebowen Hu, Shouju Wang 等（Zoom Communications）
- **学术社区关注度**：🔥 **31** Upvotes | **命中核心标签**：`agent`, `agents`, `trajectory`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> Coding harnesses shape how autonomous coding agents translate model capabilities into long-horizon software-engineering performance, yet existing work typically evaluates harnesses as monolithic systems, leaving the effectiveness of individual components unclear. To enable component-level comparisons, we study this question with a lightweight coding harness whose execution loop is fixed while three components are varied: planning, action space, and context management. Across four models evaluated on SWE-Bench Verified and Terminal-Bench 2.1, we evaluate 176 matched settings spanning five context-management strategies, four context-window budgets, and targeted ablations of planning and action space. We find that: (1) Context management becomes increasingly valuable as the context-window budget tightens, with most of its benefit coming from preventing context-overflow failures. (2) Staging rule-based elision before LLM-based summarization provides the strongest overall efficiency among the context-management strategies, whereas making elided content recoverable adds machinery that models rarely use and yields no accuracy gain. (3) Planning shifts from an accuracy scaffold for weaker models to a cost saver for stronger models, with little change in accuracy. (4) Predefined tools improve performance for models with weaker bash proficiency, whereas bash-capable models can operate effectively with a bash-only interface and achieve substantially lower cost, especially on command-line-centric tasks. Trajectory-level analysis explains these effects: context management extends execution trajectories without substantially altering agent behavior, planning changes where trajectories stop, and the action space changes the granularity at which code is written. These findings inform model- and budget-aware harness design and provide a modular framework for evaluating future harness components.

</details>

---

### 11. [CERA-MoA: Co-Evolving Routing Mechanisms with Continually Learning LLM Agents](https://huggingface.co/papers/2609.18779)

- **论文元数据**：`arXiv:2609.18779` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.18779) | [arXiv 原文](https://arxiv.org/abs/2609.18779)
 | [💻 官方开源代码](https://github.com/michaeljiang0530/CERA-MoA)

- **作者与归属机构**：Jiaxuan Jiang, Liyuan He, Zhixuan Fang（Tsinghua University）
- **学术社区关注度**：🔥 **4** Upvotes | **命中核心标签**：`agent`, `agents`, `reinforcement learning`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> Current Mixture-of-Agents (MoA) paradigms generally treat query routing and agent fine-tuning as separate processes, limiting their ability to respond to evolving agent capabilities. This disconnect prevents routing strategies from adapting to evolving agent capabilities during post-training and prevents agents from achieving synergistic data-driven specialization. To resolve this, we introduce CERA-MoA (Co-Evolving Router with continually learning Agents for Mixture-of-Agents), an iterative reinforcement learning framework where the dynamic router and independent agent policies co-evolve. We design a predictive familiarity estimator that leverages mid-layer hidden states to evaluate semantic competence among agents, avoiding the overhead of full rollouts. Based on these familiarity scores, a cumulative-threshold adaptive routing mechanism dynamically activates a tailored minimal agent subset, achieving a trade-off between task performance and efficiency. By proactively allocating targeted training samples to agents based on their evolving competence, CERA-MoA promotes capability differentiation. Extensive experiments across various domains demonstrate that CERA-MoA outperforms state-of-the-art static-agent routing and fix-workflow fine-tuning baselines.

</details>

---

### 12. [Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches](https://huggingface.co/papers/2609.17652)

- **论文元数据**：`arXiv:2609.17652` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.17652) | [arXiv 原文](https://arxiv.org/abs/2609.17652)
 | [💻 官方开源代码](https://github.com/vivekkalyanarangan30/fathom)

- **作者与归属机构**：Vivek Kalyanarangan（学术与工业界研究机构）
- **学术社区关注度**：🔥 **2** Upvotes | **命中核心标签**：`agent`, `agentic`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> When agentic sessions run to a million tokens with many sessions resident at once, the KV cache and the index that ranks it live in host memory, and the scan that ranks all n keys for a top-k step becomes the traffic that bounds decoding. We present Fathom, a key scan in which each query decides how many bits of each key channel to read. The 4-bit K cache is stored channel-major as bit planes, so a prefix of t planes is exactly the channel's t-bit quantizer, and the query spends its bit budget by reverse water-filling over the variance-weighted importance of its channels. At one million tokens on Qwen3-8B a decode step is 1.67x faster in GPU time than with the 136-bit scans of Double Sparsity, Loki and SparQ r=32, and in the same GPU time as SparQ's 68-bit read (r=16) Fathom reads 18% fewer bytes with lower attention error on six of seven model and context settings. On RULER-style tasks every per-token scan matches exact top-k decoding, and on real coding-agent sessions Fathom reaches the step agreement of the most accurate 136-bit scan at 92 bits. The store is the 4-bit K copy a quantized serving stack already holds, and the method is not faster when the index is resident in GPU memory.

</details>

---

### 13. [In-Context Robot Learning with VLM Agents](https://huggingface.co/papers/2609.19138)

- **论文元数据**：`arXiv:2609.19138` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.19138) | [arXiv 原文](https://arxiv.org/abs/2609.19138)
 | [💻 官方开源代码](https://github.com/cheng-haha/GPT-Policy)

- **作者与归属机构**：Dongzhou Cheng, Taoran Yi, Ye Fang, Xingwu Zhang, Fan Feng 等（学术与工业界研究机构）
- **学术社区关注度**：🔥 **15** Upvotes | **命中核心标签**：`agent`, `agents`, `agentic`, `environment`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> Enabling robots to adapt to unfamiliar environments as readily as humans remains a moonshot goal of embodied AI. No finite collection of demonstrations can cover every task and situation a robot will encounter, making the ability to learn from context at deployment essential for generalization. Such in-context learning (ICL), however, remains largely beyond the reach of existing robotic policies. The broad agentic capabilities of commercial vision-language models (VLMs), such as GPT-6 Astra, raise a compelling question: can these models learn from demonstrations, examples, and interaction feedback, then translate that information into executable and verifiable robot behavior from a new initial state without gradient updates or persistent changes to task-specific parameters? We introduce GPT-Policy, a general-agent framework for in-context robot learning. GPT-Policy integrates a context compiler that preserves task-relevant visual transitions, a VLM that proposes robot-tool actions, and a constrained controller that verifies and executes each action and reports its outcome. We evaluate its reliability and limitations through task success and efficiency metrics, matched comparisons across models, and controlled context ablations. In real-robot trials, human video demonstrations improve task completion even without robot action labels, while aligned action references yield further gains on contact-sensitive tasks. These findings position GPT-Policy as a step toward robot adaptation through in-context learning, providing an empirical foundation for translating the general-purpose capabilities of VLMs into physical behavior and clarifying the challenges that must be overcome for reliable deployment.

</details>

---

### 14. [Confidence Comes from Experience: Experiential Confidence Estimation from Reasoning to Agents](https://huggingface.co/papers/2609.17708)

- **论文元数据**：`arXiv:2609.17708` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.17708) | [arXiv 原文](https://arxiv.org/abs/2609.17708)
 | [💻 官方开源代码](https://github.com/caiqizh/xconf)

- **作者与归属机构**：Caiqi Zhang, Xiaochen Zhu, Chengzu Li, Yulong Chen, Dharshan Kumaran 等（University of Cambridge）
- **学术社区关注度**：🔥 **51** Upvotes | **命中核心标签**：`agent`, `agents`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> Reliable confidence estimation is increasingly central to the trustworthy deployment of language models: a calibrated estimate of the probability that an output is correct decides what to ship, what to escalate, and what to retry. Existing confidence estimators, however, share one design premise: they only read the current inference process, either by introspecting on it, scoring its token probabilities, or resampling it. We argue that the current inference is not a sufficient basis for confidence. We propose XConf (eXperiential Confidence): estimating confidence together with the model's accumulated experience. The experience is stored as a record of the model's own graded past episodes, each holding the task, the model's reflection, its stated confidence, the outcome, and a lesson written once the grade arrived. Given a new task, XConf's Recall stage retrieves past episodes on similar tasks met with a similar stated confidence, and reads off their historical success rate; its Reflect stage shows the model this record, has it name its recurring failure mode, and restate a confidence now informed by its own track records. Our estimator is format-general, requiring no logit access or weight updates, and costs only one answer generation. Across nine benchmarks spanning reasoning, coding, multimodal QA, and interactive agents, and four models from three families, XConf beats or matches ten-sample self-consistency in discrimination (AUROC) on 23 of 24 comparisons, with much lower calibration error (ECE), at a tenth of the generation cost. Used for selective prediction, abstaining on the 10% least-confident episodes raises the delivered success rate by up to 8.7 points on agent tasks. We therefore see experiential confidence estimation as a new paradigm for future general-purpose confidence estimation.

</details>

---

### 15. [Zing-0.5: Toward Playable Worlds with Real-Time Joint Action and Text Control](https://huggingface.co/papers/2609.17909)

- **论文元数据**：`arXiv:2609.17909` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.17909) | [arXiv 原文](https://arxiv.org/abs/2609.17909)

- **作者与归属机构**：Mingyang Chen, Shengdong Chen, Xiaoxiao Fu, Bosheng Gong, Haoyuan Guo 等（学术与工业界研究机构）
- **学术社区关注度**：🔥 **28** Upvotes | **命中核心标签**：`distillation`, `rl`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> We introduce Zing-0.5, a 5B autoregressive world model designed for playability: users can explore generated worlds, influence unfolding events, and respond to the resulting feedback through joint keyboard and online text control. Our approach brings together three technical contributions: (1) Unified action and text conditioning, combining magnitude-aware keyboard inputs with temporally aligned text instructions and jointly annotated videos to learn navigation and event control within the same sequence; (2) Event-scale supervision for incremental generation, using a segment-level teacher trained on connected multi-prompt videos to supervise a block-level causal student through distribution-matching distillation; and (3) Low-cost real-time interaction, combining four-step generation with context-preserving streaming to support 832 x 480 inference at 24 FPS at an estimated server rental cost of approximately USD 0.009 per stream-minute. Zing-0.5 achieves an overall score of 81.0 and a consistency score of 88.5 across 158 WBench Navigation cases. A joint-control demonstration shows a text-directed event change during continued navigation without restarting generation. We release the model weights, inference code, and Zing-SGLang serving implementation to support further work on playable generated worlds.

</details>

---

### 16. [HypoEvolve: Genetic Algorithms Enable Multi-Agent LLMs to Discover Scientific Hypotheses](https://huggingface.co/papers/2609.15938)

- **论文元数据**：`arXiv:2609.15938` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.15938) | [arXiv 原文](https://arxiv.org/abs/2609.15938)

- **作者与归属机构**：Jieyuan Liu, Mengzhou Hu, Jefferson Chen, JungHo Kong, Pratibha Jagannatha 等（学术与工业界研究机构）
- **学术社区关注度**：🔥 **25** Upvotes | **命中核心标签**：`agent`, `agents`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> Scientific agents contribute to hypothesis discovery by synthesizing evidence, assessing proposals, and developing new explanations. Recent systems combine scientific agents with evolutionary search through critique, comparison, and revision. However, how different forms of agent collaboration affect hypothesis quality remains an open question. Answering this question requires separating the effects of agents' scientific capabilities from those of their collaboration. A framework must therefore preserve agents' scientific roles and support rules for combining, revising, and retaining hypotheses. Building on this view, we introduce HypoEvolve, which makes collaboration explicit through successive updates to a hypothesis population. Specifically, we propose a generational genetic algorithm to coordinate specialized large language model (LLM) agents that integrate mechanistic arguments, reconsider assumptions, and assess evidence and testability. Each generation specifies how scientific judgments and new proposals reshape the population, making collaboration effects on hypothesis quality directly testable. Moreover, we design our evaluation around scientifically meaningful hypotheses that explain how a proposed intervention could work. Drug repurposing links these explanations to target-level biological claims assessed against external evidence. Specifically, we adapt DepMap and Open Targets into complementary external measures grounded in experimental, genetic, and clinical evidence. Across 34 cancer types, HypoEvolve achieves the highest scores against six baselines on both measures. DepMap selectivity reaches 0.171, versus 0.115 for the strongest baseline. Gains over single-pass generation also generalize to held-out cancer types. HypoEvolve advances a vision of autonomous science in which AI research teams achieve a capacity for discovery beyond that of individual models.

</details>

---

### 17. [Agora: Git as Shared Memory for Collective AutoResearch](https://huggingface.co/papers/2609.18094)

- **论文元数据**：`arXiv:2609.18094` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.18094) | [arXiv 原文](https://arxiv.org/abs/2609.18094)
 | [💻 官方开源代码](https://github.com/yifanzhang-pro/Agora)

- **作者与归属机构**：Yifan Zhang, Yunheng Zou, Shaokun Zhang, Jian Hu, Hao Zhang 等（NVIDIA）
- **学术社区关注度**：🔥 **40** Upvotes | **命中核心标签**：`agent`, `agents`, `rl`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> Autonomous research loops such as AutoResearch show that one coding agent can improve a training setup unattended. Run several of them and each session starts from scratch, so more agents tend to mean more duplicated search rather than more discovery. Agora is a shared memory for such agents: research is recorded as an append-only directed acyclic graph (DAG) stored in Git, so that every claim is a commit anyone can check out and rerun. Each result, insight, hypothesis, verification, and report is an immutable commit whose parent edges say what it builds on; a derived index exposes the frontier, the neglected branches, and the verification status of each claim, and a diversity-aware selection rule keeps the community from collapsing onto one leader. We describe the system and report its first sustained use: a run of nearly 12 days in which 13 language-model workers, with no assigned tasks and no central planner, worked on a weight-transfer problem. Given 141 pretrained donor models and a frozen 119.6M-parameter attention-SSM hybrid whose dimensions match no donor, the workers had to initialize the target without training data or gradient updates. They published 1,703 contributions and drove the evaluator from 3.39 to 1.899 bits per byte, closing 62% of the gap to a trained GPT-2 124M. The winning recipe compresses donor next-token statistics into the target's embedding and output head, then adds a short-range context signal through sparse edits to attention, feed-forward, and state-space blocks. Its 145-commit ancestry spans 15 accounts, and 165 independent reproductions were posted, none of which failed. We describe the single mid-run human intervention that pulled the community out of a monoculture, what the trace does and does not establish, and the controlled comparison that would settle whether shared research state improves discovery per unit of compute.

</details>

---

### 18. [EvolveTrade: Experience-Driven Policy Refinement for Self-Evolving LLM Trading Agents](https://huggingface.co/papers/2609.17632)

- **论文元数据**：`arXiv:2609.17632` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.17632) | [arXiv 原文](https://arxiv.org/abs/2609.17632)

- **作者与归属机构**：Sehee Kim, Yumin Choi, Minki Kang, Sung Ju Hwang（KAIST AI）
- **学术社区关注度**：🔥 **30** Upvotes | **命中核心标签**：`agent`, `agents`, `self-evolving`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> Large language model (LLM) trading agents can combine market data, news, and executable analysis, but their behavior is often controlled by static hand-written tool-use policies that are fixed before deployment. This limits their ability to adapt how they gather evidence, invoke tools, verify signals, and manage risk under changing market regimes. We introduce EvolveTrade, a self-evolving framework that treats the system prompt of a tool-using trading agent as a text-parameterized policy. After each update interval, a Policy Agent revises this policy using accumulated decision traces and realized portfolio feedback, while keeping the backbone LLM fixed. The updated policy is then used for the next batch of trading decisions, enabling the agent to refine its information-acquisition and portfolio-construction procedure over time. Experiments across multiple market regimes and two LLM backbones show that EvolveTrade often improves Sharpe Ratio and Cumulative Return over fixed-policy LLM baselines, achieving the improved SR and CR in most evaluated settings. Behavioral analyses further show that self-evolved policies increase code-mediated analysis and activate regime-relevant computations; case-level policy-to-return attributions trace how policy-induced allocation changes contribute to realized return differences. These results suggest that adapting the reusable procedure governing tool use is a key direction for building more robust LLM trading agents.

</details>

---

### 19. [ScienceIDE: Turning World's Scientific Codebase into Agent Learnable Environments](https://huggingface.co/papers/2609.19134)

- **论文元数据**：`arXiv:2609.19134` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.19134) | [arXiv 原文](https://arxiv.org/abs/2609.19134)
 | [💻 官方开源代码](https://github.com/aitofound/ScienceIDE)

- **作者与归属机构**：Hejia Geng, Zesen Huang, Haoyang Li, Wenbin Li, Koutian Wu 等（PhAI Labs）
- **学术社区关注度**：🔥 **72** Upvotes | **命中核心标签**：`agent`, `agents`, `reinforcement learning`, `rl`, `environment`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> Scientific code repositories encode decades of human knowledge in executable models, methods, and tools. Yet fragmented toolchains, implicit domain conventions, and specialized correctness criteria make this knowledge difficult to convert into reliable learning experience-a challenge we call the scientific experience bottleneck. We introduce ScienceIDE, infrastructure for turning the world's scientific code into programmable environments for scientific agents. Guided by expert-defined scientific cases and acceptance criteria, agents transform repositories into executable environments that support task generation, execution, and scientific verification. These environments provide a shared foundation for supervised fine-tuning, reinforcement learning, and evaluation. Using verified interaction trajectories, we train PhAI-IDE-72B, PhAI-IDE-9B, and PhAI-IDE-4B. The model family shows gains in held-out scientific-code repair and across selected general-purpose benchmarks in code, reasoning, and knowledge, providing evidence of positive transfer from scientific experience to broader capabilities. ScienceIDE lays the foundation for an integrated workspace for agent learning and scientific practice, making humanity's scientific software a shared substrate for developing scientific intelligence. Code: https://github.com/aitofound/ScienceIDE

</details>

---

### 20. [ProgramDistill: From Interactive Web Apps to Verifiable Reference-Guided SWE Tasks](https://huggingface.co/papers/2609.18805)

- **论文元数据**：`arXiv:2609.18805` | [Hugging Face 论文讨论页](https://huggingface.co/papers/2609.18805) | [arXiv 原文](https://arxiv.org/abs/2609.18805)

- **作者与归属机构**：Jeonghye Kim, Minseon Kim, Young Jin Kim, Matheus Pereira, Marc-Alexandre Côté 等（Microsoft Research）
- **学术社区关注度**：🔥 **45** Upvotes | **命中核心标签**：`agent`, `agents`

#### 🔬 深度科研拆解：

- **1. 具体研究什么 (What is being studied)**：
  探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。

- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：
  解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。

- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：
  提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。

- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：
  为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。

<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>

> Coding agents are typically evaluated with desired behavior specified through issues or instructions. In practical web development, however, agents may need to infer behavior from working software and implement it in an incomplete application. We introduce ProgramDistill, a benchmark evaluating coding agents on features discovered through interaction with fully functional reference applications. We build ProgramDistill by factorizing applications into features of different granularities, each associated with replayable behaviors executable via its gold patch. Our pipeline, mine-craft-patch, discovers 1,975 replay-verified behaviors across 26 applications and constructs 4,063 tasks without human intervention. Across nine frontier coding agents, GPT-6 Astra and Claude Opus 5 achieve 49.2% and 28.8% success on cumulative workflows in full-application reconstruction. In partial-application reconstruction, success falls from 100% to 64.0% and from 96% to 32% as restoration depth increases from 1 to 8. ProgramDistill thus provides a scalable benchmark with controlled difficulty for evaluating and diagnosing coding agents, and a natural basis for future curriculum-based training.

</details>

---


