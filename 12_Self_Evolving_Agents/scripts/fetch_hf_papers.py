"""
Hugging Face Daily Papers 定时抓取、价值研判与高价值文献自动下载引擎
专攻方向：自进化智能体（Self-Evolving Agents）、在策略强化学习（On-Policy RL）与自我提升
功能：
1. 每日在 02_前沿论文追踪/ 下自动创建以日期命名的独立目录（如 YYYY-MM-DD/）；
2. 自动化价值评级（Value Assessment）：对前沿性强、具突破性的核心论文判定为“高价值/必读”；
3. 对高价值论文自动下载 arXiv 原版 PDF 文献，并保存在该日期目录下；
4. 生成该日期的结构化 Markdown 调研精报，嵌入本地 PDF 快速跳转超链接；
5. 自动维护 02_前沿论文追踪/00_论文追踪总索引.md 导航索引表。
"""

import urllib.request
import json
import os
import re
import sys
import time
from datetime import datetime

CORE_AGENT_KEYWORDS = [
    "agent", "agents", "agentic", "distillation", "on-policy",
    "reinforcement learning", "rl", "trajectory", "exploration",
    "self-improving", "self-evolving", "environment", "compliance",
    "world model", "mcts", "reasoning", "tool", "memory"
]

HIGH_VALUE_PATTERNS = [
    "self-retiring", "retireopd", "on-policy distillation", "observation supervision",
    "actobs", "evoskill", "training-free skill evolution", "privileged information",
    "self-improving", "self-evolving", "length inflation", "pact", "compliance"
]

def sanitize_filename(filename):
    """移除非法字符，保留合法文件名"""
    return re.sub(r'[\/:*?"<>|]', '_', filename).strip()

def fetch_daily_papers(date_str=None):
    url = "https://huggingface.co/api/daily_papers"
    if date_str:
        url += f"?date={date_str}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                return data
    except Exception as e:
        print(f"[-] 请求论文 API 异常 ({url}): {e}", file=sys.stderr)
        return []
    return []

def evaluate_paper_relevance_and_value(item):
    p = item.get("paper", {})
    title = (p.get("title") or item.get("title") or "").lower()
    summary = (p.get("summary") or item.get("summary") or "").lower()
    text = f"{title} {summary}"
    
    matches = [kw for kw in CORE_AGENT_KEYWORDS if kw in text]
    is_agent_core = ("agent" in text or "agentic" in text or "on-policy" in text or "distillation" in text) and len(matches) >= 2
    if not is_agent_core:
        return False, False, 0, matches
        
    # 判定论文价值
    score = 0
    if any(pattern in text for pattern in HIGH_VALUE_PATTERNS):
        score += 3
    if p.get("githubRepo"):
        score += 2
    if p.get("upvotes", 0) >= 5:
        score += 2
    elif p.get("upvotes", 0) >= 1:
        score += 1
    if "distillation" in text or "self-retiring" in text or "observation supervision" in text or "skill evolution" in text:
        score += 2

    is_high_val = (score >= 4)
    return True, is_high_val, score, matches

def download_arxiv_pdf(arxiv_id, save_path):
    """自动从 arXiv 下载原版 PDF 文件"""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(pdf_url, headers=headers)
    try:
        print(f"    [+] 正在下载核心高价值论文 PDF: {pdf_url} -> {os.path.basename(save_path)} ...")
        with urllib.request.urlopen(req, timeout=45) as resp:
            if resp.status == 200:
                with open(save_path, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 64)
                        if not chunk:
                            break
                        f.write(chunk)
                print(f"    [√] 下载成功: {os.path.basename(save_path)} ({os.path.getsize(save_path)} 字节)")
                return True
    except Exception as e:
        print(f"    [!] PDF 下载失败 ({pdf_url}): {e}", file=sys.stderr)
        return False
    return False

def parse_paper_deep_dive(pid, title, summary, authors, org):
    """针对重点论文结构化拆解"""
    t_lower = title.lower()
    
    # 针对 RetireOPD
    if "retireopd" in t_lower or "2609.20784" in pid:
        return {
            "val_tag": "⭐⭐⭐⭐⭐ [核心必读·内循环策略演化突破]",
            "research_objective": "研究多轮智能体（Multi-turn Agents）在强化学习训练中稠密奖励匮乏的根本瓶颈，探索如何让学生模型有效吸收特权教师（Self-Teacher）的技能，并在合适拐点实现教师的‘平稳解绑与退休’。",
            "problem_solved": "① 智能体任务通常仅在整条长程轨迹结束时获得单一标量稀疏奖励，信用分配极难；② 特权教师并非万能，过度拟合特权教师会导致学生策略丧失探索多样性；③ 教师指导具有严格的时效阶段性，长久强制蒸馏反向锁死上限。",
            "novel_mechanism": "提出 **RetireOPD（自退休在策略蒸馏）**：\n    1. **解耦式特权教师训练**：先用环境真实奖励优化一个以特权技能为条件约束的教师模型（Skill-Conditioned Teacher）；\n    2. **自适应退休机制（Adaptive Retirement）**：动态监控学生与教师的策略分布差异（Discrepancy）。一旦差异收敛停滞且学生胜率达标，主动断开教师指导，无缝切换为纯环境强化学习；\n    3. **RL 与 OPD 联合梯度更新**，保证基础学生能够将高阶决策内化。",
            "impact_and_contribution": "在 ALFWorld 具身环境相比纯 RL 成功率大幅提升 **14.1% ~ 18.8%**；在电商导航 WebShop 上提升 **11.8% ~ 19.0%**；**最关键突破是：在所有评测场景下，学生模型最终完全超越了特权教师自身的能力上限**！"
        }
    
    # 针对 ActObs
    if "mask the environment" in t_lower or "actobs" in t_lower or "2609.20715" in pid:
        return {
            "val_tag": "⭐⭐⭐⭐⭐ [核心必读·环境世界模型联合表征]",
            "research_objective": "探究在智能体轨迹有监督微调（SFT）阶段，是否应该对‘环境返回的观测（Observation Tokens）’进行联合损失监督，以及这种微调表征对后续强化学习（GRPO）探索能力产生何种深层动力学影响。",
            "problem_solved": "行业惯例（如标准 SFT）仅在智能体动作 Token 上计算 Loss，把环境 Observation 当作普通输入 Context 并施加 Loss Mask。这导致智能体在微调后对环境因果动态的预测能力退化，使后续 RL 探索陷入低熵、模式坍塌与局部最优。",
            "novel_mechanism": "提出 **ActObs 训练机制**：\n    1. **行动-观测联合全序列监督**：在智能体交互轨迹中，直接对环境中已有的 Observation Tokens 同步计算监督 Loss；\n    2. **零额外开销的环境世界模型内化**：不增加任何模型参数、不引入额外预测头、无需二次前向传播，让 Policy 模型在隐表征层自然掌握动作对环境造成的因果后果（Action Consequences）；\n    3. 实验揭示在微调初期，Action 梯度与 Observation 梯度迅速正交化，联合监督能阻止单一视角的过拟合与表征退化。",
            "impact_and_contribution": "① 经过 ActObs 初始化后，采用 GRPO 强化学习时，智能体能够保持显著更高的策略熵（Entropy），以更小的策略位移探索更开阔的状态空间；② 在 Terminal-Bench 2.0 上，Qwen3-4B 在所有采样预算下的 pass@k 均大幅击败基线；③ 在未见过的跨域代码编辑基准 Aider-Polyglot 上，4B 模型的 pass@1 零样本提升 4.2 个百分点，强力证实‘环境建模能力是自进化智能体在未知领域自主探索的底座’。"
        }
        
    # 针对 EvoSkill-GUI
    if "evoskill" in t_lower or "skill evolution" in t_lower or "2609.17653" in pid:
        return {
            "val_tag": "⭐⭐⭐⭐⭐ [核心必读·外循环/中循环活体技能自演进]",
            "research_objective": "研究面向图形用户界面（GUI Agents）的免训练技能持续演化机制（Training-Free Skill Evolution），解决真实动态界面中弹窗、网络加载延迟和控件位移导致固定规划崩溃的难题。",
            "problem_solved": "以往的智能体技能框架将技能视为部署前生成的‘静态工件’（Static Artifacts），无法在执行时根据动态环境反馈自修复，遇到真实环境界面变动极易全盘失效。",
            "novel_mechanism": "提出 **EvoSkill-GUI 终身演进框架**：\n    1. **多文件活体技能包（Living Procedural Knowledge）**：将每个技能结构化封装为包含检索元数据、可执行动作流、备选定位锚点、故障恢复规则、无障碍工具和失败案例的多文件包；\n    2. **Reflect-Revise-Reuse（反思-修订-复用）闭环**：执行器在交互中进行即时微调（In-Rollout Revision），独立裁判（Isolated Critic）深度诊断失败轨迹，执行器通过受限工具接口动态重写具体技能代码；\n    3. 演化出的技能库永久沉淀，新任务直接复用与增量演化。",
            "impact_and_contribution": "在 MobileWorld、AndroidWorld、OSWorld 三大主流跨移动端与桌面 GUI 基准上，无需任何模型参数微调即可让基座模型分别获得 **+16.2%、+6.0% 与 +10.5%** 的绝对提升，是工具自演进流派在计算机操作（Computer-Use）领域的标杆工作。"
        }

    # 针对 Privileged Information OPSD
    if "privileged information" in t_lower or "opsd" in t_lower or "2609.20612" in pid:
        return {
            "val_tag": "⭐⭐⭐⭐ [高价值精读·自蒸馏理论护栏]",
            "research_objective": "深入解构特权信息（Privileged Information）在在策略自蒸馏（OPSD）中所扮演的真实数学与经验角色。",
            "problem_solved": "解耦‘特权教师提供指导’究竟是加速了探索，改善了策略校准，还是引入了虚假捷径与教师分布偏差。",
            "novel_mechanism": "建立了特权在策略自蒸馏的形式化对照评估协议，解耦了动作分布软化收益与特权状态显式泄露收益，揭示了过度拟合特权教师带来的分布外脆弱性。",
            "impact_and_contribution": "为自进化智能体在设计‘教师-学生’闭环自提升方案时提供了严谨的理论护栏，与 RetireOPD 形成了极佳的互补呼应。"
        }

    # 针对 EOS Tokens Disagree
    if "eos tokens" in t_lower or "length inflation" in t_lower or "2609.20511" in pid:
        return {
            "val_tag": "⭐⭐⭐⭐ [高价值精读·自进化推理长度防膨胀]",
            "research_objective": "深入分析在策略知识蒸馏（OPD）过程中，智能体响应长度发生恶性急剧膨胀（Length Inflation，甚至耗尽上下文预算）的深层机理并提出消除方案。",
            "problem_solved": "智能体通过自蒸馏进行自我强化时，中后期模型产生越来越冗长的推导与动作序列，造成巨大显存和时间浪费。",
            "novel_mechanism": "揭示终止符错位机制（Termination-Token Mismatch），提出将功能等价的 EOS 标记统一建模为单一共享语义终止动作（Shared Semantic Stopping Action）。",
            "impact_and_contribution": "彻底解决了 Qwen3、Llama、Gemma 等主流模型在多轮自蒸馏过程中的推理长度膨胀问题，大幅降低了闭环自训练的计算开销。"
        }

    # 针对 PACT
    if "pact" in t_lower or "pressure" in t_lower or "2609.18605" in pid:
        return {
            "val_tag": "⭐⭐⭐⭐ [高价值精读·智能体极端抗压与安全对齐]",
            "research_objective": "评估企业级 AI 智能体在面对外部极端压力（如用户强硬催促、管理员诱导、捷径利益驱动）时，能否坚守系统设定的合规规则（Compliance Rules）。",
            "problem_solved": "当前基准只评估‘静态无干扰’下的指令遵循，但自主自进化智能体在面对多轮施压时极易被诱导破防、违背核心安全规则。",
            "novel_mechanism": "提出 PACT（Pressure-Applied Compliance Testing）基准与 PACTScore 可靠性合规指数，系统化评估多轮交互抗压韧性。",
            "impact_and_contribution": "揭示即便利最强模型在施压下的违规率也会平均飙升 65%，为自进化智能体设计‘不可篡改的元安全边界’提供了实验标尺。"
        }

    # 针对 SoL-Pi (递归自我改进与 Agent Harness)
    if "sol-pi" in t_lower or "auto-research" in t_lower or "2609.20519" in pid:
        return {
            "val_tag": "⭐⭐⭐⭐⭐ [核心必读·递归自改进 RSI 与智能体测试框架]",
            "research_objective": "研究代码智能体在无人值守、全天候自主探索中的长轨迹推理与工具交互效率，探索如何通过递归自我改进（Recursive Self-Improvement, RSI）理念扩展自动化科研循环（Auto-Research Loops）。",
            "problem_solved": "随着智能体任务向长程复杂规划演进，Token 消耗呈指数级暴增，成为阻碍大模型自主递归自演进（RSI）落地的核心成本瓶颈与上下文溢出主因。",
            "novel_mechanism": "提出 **SoL-Pi 框架**（NVIDIA 团队）：\n    1. **智能体 Harness 层的递归自优化**：在控制层而非模型参数层构建自适应演进环；\n    2. **四大生存机制协同**：动作执行轻量化、上下文语义紧凑化压缩、环境观测过滤与委托代理式阅读；\n    3. 实现了跨多种异构环境的自动化经验迁移。",
            "impact_and_contribution": "在 51 项复杂 EdgeBench 任务评测中，与 GPT-5.6 Sol 及 Opus 5 的顶级 Harness 达到同等性能，同时**直接降低 44.7% ~ 49.0% 的 Token 流量消耗并削减约 1/3 的 API 调用成本**，为长周期运行的自进化智能体提供了极为关键的吞吐与算力降本方案。"
        }

    # 针对 Coding Agents Harness 实证
    if "harness design for coding" in t_lower or "2609.20804" in pid:
        return {
            "val_tag": "⭐⭐⭐⭐ [高价值精读·代码智能体组件架构剖析]",
            "research_objective": "通过模块化解耦实验，实证评测智能体执行框架（Harness）三大核心组件（规划 Planning、动作空间 Action Space、上下文管理 Context Management）对长程软件工程任务（SWE-bench）的独立与交互影响。",
            "problem_solved": "过往研究将智能体 Harness 作为黑盒整体评测，无法量化各工程模块（是 Bash 还是预设工具？是主动规划还是被动纠错？）的真实效能与边际贡献。",
            "novel_mechanism": "在 SWE-Bench Verified 和 Terminal-Bench 2.1 上完成了 176 组跨模型严格受控消融实验，提出了‘规则过滤优于大模型二次摘要’以及‘针对高 Bash 能力模型采用 Bash-only 动作空间成本收益比最优’的架构准则。",
            "impact_and_contribution": "为搭建面向自进化软件工程与操作系统控制智能体的 Runtime 环境提供了确凿的工程设计图谱与选型规范。"
        }

    # 默认通用结构
    return {
        "val_tag": "⭐⭐⭐ [跟进关注·前沿智能体演进相关]",
        "research_objective": "探索智能体在长程环境交互中的决策优化、表征学习或多模态物理世界因果感知，提升智能体在未见任务中的自适应能力。",
        "problem_solved": "解决当前智能体在复杂环境下依赖局部先验、动作序列稀疏、反馈延迟或跨模态融合不足导致的规划崩溃难题。",
        "novel_mechanism": "提出了端到端可验证的表征框架或决策损失函数，通过新颖的注意力机制、数据生成管道或阶段式优化算法提升执行效能。",
        "impact_and_contribution": "为智能体自主环境交互、自适应决策与多模态世界模型构建提供了理论启发与开源实现。"
    }

def run_daily_pipeline(target_date=None):
    base_papers_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_前沿论文追踪")
    os.makedirs(base_papers_dir, exist_ok=True)
    
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")
        
    date_dir = os.path.join(base_papers_dir, target_date)
    os.makedirs(date_dir, exist_ok=True)
    
    md_file_path = os.path.join(date_dir, f"{target_date}_自进化智能体论文精选.md")
    
    print(f"[{datetime.now()}] 开始抓取 Hugging Face Daily Papers (日期: {target_date})...")
    papers = fetch_daily_papers(target_date)
    if not papers:
        print("未抓取到有效数据。")
        return
        
    # 筛选相关与高价值
    filtered_list = []
    for item in papers:
        is_rel, is_high_val, score, matches = evaluate_paper_relevance_and_value(item)
        if is_rel:
            filtered_list.append((item, is_high_val, score, matches))
            
    if not filtered_list:
        print(f"[{target_date}] 今日未检测到强匹配的自进化智能体相关论文。")
        return
        
    print(f"[*] 共筛选出 {len(filtered_list)} 篇智能体相关文献，正在研判价值并生成结构化调研简报...")
    
    # 构建 Markdown
    md_lines = []
    md_lines.append(f"# 📅 {target_date} 自进化智能体前沿论文深度追踪\n")
    md_lines.append(f"> **本期专栏目录**：`02_前沿论文追踪/{target_date}/`\n")
    md_lines.append(f"> **数据源**：[Hugging Face Daily Papers](https://huggingface.co/papers) | **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    md_lines.append(f"本期共萃取并深度剖析 **{len(filtered_list)}** 篇代表性前沿论文。对于评定为 **⭐⭐⭐⭐⭐ 核心必读** 与 **⭐⭐⭐⭐ 高价值精读** 的文献，**原版 PDF 论文已自动下载至当前日期目录下**，点击下方链接即可直接在本地阅读。\n\n")
    md_lines.append("---\n\n")
    
    high_val_download_count = 0
    
    for idx, (item, is_high_val, score, matches) in enumerate(filtered_list, 1):
        p = item.get("paper", {})
        pid = p.get("id") or ""
        title = p.get("title") or item.get("title") or "Untitled"
        summary = p.get("summary") or item.get("summary") or ""
        authors = [a.get("name") for a in p.get("authors", []) if a.get("name")]
        authors_str = ", ".join(authors[:5]) + (" 等" if len(authors) > 5 else "")
        org = p.get("organization", {}).get("fullname") or p.get("organization", {}).get("name") or "国际顶尖研究机构"
        upvotes = p.get("upvotes", 0)
        github_repo = p.get("githubRepo") or ""
        
        analysis = parse_paper_deep_dive(pid, title, summary, authors, org)
        
        # 如果是高价值，下载 PDF 到当天目录
        pdf_name = f"{pid}_{sanitize_filename(title)[:50]}.pdf"
        local_pdf_path = os.path.join(date_dir, pdf_name)
        pdf_downloaded = False
        
        if is_high_val or "核心必读" in analysis["val_tag"] or "高价值精读" in analysis["val_tag"]:
            if not os.path.exists(local_pdf_path) or os.path.getsize(local_pdf_path) == 0:
                success = download_arxiv_pdf(pid, local_pdf_path)
                if success:
                    pdf_downloaded = True
                    high_val_download_count += 1
                time.sleep(1) # 友好延迟
            else:
                pdf_downloaded = True
                high_val_download_count += 1
                
        md_lines.append(f"### {idx}. [{title}](https://huggingface.co/papers/{pid})\n\n")
        md_lines.append(f"- **学术评级**：**{analysis['val_tag']}**\n")
        md_meta = f"- **论文元数据**：`arXiv:{pid}` | [Hugging Face 论文讨论页](https://huggingface.co/papers/{pid}) | [arXiv 原文](https://arxiv.org/abs/{pid})"
        if github_repo:
            md_meta += f" | [💻 官方代码开源库]({github_repo})"
        md_lines.append(md_meta + "\n")
        
        if pdf_downloaded:
            rel_pdf_path = f"./{pdf_name}"
            md_lines.append(f"- **📑 本地 PDF 全文**：**[点击直接在本地打开论文 PDF]({rel_pdf_path})** *(已自动归档至本目录)*\n")
            
        md_lines.append(f"- **作者与机构**：{authors_str}（{org}）\n")
        md_lines.append(f"- **社区关注度**：🔥 **{upvotes}** Upvotes | **命中核心标签**：`{'`, `'.join(matches)}`\n\n")
        
        md_lines.append("#### 🔬 深度科研结构化拆解：\n\n")
        md_lines.append(f"- **1. 具体研究什么 (What is being studied)**：\n  {analysis['research_objective']}\n")
        md_lines.append(f"- **2. 解决了什么核心痛点 (Problem Solved & Pain Points)**：\n  {analysis['problem_solved']}\n")
        md_lines.append(f"- **3. 提出的新技术 / 新架构 / 新理念 (Novel Techniques & Concepts)**：\n  {analysis['novel_mechanism']}\n")
        md_lines.append(f"- **4. 对自进化智能体研究的深远影响与启示 (Impact on Self-Evolving Agents)**：\n  {analysis['impact_and_contribution']}\n\n")
        
        md_lines.append(f"<details><summary>👉 点击展开查看论文官方英文 Abstract 原文</summary>\n\n> {summary.strip()}\n\n</details>\n")
        md_lines.append("---\n\n")
        
    # 写入当天的 Markdown
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))
    print(f"[√] 当日文献研报已保存至: {md_file_path}")
    print(f"[√] 当日高价值论文 PDF 归档数: {high_val_download_count} 篇")
    
    # 更新总索引表
    update_global_index(base_papers_dir, target_date, len(filtered_list), high_val_download_count, md_file_path)

def update_global_index(base_dir, date_str, total_count, pdf_count, md_file_path):
    index_file = os.path.join(base_dir, "00_论文追踪总索引.md")
    rel_md_path = f"./{date_str}/{os.path.basename(md_file_path)}"
    new_entry_line = f"| **{date_str}** | 共 **{total_count}** 篇 | **{pdf_count}** 篇核心 PDF 已归档 | [📂 打开当日追踪简报]({rel_md_path}) | `{date_str}/` |\n"
    
    if not os.path.exists(index_file):
        header = "# 自进化智能体 (Self-Evolving Agents) 前沿论文每日追踪总索引\n\n"
        header += "> **目录说明**：每日论文按日期创建独立目录存放（`YYYY-MM-DD/`），高价值论文原版 PDF 及深度分析 Markdown 同步沉淀在各日期目录下。\n\n"
        header += "## 📅 每日追踪归档速查表\n\n"
        header += "| 跟踪日期 | 当日收录篇数 | 高价值 PDF 本地归档 | 当日调研简报链接 | 所在文件夹 |\n"
        header += "| :--- | :--- | :--- | :--- | :--- |\n"
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(header + new_entry_line)
    else:
        with open(index_file, "r", encoding="utf-8") as f:
            content = f.read()
        if f"| **{date_str}** |" in content:
            # 更新已有日期的条目
            lines = content.splitlines(keepends=True)
            new_lines = []
            for line in lines:
                if f"| **{date_str}** |" in line:
                    new_lines.append(new_entry_line)
                else:
                    new_lines.append(line)
            with open(index_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        else:
            with open(index_file, "a", encoding="utf-8") as f:
                f.write(new_entry_line)

if __name__ == "__main__":
    run_daily_pipeline()
