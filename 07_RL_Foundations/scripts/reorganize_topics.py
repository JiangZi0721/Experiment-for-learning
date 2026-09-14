"""
强化学习学习笔记仓库目录结构主题化整理工具
将当前平铺的 RL_Foundations 按照 6 大核心专题提取并组织为独立的子目录体系：
01_MDP_Dynamic_Programming
02_Model_Free_Methods
03_Complex_GridWorld_Benchmark
04_Deep_Q_Networks_DQN
05_Distribution_vs_Sample_Models
06_Deep_World_Models_and_Gym
"""
import os
import shutil

ROOT_DIR = r"F:\LearningNotes\RL_Foundations"

TOPIC_CONFIGS = {
    "01_MDP_Dynamic_Programming": {
        "title": "马尔可夫决策过程与经典动态规划 (MDP & Dynamic Programming)",
        "desc": "强化学习基石理论：环境数学建模（MDP 五元组）、贝尔曼期望与最优方程、策略评估 (Policy Evaluation)、策略迭代 (Policy Iteration, PI) 与价值迭代 (Value Iteration, VI)，以及折扣因子 Gamma 对未来眼界的影响。",
        "docs": [
            "MDP_Architecture_Master.md"
        ],
        "src": [
            "src/grid_world.py",
            "src/policy_iteration.py",
            "src/value_iteration.py",
            "src/visualizer.py"
        ],
        "experiments": [
            "experiments/exp1_dynamic_programming.py",
            "experiments/exp3_gamma_comparison.py",
            "run_experiment.py",
            "run_gamma_comparison.py"
        ],
        "logs": [
            "logs/experiment_results.txt",
            "logs/experiment_output.log"
        ],
        "images": [
            "images/policy_evaluation_convergence.png",
            "images/policy_iteration_evolution.png",
            "images/value_iteration_convergence.png",
            "images/value_iteration_evolution.png",
            "images/gamma_comparison_experiment.png"
        ]
    },
    "02_Model_Free_Methods": {
        "title": "免模型时序差分三剑客与重要性采样 (Model-Free: TD(0), SARSA & Q-learning)",
        "desc": "免模型时序差分学习：蒙特卡洛 (MC) 与动态规划 (DP) 的桥梁——时间差分 (TD)、同策略控制 SARSA、异策略控制 Q-learning、重要性采样消除机理与行为策略解耦。",
        "docs": [
            "On_Policy_vs_Off_Policy_and_MC_vs_DP.md",
            "Importance_Sampling_and_Model_Free_Convergence.md"
        ],
        "src": [
            "src/model_free.py",
            "src/plot_model_free.py"
        ],
        "experiments": [
            "experiments/exp2_model_free.py",
            "run_model_free_experiment.py"
        ],
        "logs": [
            "logs/model_free_results.txt"
        ],
        "images": [
            "images/td0_complete_workflow.png",
            "images/sarsa_complete_workflow.png",
            "images/qlearning_complete_workflow.png"
        ]
    },
    "03_Complex_GridWorld_Benchmark": {
        "title": "复杂网格世界极限博弈与六大剧变时刻深度透视 (Complex GridWorld Benchmark & Critical Steps)",
        "desc": "复杂多奖励陷阱网格：岩浆坑 (Lava) 致命惩罚、地刺陷阱 (Spike Trap) 局部威慑、次级金币目标 (Subgoal) 贪心诱惑与主目标求解。逐时间步追踪 6 大决策剧变时刻的贝尔曼算术演化过程。",
        "docs": [
            "Complex_GridWorld_Analysis.md"
        ],
        "src": [
            "src/complex_grid_world.py",
            "src/complex_model_free.py",
            "src/plot_complex_analysis.py"
        ],
        "experiments": [
            "experiments/exp4_complex_gridworld.py",
            "run_complex_experiment.py",
            "generate_all_complex_figures.py",
            "scripts/find_exact_events.py"
        ],
        "logs": [
            "logs/complex_experiment_results.txt"
        ],
        "images": [
            "images/complex_env_overview_and_converged_policies.png",
            "images/six_critical_steps_overview.png",
            "images/step1_td0_subgoal_discovery.png",
            "images/step2_td0_lava_pit_fall.png",
            "images/step3_sarsa_main_goal_discovery.png",
            "images/step4_sarsa_spike_trap_deterrence.png",
            "images/step5_qlearning_bootstrap_surge.png",
            "images/step6_qlearning_lava_penalty.png"
        ]
    },
    "04_Deep_Q_Networks_DQN": {
        "title": "深度强化学习 DQN、经验回放与死亡三角 (Deep Q-Networks, Replay Buffer & Deadly Triad)",
        "desc": "从表格到深度神经网络函数逼近：致命三元组 (Deadly Triad) 的数学成因与发散灾难、经验回放池 (Experience Replay) 消除样本时序相关性、目标网络 (Target Network) 冻结自举目标。",
        "docs": [
            "Experience_Replay_and_Deadly_Triad.md"
        ],
        "src": [
            "src/dqn.py",
            "src/plot_dqn.py"
        ],
        "experiments": [
            "experiments/exp5_dqn.py"
        ],
        "logs": [
            "logs/dqn_results.txt"
        ],
        "images": [
            "images/dqn_training_curves.png",
            "images/dqn_policy_and_q_heatmap.png"
        ]
    },
    "05_Distribution_vs_Sample_Models": {
        "title": "分布模型 vs 样本模型全维度实证对比 (Distribution Model vs Sample Model Benchmark)",
        "desc": "强化学习环境建模分水岭：分布模型（全概率转移矩阵、多重积分期望更新）vs 样本模型（抽样生成单次转移、常数开销 O(1)）。实测 Sutton Fig 8.7 分界线、高分支因子组合时间爆炸与极小概率高危陷阱盲区。",
        "docs": [
            "Distribution_Model_vs_Sample_Model.md"
        ],
        "src": [
            "src/model_comparison.py",
            "src/plot_model_comparison.py"
        ],
        "experiments": [
            "experiments/exp6_distribution_vs_sample_model.py"
        ],
        "logs": [
            "logs/model_comparison_results.txt"
        ],
        "images": [
            "images/distribution_vs_sample_branching_budget.png",
            "images/distribution_vs_sample_combinatorial_time.png",
            "images/distribution_vs_sample_rare_trap.png",
            "images/distribution_vs_sample_comprehensive_dashboard.png"
        ]
    },
    "06_Deep_World_Models_and_Gym": {
        "title": "深度高斯世界模型、脑内做梦推演与 OpenAI Gym 物理实证 (Deep World Models & OpenAI Gym)",
        "desc": "连续控制下的深度世界模型：高斯概率转移网络、重参数化采样、高斯 NLL 损失反向传播、60 平行宇宙粒子滤波、GPU 矢量化 MPC 模型预测控制；接入 OpenAI Gym (Gymnasium) 连续倒立摆物理环境，实现影子环境做梦投影与三大高动态实证动画。",
        "docs": [
            "Deep_World_Model_Simulation_Report.md"
        ],
        "src": [
            "src/deep_world_model.py",
            "src/plot_world_model.py",
            "src/gym_world_model_showcase.py"
        ],
        "experiments": [
            "experiments/exp7_deep_world_model_simulation.py",
            "experiments/exp8_gym_world_model_showcase.py"
        ],
        "checkpoints": [
            "checkpoints/world_model.pt"
        ],
        "logs": [
            "logs/deep_world_model_results.txt",
            "logs/gym_world_model_showcase.txt"
        ],
        "images": [
            "images/world_model_training_curves.png",
            "images/world_model_dream_vs_real.png",
            "images/world_model_parallel_universes.png",
            "images/world_model_mpc_control.png",
            "images/world_model_master_dashboard.png",
            "images/gym_pendulum_mpc_control.gif",
            "images/gym_comparison_random_vs_mpc.gif",
            "images/gym_real_vs_dream.gif",
            "images/gym_pendulum_master_showcase.png"
        ]
    }
}

def build_topic_readme(topic_dir, config):
    readme_path = os.path.join(topic_dir, "README.md")
    folder_name = os.path.basename(topic_dir)
    
    content = f"""# {config['title']}

> **专题归属**：`RL_Foundations` 核心学习体系  
> **专题目录**：`{folder_name}`  
> **定位与目标**：{config['desc']}

---

## 目录结构导航

```
{folder_name}/
├── README.md               # [当前文档] 专题学习与运行导航
├── docs/                   # 理论讲义与详尽实验分析报告 (*.md)
├── src/                    # 核心算法实现与绘图组件 (*.py)
├── experiments/            # 独立可执行的实验脚本 (*.py)
├── logs/                   # 真实实验运行输出与数值日志 (*.txt, *.log)
└── images/                 # 出版级高清图表、动态动画与大盘 (*.png, *.gif)
```

---

## 1. 核心理论讲义 (Docs)
"""
    for doc in config.get("docs", []):
        doc_name = os.path.basename(doc)
        content += f"- [`{doc_name}`](docs/{doc_name})\n"

    content += "\n## 2. 算法源码 (Source Code)\n"
    for s in config.get("src", []):
        s_name = os.path.basename(s)
        content += f"- [`{s_name}`](src/{s_name})\n"

    content += "\n## 3. 实验脚本 (Experiments)\n"
    for exp in config.get("experiments", []):
        exp_name = os.path.basename(exp)
        content += f"- [`{exp_name}`](experiments/{exp_name})\n"

    content += "\n## 4. 真实运行日志 (Logs)\n"
    for l in config.get("logs", []):
        l_name = os.path.basename(l)
        content += f"- [`{l_name}`](logs/{l_name})\n"

    content += "\n## 5. 高清成果图表与动态动画 (Images & Visualizations)\n"
    for img in config.get("images", []):
        img_name = os.path.basename(img)
        if img_name.endswith(".gif"):
            content += f"### 动态动画：`{img_name}`\n\n![{img_name}](images/{img_name})\n\n"
        else:
            content += f"### 图表成果：`{img_name}`\n\n![{img_name}](images/{img_name})\n\n"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [Created README] {readme_path}")

def run_reorganization():
    print("=" * 80)
    print(">>> 开始执行强化学习知识库专题化目录重构...")
    print("=" * 80)
    
    for topic_folder, cfg in TOPIC_CONFIGS.items():
        topic_path = os.path.join(ROOT_DIR, topic_folder)
        print(f"\n>>> 正在处理专题: {topic_folder}")
        
        # 创建子目录结构
        subdirs = ["docs", "src", "experiments", "logs", "images"]
        if "checkpoints" in cfg:
            subdirs.append("checkpoints")
            
        for sd in subdirs:
            os.makedirs(os.path.join(topic_path, sd), exist_ok=True)
            
        # 复制文档
        for doc in cfg.get("docs", []):
            src_file = os.path.join(ROOT_DIR, doc)
            if os.path.exists(src_file):
                dst_file = os.path.join(topic_path, "docs", os.path.basename(doc))
                shutil.copy2(src_file, dst_file)
                print(f"  [Copied Doc] {doc} -> docs/")
                
        # 复制源码
        for s in cfg.get("src", []):
            src_file = os.path.join(ROOT_DIR, s)
            if os.path.exists(src_file):
                dst_file = os.path.join(topic_path, "src", os.path.basename(s))
                shutil.copy2(src_file, dst_file)
                print(f"  [Copied Src] {s} -> src/")
                
        # 复制实验
        for exp in cfg.get("experiments", []):
            src_file = os.path.join(ROOT_DIR, exp)
            if os.path.exists(src_file):
                dst_file = os.path.join(topic_path, "experiments", os.path.basename(exp))
                shutil.copy2(src_file, dst_file)
                print(f"  [Copied Exp] {exp} -> experiments/")
                
        # 复制日志
        for l in cfg.get("logs", []):
            src_file = os.path.join(ROOT_DIR, l)
            if os.path.exists(src_file):
                dst_file = os.path.join(topic_path, "logs", os.path.basename(l))
                shutil.copy2(src_file, dst_file)
                print(f"  [Copied Log] {l} -> logs/")
                
        # 复制图像
        for img in cfg.get("images", []):
            src_file = os.path.join(ROOT_DIR, img)
            if os.path.exists(src_file):
                dst_file = os.path.join(topic_path, "images", os.path.basename(img))
                shutil.copy2(src_file, dst_file)
                print(f"  [Copied Img] {img} -> images/")
                
        # 复制权重
        if "checkpoints" in cfg:
            for ckpt in cfg.get("checkpoints", []):
                src_file = os.path.join(ROOT_DIR, ckpt)
                if os.path.exists(src_file):
                    dst_file = os.path.join(topic_path, "checkpoints", os.path.basename(ckpt))
                    shutil.copy2(src_file, dst_file)
                    print(f"  [Copied Ckpt] {ckpt} -> checkpoints/")
                    
        # 生成专题专属 README.md
        build_topic_readme(topic_path, cfg)
        
    print("\n" + "=" * 80)
    print(">>> 6 大专题独立目录抽取与整理完毕！")
    print("=" * 80)

if __name__ == "__main__":
    run_reorganization()
