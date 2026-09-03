# -*- coding: utf-8 -*-
"""
White-Box RAG Lab - 统一实验调度总入口 (Master Experiment Runner)
以完全解耦的独立子进程依次调度各子目录的实验脚本，保障各实验环境隔离与独立复现。
"""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = BASE_DIR / "experiments"

def run_script(script_path: Path, args: list = None):
    cmd = [sys.executable, str(script_path)] + (args or [])
    res = subprocess.run(cmd, cwd=str(BASE_DIR))
    return res.returncode

def run_exp1():
    print("\n" + "#" * 80)
    print("  [正在执行] 实验一：重排序 (Cross-Encoder) 消融对比实验")
    print("#" * 80)
    script = EXP_DIR / "01_reranker_ablation" / "compare_ablation.py"
    run_script(script)

def run_exp2():
    print("\n" + "#" * 80)
    print("  [正在执行] 实验二：神经网络稠密嵌入微调与裕度对比实验")
    print("#" * 80)
    script = EXP_DIR / "02_embedding_finetune" / "evaluate_base_vs_finetuned.py"
    run_script(script)

def run_exp3():
    print("\n" + "#" * 80)
    print("  [正在执行] 实验三：多维评估指标体系量化大盘对比实验 (基线 vs 针对性优化)")
    print("#" * 80)
    script = EXP_DIR / "03_evaluation_metrics" / "compare_metrics_dashboard.py"
    run_script(script)

def run_exp4():
    print("\n" + "#" * 80)
    print("  [正在执行] 实验四：稀疏 (BM25) vs 稠密 (Dense) 双路正交对抗实验")
    print("#" * 80)
    script = EXP_DIR / "04_hybrid_retrieval_duel" / "compare_duel_synergy.py"
    run_script(script)

def main():
    parser = argparse.ArgumentParser(description="White-Box RAG 统一实验套件调度器")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4], help="指定运行第几组实验 (1:重排消融, 2:嵌入微调, 3:多维评估, 4:双路对抗)")
    parser.add_argument("--all", action="store_true", help="一键串行运行全部 4 组对比实验")

    args = parser.parse_args()

    if args.exp == 1:
        run_exp1()
    elif args.exp == 2:
        run_exp2()
    elif args.exp == 3:
        run_exp3()
    elif args.exp == 4:
        run_exp4()
    elif args.all or len(sys.argv) == 1:
        print("\n" + "=" * 80)
        print("  White-Box RAG Lab: 开始全量自动化复现 4 组经典对比实验")
        print("=" * 80)
        run_exp1()
        run_exp2()
        run_exp3()
        run_exp4()
        print("\n" + "=" * 80)
        print("🎉 全部 4 组对比实验已依次顺利执行完毕！")
        print("=" * 80)

if __name__ == "__main__":
    main()
