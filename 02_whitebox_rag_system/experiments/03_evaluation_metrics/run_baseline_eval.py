# -*- coding: utf-8 -*-
"""
实验三：多维评估体系量化评测 - 改进前 (Baseline: 静态 Top-5 全量塞入 Context)
运行本脚本：在 8 组标准化基准题库上客观评测基线系统的 Context Recall, Context Precision, Noise Rate, Faithfulness, Completeness 等全量指标。
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.evaluate_rag_metrics import run_evaluation_pipeline

def main():
    print("\n" + "=" * 70)
    print("  [实验三·改进前] 运行基线系统全量指标量化评测 (静态 Top-5 注入)")
    print("=" * 70)
    results, avg = run_evaluation_pipeline(mode="baseline")

    print("\n" + "=" * 70)
    print("  [基线评测平均指标看板]:")
    print(f"  • 首位命中率 HitRate@1:      {avg.get('HitRate@1', 0):.4f}")
    print(f"  • 平均倒数排名 MRR:          {avg.get('MRR', 0):.4f}")
    print(f"  • 上下文召回率 Context Recall: {avg.get('Context_Recall', 0):.4f}")
    print(f"  • ⚠️ 上下文精确率 Context Precision: {avg.get('Context_Precision', 0):.4f} (显著偏低！)")
    print(f"  • ⚠️ 上下文噪声率 Noise Rate:        {avg.get('Context_Noise_Rate', 0):.4f} (噪声过半！)")
    print(f"  • 生成忠实度 Faithfulness:   {avg.get('Faithfulness', 0):.4f}")
    print(f"  • 要点完整度 Completeness:   {avg.get('Completeness', 0):.4f}")
    print(f"  • 引用精准度 Citation Acc:   {avg.get('Citation_Accuracy', 0):.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
