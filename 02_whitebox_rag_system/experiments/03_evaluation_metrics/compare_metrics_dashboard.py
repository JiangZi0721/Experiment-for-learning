# -*- coding: utf-8 -*-
"""
实验三：多维评估体系量化评测 - 综合对比大盘 (Side-by-Side Metrics Dashboard)
运行本脚本：先后执行基线与优化评测，输出完整的指标增益对比表格，并将全量数据写入缓存。
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.evaluate_rag_metrics import main

if __name__ == "__main__":
    main()
