# -*- coding: utf-8 -*-
"""
实验二：神经网络稠密嵌入微调实验 - 训练套件 (Live Contrastive Fine-Tuning)
运行本脚本：基于原生 PyTorch + Transformers 在本地 CPU 执行基于 InfoNCE 损失的三元组对比学习微调，
将生成的私有模型权重持久化保存到 models/bge-small-finetuned/。
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.train_bge_live import train_live

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  [实验二：稠密嵌入模型现场微调] 启动 InfoNCE 对比学习训练")
    print("=" * 70)
    train_live()
