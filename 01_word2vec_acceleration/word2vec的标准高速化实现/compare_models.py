"""
=============================================================================
脚本名称: compare_models.py
核心功能: CBOW 与 Skip-Gram 高速化实现在 PTB 数据集上的全方位效果对比与基准评测
评测维度:
    1. 训练吞吐率与耗时对比 (Speed: samples/sec, Total Time)
    2. 损失收敛曲线对比 (Loss Convergence)
    3. 语义相似度表现对比 (Semantic Similarity on key domain words)
    4. 语法与语义类比推理能力对比 (Analogy Reasoning: a:b :: c:?)
=============================================================================
"""

import sys
import os
import time
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from dataset import load_ptb_data
from models import CBOWModel, SkipGramModel
from optimizer import Adam
from trainer import Trainer
from eval import most_similar, analogy


def run_comparison(epochs: int = 5, max_words: int = 15000, hidden_size: int = 64, window_size: int = 3):
    # 确保 Windows 终端 UTF-8 打印兼容
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 80)
    print(" 📊 CBOW vs Skip-Gram 架构全方位效果对比与 Benchmark 基准测试")
    print(f" 参数配置: 语料规模={max_words} 词 | 嵌入维度={hidden_size} | 窗口大小={window_size} | 轮数={epochs}")
    print("=" * 80)

    # 1. 统一加载相同的数据集
    print("\n[Step 1] 加载基准评测语料 (PTB)...")
    contexts, target, corpus, word_to_id, id_to_word = load_ptb_data(
        window_size=window_size,
        max_words=max_words,
        use_subsampling=True,
        subsample_threshold=1e-3
    )
    vocab_size = len(word_to_id)
    data_size = len(contexts)
    print(f"[Dataset] 词表大小: {vocab_size} | 样本对数量: {data_size:,}")

    # =========================================================================
    # 2. 评测 CBOW 模型
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 🚀 [1/2] 开始评测 CBOW (Continuous Bag of Words) 模型...")
    print("=" * 80)
    np.random.seed(42)
    cbow_model = CBOWModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        corpus=corpus,
        window_size=window_size,
        sample_size=5,
        power=0.75
    )
    cbow_opt = Adam(lr=0.005)
    cbow_trainer = Trainer(cbow_model, cbow_opt)

    start_cbow = time.time()
    cbow_losses = cbow_trainer.fit(
        contexts=contexts,
        target=target,
        max_epoch=epochs,
        batch_size=64,
        log_interval=40
    )
    cbow_time = time.time() - start_cbow

    # =========================================================================
    # 3. 评测 Skip-Gram 模型
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 🚀 [2/2] 开始评测 Skip-Gram 模型...")
    print("=" * 80)
    np.random.seed(42)
    skipgram_model = SkipGramModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        corpus=corpus,
        window_size=window_size,
        sample_size=5,
        power=0.75
    )
    skipgram_opt = Adam(lr=0.002)
    skipgram_trainer = Trainer(skipgram_model, skipgram_opt)

    start_sg = time.time()
    sg_losses = skipgram_trainer.fit(
        contexts=contexts,
        target=target,
        max_epoch=epochs,
        batch_size=64,
        log_interval=40
    )
    sg_time = time.time() - start_sg

    # =========================================================================
    # 4. 汇总对比报告
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [*] 对比测试结果汇总分析报告")
    print("=" * 80)

    cbow_speed = (data_size * epochs) / cbow_time
    sg_speed = (data_size * epochs) / sg_time

    print(f"{'对比指标':<25s} | {'CBOW 模型':<20s} | {'Skip-Gram 模型':<20s}")
    print("-" * 75)
    print(f"{'总训练耗时 (Total Time)':<25s} | {cbow_time:18.2f}s | {sg_time:18.2f}s")
    print(f"{'平均训练吞吐量 (Samples/s)':<25s} | {cbow_speed:16.0f} /s | {sg_speed:16.0f} /s")
    print(f"{'速度倍率 (Relative Speed)':<25s} | {'基准 (1.0x)':<20s} | {f'{sg_speed/cbow_speed:.2f}x':<20s}")
    print(f"{'初始轮次损失 (Initial Loss)':<25s} | {cbow_trainer.epoch_loss_list[0]:18.4f} | {skipgram_trainer.epoch_loss_list[0]:18.4f}")
    print(f"{'最终轮次损失 (Final Loss)':<25s} | {cbow_trainer.epoch_loss_list[-1]:18.4f} | {skipgram_trainer.epoch_loss_list[-1]:18.4f}")
    print("-" * 75)

    # 5. 近义词语义质量对比
    test_word = "market" if "market" in word_to_id else ("company" if "company" in word_to_id else list(word_to_id.keys())[10])
    print(f"\n[语义检索对比] 查询词: '{test_word}'")
    print("\n--- CBOW 模型的 Top-5 最相似词 ---")
    most_similar(test_word, word_to_id, id_to_word, cbow_model.word_vecs, top=5)

    print("\n--- Skip-Gram 模型的 Top-5 最相似词 ---")
    most_similar(test_word, word_to_id, id_to_word, skipgram_model.word_vecs, top=5)

    # 6. 类比推理能力对比
    analogy_triplet = ("he", "his", "she")
    if all(w in word_to_id for w in analogy_triplet):
        print(f"\n[类比推理对比] '{analogy_triplet[0]}' 之于 '{analogy_triplet[1]}' 犹如 '{analogy_triplet[2]}' 之于 [?]")
        print("\n--- CBOW 词类比推理 ---")
        analogy(*analogy_triplet, word_to_id, id_to_word, cbow_model.word_vecs, top=3)

        print("\n--- Skip-Gram 词类比推理 ---")
        analogy(*analogy_triplet, word_to_id, id_to_word, skipgram_model.word_vecs, top=3)


if __name__ == "__main__":
    run_comparison(epochs=5, max_words=12000, hidden_size=64, window_size=3)
