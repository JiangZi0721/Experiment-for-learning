"""
=============================================================================
脚本名称: quick_demo.py
核心功能: 极速上手体验脚本 (5~10秒内完整跑通 数据预处理 -> CBOW训练 -> 语义检索)
适用场景: 初学者第一次运行、测试环境、代码逻辑即时自检
=============================================================================
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from dataset import load_ptb_data
from models import CBOWModel
from optimizer import Adam
from trainer import Trainer
from eval import most_similar, analogy


def run_quick_demo():
    # 确保 Windows 终端输出 UTF-8 编码
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[*] Word2Vec 高速化实现 - 极速体验 Demo (CBOW + Negative Sampling)")
    print("=" * 70)

    # 1. 加载 PTB 前 12,000 个词作为轻量语料，确保极速收敛与秒级体验
    print("\n[Step 1] 加载与预处理轻量 PTB 数据集...")
    window_size = 2
    contexts, target, corpus, word_to_id, id_to_word = load_ptb_data(
        window_size=window_size,
        max_words=12000,           # 读取前 12,000 词
        use_subsampling=True,      # 启用高频词下采样
        subsample_threshold=1e-3
    )

    vocab_size = len(word_to_id)
    hidden_size = 64               # 快速实验使用 64 维词向量

    # 2. 实例化高速 CBOW 模型
    print(f"\n[Step 2] 构建高速化 CBOW 模型 (Vocab={vocab_size}, Dim={hidden_size}, NegSamples=5)...")
    model = CBOWModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        corpus=corpus,
        window_size=window_size,
        sample_size=5,
        power=0.75
    )

    # 3. 配置 Adam 优化器与训练器
    optimizer = Adam(lr=0.005)
    trainer = Trainer(model, optimizer)

    # 4. 开始快速训练 5 个 Epoch
    print("\n[Step 3] 开始极速训练 (5 个 Epoch)...")
    trainer.fit(
        contexts=contexts,
        target=target,
        max_epoch=5,
        batch_size=64,
        max_grad_norm=5.0,
        log_interval=20
    )

    # 5. 保存模型
    save_path = os.path.join(CURRENT_DIR, "weights", "quick_demo_weights.pkl")
    trainer.save_model(save_path, word_to_id, id_to_word)

    # 6. 词向量效果演示
    print("\n" + "=" * 70)
    print("[*] [Step 4] 词向量语义检索与相似度测试")
    print("=" * 70)
    word_matrix = model.word_vecs

    # 测试若干高频常见词的近义词检索
    test_queries = ["bank", "company", "year", "market"]
    for q in test_queries:
        if q in word_to_id:
            most_similar(q, word_to_id, id_to_word, word_matrix, top=5)

    print("\n[DONE] Quick Demo 成功运行完毕！如需在完整 PTB 语料上训练，请运行: python train.py")


if __name__ == "__main__":
    run_quick_demo()
