"""
=============================================================================
脚本名称: train.py
核心功能: PTB 完整语料库上的 Word2Vec 高速化标准训练与评估脚本
支持特性:
    1. 支持 CBOW 与 Skip-Gram 两种模型架构切换 (--model cbow / skipgram)
    2. 支持 Adam 与 SGD 优化器切换 (--optimizer adam / sgd)
    3. 支持滑动窗口大小、词向量维度、负样本数、学习率等超参数命令行自由配置
    4. 自动高频词下采样与词频 0.75 次幂负采样
    5. 训练完毕后自动评估核心语义词的近邻检索与词类比 (Word Analogy) 推理
=============================================================================
"""

import sys
import os
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from dataset import load_ptb_data
from models import CBOWModel, SkipGramModel
from optimizer import Adam, SGD
from trainer import Trainer
from eval import most_similar, analogy


def parse_args():
    parser = argparse.ArgumentParser(description="Word2Vec 高速化标准实现训练器 (PTB 数据集)")

    # 模型与数据超参数
    parser.add_argument("--model", type=str, default="cbow", choices=["cbow", "skipgram"],
                        help="模型类型: cbow (速度更快) 或 skipgram (低频词表征更好)")
    parser.add_argument("--hidden_size", type=int, default=100,
                        help="词向量维度 H (默认 100)")
    parser.add_argument("--window_size", type=int, default=5,
                        help="单侧上下文窗口大小 (默认 5，即单侧 5 个词，总窗口 10)")
    parser.add_argument("--negative_samples", type=int, default=5,
                        help="负采样抽取数量 K (默认 5)")
    parser.add_argument("--power", type=float, default=0.75,
                        help="负采样分布平滑指数 (默认 0.75)")

    # 训练过程超参数
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数 (默认 10)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批次大小 (默认 128)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="优化器学习率 (默认 0.001)")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="优化器类型 (默认 adam)")
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                        help="梯度裁剪最大范数 (默认 5.0)")

    # 数据集选项
    parser.add_argument("--max_words", type=int, default=None,
                        help="限制读取的最大词数 (调试时可设为如 50000，默认全部读取)")
    parser.add_argument("--no_subsampling", action="store_true",
                        help="若指定则关闭高频词下采样")
    parser.add_argument("--subsample_threshold", type=float, default=1e-4,
                        help="高频词下采样阈值 (默认 1e-4)")

    # 存储与评估
    parser.add_argument("--save_path", type=str, default=None,
                        help="训练权重保存路径 (默认保存到 weights/ 目录)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="是否跳过训练后的语义评估")

    return parser.parse_args()


def main():
    # 确保 Windows 终端 UTF-8 打印兼容
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()

    print("=" * 75)
    print("[*] Word2Vec 标准高速化实现 - PTB 语料库训练任务")
    print(f"   架构: {args.model.upper()} | 嵌入维度: {args.hidden_size} | 上下文窗口: {args.window_size}")
    print(f"   负采样数: {args.negative_samples} | 优化器: {args.optimizer.upper()} | 学习率: {args.lr}")
    print("=" * 75)

    # 1. 加载与预处理 PTB 数据集
    use_sub = not args.no_subsampling
    contexts, target, corpus, word_to_id, id_to_word = load_ptb_data(
        window_size=args.window_size,
        max_words=args.max_words,
        use_subsampling=use_sub,
        subsample_threshold=args.subsample_threshold
    )
    vocab_size = len(word_to_id)

    # 2. 构建模型
    if args.model.lower() == "cbow":
        model = CBOWModel(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            corpus=corpus,
            window_size=args.window_size,
            sample_size=args.negative_samples,
            power=args.power
        )
    else:
        model = SkipGramModel(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            corpus=corpus,
            window_size=args.window_size,
            sample_size=args.negative_samples,
            power=args.power
        )

    # 3. 构建优化器
    if args.optimizer.lower() == "adam":
        optimizer = Adam(lr=args.lr)
    else:
        optimizer = SGD(lr=args.lr)

    # 4. 实例化训练器并启动训练
    trainer = Trainer(model, optimizer)
    trainer.fit(
        contexts=contexts,
        target=target,
        max_epoch=args.epochs,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        log_interval=50
    )

    # 5. 保存模型权重
    if args.save_path is None:
        save_filename = f"word2vec_{args.model}_h{args.hidden_size}_w{args.window_size}.pkl"
        args.save_path = os.path.join(CURRENT_DIR, "weights", save_filename)

    trainer.save_model(args.save_path, word_to_id, id_to_word)

    # 6. 语义与类比推理评估
    if not args.skip_eval:
        print("\n" + "=" * 75)
        print("[*] 训练后语义相似度与类比推理全面评测")
        print("=" * 75)
        word_matrix = model.word_vecs

        # 核心金融与商业测试词 (PTB 为华尔街日报语料)
        eval_queries = ["bank", "company", "year", "market", "stock", "dollar", "president"]
        for q in eval_queries:
            if q in word_to_id:
                most_similar(q, word_to_id, id_to_word, word_matrix, top=5)

        # 词类比推理评测
        analogy_triplets = [
            ("man", "king", "woman"),
            ("he", "his", "she"),
            ("take", "took", "go"),
        ]
        for a, b, c in analogy_triplets:
            if a in word_to_id and b in word_to_id and c in word_to_id:
                analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5)


if __name__ == "__main__":
    main()
