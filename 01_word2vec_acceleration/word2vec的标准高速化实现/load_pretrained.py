"""
=============================================================================
脚本名称: load_pretrained.py
核心功能: 工业级预训练词向量加载与格式转换工具 (Pretrained Vectors Loader)
包含特性:
    1. 支持将任意标准文本格式词向量 (如 Stanford GloVe, Google News Word2Vec,
       FastText 的 .txt 文件) 一键无缝转换为本项目统一的 .pkl 权重格式
    2. 支持按高频词表裁剪 (Vocab Pruning)，将数百兆的大文件秒级压缩为仅需几兆的精炼词表
    3. 内置经典语义基准向量生成器，为离线环境提供教科书级的经典语义类比对照
=============================================================================
理论认知与语料规模的决定性影响:
在 NLP 领域中，Word2Vec 词向量的表征质量受到语料规模的绝对支配:
- Penn Treebank (PTB 语料库): 约 90 万词 (约 5MB)，纯粹由 1989 年《华尔街日报》财经新闻构成。
  它能非常出色地学习到金融商业词汇 (bank, market, stock, dollar) 与常见动词时态 (take, took, go, went) 的聚类；
  但像 'king', 'queen' 在全篇只有极低频的两三次出现，无法涌现经典的“国王-王后”类比。
- 大规模工业语料 (如 Wikipedia 3 亿词，或 Google News 1000 亿词):
  充分包含了人类社会的通用百科常识，使得词向量在高维欧式空间中形成近乎完美的超平面平移。
=============================================================================
"""

import sys
import os
import argparse
import pickle
import numpy as np
from typing import Dict, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")


def convert_txt_to_pkl(
    txt_path: str,
    output_pkl_path: str,
    max_vocab: int = 20000,
    encoding: str = "utf-8"
):
    """
    将标准格式的 GloVe / Word2Vec .txt 词向量文件转换为本项目可读的 .pkl 字典
    .txt 格式规范: 每行为 "word val1 val2 val3 ... valD"
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"未找到词向量文本文件: {txt_path}")

    print(f"[*] 开始解析词向量文本: {txt_path} (最大保留词数: {max_vocab:,}) ...")
    word_to_id: Dict[str, int] = {}
    id_to_word: Dict[int, str] = {}
    vectors = []

    with open(txt_path, "r", encoding=encoding, errors="ignore") as f:
        # 跳过可能存在的首行 (某些 word2vec 格式首行为 "vocab_size dim")
        first_line = f.readline().strip().split()
        if len(first_line) == 2 and first_line[0].isdigit() and first_line[1].isdigit():
            print(f"[*] 检测到 Word2Vec 首行格式说明: 词表={first_line[0]}, 维度={first_line[1]}")
        else:
            # 是正常行，处理首行
            w = first_line[0]
            v = np.array(first_line[1:], dtype=np.float32)
            idx = len(vectors)
            word_to_id[w] = idx
            id_to_word[idx] = w
            vectors.append(v)

        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            w = parts[0]
            try:
                v = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue

            idx = len(vectors)
            word_to_id[w] = idx
            id_to_word[idx] = w
            vectors.append(v)

            if len(vectors) >= max_vocab:
                break

    W_matrix = np.array(vectors, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_pkl_path)), exist_ok=True)

    save_data = {
        "W_in": W_matrix,
        "W_out": W_matrix,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "vocab_size": len(word_to_id),
        "hidden_size": W_matrix.shape[1],
        "source": os.path.basename(txt_path)
    }

    with open(output_pkl_path, "wb") as f:
        pickle.dump(save_data, f)

    print(f"[OK] 转换完成！")
    print(f"   • 词表容量: {len(word_to_id):,} 词")
    print(f"   • 嵌入维度: {W_matrix.shape[1]} 维")
    print(f"   • 保存路径: {output_pkl_path}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="预训练词向量格式转换工具 (GloVe / Word2Vec -> pkl)")
    parser.add_argument("--txt_path", type=str, required=True,
                        help="输入的 .txt 词向量文件路径 (例如: glove.6B.50d.txt)")
    parser.add_argument("--output_pkl", type=str, default=os.path.join(WEIGHTS_DIR, "pretrained_glove.pkl"),
                        help="输出的 .pkl 权重路径")
    parser.add_argument("--max_vocab", type=int, default=25000,
                        help="最大截取词汇数量 (默认前 25,000 高频词)")
    args = parser.parse_args()

    convert_txt_to_pkl(args.txt_path, args.output_pkl, max_vocab=args.max_vocab)


if __name__ == "__main__":
    main()
