"""
=============================================================================
脚本名称: word_math_cli.py
核心功能: 交互式词向量计算器 (Word Vector Arithmetic Calculator)
支持特性:
    1. 任意形式的语义代数运算: king - man + woman, tokyo - japan + france
    2. 语义叠加运算: bank + money, computer + software
    3. 单词近义词检索: market, company
    4. 自动未登录词 (OOV) 智能近邻拼写推荐
    5. 控制台动态渲染 ASCII 相似度置信柱状图
    6. 既支持交互式 REPL 终端，也支持命令行一次性传参求值
=============================================================================
理论重点:
词向量的“线性空间平移特性” (Linear Translation Property):
在 Mikolov 2013 的经典论文中，证明了词向量在高维流形空间中，不同词汇之间的语法与语义关系
表现为平行的空间位移向量:
    vec("king") - vec("man") ≈ vec("queen") - vec("woman")
因此通过向量加减:
    vec(?) ≈ vec("king") - vec("man") + vec("woman")
在全词表空间中检索距离该合成向量最近的词，即可完成奇迹般的语义类比推理！
=============================================================================
"""

import sys
import os
import re
import argparse
import pickle
import numpy as np
from typing import List, Tuple, Dict, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights", "pretrained_glove.pkl")
PTB_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights", "word2vec_cbow_h100_w5.pkl")
ALT_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights", "quick_demo_weights.pkl")

# 默认优先加载通用百科模型 (满足常识类比需求)，若不存在则自动加载本地 PTB 财经模型
DEFAULT_WEIGHTS_PATH = PRETRAINED_WEIGHTS_PATH if os.path.exists(PRETRAINED_WEIGHTS_PATH) else PTB_WEIGHTS_PATH


class WordVectorCalculator:
    """
    词向量代数运算引擎
    """

    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.W = None
        self.norm_W = None
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.dim = 0
        self.load_model()

    def load_model(self):
        """加载模型权重与词表，并预先对词矩阵进行 L2 归一化"""
        if not os.path.exists(self.weights_path):
            if os.path.exists(ALT_WEIGHTS_PATH):
                print(f"[*] 提示: 默认权重不存在，自动切换至轻量权重: {ALT_WEIGHTS_PATH}")
                self.weights_path = ALT_WEIGHTS_PATH
            else:
                raise FileNotFoundError(f"未找到词向量权重文件: {self.weights_path}，请先运行 train.py 或 quick_demo.py 进行训练！")

        print(f"[*] 正在载入词向量模型: {os.path.basename(self.weights_path)} ...")
        with open(self.weights_path, "rb") as f:
            data = pickle.load(f)

        self.W = data["W_in"]
        self.word_to_id = data["word_to_id"]
        self.id_to_word = data["id_to_word"]
        self.vocab_size, self.dim = self.W.shape

        # 预计算单位 L2 范数归一化矩阵，加速后续所有余弦相似度点乘运算
        eps = 1e-8
        norms = np.sqrt(np.sum(self.W ** 2, axis=1, keepdims=True)) + eps
        self.norm_W = self.W / norms
        print(f"[OK] 成功载入! 词汇表容量: {self.vocab_size:,} 词 | 向量维度: {self.dim} 维\n")

    def find_close_matches(self, word: str, max_matches: int = 4) -> List[str]:
        """为未登录词提供相似前缀或子串建议"""
        matches = []
        for w in self.word_to_id.keys():
            if word in w or w in word:
                matches.append(w)
                if len(matches) >= max_matches:
                    break
        return matches

    def parse_expression(self, expr: str) -> Tuple[List[str], List[str]]:
        """
        解析代数表达式为正负词列表
        例如: "king - man + woman" -> 正词: ["king", "woman"], 负词: ["man"]
        """
        expr = expr.strip().lower()
        # 正则匹配形如 "+ word", "- word", "word"
        tokens = re.findall(r'([+-]?)\s*([a-zA-Z0-9_\'\.<>]+)', expr)

        positive_words = []
        negative_words = []

        for sign, word in tokens:
            if sign == "-":
                negative_words.append(word)
            else:
                positive_words.append(word)

        return positive_words, negative_words

    def evaluate(self, expr: str, top_n: int = 5) -> Optional[List[Tuple[str, float]]]:
        """
        执行词向量代数运算并在全词表中检索 Top-N 最相似词
        """
        pos_words, neg_words = self.parse_expression(expr)
        if not pos_words and not neg_words:
            print("[!] 错误: 输入表达式为空或格式无法识别。")
            return None

        # 检查单词是否存在于词表中
        all_words = pos_words + neg_words
        missing_words = [w for w in all_words if w not in self.word_to_id]
        if missing_words:
            for mw in missing_words:
                suggestions = self.find_close_matches(mw)
                sugg_str = f" (您是否想输入: {', '.join(suggestions)} ?)" if suggestions else ""
                print(f"[!] 词典中未找到单词: '{mw}'{sugg_str}")
            return None

        # 按照 Mikolov 论文推荐的标准方式进行单位向量加减:
        # 每个词先自身做 L2 归一化，再按符号加减，最后对总向量再做一次归一化
        target_vec = np.zeros(self.dim, dtype=np.float32)

        formula_parts = []
        for w in pos_words:
            idx = self.word_to_id[w]
            target_vec += self.norm_W[idx]
            formula_parts.append(f"+ '{w}'")
        for w in neg_words:
            idx = self.word_to_id[w]
            target_vec -= self.norm_W[idx]
            formula_parts.append(f"- '{w}'")

        formula_repr = " ".join(formula_parts)
        if formula_repr.startswith("+ "):
            formula_repr = formula_repr[2:]

        # 目标向量单位归一化
        target_norm = np.linalg.norm(target_vec)
        if target_norm < 1e-8:
            print("[!] 错误: 运算结果向量范数接近于 0，无法计算方向。")
            return None
        target_vec = target_vec / target_norm

        # 向量化点乘全词表矩阵，耗时 O(V)
        sim_scores = np.dot(self.norm_W, target_vec)

        # 降序排序
        sorted_indices = np.argsort(-sim_scores)

        # 排除输入算式中包含的原词，提取 Top-N 结果
        excluded_ids = {self.word_to_id[w] for w in all_words}
        results = []
        for idx in sorted_indices:
            if idx in excluded_ids:
                continue

            word = self.id_to_word[idx]
            score = float(sim_scores[idx])
            results.append((word, score))
            if len(results) >= top_n:
                break

        return results


def print_results(expr: str, results: List[Tuple[str, float]]):
    """打印漂亮的带百分比和 ASCII 进度条的结果面板"""
    print("\n" + "-" * 70)
    print(f"  [*] 算式求解: {expr}")
    print("-" * 70)
    if not results:
        print("  未检索到结果。")
        return

    for rank, (word, score) in enumerate(results, start=1):
        # 构造 ASCII 相似度置信条
        bar_len = 22
        # 将 -1~1 的分数映射为 0~1 长度
        norm_val = max(0.0, min(1.0, (score + 1.0) / 2.0 if score < 0 else score))
        filled = int(norm_val * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"  Rank {rank:2d} | 预测词: {word:<16s} | 相似度: {score:+.4f} [{bar}]")
    print("-" * 70 + "\n")


def interactive_repl(calc: WordVectorCalculator):
    """交互式命令行循环"""
    banner = r"""
=============================================================================
  __          __           _  __  __       _   _     
  \ \        / /          | ||  \/  |     | | | |    
   \ \  /\  / /___  _ __ _| || \  / | __ _| |_| |__  
    \ \/  \/ // _ \| '__/ _` || |\/| |/ _` | __| '_ \ 
     \  /\  /| (_) | | | (_| || |  | | (_| | |_| | | |
      \/  \/  \___/|_|  \__,_||_|  |_|\__,_|\__|_| |_|
            Word2Vec 交互式语义代数计算器 (Word Vector Calculator)
=============================================================================
 指令说明:
   • 经典类比推理 : 输入 'king - man + woman' 或 'took - take + go'
   • 概念叠加合成 : 输入 'bank + company' 或 'market + stock'
   • 单词近邻查询 : 输入单个词如 'queen' 或 'doctor'
   • 模型即时切换 : 输入 'switch' 自由在 [通用百科模型] 与 [PTB财经模型] 间切换
   • 快捷命令     : 输入 'demo' 运行经典案例 | 输入 'q' 或 'exit' 退出
=============================================================================
"""
    print(banner)

    demo_cases_pretrained = [
        "king - man + woman",
        "paris - france + japan",
        "took - take + go",
        "his - he + she",
        "bigger - big + small",
    ]
    demo_cases_ptb = [
        "stock",
        "bank",
        "company",
        "market",
    ]

    while True:
        model_name = os.path.basename(calc.weights_path)
        tag = "通用百科" if "glove" in model_name or "pretrain" in model_name else "PTB财经"
        prompt_str = f"VectorCalc [{tag}] >>> "

        try:
            user_input = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出计算器，欢迎再次使用！")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit", "q"):
            print("再见！感谢使用 Word2Vec 词向量计算器。")
            break

        if cmd in ("switch", "model", "toggle"):
            # 切换模型
            if "glove" in model_name or "pretrain" in model_name:
                target_path = PTB_WEIGHTS_PATH
            else:
                target_path = PRETRAINED_WEIGHTS_PATH

            if not os.path.exists(target_path):
                print(f"[!] 切换失败: 未找到目标权重文件: {target_path}")
                continue

            print(f"\n[*] 正在切换至新模型: {os.path.basename(target_path)} ...")
            calc.weights_path = target_path
            calc.load_model()
            continue

        if cmd in ("help", "h", "?"):
            print("\n支持格式示例:")
            print("  1. 词类比: a - b + c (例如: king - man + woman)")
            print("  2. 词融合: word1 + word2 (例如: computer + science)")
            print("  3. 查同义: word (例如: queen)")
            print("  4. switch: 随时切换当前底层模型 ([通用百科] 或 [PTB财经])")
            print("  5. demo  : 运行当前模型的一组经典案例\n")
            continue

        if cmd == "demo":
            print("\n--- 正在批量运行预设经典语义演示 ---")
            cases = demo_cases_pretrained if "glove" in model_name or "pretrain" in model_name else demo_cases_ptb
            for demo_expr in cases:
                res = calc.evaluate(demo_expr, top_n=3)
                if res:
                    print_results(demo_expr, res)
            continue

        # 执行用户自定义算式
        results = calc.evaluate(user_input, top_n=5)
        if results:
            print_results(user_input, results)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Word2Vec 交互式词向量计算器")
    parser.add_argument("expr", nargs="*", type=str, default=None,
                        help="可选: 直接传入要求值的算式，例如 'he - his + she'")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH,
                        help="模型权重文件路径 (.pkl)")
    parser.add_argument("--top_n", type=int, default=5,
                        help="返回相似词数量 (默认 5)")
    args = parser.parse_args()

    calc = WordVectorCalculator(weights_path=args.weights)

    if args.expr:
        query_expr = " ".join(args.expr)
        results = calc.evaluate(query_expr, top_n=args.top_n)
        if results:
            print_results(query_expr, results)
    else:
        interactive_repl(calc)


if __name__ == "__main__":
    main()
