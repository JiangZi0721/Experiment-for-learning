"""
=============================================================================
模块名称: eval.py
核心功能: 词向量评估与语义相似度分析工具
包含功能:
    1. cos_similarity : 计算两个向量的余弦相似度
    2. most_similar    : 检索与目标词在语义几何空间中最相似的 Top-N 单词
    3. analogy         : 经典词类比推理计算 (a 之于 b 犹如 c 之于 ?)
=============================================================================
理论重点:
1. 余弦相似度 (Cosine Similarity):
   - 度量两个向量在方向上的夹角余弦值，范围在 [-1, 1] 之间:
         cos_sim(u, v) = (u · v) / (||u||_2 * ||v||_2)
   - 越接近 1 表示两词在语料上下文中的语义与语法角色越相似。

2. 词类比推理 (Word Analogy):
   - Mikolov 2013 论文中最引人注目的发现: Word2Vec 词向量具有线性的语义代数运算特性。
   - 典型例子:
     vec("king") - vec("man") + vec("woman") ≈ vec("queen")
     vec("tokyo") - vec("japan") + vec("france") ≈ vec("paris")
   - 在高维空间中，不同词汇之间的语法关系 (如单复数、时态) 与语义关系 (如首都-国家、性别)
     被映射成了近乎平行的位移向量。
=============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def cos_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    """
    计算两个一维向量之间的余弦相似度

    参数:
        x (np.ndarray): 向量 1，形状 (H,)
        y (np.ndarray): 向量 2，形状 (H,)
        eps (float): 防止除以 0 的极小常数

    返回:
        sim (float): 余弦相似度值 (-1.0 ~ 1.0)
    """
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return float(np.dot(nx, ny))


def most_similar(
    query: str,
    word_to_id: Dict[str, int],
    id_to_word: Dict[int, str],
    word_matrix: np.ndarray,
    top: int = 5
) -> List[Tuple[str, float]]:
    """
    检索与指定词最相似的前 top 个单词并打印

    参数:
        query (str): 查询词 (例如 "bank" 或 "market")
        word_to_id (dict): 词到 ID 的映射表
        id_to_word (dict): ID 到词的映射表
        word_matrix (np.ndarray): 词向量矩阵，形状 (V, H)
        top (int): 返回前 top 个相似词

    返回:
        results (List[Tuple[str, float]]): [(词, 相似度得分), ...]
    """
    if query not in word_to_id:
        print(f"[Eval] 提示: 单词 '{query}' 不在当前词汇表中！")
        return []

    print(f"\n[Eval] === 检索与 '{query}' 最相近的单词 (Top {top}) ===")
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 对全词表进行 L2 归一化: (V, H) / (V, 1)
    eps = 1e-8
    norms = np.sqrt(np.sum(word_matrix ** 2, axis=1, keepdims=True)) + eps
    normalized_matrix = word_matrix / norms

    # 对查询向量进行 L2 归一化: (H,)
    query_norm = query_vec / (np.sqrt(np.sum(query_vec ** 2)) + eps)

    # 向量化矩阵-向量点积得到与全词表的余弦相似度: 形状 (V,)
    similarity = np.dot(normalized_matrix, query_norm)

    # 相似度降序排序
    sorted_indices = np.argsort(-similarity)

    results = []
    count = 0
    for idx in sorted_indices:
        if idx == query_id:
            continue  # 跳过查询词自身

        word = id_to_word[idx]
        score = float(similarity[idx])
        results.append((word, score))
        print(f"  Rank {count + 1:2d} | 词: {word:<15s} | 余弦相似度: {score:.4f}")

        count += 1
        if count >= top:
            break

    return results


def analogy(
    a: str,
    b: str,
    c: str,
    word_to_id: Dict[str, int],
    id_to_word: Dict[int, str],
    word_matrix: np.ndarray,
    top: int = 5
) -> List[Tuple[str, float]]:
    """
    词类比推理计算: a 之于 b，犹如 c 之于 ?
    向量运算: vec(d) = vec(b) - vec(a) + vec(c)

    典型案例:
        - man : king  ->  woman : queen
        - japan : tokyo  ->  france : paris
        - take : took  ->  go : went

    参数:
        a (str): 基础概念 A
        b (str): 目标属性 B
        c (str): 类比概念 C
        word_to_id (dict): 词到 ID
        id_to_word (dict): ID 到词
        word_matrix (np.ndarray): 词向量矩阵
        top (int): 返回前 top 个类比词

    返回:
        results (List[Tuple[str, float]]): [(预测词, 相似度得分), ...]
    """
    for word in [a, b, c]:
        if word not in word_to_id:
            print(f"[Eval] 提示: 单词 '{word}' 不在词表中，无法进行类比推理！")
            return []

    print(f"\n[Eval] === 类比推理: [{a}] 之于 [{b}] 犹如 [{c}] 之于 [?] ===")
    print(f"       向量运算: vec(?) ≈ vec({b}) - vec({a}) + vec({c})")

    # 提取对应向量
    vec_a = word_matrix[word_to_id[a]]
    vec_b = word_matrix[word_to_id[b]]
    vec_c = word_matrix[word_to_id[c]]

    # 目标合成向量: target_vec = b - a + c
    target_vec = vec_b - vec_a + vec_c

    # 归一化
    eps = 1e-8
    target_norm = target_vec / (np.sqrt(np.sum(target_vec ** 2)) + eps)
    norms = np.sqrt(np.sum(word_matrix ** 2, axis=1, keepdims=True)) + eps
    normalized_matrix = word_matrix / norms

    # 计算与所有词的余弦相似度
    similarity = np.dot(normalized_matrix, target_norm)

    # 降序排序
    sorted_indices = np.argsort(-similarity)

    # 排除输入词自身
    exclude_ids = {word_to_id[a], word_to_id[b], word_to_id[c]}

    results = []
    count = 0
    for idx in sorted_indices:
        if idx == exclude_ids:
            continue

        word = id_to_word[idx]
        score = float(similarity[idx])
        results.append((word, score))
        print(f"  Rank {count + 1:2d} | 候选词: {word:<15s} | 相似度: {score:.4f}")

        count += 1
        if count >= top:
            break

    return results
