# -*- coding: utf-8 -*-
"""
WhiteBox Transformer 基础数学算子库
纯 Python 原生数学库实现，保证零第三方环境依赖，具备高度透明性与数值稳定性。
"""
import math
from typing import List, Optional, Tuple

Matrix = List[List[float]]
Vector = List[float]

def matmul(A: Matrix, B: Matrix) -> Matrix:
    """计算矩阵乘法 C = A @ B (A: M x K, B: K x N -> C: M x N)"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, f"矩阵维度不匹配，无法相乘: ({rows_A}x{cols_A}) 与 ({rows_B}x{cols_B})"
    
    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return C

def mat_transpose(A: Matrix) -> Matrix:
    """矩阵转置 A^T"""
    rows, cols = len(A), len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]

def mat_add(A: Matrix, B: Matrix) -> Matrix:
    """矩阵逐元素相加 A + B"""
    rows, cols = len(A), len(A[0])
    return [[A[i][j] + B[i][j] for j in range(cols)] for i in range(rows)]

def mat_scale(A: Matrix, factor: float) -> Matrix:
    """矩阵标量缩放 factor * A"""
    return [[val * factor for val in row] for row in A]

def softmax_row(row: Vector, mask_row: Optional[List[bool]] = None) -> Vector:
    """
    带可选掩码的数值稳定 Softmax
    mask_row: 若为 True，表示该位置被 Mask 掩蔽（置为 -无穷大，权重为 0）
    """
    # 过滤被掩盖的位置
    effective_vals = []
    for idx, val in enumerate(row):
        if mask_row and mask_row[idx]:
            continue
        effective_vals.append(val)
    
    if not effective_vals:
        return [1.0 / len(row) for _ in row]
    
    max_val = max(effective_vals)
    exp_vals = []
    for idx, val in enumerate(row):
        if mask_row and mask_row[idx]:
            exp_vals.append(0.0)
        else:
            exp_vals.append(math.exp(val - max_val))
            
    sum_exp = sum(exp_vals)
    if sum_exp == 0.0:
        return [0.0 for _ in row]
    return [x / sum_exp for x in exp_vals]

def softmax_matrix(mat: Matrix, mask: Optional[List[List[bool]]] = None) -> Matrix:
    """对矩阵的每一行进行 Softmax 归一化（支持行级 Mask）"""
    res = []
    for i, row in enumerate(mat):
        mask_row = mask[i] if mask else None
        res.append(softmax_row(row, mask_row))
    return res

def create_causal_mask(seq_len: int) -> List[List[bool]]:
    """
    生成因果自注意力下三角掩码 (Causal Mask / Look-ahead Mask)
    True 表示被掩盖 (未来 Token，不可见)
    False 表示可见 (当前及过去 Token)
    示例 (seq_len = 3):
    [
      [False, True,  True ],  # Token 0 只能看 Token 0
      [False, False, True ],  # Token 1 能看 0, 1
      [False, False, False]   # Token 2 能看 0, 1, 2
    ]
    """
    mask = []
    for i in range(seq_len):
        row = [j > i for j in range(seq_len)]
        mask.append(row)
    return mask

def layer_norm(X: Matrix, eps: float = 1e-5, 
               gamma: Optional[Vector] = None, 
               beta: Optional[Vector] = None) -> Tuple[Matrix, List[Tuple[float, float]]]:
    """
    层归一化 (Layer Normalization)
    对每个样本（每个 Token 向量）沿特征维度计算均值和方差，并标准化。
    返回: (标准化后的矩阵, 每行的 (均值, 方差) 列表)
    """
    d_model = len(X[0])
    normed = []
    stats = []
    
    for row in X:
        mean = sum(row) / d_model
        var = sum((x - mean) ** 2 for x in row) / d_model
        std = math.sqrt(var + eps)
        
        norm_row = []
        for j in range(d_model):
            val = (row[j] - mean) / std
            g = gamma[j] if gamma else 1.0
            b = beta[j] if beta else 0.0
            norm_row.append(val * g + b)
            
        normed.append(norm_row)
        stats.append((mean, var))
        
    return normed, stats

def relu_matrix(X: Matrix) -> Matrix:
    """ReLU 激活函数 max(0, x)"""
    return [[max(0.0, val) for val in row] for row in X]
