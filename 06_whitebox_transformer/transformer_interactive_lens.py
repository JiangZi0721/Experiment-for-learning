# -*- coding: utf-8 -*-
"""
================================================================================
               Transformer 架构“白盒透视”交互式学习系统
                        (Transformer Interactive Lens)
================================================================================
本程序专为深入透视 Transformer 内部数学原理与张量流转而设计。
特点：
1. 零第三方依赖：纯 Python 原生数学库实现，无需额外 pip 安装，直接运行。
2. 全流程白盒化：每个环节提供直观原理讲解、数学公式，并支持主动提示用户
   展开查看真实的微型数值矩阵与具体每一步的推导过程。
3. 自由交互模式：支持按序单步通关体验，也支持章节式自由跳转。
================================================================================
"""

import math
import sys

# 格式化打印辅助函数
def print_banner(title):
    print("\n" + "=" * 76)
    print(f"  📌 {title}")
    print("=" * 76)

def print_section(title):
    print(f"\n--- 🔹 {title} ---")

def format_vector(vec, precision=4):
    return "[" + ", ".join([f"{x:+.{precision}f}" for x in vec]) + "]"

def print_matrix(mat, row_labels=None, col_labels=None, precision=4, name=""):
    if name:
        print(f"\n【矩阵: {name}】 维度: ({len(mat)} 行 x {len(mat[0])} 列)")
    if col_labels:
        col_header = "       " + "   ".join([f"{col[:8]:>8}" for col in col_labels])
        print(col_header)
        print("       " + "-" * (len(col_labels) * 11))
    
    for idx, row in enumerate(mat):
        label = f"{row_labels[idx][:6]:>6}: " if row_labels else f"Row {idx}: "
        row_str = "  ".join([f"{x:+.{precision}f}" for x in row])
        print(f"{label}[ {row_str} ]")

def ask_expand(step_name="这其中的具体计算过程与数值矩阵"):
    while True:
        choice = input(f"\n👉 是否展开查看【{step_name}】？(y: 展开查看 / n: 跳过查看公式 / q: 返回主菜单): ").strip().lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no', '']:
            return False
        elif choice in ['q', 'quit']:
            return 'quit'
        else:
            print("请输入 y (是), n (否) 或 q (退出到主菜单)。")

def pause_step():
    input("\n[按 Enter 键继续推进到下一步...]")

# ----------------- 基础矩阵数学运算 (纯原生实现，保证零环境依赖) -----------------
def matmul(A, B):
    """计算矩阵 A (M x K) 与 B (K x N) 的乘积"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, f"矩阵维度不匹配: ({rows_A}x{cols_A}) 与 ({rows_B}x{cols_B})"
    
    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return C

def mat_transpose(A):
    """矩阵转置"""
    rows, cols = len(A), len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]

def mat_add(A, B):
    """矩阵逐元素相加"""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def softmax_row(row):
    """数值稳定的 Softmax"""
    max_val = max(row)
    exp_vals = [math.exp(x - max_val) for x in row]
    sum_exp = sum(exp_vals)
    return [x / sum_exp for x in exp_vals]

def softmax_matrix(mat):
    """对矩阵的每一行进行 Softmax 归一化"""
    return [softmax_row(row) for row in mat]

# ==============================================================================
#                               微型玩具模型定义
# ==============================================================================
# 示例句子：4 个 Token
SAMPLE_TOKENS = ["我", "喜欢", "机器", "学习"]
N_TOKENS = len(SAMPLE_TOKENS)  # N = 4
D_MODEL = 4                   # D = 4 (便于全览数值)
N_HEADS = 2                   # h = 2
D_K = D_MODEL // N_HEADS      # d_k = 2

# 静态词嵌入表 (4 x 4)
EMBEDDING_TABLE = [
    [ 0.82, -0.41,  0.15,  0.93],  # 我
    [ 0.12,  0.95, -0.62,  0.31],  # 喜欢
    [-0.75,  0.22,  0.88, -0.45],  # 机器
    [-0.68,  0.18,  0.92, -0.38],  # 学习
]

# 权重矩阵定义 (用于投影 Q, K, V)
W_Q = [
    [ 0.5, -0.2,  0.3,  0.1],
    [-0.1,  0.4,  0.2, -0.3],
    [ 0.3,  0.1, -0.4,  0.2],
    [ 0.2, -0.3,  0.1,  0.5]
]

W_K = [
    [ 0.4,  0.1, -0.2,  0.3],
    [-0.2,  0.5,  0.1, -0.1],
    [ 0.1, -0.3,  0.4,  0.2],
    [ 0.3,  0.2, -0.1,  0.4]
]

W_V = [
    [ 0.2,  0.4, -0.1,  0.3],
    [ 0.3, -0.2,  0.4,  0.1],
    [-0.1,  0.3,  0.2, -0.4],
    [ 0.4,  0.1, -0.3,  0.2]
]

W_O = [
    [ 0.4, -0.1,  0.2,  0.3],
    [ 0.1,  0.5, -0.2,  0.1],
    [-0.2,  0.1,  0.4, -0.1],
    [ 0.3, -0.3,  0.1,  0.4]
]


# ==============================================================================
#                             各阶段透视流程实现
# ==============================================================================

def chapter_1_embedding_and_pe():
    print_banner("第一章：输入预处理层 —— 词嵌入 (Embedding) 与位置编码 (Positional Encoding)")
    
    print("""
【1.1 词嵌入 (Input Embedding) 的本质】
* 计算机无法直接理解字符文本，需要将其映射为稠密连续的特征向量。
* 输入句子由 N 个 Token 组成：['我', '喜欢', '机器', '学习'] (N=4)。
* 每个 Token 首先转化为整数 ID，再通过查找表转换为 D 维向量 (此处设 D=4)。
* ⚠️ 关键特性：此阶段的词向量【完全没有位置顺序信息】！
  如果把句子打乱成 ['学习', '机器', '喜欢', '我']，它们对应的词向量集合是完全一模一样的。
  这种性质称为“置换不变性 (Permutation Invariance)”。
""")
    
    print("""
【1.2 位置编码方案辨析与纠偏】
💡 用户敏锐的直觉探索：
   “能否用 One-Hot 独热编码表示位置，将词嵌入与 One-Hot 拼接，再乘以权重矩阵降维？”
👉 纠偏与数学深度透视：
   1. 您的直觉在数学上是【完全走得通且非常巧妙的】！
      因为：[词向量, OneHot] @ [W_token; W_pos] = 词向量 @ W_token + OneHot @ W_pos。
      OneHot 乘以矩阵本质上就是在查找“第 pos 个可学习位置向量”！
   2. 但在 Vaswani 原论文的标准 Transformer 中，采用的是【直接逐元素相加】：
      X = Token_Embedding * sqrt(D) + Positional_Encoding
   3. 为什么是相加而不是拼接？
      * 避免维度翻倍：拼接会使输入维度变成 D + N，后续各层参数量与计算量显著增加。
      * 高维几何正交性：在 D=512（甚至 D=64）的高维空间中，特征空间极其空旷且近似正交。
        相加并不会混淆词义和位置，后续的投影矩阵（W_Q, W_K）能非常自如地将语义特征和位置特征解耦！
   4. 原版正弦余弦位置编码公式：
      PE(pos, 2i)   = sin( pos / (10000^(2i / D)) )
      PE(pos, 2i+1) = cos( pos / (10000^(2i / D)) )
      好处：无需任何训练参数、天然支持超长外推、三角和角公式让相对位置变换成为简单的线性旋转。
""")

    res = ask_expand("词嵌入矩阵、正弦位置编码矩阵与逐元素相加后的输入矩阵 X")
    if res == 'quit': return None
    if res:
        print_matrix(EMBEDDING_TABLE, row_labels=SAMPLE_TOKENS, col_labels=[f"dim_{i}" for i in range(D_MODEL)], name="词嵌入矩阵 E (Embedding)")
        
        # 计算正弦位置编码
        pe_matrix = []
        for pos in range(N_TOKENS):
            row = []
            for i in range(D_MODEL // 2):
                denom = 10000.0 ** ((2 * i) / D_MODEL)
                row.append(math.sin(pos / denom))
                row.append(math.cos(pos / denom))
            pe_matrix.append(row)
        
        print_matrix(pe_matrix, row_labels=[f"Pos_{i}" for i in range(N_TOKENS)], col_labels=[f"dim_{i}" for i in range(D_MODEL)], name="位置编码矩阵 PE (Positional Encoding)")
        
        # 相加 (此处按教学简化，不乘 sqrt(D)，直接直观呈现加和)
        X_input = mat_add(EMBEDDING_TABLE, pe_matrix)
        print_matrix(X_input, row_labels=SAMPLE_TOKENS, col_labels=[f"dim_{i}" for i in range(D_MODEL)], name="预处理最终输出矩阵 X = E + PE")
        print("\n💡 观察：每个词向量此时都深深烙上了自己所在位置的特征印记！")
    else:
        # 直接计算供后续使用
        pe_matrix = []
        for pos in range(N_TOKENS):
            row = []
            for i in range(D_MODEL // 2):
                denom = 10000.0 ** ((2 * i) / D_MODEL)
                row.append(math.sin(pos / denom))
                row.append(math.cos(pos / denom))
            pe_matrix.append(row)
        X_input = mat_add(EMBEDDING_TABLE, pe_matrix)

    pause_step()
    return X_input


def chapter_2_qkv_projection(X_input):
    print_banner("第二章：核心角色诞生 —— Q (Query)、K (Key)、V (Value) 线性投影")
    
    print("""
【2.1 为什么要将输入 X 投影为三个矩阵？】
如果让词向量之间直接做点积，就好比让一个人的全部资产直接和别人的全部资产比拼，缺乏视角与角色。
Transformer 引入了三组不同的线性投影，赋予每个 Token 三个截然不同的职能：
1. 🔍 Query (q_i)：当前词“想寻找什么样的上下文信息”（发出检索请求的搜索词）。
2. 🏷️  Key (k_j)：当前词“对外展示的特征标签”（供别人检索匹配的关键词）。
3. 📦 Value (v_j)：当前词“真正蕴含并准备贡献给全局的语义信息”（实际携带的内容）。

【2.2 数学投影公式】
输入矩阵 X 维度为 (N x D)，与三个权重矩阵相乘：
* Q = X @ W_Q    (维度: N x D)
* K = X @ W_K    (维度: N x D)
* V = X @ W_V    (维度: N x D)
其中每一行 q_i, k_i, v_i 分别代表第 i 个 Token 对应的 Query、Key 和 Value 向量。
""")

    res = ask_expand("W_Q, W_K, W_V 权重矩阵与投影生成的 Q, K, V 真实数值")
    if res == 'quit': return None
    
    Q = matmul(X_input, W_Q)
    K = matmul(X_input, W_K)
    V = matmul(X_input, W_V)
    
    if res:
        print_matrix(W_Q, name="投影权重 W_Q (4x4)")
        print_matrix(Q, row_labels=[f"q({t})" for t in SAMPLE_TOKENS], name="Query 矩阵 Q = X @ W_Q (4x4)")
        print_matrix(K, row_labels=[f"k({t})" for t in SAMPLE_TOKENS], name="Key 矩阵 K = X @ W_K (4x4)")
        print_matrix(V, row_labels=[f"v({t})" for t in SAMPLE_TOKENS], name="Value 矩阵 V = X @ W_V (4x4)")
        print("\n💡 观察：以第 1 个词“我”为例，它产生了三个专属向量：q_我, k_我, v_我。")

    pause_step()
    return Q, K, V


def chapter_3_scaled_dot_product(Q, K, V):
    print_banner("第三章：灵魂算法 —— 缩放点积注意力 (Scaled Dot-Product Attention) 四步透视")
    
    print("""
【3.1 为什么点积可以衡量相关性？】
向量的点积公式为：a · b = |a| * |b| * cos(θ)。
当两个向量方向越一致（夹角 θ 越小），点积数值越大。
因此，用 q_i 与 k_j 做点积，能够精准评估：“第 i 个词想找的信息，与第 j 个词的特征匹配度有多高”。

【3.2 深度追问：为什么必须除以 sqrt(d_k)？（李宏毅教授核心考点）】
设 q 和 k 的每个分量均值为 0，方差为 1 的独立随机变量：
* 两个维度为 d_k 的向量点积：q · k = sum_{m=1}^{d_k} (q_m * k_m)
* 每一项 q_m * k_m 的方差为 1，共 d_k 项相加，点积总方差直接膨胀为 d_k（标准差为 sqrt(d_k)）！
* 当维度 d_k 较大时，点积数值的绝对值会变得极大（例如几十）。
* 这会导致 Softmax 函数进入【饱和两极区】—— 其输出要么极度趋近 1，要么趋近 0。
* 在饱和区，Softmax 的导数几乎为 0，反向传播将遭遇毁灭性的【梯度消失 (Gradient Vanishing)】！
* 因此，除以 sqrt(d_k) 能将方差重新稳定拉回到 1，让梯度落在最敏感的黄金学习区域！
""")

    res = ask_expand("单行点积拆解、完整得分矩阵、Softmax 归一化与加权求和全过程")
    if res == 'quit': return None
    
    # 1. 计算点积矩阵 Scores = Q @ K^T
    K_T = mat_transpose(K)
    raw_scores = matmul(Q, K_T)
    
    # 2. 尺度缩放
    scale_factor = math.sqrt(D_MODEL)  # 此处单头全维示例 D=4, sqrt(4)=2.0
    scaled_scores = [[val / scale_factor for val in row] for row in raw_scores]
    
    # 3. Softmax 归一化
    attention_weights = softmax_matrix(scaled_scores)
    
    # 4. 加权聚合上下文 B = Weights @ V
    context_B = matmul(attention_weights, V)

    if res:
        print_section("步骤 1：第一行详细单步推演 —— 以“我”的 q1 为例")
        q1 = Q[0]
        print(f"q_我: {format_vector(q1)}")
        for j in range(N_TOKENS):
            kj = K[j]
            dot_product = sum(q1[m] * kj[m] for m in range(D_MODEL))
            scaled = dot_product / scale_factor
            print(f"  -> 与 k_{SAMPLE_TOKENS[j]:<2} 点积: {dot_product:+.4f}  |  除以 sqrt({D_MODEL}) 缩放后: {scaled:+.4f}")
        
        print_section("步骤 2：全序列缩放后得分矩阵 (N x N = 4 x 4)")
        print_matrix(scaled_scores, row_labels=SAMPLE_TOKENS, col_labels=SAMPLE_TOKENS, name="Scaled Scores (Q @ K^T / sqrt(d))")
        
        print_section("步骤 3：Softmax 归一化得到注意力权重矩阵 (Attention Map)")
        print_matrix(attention_weights, row_labels=SAMPLE_TOKENS, col_labels=SAMPLE_TOKENS, name="Attention Weights (每行之和为 1.0)")
        print("💡 解读：查看每一行，数值越高代表该词将多少注意力倾注到了对应的词身上！")
        
        print_section("步骤 4：与 Value 矩阵加权求和，输出上下文向量矩阵 B")
        print_matrix(context_B, row_labels=[f"b({t})" for t in SAMPLE_TOKENS], col_labels=[f"dim_{i}" for i in range(D_MODEL)], name="上下文矩阵 B = Attention @ V (4 x 4)")
        print("\n💡 此时的 b_我 已经不再是单纯孤立的“我”，而是融合了全句所有相关词信息的全局表征！")

    pause_step()
    return context_B


def chapter_4_multi_head_attention(X_input):
    print_banner("第四章：拓宽视野 —— 多头注意力机制 (Multi-Head Attention) 的子空间智慧")
    
    print("""
【4.1 多头注意力到底在干什么？】
* 如果只有“一个头”，注意力只能在单一维度视角下建立关联（例如当前头可能只被强行训练去关注邻近主谓搭配）。
* 人类语言极其复杂：一个词同时存在语法依存关系、远距离代词指代、同义反义、情感逻辑等多种视角。
* 多头机制（Multi-Head Attention, MHA）将维度 D 拆分成 h 个互不干扰的子空间 (每个子空间维度 d_k = D / h)：
  - Head 1 可以在第 1 个子空间中专门捕获【局部短距离修饰】；
  - Head 2 可以在第 2 个子空间中专门捕获【远距离逻辑依赖】。

【4.2 计算量有增加吗？】
* 答案是：【几乎没有增加】！
* 假设原本单头计算复杂度是 O(N^2 * D)。
* 分成 h 个头后，每个头维度为 D/h，单个头计算量为 O(N^2 * (D/h))。
* h 个头的总计算量：h * O(N^2 * (D/h)) = O(N^2 * D)，总浮点运算量 (FLOPs) 完全等价！
* 最终将 h 个头的输出横向拼接 (Concat)，再乘以一个输出投影矩阵 W_O (D x D)，完成多视角信息融合。
""")

    res = ask_expand(f"拆分 {N_HEADS} 个头、独立计算注意力图、横向拼接与 W_O 融合全过程")
    if res == 'quit': return None

    # 多头具体推演：将 Q, K, V 在特征维度均分给 2 个头 (每个头维度 2)
    Q = matmul(X_input, W_Q)
    K = matmul(X_input, W_K)
    V = matmul(X_input, W_V)
    
    # 拆分给 Head 1 (前 2 维) 和 Head 2 (后 2 维)
    Q_h1 = [row[:D_K] for row in Q]
    K_h1 = [row[:D_K] for row in K]
    V_h1 = [row[:D_K] for row in V]

    Q_h2 = [row[D_K:] for row in Q]
    K_h2 = [row[D_K:] for row in K]
    V_h2 = [row[D_K:] for row in V]

    scale_h = math.sqrt(D_K)
    
    # Head 1 计算
    scores_h1 = matmul(Q_h1, mat_transpose(K_h1))
    scaled_h1 = [[v / scale_h for v in r] for r in scores_h1]
    attn_h1 = softmax_matrix(scaled_h1)
    out_h1 = matmul(attn_h1, V_h1)

    # Head 2 计算
    scores_h2 = matmul(Q_h2, mat_transpose(K_h2))
    scaled_h2 = [[v / scale_h for v in r] for r in scores_h2]
    attn_h2 = softmax_matrix(scaled_h2)
    out_h2 = matmul(attn_h2, V_h2)

    # 拼接 Concat (4 x 4)
    concat_out = [out_h1[i] + out_h2[i] for i in range(N_TOKENS)]
    
    # 经过最终线性层 W_O
    mha_final = matmul(concat_out, W_O)

    if res:
        print_section(f"Head 1 注意力图 (子空间维度 d_k={D_K})")
        print_matrix(attn_h1, row_labels=SAMPLE_TOKENS, col_labels=SAMPLE_TOKENS, name="Head 1 Attention Map")
        
        print_section(f"Head 2 注意力图 (子空间维度 d_k={D_K})")
        print_matrix(attn_h2, row_labels=SAMPLE_TOKENS, col_labels=SAMPLE_TOKENS, name="Head 2 Attention Map")
        print("💡 对比观察：Head 1 与 Head 2 对同一个词的关注权重分布完全不同，体现出多视角的特征捕捉！")
        
        print_section("多头拼接 (Concat) 输出矩阵 (4 x 4)")
        print_matrix(concat_out, row_labels=SAMPLE_TOKENS, col_labels=["H1_d0", "H1_d1", "H2_d0", "H2_d1"], name="Concat(Head1, Head2)")
        
        print_section("乘以融合投影矩阵 W_O 后的多头注意力最终输出")
        print_matrix(mha_final, row_labels=SAMPLE_TOKENS, col_labels=[f"dim_{i}" for i in range(D_MODEL)], name="MHA Output = Concat @ W_O")

    pause_step()
    return mha_final


def chapter_5_add_and_norm(X_input, sublayer_output, layer_name="Self-Attention"):
    print_banner("第五章：基石与稳定器 —— 残差连接 (Add) 与层归一化 (LayerNorm)")
    
    print(f"""
【5.1 残差连接 (Add / Residual Connection) 的作用】
* 公式：Output = X + SubLayer(X)
* 为什么需要？
  1. 解决深层网络的退化问题：随着网络堆叠变深，梯度需要逆向传播数十层。
     残差连接提供了一条无衰减的“直连绿色通道”：d(X + F(X))/dX = 1 + dF(X)/dX，梯度永远保底有 1，杜绝梯度消失。
  2. 保持低阶特征：确保高层不仅能学到抽象的注意力模式，还能随时调取原始词的初始特征。

【5.2 层归一化 (LayerNorm) 的作用与对比】
* 为什么要 Norm？
  在深层神经网络中，各层激活值的方差和均值容易随着计算层层漂移（内部协变量偏移 Internal Covariate Shift），导致训练极不稳定。
* 为什么不用 BatchNorm，而坚决使用 LayerNorm？
  - BatchNorm 是“跨样本、沿 Batch 维度”归一化：在 CV 中有效，但在 NLP 中每个句子的长度长短不一，Batch 大小经常变化，导致统计量剧烈抖动。
  - LayerNorm 是“在单个样本内部，沿特征维度 D”归一化：完全独立于 Batch 大小和其他句子，每个 Token 的向量自己计算均值 μ 和方差 σ^2，即使单条推演也极为稳定！
* LayerNorm 计算三部曲：
  1. μ = (1/D) * sum(x_i)
  2. σ^2 = (1/D) * sum((x_i - μ)^2)
  3. y = ((x - μ) / sqrt(σ^2 + ε)) * γ + β  (γ为缩放参数，β为平移偏置，初始 γ=1, β=0)
""")

    res = ask_expand(f"残差加和、逐行均值/方差与归一化标准化后数值")
    if res == 'quit': return None

    # 1. 残差相加
    residual_added = mat_add(X_input, sublayer_output)
    
    # 2. LayerNorm 逐行计算
    eps = 1e-5
    normed = []
    stats = []
    for row in residual_added:
        mean = sum(row) / len(row)
        variance = sum((x - mean) ** 2 for x in row) / len(row)
        std = math.sqrt(variance + eps)
        norm_row = [(x - mean) / std for x in row]  # 默认 gamma=1, beta=0
        normed.append(norm_row)
        stats.append((mean, variance))

    if res:
        print_section(f"步骤 1：残差连接相加 X + {layer_name}(X)")
        print_matrix(residual_added, row_labels=SAMPLE_TOKENS, col_labels=[f"dim_{i}" for i in range(D_MODEL)], name="Residual Added (X + SubLayer)")
        
        print_section("步骤 2：每个 Token 内部统计量 (沿 D=4 维度计算)")
        for idx, (m, v) in enumerate(stats):
            print(f"  Token [{SAMPLE_TOKENS[idx]}]: 均值 μ = {m:+.4f}, 方差 σ^2 = {v:.4f}")
            
        print_section("步骤 3：LayerNorm 输出结果 (均值归 0，方差归 1)")
        print_matrix(normed, row_labels=SAMPLE_TOKENS, col_labels=[f"dim_{i}" for i in range(D_MODEL)], name="LayerNorm Output")
        print("💡 验证：您可以检查每一行，各元素的均值严格为 0，方差严格为 1！激活值重新恢复健康分布。")

    pause_step()
    return normed


def chapter_6_ffn(X_norm):
    print_banner("第六章：逐位置前馈网络 (Position-wise Feed-Forward Network, FFN)")
    
    print("""
【6.1 为什么有了注意力机制，还需要 FFN？】
* 注意力机制本质上只负责【空间信息重组与聚合】（把别的词的信息收集过来），但它全部都是线性加权组合。
* 为了赋予模型强大的【非线性表征与长期记忆存储能力】，必须在注意力层后接一个两层的多层感知机 (MLP)。
* 论文中的经典设计：
  FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
  - 先把特征维度从 D (如 512) 放大 4 倍至 4D (如 2048)；
  - 经过 ReLU / GELU 激活函数激活；
  - 再把维度从 4D 投影压缩回 D (512)。
* “Position-wise (逐位置)”的含义：
  每个 Token 独立通过这套完全相同的两层权重，词与词之间不发生任何横向交流。
""")

    res = ask_expand("两层升维降维变换与最终 Add & LayerNorm 计算")
    if res == 'quit': return None

    # 微型 FFN 权重：升维到 8 (2*D)，再降维回 4
    D_FFN = 8
    W1 = [[0.3 * (i - j) for j in range(D_FFN)] for i in range(D_MODEL)]
    W2 = [[0.2 * (j - i) for j in range(D_MODEL)] for i in range(D_FFN)]

    # 升维与 ReLU
    h1 = matmul(X_norm, W1)
    h1_relu = [[max(0.0, val) for val in row] for row in h1]
    
    # 降维回 D_MODEL
    ffn_out = matmul(h1_relu, W2)

    # 经过 FFN 后的残差连接与 LayerNorm
    final_encoder_layer_out = chapter_5_add_and_norm(X_norm, ffn_out, layer_name="FFN")

    if res:
        print_section(f"步骤 1：第一层投影并升维到 D_ffn={D_FFN}，并经 ReLU 激活")
        print_matrix(h1_relu, row_labels=SAMPLE_TOKENS, name="ReLU(X @ W1)")
        
        print_section("步骤 2：第二层降维回 D=4 维度")
        print_matrix(ffn_out, row_labels=SAMPLE_TOKENS, col_labels=[f"dim_{i}" for i in range(D_MODEL)], name="FFN 最终输出")

    return final_encoder_layer_out


def chapter_7_summary(X_final):
    print_banner("第七章：Encoder 完整生命周期闭环总览")
    
    print("""
🎉 恭喜！您已经完整走完了一个标准 Transformer 编码器层 (Encoder Layer) 的全部微观推演：

输入文本: ['我', '喜欢', '机器', '学习']
    │
    ▼
[1. 词嵌入 (Embedding) + 正弦位置编码 (PE)]   --> 矩阵维度: (4, 4)
    │
    ├───────────────────────┐ (残差高速通道)
    ▼                       │
[2. 多头自注意力 (Multi-Head Attention)]        │
    - 投影得到 Q, K, V       │
    - Scaled Dot-Product: Softmax(Q K^T / sqrt(d)) * V
    - 2 个独立子空间聚合与 W_O 投影
    │                       │
    ▼                       ▼
[3. 残差相加 (Add) & 层归一化 (LayerNorm)]    --> 矩阵维度: (4, 4)
    │
    ├───────────────────────┐ (残差高速通道)
    ▼                       │
[4. 逐位置前馈网络 (FFN: 升维 -> ReLU -> 降维)]│
    │                       │
    ▼                       ▼
[5. 残差相加 (Add) & 层归一化 (LayerNorm)]    --> 最终输出: (4, 4)
    │
    ▼
输出编码向量 (Memory 矩阵，供下一层 Encoder 或 Decoder 交叉注意力调用)
""")
    print_matrix(X_final, row_labels=SAMPLE_TOKENS, col_labels=[f"dim_{i}" for i in range(D_MODEL)], name="当前 Encoder Block 最终输出语义矩阵")
    print("\n💡 核心结论：无论是单层还是堆叠 6 层、12 层，输入形状永远是 (N, D)，输出形状永远是 (N, D)！各层之间无缝平滑串联。")
    pause_step()


def run_full_pipeline():
    X_input = chapter_1_embedding_and_pe()
    if X_input is None: return
    
    qkv = chapter_2_qkv_projection(X_input)
    if qkv is None: return
    Q, K, V = qkv
    
    context_B = chapter_3_scaled_dot_product(Q, K, V)
    if context_B is None: return
    
    mha_out = chapter_4_multi_head_attention(X_input)
    if mha_out is None: return
    
    norm1_out = chapter_5_add_and_norm(X_input, mha_out, layer_name="Self-Attention")
    if norm1_out is None: return
    
    final_out = chapter_6_ffn(norm1_out)
    if final_out is None: return
    
    chapter_7_summary(final_out)


def main():
    while True:
        print("\n" + "=" * 76)
        print("         🧭 Transformer 编码器 (Encoder) 白盒透视学习系统 - 主菜单")
        print("=" * 76)
        print("  [1] 全流程沉浸式推演 (推荐：跟随系统引导逐章步进，每步主动提问是否展开)")
        print("  [2] 单独透视 第一章：词嵌入 (Embedding) 与位置编码 (Positional Encoding)")
        print("  [3] 单独透视 第二章：Q, K, V 投影的物理本质与矩阵生成")
        print("  [4] 单独透视 第三章：缩放点积注意力 (除以 sqrt(d) 的数学根源与点积推演)")
        print("  [5] 单独透视 第四章：多头注意力机制 (Multi-Head) 的子空间分治")
        print("  [6] 单独透视 第五章：残差连接 (Add) 与层归一化 (LayerNorm)")
        print("  [7] 单独透视 第六章：前馈神经网络 (FFN) 的非线性升降维")
        print("  [0] 退出系统")
        print("=" * 76)
        
        choice = input("请选择要进入的模块编号 (0-7): ").strip()
        
        if choice == '1':
            run_full_pipeline()
        elif choice == '2':
            chapter_1_embedding_and_pe()
        elif choice == '3':
            X = chapter_1_embedding_and_pe()
            if X: chapter_2_qkv_projection(X)
        elif choice == '4':
            X = chapter_1_embedding_and_pe()
            if X:
                qkv = chapter_2_qkv_projection(X)
                if qkv: chapter_3_scaled_dot_product(*qkv)
        elif choice == '5':
            X = chapter_1_embedding_and_pe()
            if X: chapter_4_multi_head_attention(X)
        elif choice == '6':
            X = chapter_1_embedding_and_pe()
            if X:
                mha = chapter_4_multi_head_attention(X)
                if mha: chapter_5_add_and_norm(X, mha)
        elif choice == '7':
            X = chapter_1_embedding_and_pe()
            if X:
                mha = chapter_4_multi_head_attention(X)
                if mha:
                    norm1 = chapter_5_add_and_norm(X, mha)
                    if norm1: chapter_6_ffn(norm1)
        elif choice == '0':
            print("\n感谢使用 Transformer 白盒透视学习系统，祝您科研与学习顺利！再见。")
            break
        else:
            print("输入无效，请输入 0 到 7 之间的数字。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已被用户中断。")
