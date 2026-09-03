"""
绘制 Word2Vec CBOW (窗口为6) 完整数据流与运算过程高清架构图 (排版终极微调版)
"""
import os
from PIL import Image, ImageDraw, ImageFont

def create_cbow_flow_diagram():
    width = 1680
    height = 1060
    img = Image.new("RGB", (width, height), "#F8FAFC")
    draw = ImageDraw.Draw(img)

    # 载入系统微软雅黑字体
    font_path = "C:/Windows/Fonts/msyh.ttc"
    title_font = ImageFont.truetype(font_path, 28)
    subtitle_font = ImageFont.truetype(font_path, 16)
    sec_font = ImageFont.truetype(font_path, 19)
    card_title_font = ImageFont.truetype(font_path, 16)
    body_font = ImageFont.truetype(font_path, 13)
    code_font = ImageFont.truetype(font_path, 12)
    badge_font = ImageFont.truetype(font_path, 14)

    # 1. 顶部 Header 栏
    draw.rectangle([(0, 0), (width, 85)], fill="#0F172A")
    draw.text((40, 16), "Word2Vec (CBOW) 窗口大小为 6 的完整推导与数据流图解", fill="#FFFFFF", font=title_font)
    draw.text((40, 54), "涵盖核心疑问：上下文 12 向量是【相加/平均】还是【并列拼接】？ 与 W_out 运算后得到的是什么？", fill="#94A3B8", font=subtitle_font)

    # 辅助函数: 绘制圆角矩形卡片
    def draw_card(box, fill_color, border_color="#CBD5E1", radius=10):
        draw.rounded_rectangle(box, radius=radius, fill=fill_color, outline=border_color, width=2)

    # -------------------------------------------------------------
    # 2. 第一阶段：输入上下文 (窗口 6 -> 左右各 6，共 12 个上下文词)
    # -------------------------------------------------------------
    col1_box = (40, 105, 520, 1020)
    draw_card(col1_box, "#FFFFFF")
    draw.rectangle([(40, 105), (520, 150)], fill="#E2E8F0")
    draw.text((60, 117), "第一阶段：输入与词嵌入 (Embedding)", fill="#0F172A", font=sec_font)

    # 示例文本切片
    draw.text((60, 165), "【示例文本语境】单侧窗口 window_size = 6", fill="#1E293B", font=card_title_font)
    draw_card((60, 190, 500, 240), "#F1F5F9", border_color="#94A3B8", radius=6)
    draw.text((70, 198), "... federal reserve chairman said the us", fill="#0F172A", font=code_font)
    draw.text((70, 218), "    [ market ] is growing very steadily in recent ...", fill="#2563EB", font=code_font)

    draw.text((60, 255), "1. 提取中心词周围 12 个上下文词 (6 个左词 + 6 个右词):", fill="#475569", font=body_font)
    left_words = ["w-6: federal", "w-5: reserve", "w-4: chairman", "w-3: said", "w-2: the", "w-1: us"]
    right_words = ["w+1: is", "w+2: growing", "w+3: very", "w+4: steadily", "w+5: in", "w+6: recent"]

    y_w = 280
    for i in range(6):
        draw_card((60, y_w, 270, y_w + 28), "#EFF6FF", border_color="#BFDBFE", radius=4)
        draw.text((70, y_w + 5), left_words[i], fill="#1D4ED8", font=code_font)
        draw_card((290, y_w, 500, y_w + 28), "#EFF6FF", border_color="#BFDBFE", radius=4)
        draw.text((300, y_w + 5), right_words[i], fill="#1D4ED8", font=code_font)
        y_w += 35

    draw.text((60, 500), "2. 查表获取词向量 (Embedding Lookup):", fill="#475569", font=body_font)
    draw.text((60, 525), "输入权重矩阵 W_in 的形状为 (V, H) ，其中:\n- V: 词典词表大小 (如 10,000 ~ 100,000)\n- H: 隐藏层词向量维度 (如 100 维)", fill="#64748B", font=body_font)

    draw_card((60, 590, 500, 725), "#F8FAFC", border_color="#CBD5E1", radius=6)
    draw.text((75, 603), "提取出来的 12 个词向量形状:", fill="#0F172A", font=card_title_font)
    draw.text((75, 633), "v_{-6} = W_in[idx_{-6}]   ->   形状: (1, H)", fill="#0369A1", font=code_font)
    draw.text((75, 653), "v_{-5} = W_in[idx_{-5}]   ->   形状: (1, H)", fill="#0369A1", font=code_font)
    draw.text((75, 673), " ... (中间 8 个词向量同样各为 (1, H))", fill="#64748B", font=code_font)
    draw.text((75, 693), "v_{+6} = W_in[idx_{+6}]   ->   形状: (1, H)", fill="#0369A1", font=code_font)

    # 关键提示卡片
    draw_card((60, 745, 500, 995), "#FEF3C7", border_color="#F59E0B", radius=8)
    draw.text((75, 760), "【核心高速化对比】", fill="#B45309", font=card_title_font)
    notes_in = (
        "• 传统低效做法:\n"
        "  将 12 个词转为 12 个 (1, V) 的 One-hot 稀疏向量，\n"
        "  分别与 W_in (V, H) 做矩阵乘法。由于 99.9% 都是 0，\n"
        "  耗费了 12 × V × H 次毫无意义的乘加！\n\n"
        "• 高速化方案 (本项目实现):\n"
        "  直接用整数索引进行内存行切片 W_in[idx]，\n"
        "  耗时 O(1) 瞬间获取这 12 个 (1, H) 的稠密向量！"
    )
    draw.text((75, 790), notes_in, fill="#92400E", font=body_font)

    # -------------------------------------------------------------
    # 3. 第二阶段：隐藏层聚合运算 (Sum/Average vs Concat)
    # -------------------------------------------------------------
    col2_box = (555, 105, 1080, 1020)
    draw_card(col2_box, "#FFFFFF")
    draw.rectangle([(555, 105), (1080, 150)], fill="#E2E8F0")
    draw.text((575, 117), "第二阶段：隐层聚合 (相加/求平均 vs 并列拼接)", fill="#0F172A", font=sec_font)

    # 直截了当给出答案的醒目绿色卡片
    draw_card((575, 165, 1060, 225), "#DCFCE7", border_color="#22C55E", radius=8)
    draw.text((590, 175), "核心提问解答：这 12 个 (1, H) 向量是相加还是并列？", fill="#15803D", font=card_title_font)
    draw.text((590, 200), "权威结论：在 Word2Vec (CBOW) 中，必须【逐元素相加 / 求平均】！", fill="#166534", font=badge_font)

    # 正确方案: 相加/求平均
    draw_card((575, 238, 1060, 505), "#F0FDF4", border_color="#86EFAC", radius=8)
    draw.text((590, 248), "【正解】向量逐元素求和 / 求平均 (Sum / Average)", fill="#166534", font=card_title_font)
    cbow_desc = (
        "聚合公式:  h = (1 / 12) * sum_{c=1}^{12} v_c    (最终形状保持 1 × H)\n\n"
        "深度原理解析 (为什么必须是相加/平均？):\n"
        "1. 词袋模型 (Bag-of-Words) 本义:\n"
        "   把上下文视为无序词袋，向量叠加代表概念融合，对局部语序颠倒更鲁棒。\n"
        "2. 维度守恒与参数定长 (最关键原因):\n"
        "   无论窗口是 2、6 还是 10，输出向量 h 严格为 (1, H)！\n"
        "   与后层连接的权重矩阵尺寸永远固定为 (H, V)，不与窗口长度耦合。\n"
        "3. 边界自适应能力:\n"
        "   在句首句尾窗口不足 12 个词时，直接除以实际词数 C 即可，无需 Padding。"
    )
    draw.text((590, 273), cbow_desc, fill="#14532D", font=body_font)

    # 反面教材: 并列拼接的严重弊端
    draw_card((575, 520, 1060, 835), "#FEE2E2", border_color="#F87171", radius=8)
    draw.text((590, 532), "【反面教材】如果采用并列拼接 (Concatenation) 会怎样？", fill="#B91C1C", font=card_title_font)
    concat_desc = (
        "拼接公式:  h_concat = [v_{-6} ; v_{-5} ; ... ; v_{+6}]   (形状变为 1 × 12H)\n\n"
        "引发的严重灾难:\n"
        "1. 模型参数量爆炸 12 倍:\n"
        "   输出层矩阵将被迫从 (H, V) 膨胀为 (12H, V)！\n"
        "   当 V=100,000 时，单层参数量瞬间突破十亿级！\n"
        "2. 上下文长度被死死锁死:\n"
        "   句首缺少词时必须强行补零 (Padding)，破坏语义分布并丧失灵活性。\n\n"
        "历史注记: 2003年 Bengio 的 NNLM 曾用拼接保持语序，但 Mikolov 2013 在\n"
        "Word2Vec 中正是用【求和/平均】革除拼接弊端，换来了千百倍的极速训练！"
    )
    draw.text((590, 558), concat_desc, fill="#7F1D1D", font=body_font)

    # 阶段产出
    draw_card((575, 850, 1060, 995), "#F1F5F9", border_color="#94A3B8", radius=8)
    draw.text((590, 862), "第二阶段最终产出的隐层向量:", fill="#0F172A", font=card_title_font)
    draw.text((590, 890), "h = np.mean(h_all, axis=1)    ->   形状: (1, H)", fill="#2563EB", font=badge_font)
    draw.text((590, 918), "该 (1, H) 向量已高度浓缩了上下文 12 个词的综合语义信息，\n将作为下一阶段与输出矩阵 W_out 进行交互的唯一输入载体。", fill="#475569", font=body_font)

    # -------------------------------------------------------------
    # 4. 第三阶段：与 W_out 的运算及产出结果 (两路径对比)
    # -------------------------------------------------------------
    col3_box = (1115, 105, 1640, 1020)
    draw_card(col3_box, "#FFFFFF")
    draw.rectangle([(1115, 105), (1640, 150)], fill="#E2E8F0")
    draw.text((1135, 117), "第三阶段：与 W_out 的运算及最终产物", fill="#0F172A", font=sec_font)

    # 分支 A：传统朴素 Softmax
    draw_card((1135, 165, 1620, 510), "#F8FAFC", border_color="#CBD5E1", radius=8)
    draw.text((1150, 178), "分支 A: 传统朴素 Softmax (全词表多分类)", fill="#0F172A", font=card_title_font)
    softmax_desc = (
        "1. 运算方式: 矩阵乘法 (Matrix Multiplication)\n"
        "   Score = h @ W_out    (其中 W_out 的形状为 (H, V))\n"
        "   运算后得到的是: 一个长度为 V 的得分向量 (1 × V)。\n"
        "   里面的每一个数字，代表词典中第 i 个词作为中心词的未归一化得分。\n\n"
        "2. 归一化: 全局 Softmax 函数\n"
        "   P(w_i | 上下文) = exp(Score_i) / sum_{j=1}^V exp(Score_j)\n"
        "   运算后得到的是: 全词表的概率分布向量 (1 × V)。\n"
        "   每个元素都在 0~1 之间，且所有元素相加严格等于 1.0。\n\n"
        "3. 损失计算: 多分类交叉熵损失 (Cross Entropy Loss)\n"
        "   Loss = - log( P(真实目标词 | 上下文) )\n\n"
        "致命缺陷: 分母计算 sum_{j=1}^V 必须在全词表上累加，复杂度 O(V)！"
    )
    draw.text((1150, 206), softmax_desc, fill="#334155", font=body_font)

    # 分支 B：高速化负采样
    draw_card((1135, 525, 1620, 995), "#EFF6FF", border_color="#3B82F6", radius=8)
    draw.text((1150, 538), "分支 B: 高速化负采样 (二分类逻辑回归 - 本项目实现)", fill="#1D4ED8", font=card_title_font)
    ns_desc = (
        "革命性转变: 绝不进行全词表矩阵乘法！只做几个点积(内积)运算。\n\n"
        "1. 正样本运算 (真实中心词 target=market, 标签 t=1):\n"
        "   • 从 W_out 提取 target 行: w_pos = W_out[target]  (形状: 1 × H)\n"
        "   • 计算向量点积: score_pos = sum(h * w_pos)  -> 得到一个【标量数值】！\n"
        "   • 经 Sigmoid 映射: y_pos = 1 / (1 + exp(-score_pos))  -> 预测概率\n"
        "   • 正样本损失: Loss_pos = - log(y_pos)  (目标让 y_pos 逼近 1.0)\n\n"
        "2. 负样本运算 (随机抽样 K 个负词，如 neg=[cat, apple, ...], 标签 t=0):\n"
        "   • 从 W_out 提取对应行: w_neg_k = W_out[neg_k]     (形状: 1 × H)\n"
        "   • 计算向量点积: score_neg_k = sum(h * w_neg_k)  -> 得到 K 个【标量数值】！\n"
        "   • 经 Sigmoid 映射: y_neg_k = 1 / (1 + exp(-score_neg_k))\n"
        "   • 负样本损失: Loss_neg = sum_{k=1}^K - log(1 - y_neg_k)  (逼近 0.0)\n\n"
        "3. 最终产出结果:\n"
        "   • 综合损失: Total_Loss = Loss_pos + Loss_neg\n"
        "   • 运算量从 V 次矩阵乘法骤降为 (K+1) 次向量点积，提速数万倍！"
    )
    draw.text((1150, 566), ns_desc, fill="#1E3A8A", font=body_font)

    # 绘制流程指示箭头
    def draw_arrow(start_pt, end_pt, color="#64748B", text=""):
        draw.line([start_pt, end_pt], fill=color, width=3)
        ex, ey = end_pt
        draw.polygon([(ex, ey), (ex - 8, ey - 5), (ex - 8, ey + 5)], fill=color)
        if text:
            tx = (start_pt[0] + end_pt[0]) // 2 - 25
            ty = start_pt[1] - 18
            draw.text((tx, ty), text, fill=color, font=code_font)

    draw_arrow((520, 520), (555, 520), color="#2563EB", text="输入 12 向量")
    draw_arrow((1080, 340), (1115, 340), color="#64748B", text="全量路径")
    draw_arrow((1080, 750), (1115, 750), color="#2563EB", text="负采样路径")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cbow_simulation_flow.png")
    img.save(output_path, quality=95)
    print(f"流程图已成功生成至: {output_path}")

if __name__ == "__main__":
    create_cbow_flow_diagram()
