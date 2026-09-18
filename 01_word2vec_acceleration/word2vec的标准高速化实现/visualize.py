"""
=============================================================================
脚本名称: visualize.py
核心功能: 词向量空间 2D 降维可视化分析工具 (PCA / t-SNE)
包含特性:
    1. 支持 PCA (主成分分析) 与 t-SNE (t-分布随机邻域嵌入) 两种主流降维算法
    2. 支持语义分类聚类模式 (--mode cluster) 与 高频词全景模式 (--mode top)
    3. 自动生成 1800x1100 超高清静态分析图 (assets/word_embeddings_2d.png)
    4. 自动生成零依赖、完全离线可交互的 HTML5 可视化页面 (assets/word_embeddings_interactive.html)
    5. 支持自定义权重路径、降维算法、词数与保存文件名
=============================================================================
理论重点:
词向量之所以能在 2D 散点图上呈现语义聚类，是因为模型在最大化上下文预测概率的过程中，
强迫具有相似上下文环境的词汇在高维欧几里得流形中靠拢 (分布假说 Distributional Hypothesis:
'You shall know a word by the company it keeps')。降维算法 (如 PCA 寻找最大方差投影、
t-SNE 保留局部邻域概率拓扑) 能够将 100 维隐藏空间压缩至 2 维平面，从而直观验证模型学习成效。
=============================================================================
"""

import sys
import os
import argparse
import pickle
import numpy as np
from typing import Dict, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights", "word2vec_cbow_h100_w5.pkl")
ALT_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights", "quick_demo_weights.pkl")
ASSETS_DIR = os.path.join(CURRENT_DIR, "assets")

# 预设的语义测试分类聚类词组 (涵盖 PTB 财经新闻语料典型特征)
SEMANTIC_CLUSTERS = {
    "金融与商业 (Financial)": {
        "color": "#2563EB",  # 科技蓝
        "bg_color": "#EFF6FF",
        "words": ["bank", "market", "stock", "dollar", "fund", "price", "trade", "money", "bond", "debt", "mortgage", "cost"]
    },
    "机构与实体 (Entities)": {
        "color": "#D97706",  # 琥珀橙
        "bg_color": "#FEF3C7",
        "words": ["government", "company", "state", "court", "firm", "department", "board", "federal", "corp.", "agency"]
    },
    "人物与职位 (Roles)": {
        "color": "#059669",  # 翡翠绿
        "bg_color": "#ECFDF5",
        "words": ["president", "chairman", "executive", "director", "manager", "officer", "lawyer", "worker", "man", "woman"]
    },
    "时间与周期 (Temporal)": {
        "color": "#7C3AED",  # 紫水晶
        "bg_color": "#F5F3FF",
        "words": ["year", "month", "day", "week", "quarter", "yesterday", "recent", "time", "annual", "period"]
    },
    "动作与变化 (Dynamics)": {
        "color": "#DC2626",  # 胭脂红
        "bg_color": "#FEF2F2",
        "words": ["rose", "fell", "grew", "declined", "reported", "expected", "said", "take", "took", "increased"]
    }
}


def load_model_data(weights_path: str) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """载入模型权重矩阵与词典"""
    if not os.path.exists(weights_path):
        if os.path.exists(ALT_WEIGHTS_PATH):
            print(f"[*] 提示: 默认权重不存在，切换至轻量权重: {ALT_WEIGHTS_PATH}")
            weights_path = ALT_WEIGHTS_PATH
        else:
            raise FileNotFoundError(f"未找到权重文件: {weights_path}，请先执行训练。")

    print(f"[*] 正在载入词向量模型: {os.path.basename(weights_path)} ...")
    with open(weights_path, "rb") as f:
        data = pickle.load(f)

    W = data["W_in"]
    word_to_id = data["word_to_id"]
    id_to_word = data["id_to_word"]
    print(f"[OK] 成功载入! 词表规模: {len(word_to_id):,} 词 | 向量维度: {W.shape[1]} 维")
    return W, word_to_id, id_to_word


def reduce_dimensions(vectors: np.ndarray, method: str = "pca", random_state: int = 42) -> np.ndarray:
    """
    将高维词向量降至 2D 平面
    优先使用 sklearn，若未安装则自动回退至纯 NumPy SVD 实现的 PCA
    """
    print(f"[*] 正在使用 {method.upper()} 算法对 {vectors.shape[0]} 个词向量进行降维 ( {vectors.shape[1]}D -> 2D ) ...")
    
    if method.lower() == "tsne":
        try:
            from sklearn.manifold import TSNE
            perplexity = min(30, max(5, vectors.shape[0] // 4))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, max_iter=1000)
            coords_2d = tsne.fit_transform(vectors)
            return coords_2d
        except ImportError:
            print("[!] 警告: 未找到 sklearn，自动回退至纯 NumPy SVD-PCA 降维。")

    # PCA 算法实现 (纯 NumPy SVD，零第三方依赖)
    # 1. 中心化
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    # 2. 奇异值分解 SVD: centered = U * S * Vt
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    # 3. 投影到前两个主成分空间
    coords_2d = np.dot(centered, vt[:2].T)
    return coords_2d


def render_pillow_plot(
    coords_2d: np.ndarray,
    words: List[str],
    categories: List[str],
    output_path: str,
    method_name: str
):
    """使用 Pillow 绘制出版物级别的高清 2D 词向量散点图 (无 Matplotlib 依赖)"""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1800, 1100
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # 加载系统微软雅黑
    font_path = "C:/Windows/Fonts/msyh.ttc"
    title_font = ImageFont.truetype(font_path, 28)
    subtitle_font = ImageFont.truetype(font_path, 16)
    legend_font = ImageFont.truetype(font_path, 15)
    word_font = ImageFont.truetype(font_path, 13)
    axis_font = ImageFont.truetype(font_path, 12)

    # 1. 绘制顶部信息栏
    draw.rectangle([(0, 0), (width, 80)], fill="#0F172A")
    draw.text((40, 15), f"Word2Vec 词向量空间 2D 语义流形可视化 ({method_name.upper()} 降维)", fill="#FFFFFF", font=title_font)
    draw.text((40, 52), "语义聚合特性观察: 语义相关的词汇在流形空间中自动聚集成簇 | 语料: Penn Treebank (PTB)", fill="#94A3B8", font=subtitle_font)

    # 2. 计算坐标映射到图像画布范围
    margin_left, margin_right = 90, 280
    margin_top, margin_bottom = 120, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    xs = coords_2d[:, 0]
    ys = coords_2d[:, 1]
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)

    # 给边界留 10% 余量
    pad_x = (max_x - min_x) * 0.1 if max_x != min_x else 1.0
    pad_y = (max_y - min_y) * 0.1 if max_y != min_y else 1.0
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    def to_screen(x, y):
        sx = margin_left + (x - min_x) / (max_x - min_x) * plot_w
        # y 轴向上为正，屏幕坐标向下为正，需要反转
        sy = margin_top + (max_y - y) / (max_y - min_y) * plot_h
        return sx, sy

    # 3. 绘制背景网格和边框
    draw.rectangle([(margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h)], outline="#E2E8F0", width=2)
    grid_steps = 8
    for i in range(1, grid_steps):
        gx = margin_left + (plot_w / grid_steps) * i
        draw.line([(gx, margin_top), (gx, margin_top + plot_h)], fill="#F1F5F9", width=1)
        gy = margin_top + (plot_h / grid_steps) * i
        draw.line([(margin_left, gy), (margin_left + plot_w, gy)], fill="#F1F5F9", width=1)

    # 绘制坐标轴零刻度线 (如果包含原点)
    if min_x <= 0 <= max_x:
        zx, _ = to_screen(0, min_y)
        draw.line([(zx, margin_top), (zx, margin_top + plot_h)], fill="#CBD5E1", width=2)
        draw.text((zx + 5, margin_top + plot_h - 20), "X = 0", fill="#94A3B8", font=axis_font)
    if min_y <= 0 <= max_y:
        _, zy = to_screen(min_x, 0)
        draw.line([(margin_left, zy), (margin_left + plot_w, zy)], fill="#CBD5E1", width=2)
        draw.text((margin_left + 8, zy - 18), "Y = 0", fill="#94A3B8", font=axis_font)

    # 4. 绘制右侧分类图例面板 (Legend)
    legend_x = width - margin_right + 25
    legend_y = margin_top + 10
    draw.rounded_rectangle([(legend_x - 10, legend_y - 10), (width - 30, legend_y + 260)], radius=8, fill="#F8FAFC", outline="#E2E8F0", width=1)
    draw.text((legend_x, legend_y), "语义分类图例", fill="#1E293B", font=legend_font)
    legend_y += 35

    color_map = {}
    for cat_name, cat_info in SEMANTIC_CLUSTERS.items():
        color_map[cat_name] = (cat_info["color"], cat_info["bg_color"])
        # 绘制图例颜色圆点与文字
        draw.ellipse([(legend_x, legend_y + 2), (legend_x + 14, legend_y + 16)], fill=cat_info["color"])
        draw.text((legend_x + 24, legend_y), cat_name, fill="#334155", font=legend_font)
        legend_y += 38

    # 5. 绘制各个单词节点 (散点 + 气泡文字标签)
    for i, (word, cat) in enumerate(zip(words, categories)):
        x, y = coords_2d[i]
        sx, sy = to_screen(x, y)
        dot_color, pill_bg = color_map.get(cat, ("#2563EB", "#EFF6FF"))

        # 绘制扩散半透明光晕外圈
        draw.ellipse([(sx - 8, sy - 8), (sx + 8, sy + 8)], fill=pill_bg, outline=dot_color, width=1)
        # 绘制核心圆点
        draw.ellipse([(sx - 4, sy - 4), (sx + 4, sy + 4)], fill=dot_color)

        # 计算文字包围盒，绘制精致圆角药丸标签 (Pill Label)
        # 错开位置放置，右上角偏移
        tx, ty = sx + 8, sy - 18
        # 获取文字宽度
        bbox = word_font.getbbox(word)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # 绘制药丸气泡底色
        draw.rounded_rectangle([(tx - 4, ty - 2), (tx + tw + 6, ty + th + 4)], radius=4, fill=pill_bg, outline=dot_color, width=1)
        draw.text((tx, ty), word, fill=dot_color, font=word_font)

    # 6. 底部状态说明
    draw.text((margin_left, height - 40), f"降维算法: {method_name.upper()} | 词向量数量: {len(words)} | 图像分辨率: {width}x{height}", fill="#64748B", font=axis_font)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path, quality=95)
    print(f"[OK] 高清静态散点图已生成: {output_path}")


def render_interactive_html(
    coords_2d: np.ndarray,
    words: List[str],
    categories: List[str],
    output_path: str,
    method_name: str
):
    """生成完全独立的交互式 HTML5 散点图 (支持鼠标悬停、缩放查看详细信息)"""
    # 归一化坐标到 0~1000 范围
    xs = coords_2d[:, 0]
    ys = coords_2d[:, 1]
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    
    pad_x = (max_x - min_x) * 0.08 if max_x != min_x else 1.0
    pad_y = (max_y - min_y) * 0.08 if max_y != min_y else 1.0
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    data_points = []
    for i, (w, c) in enumerate(zip(words, categories)):
        norm_x = (coords_2d[i, 0] - min_x) / (max_x - min_x) * 900 + 50
        norm_y = (max_y - coords_2d[i, 1]) / (max_y - min_y) * 550 + 50
        color = SEMANTIC_CLUSTERS.get(c, {}).get("color", "#2563EB")
        data_points.append({
            "word": w,
            "cat": c,
            "x": round(float(norm_x), 1),
            "y": round(float(norm_y), 1),
            "raw_x": round(float(coords_2d[i, 0]), 4),
            "raw_y": round(float(coords_2d[i, 1]), 4),
            "color": color
        })

    import json
    data_json = json.dumps(data_points, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Word2Vec 词向量 2D 交互式可视化</title>
    <style>
        body {{
            margin: 0;
            padding: 24px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #F8FAFC;
            color: #0F172A;
        }}
        .header {{
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #E2E8F0;
        }}
        .header h1 {{
            margin: 0 0 6px 0;
            font-size: 24px;
        }}
        .header p {{
            margin: 0;
            color: #64748B;
            font-size: 14px;
        }}
        .container {{
            display: flex;
            gap: 20px;
        }}
        .chart-box {{
            flex: 1;
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            padding: 16px;
            position: relative;
        }}
        svg {{
            width: 100%;
            height: 650px;
            background: #FAFAFA;
            border-radius: 8px;
        }}
        .dot {{
            cursor: pointer;
            transition: r 0.2s, stroke-width 0.2s;
        }}
        .dot:hover {{
            r: 8px;
            stroke: #0F172A;
            stroke-width: 2px;
        }}
        .word-tag {{
            font-size: 12px;
            fill: #334155;
            user-select: none;
            pointer-events: none;
            font-weight: 500;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(15, 23, 42, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
            pointer-events: none;
            display: none;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
            z-index: 100;
        }}
        .sidebar {{
            width: 280px;
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 12px;
            padding: 20px;
            height: fit-content;
        }}
        .sidebar h3 {{
            margin-top: 0;
            font-size: 16px;
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            font-size: 14px;
        }}
        .legend-badge {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Word2Vec 词向量空间 2D 交互流形 ({method_name.upper()} 降维)</h1>
        <p>鼠标悬停在点上可高亮词汇及精确坐标，直观感受语义聚类分布规律</p>
    </div>
    <div class="container">
        <div class="chart-box">
            <div id="tooltip" class="tooltip"></div>
            <svg id="chart" viewBox="0 0 1000 650"></svg>
        </div>
        <div class="sidebar">
            <h3>语义分类图例</h3>
            <div class="legend-item"><span class="legend-badge" style="background:#2563EB;"></span>金融与商业 (Financial)</div>
            <div class="legend-item"><span class="legend-badge" style="background:#D97706;"></span>机构与实体 (Entities)</div>
            <div class="legend-item"><span class="legend-badge" style="background:#059669;"></span>人物与职位 (Roles)</div>
            <div class="legend-item"><span class="legend-badge" style="background:#7C3AED;"></span>时间与周期 (Temporal)</div>
            <div class="legend-item"><span class="legend-badge" style="background:#DC2626;"></span>动作与变化 (Dynamics)</div>
            <hr style="border:none;border-top:1px solid #E2E8F0;margin:20px 0;">
            <p style="font-size:12px;color:#64748B;line-height:1.6;">
                <b>交互提示:</b><br>
                每个点的欧几里得距离反映了神经网络所学到的词义相似度。同类词汇在空间中自然聚集，体现了经典的“分布式表征假说”。
            </p>
        </div>
    </div>

    <script>
        const data = {data_json};
        const svg = document.getElementById('chart');
        const tooltip = document.getElementById('tooltip');

        data.forEach(d => {{
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', d.x);
            circle.setAttribute('cy', d.y);
            circle.setAttribute('r', '5');
            circle.setAttribute('fill', d.color);
            circle.setAttribute('class', 'dot');

            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', d.x + 8);
            text.setAttribute('y', d.y + 4);
            text.setAttribute('class', 'word-tag');
            text.textContent = d.word;

            circle.addEventListener('mousemove', (e) => {{
                tooltip.style.display = 'block';
                tooltip.style.left = (e.pageX + 15) + 'px';
                tooltip.style.top = (e.pageY - 20) + 'px';
                tooltip.innerHTML = `<b>${{d.word}}</b><br>类别: ${{d.cat}}<br>坐标: [${{d.raw_x}}, ${{d.raw_y}}]`;
            }});

            circle.addEventListener('mouseleave', () => {{
                tooltip.style.display = 'none';
            }});

            svg.appendChild(circle);
            svg.appendChild(text);
        }});
    </script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] 交互式网页已生成: {output_path}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Word2Vec 词向量 2D 降维可视化工具")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH,
                        help="模型权重文件路径 (.pkl)")
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"],
                        help="降维算法: pca (全局方差最大) 或 tsne (局部拓扑保持)")
    parser.add_argument("--mode", type=str, default="cluster", choices=["cluster", "top"],
                        help="可视化模式: cluster (语义分类聚类) 或 top (高频词概览)")
    parser.add_argument("--top_n", type=int, default=60,
                        help="top 模式下可视化的单词数量")
    parser.add_argument("--output_png", type=str, default=os.path.join(ASSETS_DIR, "word_embeddings_2d.png"),
                        help="输出静态图像路径")
    parser.add_argument("--output_html", type=str, default=os.path.join(ASSETS_DIR, "word_embeddings_interactive.html"),
                        help="输出交互式网页路径")
    args = parser.parse_args()

    # 1. 载入模型
    W, word_to_id, id_to_word = load_model_data(args.weights)

    # 2. 筛选待可视化的单词与分类
    selected_words = []
    selected_categories = []
    selected_vectors = []

    if args.mode == "cluster":
        print("[*] 正在筛选预设的 5 大语义特征分类词汇...")
        for cat_name, cat_info in SEMANTIC_CLUSTERS.items():
            for w in cat_info["words"]:
                if w in word_to_id:
                    idx = word_to_id[w]
                    selected_words.append(w)
                    selected_categories.append(cat_name)
                    selected_vectors.append(W[idx])
        print(f"[OK] 共在词表中命中 {len(selected_words)} 个典型语义评测词。")
    else:
        print(f"[*] 正在提取词表中前 {args.top_n} 个高频词汇...")
        for idx in range(min(args.top_n, len(word_to_id))):
            w = id_to_word[idx]
            selected_words.append(w)
            selected_categories.append("高频词汇")
            selected_vectors.append(W[idx])

    if not selected_vectors:
        print("[!] 错误: 未能在词表中匹配到有效词汇。")
        return

    selected_vectors = np.array(selected_vectors, dtype=np.float32)

    # 3. 执行降维
    coords_2d = reduce_dimensions(selected_vectors, method=args.method)

    # 4. 生成出版级静态高清图
    render_pillow_plot(coords_2d, selected_words, selected_categories, args.output_png, method_name=args.method)

    # 5. 生成零依赖交互式 HTML5 页面
    render_interactive_html(coords_2d, selected_words, selected_categories, args.output_html, method_name=args.method)

    print("\n" + "=" * 75)
    print("✨ 可视化任务全部成功完成！")
    print(f"   • 静态高清图: {args.output_png}")
    print(f"   • 交互式网页: {args.output_html} (可在任意浏览器中双击直接打开)")
    print("=" * 75)


if __name__ == "__main__":
    main()
