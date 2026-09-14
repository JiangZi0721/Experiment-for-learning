# -*- coding: utf-8 -*-
"""
KV Cache Compression Benchmark (MHA vs MQA vs GQA vs MLA)
严密数学建模与显存占用对比分析
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_kv_cache_bytes_per_token(num_layers, num_heads_q, num_heads_kv, head_dim, precision_bytes=2):
    """
    经典 KV Cache 每 Token 显存开销 (Bytes/Token):
    Size = 2 (K and V) * num_layers * num_heads_kv * head_dim * precision_bytes
    """
    return 2 * num_layers * num_heads_kv * head_dim * precision_bytes

def calculate_mla_bytes_per_token(num_layers, d_latent_kv, d_rope_k, precision_bytes=2):
    """
    DeepSeek MLA (Multi-Head Latent Attention) 每 Token 显存开销:
    仅需缓存联合压缩潜变量 c_t^{KV} 以及解耦位置编码 k_t^R:
    Size = num_layers * (d_latent_kv + d_rope_k) * precision_bytes
    """
    return num_layers * (d_latent_kv + d_rope_k) * precision_bytes

def run_kv_cache_analysis(save_path):
    # 以 LLaMA-3-70B / DeepSeek-V3 规模参数建模:
    num_layers = 64
    num_heads_q = 64
    head_dim = 128
    precision_bytes = 2 # FP16

    # 1. MHA (Multi-Head Attention: 64 Q, 64 KV)
    b_mha = calculate_kv_cache_bytes_per_token(num_layers, num_heads_q, 64, head_dim, precision_bytes)

    # 2. GQA (Grouped-Query Attention: 64 Q, 8 KV)
    b_gqa = calculate_kv_cache_bytes_per_token(num_layers, num_heads_q, 8, head_dim, precision_bytes)

    # 3. MQA (Multi-Query Attention: 64 Q, 1 KV)
    b_mqa = calculate_kv_cache_bytes_per_token(num_layers, num_heads_q, 1, head_dim, precision_bytes)

    # 4. MLA (DeepSeek Multi-head Latent Attention: d_c = 512, d_rope = 64)
    b_mla = calculate_mla_bytes_per_token(num_layers, d_latent_kv=512, d_rope_k=64, precision_bytes=precision_bytes)

    archs = ["MHA (标准)", "GQA (LLaMA-3)", "MQA (极简)", "MLA (DeepSeek-V3)"]
    bytes_per_tok = [b_mha, b_gqa, b_mqa, b_mla]
    kb_per_tok = [b / 1024.0 for b in bytes_per_tok]

    # 计算 128k 上下文时的单并发显存 (GB)
    gb_128k = [b * 128 * 1024 / (1024**3) for b in bytes_per_tok]

    print("=" * 80)
    print(">>> 【KV Cache 架构演进与显存压缩深度全景对比 (64层模型基准)】")
    print("=" * 80)
    print(f"{'架构范式':<16} | {'单 Token 开销 (KB)':<18} | {'128K 单并发显存 (GB)':<20} | {'显存压缩率':<12} | {'模型精度表现'}")
    print("-" * 88)
    for name, kb, gb in zip(archs, kb_per_tok, gb_128k):
        compression = (1.0 - kb / kb_per_tok[0]) * 100
        perf = "基准 (全无损)" if name.startswith("MHA") else ("近无损 (工业主流)" if "GQA" in name else ("微损" if "MQA" in name else "全无损且长文本卓越 (SOTA)"))
        print(f"{name:<16} | {kb:>14.2f} KB | {gb:>16.2f} GB | {compression:>9.1f}% | {perf}")
    print("=" * 80)

    # 绘制可视化柱状图
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 图 1: 单 Token 显存
    bars1 = axes[0].bar(archs, kb_per_tok, color=['#EF4444', '#F59E0B', '#3B82F6', '#10B981'], width=0.5)
    axes[0].set_title("每 Token 缓存显存开销 (KB/Token)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("KB / Token", fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars1:
        y = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, y + 5, f"{y:.1f} KB", ha='center', fontweight='bold')

    # 图 2: 128k 显存
    bars2 = axes[1].bar(archs, gb_128k, color=['#EF4444', '#F59E0B', '#3B82F6', '#10B981'], width=0.5)
    axes[1].set_title("128K 超长上下文单并发显存占用 (GB)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("显存占用 (GB)", fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        y = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, y + 0.8, f"{y:.1f} GB", ha='center', fontweight='bold')

    plt.suptitle("大模型 KV Cache 演进 (MHA -> GQA -> MQA -> DeepSeek MLA) 显存革命", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
