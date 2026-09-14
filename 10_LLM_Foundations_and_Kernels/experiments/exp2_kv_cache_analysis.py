# -*- coding: utf-8 -*-
"""
Experiment 2: KV Cache 架构演进与显存压缩深度实测
"""
import os
from src.kv_cache_benchmark import run_kv_cache_analysis

def run():
    img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
    os.makedirs(img_dir, exist_ok=True)
    save_fig = os.path.join(img_dir, "llm_kv_cache_compression.png")
    run_kv_cache_analysis(save_fig)
    print(f">>> KV Cache 显存图已生成至: {save_fig}\n")

if __name__ == "__main__":
    run()
