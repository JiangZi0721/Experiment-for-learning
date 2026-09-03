# -*- coding: utf-8 -*-
"""
White-Box RAG Lab 100篇语料整合构建脚本
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(BASE_DIR, "data", "corpus")

# 引入4个模块
from corpus_part1 import DISTRIBUTED_SYSTEMS
from corpus_part2 import STORAGE_ENGINES
from corpus_part3 import LLM_INFRA
from corpus_part4 import OS_KERNEL_NETWORK

def write_topics(topic_list, subfolder):
    target_dir = os.path.join(CORPUS_DIR, subfolder)
    os.makedirs(target_dir, exist_ok=True)
    count = 0
    for slug, title, content in topic_list:
        file_path = os.path.join(target_dir, f"{slug}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        count += 1
    print(f"[*] 成功生成 {count} 篇文档 -> {subfolder}")
    return count

if __name__ == "__main__":
    print("[+] 开始生成 White-Box RAG Lab 100篇深度技术语料库...")
    total = 0
    total += write_topics(DISTRIBUTED_SYSTEMS, "01_distributed_systems")
    total += write_topics(STORAGE_ENGINES, "02_storage_engines")
    total += write_topics(LLM_INFRA, "03_llm_infra")
    total += write_topics(OS_KERNEL_NETWORK, "04_os_kernel_network")
    print(f"\n[OK] 全量 100 篇高质量技术拆解文档生成完毕！总计：{total} 篇。")
