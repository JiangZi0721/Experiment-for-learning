# -*- coding: utf-8 -*-
"""
一键下载 BAAI/bge-small-zh-v1.5 本地模型至 models/bge-small-zh-v1.5
自动处理国内镜像源与代理 SSL 证书问题，实现 100% 离线秒级加载。
"""
import os
import sys
import requests
import urllib3

urllib3.disable_warnings()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "bge-small-zh-v1.5")
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_URL = "https://hf-mirror.com/BAAI/bge-small-zh-v1.5/resolve/main"
FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "model.safetensors"
]

def download_file(filename):
    dest_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[*] 文件已存在，跳过: {filename} ({os.path.getsize(dest_path):,} bytes)")
        return

    url = f"{BASE_URL}/{filename}"
    print(f"[+] 正在下载: {filename} ...")
    resp = requests.get(url, verify=False, stream=True, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"下载失败 [{resp.status_code}]: {url}")

    total = int(resp.headers.get('content-length', 0))
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = (downloaded / total) * 100
                    sys.stdout.write(f"\r    -> 进度: {downloaded / 1024 / 1024:.1f}MB / {total / 1024 / 1024:.1f}MB ({percent:.1f}%)")
                    sys.stdout.flush()
    print(f"\n[OK] 完成: {filename}")

if __name__ == "__main__":
    print(f"[*] 开始下载 BAAI/bge-small-zh-v1.5 模型至: {MODEL_DIR}")
    for fname in FILES:
        download_file(fname)
    print("\n🎉 模型全部下载完毕！已就绪可离线加载。")
