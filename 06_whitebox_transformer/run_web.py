# -*- coding: utf-8 -*-
"""
WhiteBox Transformer 根目录快捷网页启动入口
在根目录下直接运行:
    python run_web.py
"""
import sys
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket
from functools import partial

PORT = 8000

def find_free_port(start_port=8000, max_port=8050):
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return start_port

def locate_web_dir():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_file_dir, "whitebox_transformer", "web"),
        os.path.join(current_file_dir, "web"),
        os.path.join(os.getcwd(), "whitebox_transformer", "web"),
        os.path.join(os.getcwd(), "web")
    ]
    for cand in candidates:
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "index.html")):
            return cand
    raise FileNotFoundError(f"未找到 web 静态资源目录，已尝试检查路径: {candidates}")

def run_server():
    web_dir = locate_web_dir()
    port = find_free_port(PORT)
    server_address = ('', port)
    
    handler_class = partial(SimpleHTTPRequestHandler, directory=web_dir)
    httpd = HTTPServer(server_address, handler_class)
    url = f"http://localhost:{port}/index.html"
    
    print("\n" + "═" * 70)
    print("  🚀 WhiteBox Transformer 可视化网页学习服务已就绪！")
    print("═" * 70)
    print(f"  🌐 本地访问地址: {url}")
    print(f"  📂 静态资源目录: {web_dir}")
    print(f"  💡 提示: 您也可以直接在浏览器中打开:")
    print(f"     {os.path.join(web_dir, 'index.html')}")
    print("  ⌨️  按 Ctrl+C 可停止网页服务。")
    print("═" * 70 + "\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n服务器已安全停止。感谢使用 WhiteBox Transformer！")

if __name__ == "__main__":
    run_server()
