# -*- coding: utf-8 -*-
"""
WhiteBox Transformer 根目录快捷启动入口
用户无论在根目录还是子目录，均可直接运行:
    python main.py
"""
import sys
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(root_dir, "whitebox_transformer")

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from whitebox_transformer.main import main

if __name__ == "__main__":
    main()
