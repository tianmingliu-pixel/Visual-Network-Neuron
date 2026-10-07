"""
让测试无论从哪里、用哪种方式运行（pytest / python -m pytest / IDE）都能找到项目包。
Make the project root importable no matter how/where pytest is launched.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ 的上一级 = 项目根目录
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # 放在最前面，优先于同名的其他安装
