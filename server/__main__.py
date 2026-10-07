"""
启动可视化界面 / Launch the visual training UI
    python -m server              # http://127.0.0.1:8765
    python -m server --open       # 并自动打开浏览器 / and open the browser
"""
import argparse
import os
import threading
import webbrowser

import uvicorn

from .app import create_app


def _utf8_console():
    """输出被重定向到日志文件时，Windows 默认用 cp1252 编码，打印中文会崩溃；统一改成 UTF-8。
    On Windows, redirected stdout defaults to cp1252 and crashes on Chinese text; force UTF-8."""
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    _utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    url = f"http://{a.host}:{a.port}"
    from .config import CLOUD, DATA_DIR, TOKEN
    print(f"NeuroCore UI -> {url}   (Ctrl+C 退出 / quit)")
    print(f"  数据目录 {DATA_DIR} · 云端模式 {'开' if CLOUD else '关'} · 写操作口令 {'已设置' if TOKEN else '未设置'}")
    if a.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
