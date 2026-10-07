"""
逐行追踪器 / Line-by-line tracer
================================
用 sys.settrace 记录一次训练迭代中 *我们自己的代码*（neurocore/ 与 server/tasks.py）
执行到的每一行，以及每行执行后新产生/改变的局部变量（张量记录形状与统计量）。
PyTorch 内部（C++/autograd）不追踪 —— 只看我们写的网络代码。

只在“慢动作单步讲解”时开启：追踪会让这一步变慢数十倍，平时训练不受影响。
Tracing is enabled only for the single "slow-motion" step; normal training runs untraced.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional

MAX_EVENTS = 6000          # 单次追踪最多记录多少个事件 / hard cap on recorded events
SKIP_NAMES = {"self", "__class__"}


def _num(x) -> Optional[float]:
    """转成 JSON 安全的浮点数（NaN/Inf -> None）/ JSON-safe float."""
    try:
        f = float(x)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def summarize(value):
    """把一个变量压缩成小字典，便于发给前端 / Compact, JSON-safe summary of a value."""
    # 张量 / 数组：记录形状、类型、统计量 / tensors & arrays
    if hasattr(value, "shape") and hasattr(value, "dtype") and not isinstance(value, type):
        out = {"k": "tensor", "s": [int(d) for d in value.shape], "dt": str(value.dtype).replace("torch.", "")}
        try:
            t = value
            if hasattr(t, "detach"):                                  # torch.Tensor：脱离计算图并转 float 便于统计
                t = t.detach().float()
            n = int(t.numel()) if hasattr(t, "numel") else int(t.size)
            if 0 < n <= 4_000_000:
                out.update(mean=_num(t.mean()), std=_num(t.std()) if n > 1 else 0.0,
                           min=_num(t.min()), max=_num(t.max()))
            if hasattr(value, "requires_grad"):
                out["grad"] = bool(value.requires_grad)
        except Exception:
            pass
        return out
    if isinstance(value, bool):
        return {"k": "val", "val": value}
    if isinstance(value, (int, float)):
        return {"k": "val", "val": _num(value) if isinstance(value, float) else value}
    if isinstance(value, str):
        return {"k": "val", "val": value[:60]}
    if isinstance(value, (tuple, list)) and len(value) <= 8:
        if all(isinstance(v, (int, float, bool)) for v in value):
            return {"k": "val", "val": [(_num(v) if isinstance(v, float) else v) for v in value]}
        if all(hasattr(v, "shape") for v in value):
            return {"k": "list", "items": [summarize(v) for v in value]}
    if isinstance(value, dict) and len(value) <= 8 and all(isinstance(v, (int, float)) for v in value.values()):
        return {"k": "val", "val": {str(k): _num(v) for k, v in value.items()}}
    return None   # 其他类型（模块、函数等）不展示 / modules, functions… are skipped


def _key(value):
    """判断变量是否变化 / change-detection key."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return ("t", id(value), tuple(value.shape))
    if isinstance(value, (int, float, bool, str)):
        return ("v", value)
    if isinstance(value, (tuple, list)):
        return ("l", id(value), len(value))
    return ("o", id(value))



def _rel(f, root):
    """相对路径；Windows 跨盘时退回绝对路径 / relative path, absolute if on another drive."""
    try:
        return os.path.relpath(f, root).replace(os.sep, "/")
    except ValueError:
        return f.replace(os.sep, "/")

class LineTracer:
    def __init__(self, include_dirs: List[str], max_events: int = MAX_EVENTS):
        self.include = [os.path.normcase(os.path.abspath(d)) for d in include_dirs]
        self.max_events = max_events
        self.files: List[str] = []
        self._file_idx: Dict[str, int] = {}
        self.events: List[dict] = []
        self.truncated = False
        self._frames: Dict[int, dict] = {}    # id(frame) -> {"last": event idx, "keys": {...}}
        self._depth = 0
        self._want_cache: Dict[str, bool] = {}
        self._old = None

    # ---- 文件过滤 / file filter -------------------------------------------------
    def _wanted(self, filename: str) -> bool:
        r = self._want_cache.get(filename)
        if r is None:
            p = os.path.normcase(os.path.abspath(filename))
            r = any(p.startswith(d) for d in self.include)
            self._want_cache[filename] = r
        return r

    def _fidx(self, filename: str) -> int:
        i = self._file_idx.get(filename)
        if i is None:
            i = len(self.files)
            self._file_idx[filename] = i
            self.files.append(filename)
        return i

    def _diff(self, frame, st):
        """找出自上一行以来新产生或改变的局部变量 / locals that changed since the last line."""
        changed = []
        keys = st["keys"]
        for name, val in frame.f_locals.items():
            if name in SKIP_NAMES or name.startswith("__"):
                continue
            k = _key(val)
            if keys.get(name) != k:
                keys[name] = k
                s = summarize(val)
                if s is not None:
                    s["n"] = name
                    changed.append(s)
        return changed

    def _push(self, ev):
        if len(self.events) >= self.max_events:
            self.truncated = True
            return -1
        ev["i"] = len(self.events)
        self.events.append(ev)
        return ev["i"]

    # ---- 追踪回调 / trace callbacks -----------------------------------------------
    def _global(self, frame, event, arg):
        if event != "call" or not self._wanted(frame.f_code.co_filename):
            return None
        self._depth += 1
        slf = frame.f_locals.get("self")
        st = {"last": -1, "keys": {}, "cls": type(slf).__name__ if slf is not None else ""}
        self._frames[id(frame)] = st
        args = self._diff(frame, st)                                # 函数参数 / arguments
        st["last"] = self._push({"t": "call", "f": self._fidx(frame.f_code.co_filename),
                                 "l": frame.f_code.co_firstlineno, "fn": frame.f_code.co_name,
                                 "c": st["cls"], "d": self._depth, "v": args})
        return self._local

    def _local(self, frame, event, arg):
        st = self._frames.get(id(frame))
        if st is None:
            return None
        if event == "line":
            changed = self._diff(frame, st)
            if changed and st["last"] >= 0:
                self.events[st["last"]].setdefault("v", []).extend(changed)   # 归属上一行 / produced by previous line
            st["last"] = self._push({"t": "line", "f": self._fidx(frame.f_code.co_filename),
                                     "l": frame.f_lineno, "fn": frame.f_code.co_name, "d": self._depth, "v": []})
        elif event == "return":
            changed = self._diff(frame, st)
            if changed and st["last"] >= 0:
                self.events[st["last"]].setdefault("v", []).extend(changed)
            ret = summarize(arg)
            if st["last"] >= 0 and ret is not None:
                self.events[st["last"]]["ret"] = ret                  # 返回值挂在 return 行 / return value
            self._frames.pop(id(frame), None)
            self._depth -= 1
        return self._local

    def __enter__(self):
        self._old = sys.gettrace()
        sys.settrace(self._global)
        return self

    def __exit__(self, *exc):
        sys.settrace(self._old)
        return False

    def add_synthetic(self, title: str, detail: str, values: Optional[list] = None):
        """追加一个“非 Python 行”的事件（如 C++ 中的反向传播）/ synthetic event."""
        self._push({"t": "note", "title": title, "detail": detail, "d": 1, "v": values or []})

    def result(self, root: str) -> dict:
        rel = [_rel(f, root) for f in self.files]
        return {"files": rel, "events": self.events, "truncated": self.truncated}
