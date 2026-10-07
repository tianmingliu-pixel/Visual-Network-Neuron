"""读取源代码并把每行拆成“代码 + 中文注释” / split source lines into code + comment."""
from __future__ import annotations

import io
import os
import tokenize
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED = ("neurocore/", "server/tasks.py")


def resolve(rel: str) -> str:
    rel = rel.replace("\\", "/").lstrip("/")
    if not rel.endswith(".py") or ".." in rel.split("/") or not rel.startswith(ALLOWED):
        raise PermissionError(rel)
    path = os.path.normpath(os.path.join(ROOT, rel))
    if not path.startswith(ROOT) or not os.path.isfile(path):
        raise FileNotFoundError(rel)
    return path


@lru_cache(maxsize=64)
def _parse(path: str, mtime: float):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")
    comments, docs = {}, set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                comments[tok.start[0]] = (tok.start[1], tok.string.lstrip("#").strip())
            elif tok.type == tokenize.STRING and tok.start[0] != tok.end[0]:
                docs.update(range(tok.start[0], tok.end[0] + 1))        # 多行字符串（文档说明）
    except (tokenize.TokenError, IndentationError):
        pass
    out = []
    for i, line in enumerate(lines, 1):
        if i in comments:
            col, c = comments[i]
            code = line[:col].rstrip()
        else:
            code, c = line.rstrip(), ""
        out.append({"n": i, "code": code, "comment": c, "doc": i in docs})
    return out


def read_source(rel: str) -> dict:
    path = resolve(rel)
    return {"file": rel, "lines": _parse(path, os.path.getmtime(path))}
