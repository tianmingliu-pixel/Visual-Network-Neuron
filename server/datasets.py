"""
通用数据导入与分析 / Universal data import & analysis
====================================================
支持的数据（自动识别）/ Supported inputs (auto-detected):
  · 表格：.csv .tsv .txt .json .jsonl .xlsx/.xls* .parquet*      （* 需要 pandas / openpyxl）
          自动判断每列类型：数值 / 类别(独热) / 文本(哈希词袋) / ID(丢弃)；默认最后一列为目标
  · 图片 / 音频：文件夹（或上传的整个文件夹 / zip），标签来源按优先级自动识别：
          ① 元数据表：文件夹里的 CSV/Excel/JSON，有一列是文件名、另一列是标签（labels.csv: filename,label）
          ② 类别子文件夹：root/猫/*.jpg  root/狗/*.jpg（也支持 train/ val/ 再分类别）
          ③ 文件名前缀：cat_001.jpg、dog.12.jpg → cat / dog
     音频支持 .wav（PCM/浮点，无需依赖）；.flac .ogg .mp3 需要 pip install soundfile
  · 数组：.npz / .npy（X 与 y；或 x_train/y_train/x_test/y_test；或单个二维数组，最后一列为目标）
  · 压缩包：.zip 会先解压再按上面规则识别
只依赖 numpy；图片需要 Pillow。全部在本机处理，不上传到任何地方。
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import wave
import zipfile
import zlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from .config import DATA_DIR  # noqa: E402
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

TAB_EXT = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".parquet"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
AUD_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".aiff", ".aif"}
ARR_EXT = {".npz", ".npy"}
LABEL_COLS = ("label", "labels", "class", "classes", "category", "target", "y", "species", "breed", "tag",
              "类别", "标签", "分类", "类型", "目标")
FILE_COLS = ("filename", "file", "file_name", "fname", "path", "filepath", "image", "img", "image_id", "audio",
             "id", "name", "文件名", "文件", "路径", "图片")
MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "-", "--", "nil", "缺失", "无"}
SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val", "test": "val"}

MAX_ROWS = 200_000
MAX_CATEGORIES = 30
TEXT_HASH_DIM = 32
SR = 16000
N_BANDS = 24


class DataError(Exception):
    """给用户看的友好错误 / user-facing error."""


@dataclass
class Dataset:
    kind: str                       # tabular | image | audio | array
    task_type: str                  # classification | regression
    X: np.ndarray                   # (N, D) 或 (N, C, H, W)，已标准化
    y: np.ndarray                   # 类别下标 int64 或 回归值 float32（已标准化）
    feature_names: List[str]
    class_names: List[str]
    source: str
    target_name: str = "label"
    train_idx: np.ndarray = None
    val_idx: np.ndarray = None
    columns: list = field(default_factory=list)     # 原始列统计 / raw column stats
    dropped: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    preview_rows: Optional[dict] = None
    y_mean: float = 0.0
    y_std: float = 1.0
    extra: dict = field(default_factory=dict)

    @property
    def is_image(self):
        return self.X.ndim == 4

    def describe_model(self) -> str:
        if self.is_image:
            return f"ViT · 输入 {self.X.shape[1]}×{self.X.shape[2]}×{self.X.shape[3]} → {len(self.class_names)} 类"
        out = len(self.class_names) if self.task_type == "classification" else 1
        h1, h2 = hidden_sizes(self.X.shape[1])
        return f"MLP {self.X.shape[1]} → {h1} → {h2} → {out}"


def hidden_sizes(d: int):
    h1 = int(min(128, max(16, 2 ** math.ceil(math.log2(max(2, d))))))
    return h1, max(8, h1 // 2)


# =============================================================================
# 入口 / entry points
# =============================================================================
def safe_name(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    base = re.sub(r"[^\w.\-一-鿿]+", "_", base).strip("._") or "upload"
    return base[:120]


def safe_relpath(rel: str) -> str:
    """上传文件夹时保留相对路径（类别子文件夹），但清洗每一段并阻止 .. 穿越 / sanitize a relative path."""
    parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    parts = [safe_name(p) for p in parts][-6:]
    return os.path.join(*parts) if parts else "upload"


def save_upload(name: str, data: bytes) -> str:
    """保存上传的文件；zip 自动安全解压 / save an upload, extracting zips safely."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    fname = safe_name(name)
    path = os.path.join(UPLOAD_DIR, fname)
    with open(path, "wb") as f:
        f.write(data)
    if fname.lower().endswith(".zip"):
        return extract_zip(path)
    return path


def extract_zip(path: str) -> str:
    dest = os.path.splitext(path)[0] + "_unzipped"                 # 不覆盖同名文件夹 / never clobber a same-named folder
    os.makedirs(dest, exist_ok=True)
    real_dest = os.path.realpath(dest)
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename
            try:                                     # 修正中文文件名（zip 默认 cp437）
                if not (info.flag_bits & 0x800):
                    name = name.encode("cp437").decode("gbk")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            target = os.path.realpath(os.path.join(dest, name))
            if not target.startswith(real_dest + os.sep) and target != real_dest:
                continue                             # 防止 zip 路径穿越 / zip-slip guard
            if name.endswith("/") or info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
    # 若压缩包里只有一个顶层文件夹，直接进入它 / descend into a single top-level folder
    entries = [e for e in os.listdir(dest) if not e.startswith(("__MACOSX", "."))]
    if len(entries) == 1 and os.path.isdir(os.path.join(dest, entries[0])):
        return os.path.join(dest, entries[0])
    return dest


def _walk_files(root, limit=200_000):
    n = 0
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if not d.startswith((".", "__MACOSX"))]
        for fn in fns:
            if fn.startswith("."):
                continue
            yield os.path.join(dp, fn)
            n += 1
            if n >= limit:
                return


def detect(path: str, target: Optional[str] = None) -> dict:
    """快速识别数据类型与标签来源（不做重计算）/ quick inspection incl. where labels come from."""
    path = os.path.expanduser(str(path).strip().strip('"').strip("'"))
    if not path:
        raise DataError("请先输入路径或上传文件。")
    if not os.path.exists(path):
        raise DataError(f"路径不存在：{path}")
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".zip":
            return detect(extract_zip(path), target)
        if ext in TAB_EXT:
            header, rows = read_table(path, max_rows=200)
            return {"path": path, "kind": "tabular", "columns": header, "n_preview": len(rows),
                    "suggested_target": header[-1] if header else None}
        if ext in ARR_EXT:
            arrs = _array_shapes(path)
            return {"path": path, "kind": "array", "arrays": arrs}
        if ext in IMG_EXT or ext in AUD_EXT:
            raise DataError("只有一个图片/音频文件，无法训练。请上传整个文件夹（「上传文件夹」按钮或拖入文件夹），"
                            "或 zip 压缩包；类别用子文件夹（数据/猫/1.jpg）、文件名前缀（cat_1.jpg）"
                            "或一张元数据表（labels.csv：filename,label）来表示。")
        raise DataError(f"暂不支持的文件类型：{ext or '（无扩展名）'}")
    counts = {"image": 0, "audio": 0, "table": 0, "array": 0}
    for f in _walk_files(path, 20000):
        e = os.path.splitext(f)[1].lower()
        if e in IMG_EXT:
            counts["image"] += 1
        elif e in AUD_EXT:
            counts["audio"] += 1
        elif e in TAB_EXT:
            counts["table"] += 1
        elif e in ARR_EXT:
            counts["array"] += 1
    media = "image" if counts["image"] >= counts["audio"] else "audio"
    if counts[media] == 0:
        if counts["array"]:
            arrs = _array_shapes(path)
            return {"path": path, "kind": "array", "arrays": arrs}
        tabs = [f for f in _walk_files(path, 2000) if os.path.splitext(f)[1].lower() in TAB_EXT]
        if len(tabs) == 1:                               # 例如 zip 里只有一个 CSV
            return detect(tabs[0], target)
        if tabs:
            raise DataError("这个文件夹里有多个表格文件，请直接选择其中一个：" + "、".join(os.path.basename(t) for t in tabs[:5]))
        raise DataError("文件夹里没有找到可识别的图片(.jpg/.png…)、音频(.wav…)、表格或数组(.npz/.npy)文件。")
    tree = folder_tree(path, IMG_EXT | AUD_EXT | TAB_EXT)
    try:
        lab = _label_files(path, IMG_EXT if media == "image" else AUD_EXT, target)
    except DataError as e:                          # 还不能训练，但把已上传的内容和下一步告诉用户
        return {"path": path, "kind": media, "files": counts[media], "ready": False, "problem": str(e),
                "tree": tree, "classes": {}}
    out = {"path": path, "kind": media, "files": counts[media], "ready": True, "tree": tree, "label_source": lab["source"],
           "label_desc": lab["desc"], "has_split": bool(lab["splits"]), "classes": {}, "numeric_target": lab["numeric"]}
    if lab["meta"]:
        out.update(meta_file=os.path.relpath(lab["meta"]["file"], path), columns=lab["meta"]["columns"],
                   file_column=lab["meta"]["file_col"], suggested_target=lab["meta"]["target"],
                   unmatched=lab["meta"]["unmatched"])
    if not lab["numeric"]:
        for _, v in lab["pairs"]:
            out["classes"][str(v)] = out["classes"].get(str(v), 0) + 1
    return out


def folder_tree(root, exts, depth=2):
    """已上传文件夹的结构：每个子文件夹里有多少个可用文件 / per-subfolder file counts."""
    out = {}
    for f in _walk_files(root, 50000):
        if os.path.splitext(f)[1].lower() not in exts:
            continue
        rel = os.path.relpath(os.path.dirname(f), root).replace(os.sep, "/")
        key = "/".join(rel.split("/")[:depth]) if rel != "." else "（根目录）"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def list_uploads(limit=30):
    """data/uploads 下已上传的数据（新的在前）/ previously uploaded datasets."""
    if not os.path.isdir(UPLOAD_DIR):
        return []
    items = []
    for name in os.listdir(UPLOAD_DIR):
        p = os.path.join(UPLOAD_DIR, name)
        if name.startswith(".") or name.endswith("_unzipped") and os.path.exists(p[:-9] + ".zip"):
            continue
        n = sum(1 for _ in _walk_files(p, 100000)) if os.path.isdir(p) else 1
        items.append({"name": name, "path": p, "is_dir": os.path.isdir(p), "files": n, "mtime": os.path.getmtime(p),
                      "tree": folder_tree(p, IMG_EXT | AUD_EXT | TAB_EXT | ARR_EXT, 1) if os.path.isdir(p) else {}})
    items.sort(key=lambda x: -x["mtime"])
    return items[:limit]


def load_dataset(path: str, target: Optional[str] = None, task_type: str = "auto", img_size: int = 32,
                 max_per_class: int = 1000, val_ratio: float = 0.2, seed: int = 0) -> Dataset:
    info = detect(path, target)
    if info.get("ready") is False:
        raise DataError(info["problem"])
    path, kind = info["path"], info["kind"]
    if kind == "tabular":
        ds = load_tabular(path, target, task_type)
    elif kind == "array":
        ds = load_arrays(path, task_type, img_size)
    elif kind == "image":
        ds = load_images(path, img_size, max_per_class, target)
    else:
        ds = load_audio(path, max_per_class, target, task_type)
    if len(ds.y) < 10:
        raise DataError(f"样本太少（{len(ds.y)} 条），至少需要 10 条。")
    if ds.train_idx is None:
        ds.train_idx, ds.val_idx = split(ds.y, ds.task_type, val_ratio, seed)
    _standardize(ds)
    return ds


# =============================================================================
# 表格 / tabular
# =============================================================================
def _read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def read_table(path: str, max_rows: int = MAX_ROWS):
    """返回 (表头, 行列表)；值为字符串/数字/None / returns (header, rows)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls", ".parquet"):
        try:
            import pandas as pd
        except ImportError:
            if ext == ".xlsx":                       # 没装 pandas 也能读 xlsx / built-in xlsx reader
                return _read_xlsx(path, max_rows)
            raise DataError("读取 .xls / Parquet 需要 pandas：pip install -r requirements-data.txt（或先另存为 CSV / .xlsx）")
        try:
            df = pd.read_parquet(path) if ext == ".parquet" else pd.read_excel(path)
        except ImportError as e:
            if ext == ".xlsx":
                return _read_xlsx(path, max_rows)
            raise DataError(f"缺少依赖：{e}。可运行 pip install -r requirements-data.txt")
        df = df.head(max_rows)
        header = [str(c) for c in df.columns]
        rows = [[None if (isinstance(v, float) and math.isnan(v)) else v for v in r] for r in df.itertuples(index=False)]
        return header, rows
    text = _read_text(path)
    if ext in (".json", ".jsonl", ".ndjson") or text.lstrip()[:1] in ("[", "{"):
        return _read_json(text, ext, max_rows)
    sample = text[:20000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        delim = dialect.delimiter
    except csv.Error:
        delim = "\t" if ext == ".tsv" or sample.count("\t") > sample.count(",") else ","
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows = []
    for i, r in enumerate(reader):
        if not any(c.strip() for c in r):
            continue
        rows.append([c.strip() for c in r])
        if len(rows) > max_rows:
            break
    if not rows:
        raise DataError("表格是空的。")
    header = rows[0]
    if all(_to_float(h) is not None for h in header if h):          # 没有表头 / no header row
        header = [f"col_{i + 1}" for i in range(len(rows[0]))]
    else:
        rows = rows[1:]
    width = len(header)
    rows = [(r + [""] * width)[:width] for r in rows]
    header = [h or f"col_{i + 1}" for i, h in enumerate(header)]
    return header, rows


def _read_xlsx(path, max_rows=MAX_ROWS):
    """只用标准库读取 .xlsx 第一个工作表 / minimal stdlib .xlsx reader (first sheet)."""
    import xml.etree.ElementTree as ET
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise DataError("这个 .xlsx 文件已损坏或不是真正的 Excel 文件。")
    with z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
        sheets = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
        if not sheets:
            raise DataError("Excel 里没有工作表。")
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml" if "xl/worksheets/sheet1.xml" in sheets else sheets[0]))
    rows = []
    for row in root.iter("{%s}row" % ns["m"]):
        vals = {}
        for c in row.findall("m:c", ns):
            ref, t = c.get("r", ""), c.get("t")
            col = 0
            for ch in re.match(r"[A-Z]*", ref).group():
                col = col * 26 + ord(ch) - 64
            col = col - 1 if col else len(vals)
            v = c.find("m:v", ns)
            if t == "s" and v is not None:
                val = shared[int(v.text)]
            elif t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter("{%s}t" % ns["m"]))
            elif t == "b" and v is not None:
                val = v.text == "1"
            else:
                val = v.text if v is not None else None
                f = _to_float(val) if t != "str" else None
                val = f if f is not None else val
            vals[col] = val
        if vals:
            width = max(vals) + 1
            rows.append([vals.get(i) for i in range(width)])
        if len(rows) > max_rows:
            break
    if not rows:
        raise DataError("表格是空的。")
    width = max(len(r) for r in rows)
    rows = [r + [None] * (width - len(r)) for r in rows]
    header = rows[0]
    if all(isinstance(h, (int, float)) for h in header if h is not None):
        header = [f"col_{i + 1}" for i in range(width)]
    else:
        rows = rows[1:]
    header = [str(h) if h not in (None, "") else f"col_{i + 1}" for i, h in enumerate(header)]
    return header, rows


def _read_json(text, ext, max_rows):
    if ext in (".jsonl", ".ndjson") or (text.lstrip().startswith("{") and "\n{" in text):
        recs = [json.loads(l) for l in text.splitlines() if l.strip()][:max_rows]
    else:
        obj = json.loads(text)
        if isinstance(obj, dict):
            lists = {k: v for k, v in obj.items() if isinstance(v, list)}
            if "data" in obj and isinstance(obj["data"], list):
                obj = obj["data"]
            elif lists and len({len(v) for v in lists.values()}) == 1:   # 按列存储 / columnar
                keys = list(lists)
                obj = [dict(zip(keys, vals)) for vals in zip(*lists.values())]
            else:
                raise DataError("JSON 需要是“记录列表”[{...},{...}] 或按列存储 {列名: [...]}。")
        recs = obj[:max_rows]
    if not recs or not isinstance(recs[0], dict):
        raise DataError("JSON 里没有找到记录（字典）列表。")
    header = []
    for r in recs[:1000]:
        for k in r:
            if k not in header:
                header.append(k)
    rows = [[r.get(k) for k in header] for r in recs]
    rows = [[json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for v in row] for row in rows]
    return [str(h) for h in header], rows


def _to_float(v):
    if v is None or isinstance(v, bool):
        return None if v is None else float(v)
    if isinstance(v, (int, float, np.number)):
        f = float(v)
        return f if math.isfinite(f) else None
    s = str(v).strip().replace(",", "") if isinstance(v, str) else str(v)
    if s.lower() in MISSING:
        return None
    if s.endswith("%"):
        s = s[:-1]
    try:
        f = float(s)
        return f if math.isfinite(f) else None
    except ValueError:
        return None


def _is_missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip().lower() in MISSING)


def _hist(vals, bins=20):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return None
    lo, hi = float(a.min()), float(a.max())
    if lo == hi:
        hi = lo + 1
    counts, _ = np.histogram(a, bins=bins, range=(lo, hi))
    return {"counts": counts.tolist(), "min": lo, "max": hi}


def _profile_column(name, values):
    n = len(values)
    present = [v for v in values if not _is_missing(v)]
    nums = [f for f in (_to_float(v) for v in present) if f is not None]
    info = {"name": name, "missing": n - len(present), "unique": len({str(v) for v in present})}
    numeric_ratio = len(nums) / max(1, len(present))
    if present and numeric_ratio >= 0.95:
        a = np.asarray(nums)
        info.update(type="numeric", mean=float(a.mean()), std=float(a.std()), min=float(a.min()), max=float(a.max()),
                    hist=_hist(a), integer=bool(np.all(np.mod(a, 1) == 0)))
        return info
    strs = [str(v).strip() for v in present]
    counts = {}
    for s in strs:
        counts[s] = counts.get(s, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])
    info["top"] = [[k[:40], c] for k, c in top[:12]]
    avg_len = sum(len(s) for s in strs) / max(1, len(strs))
    if info["unique"] <= max(MAX_CATEGORIES, 2) or info["unique"] <= 0.05 * len(strs):
        info["type"] = "categorical"
    elif avg_len >= 12 or any(" " in s for s in strs[:200]) or re.search(r"[一-鿿]{4,}", "".join(strs[:50])):
        info["type"] = "text"
    elif info["unique"] >= 0.9 * len(strs):
        info["type"] = "id"
    else:
        info["type"] = "categorical"
    return info


_ID_EXACT = re.compile(r"^(id|index|idx|uuid|key|编号|序号|行号|no\.?)$", re.I)
_ID_SUFFIX = re.compile(r"(_id|_ID|[a-z]Id|ID)$")


class _IdName:
    @staticmethod
    def search(name):
        return bool(_ID_EXACT.search(name) or _ID_SUFFIX.search(name))


ID_NAME = _IdName()


def _hash_text(s, dim=TEXT_HASH_DIM):
    v = np.zeros(dim, np.float32)
    toks = re.findall(r"[a-zA-Z0-9]+|[一-鿿]", str(s).lower())
    for t in toks:
        v[zlib.crc32(t.encode()) % dim] += 1.0
    return np.log1p(v)


def load_tabular(path, target=None, task_type="auto") -> Dataset:
    header, rows = read_table(path)
    if len(header) < 2:
        raise DataError("表格至少需要两列（特征 + 目标）。")
    cols = {h: [r[i] for r in rows] for i, h in enumerate(header)}
    profiles = [_profile_column(h, cols[h]) for h in header]
    prof = {p["name"]: p for p in profiles}
    if target is None or target not in cols:
        cands = [p["name"] for p in profiles if p["type"] != "id" and not ID_NAME.search(p["name"])]
        target = cands[-1] if cands else header[-1]
    tprof = prof[target]
    # ---- 目标列 / target -------------------------------------------------------
    traw = cols[target]
    keep = [i for i, v in enumerate(traw) if not _is_missing(v)]
    notes = []
    if len(keep) < len(traw):
        notes.append(f"目标列 {target} 有 {len(traw) - len(keep)} 行缺失，已去掉这些行。")
    if task_type == "auto":
        if tprof["type"] == "numeric" and not (tprof.get("integer") and tprof["unique"] <= 20):
            task_type = "regression"
        else:
            task_type = "classification"
    if task_type == "regression":
        if tprof["type"] != "numeric":
            raise DataError(f"目标列 {target} 不是数值，不能做回归；请改选“分类”。")
        y = np.array([_to_float(traw[i]) for i in keep], np.float64)
        ok = ~np.isnan(y)
        keep = [k for k, o in zip(keep, ok) if o]
        y = y[ok].astype(np.float32)
        class_names = []
    else:
        labels = [str(traw[i]).strip() for i in keep]
        if tprof["type"] == "numeric":
            labels = [str(int(f)) if f is not None and float(f).is_integer() else str(f)
                      for f in (_to_float(traw[i]) for i in keep)]
        uniq = sorted(set(labels), key=lambda s: (_to_float(s) is None, _to_float(s) or 0, s))
        if len(uniq) > 200:
            raise DataError(f"目标列 {target} 有 {len(uniq)} 个不同取值，太多了不适合分类；如果是数值请选择“回归”。")
        if len(uniq) < 2:
            raise DataError(f"目标列 {target} 只有一个取值，无法学习。")
        idx = {c: i for i, c in enumerate(uniq)}
        y = np.array([idx[l] for l in labels], np.int64)
        class_names = uniq
    # ---- 特征列 / features ----------------------------------------------------
    feats, names, dropped = [], [], []
    for p in profiles:
        h = p["name"]
        if h == target:
            continue
        vals = [cols[h][i] for i in keep]
        if p["type"] == "id" or (ID_NAME.search(h) and p["unique"] >= 0.9 * len(rows)):
            dropped.append({"name": h, "reason": "ID/编号列（每行都不同，没有可学习的规律）"})
            continue
        if p["missing"] >= 0.95 * len(rows):
            dropped.append({"name": h, "reason": "几乎全部缺失"})
            continue
        if p["type"] == "numeric":
            if p["std"] == 0:
                dropped.append({"name": h, "reason": "常数列（没有变化）"})
                continue
            a = np.array([np.nan if _to_float(v) is None else _to_float(v) for v in vals], np.float64)
            miss = np.isnan(a)
            a[miss] = np.nanmean(a) if (~miss).any() else 0.0
            feats.append(a[:, None]); names.append(h)
            if miss.mean() > 0.05:                                     # 缺失指示列 / missing indicator
                feats.append(miss[:, None].astype(np.float64)); names.append(f"{h}·缺失")
        elif p["type"] == "categorical":
            sv = ["" if _is_missing(v) else str(v).strip() for v in vals]
            counts = {}
            for v in sv:
                if v:
                    counts[v] = counts.get(v, 0) + 1
            if len(counts) <= 1:
                dropped.append({"name": h, "reason": "只有一个取值"})
                continue
            cats = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:MAX_CATEGORIES]
            for c in cats:                                              # 独热编码 / one-hot
                feats.append(np.array([v == c for v in sv], np.float64)[:, None]); names.append(f"{h}={c[:30]}")
            if len(counts) > len(cats):
                cs = set(cats)
                feats.append(np.array([v not in cs and v != "" for v in sv], np.float64)[:, None]); names.append(f"{h}=其他")
        elif p["type"] == "text":
            feats.append(np.stack([_hash_text(v) for v in vals]).astype(np.float64))
            names += [f"{h}#词{i}" for i in range(TEXT_HASH_DIM)]
            notes.append(f"文本列 {h} 已用哈希词袋编码成 {TEXT_HASH_DIM} 维。")
    if not feats:
        raise DataError("去掉 ID/常数/缺失列后没有剩下可用的特征列。")
    X = np.concatenate(feats, 1).astype(np.float32)
    preview = {"header": header, "rows": [[_cell(v) for v in r] for r in rows[:8]]}
    ds = Dataset("tabular", task_type, X, y, names, class_names, path, target_name=target,
                 columns=profiles, dropped=dropped, notes=notes, preview_rows=preview)
    return ds


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return round(v, 4) if math.isfinite(v) else ""
    s = str(v)
    return s if len(s) <= 40 else s[:38] + "…"


# =============================================================================
# 图片 / images
# =============================================================================
def _split_of(root, f):
    """路径中出现 train/ val/ test/ 目录时记下划分 / split from a train|val|test path component."""
    for part in os.path.relpath(f, root).replace(os.sep, "/").split("/")[:-1]:
        if part.lower() in SPLIT_NAMES:
            return SPLIT_NAMES[part.lower()]
    return None


def _class_dirs(root, exts):
    """类别子文件夹：返回 {类别: [文件]}；不满足条件返回 {} / class sub-folders (optionally under train/val)."""
    subs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith((".", "__")))
    classes = {}
    if subs and all(s.lower() in SPLIT_NAMES for s in subs):        # root/train/类别/…
        for s in subs:
            for c, files in _class_dirs(os.path.join(root, s), exts).items():
                classes.setdefault(c, []).extend(files)
        return classes
    for d in subs:
        files = sorted(f for f in _walk_files(os.path.join(root, d)) if os.path.splitext(f)[1].lower() in exts)
        if files:
            classes[d] = files
    return classes


def _norm_key(v):
    s = str(v).strip().replace("\\", "/").lower()
    return s[2:] if s.startswith("./") else s


def _find_metadata(root, files, target=None):
    """找一张“文件名 → 标签”的元数据表 / find a table whose column matches the media file names."""
    keys, dup = {}, set()
    for f in files:
        rel = os.path.relpath(f, root).replace(os.sep, "/").lower()
        base = os.path.basename(rel)
        for k in {rel, base, os.path.splitext(rel)[0], os.path.splitext(base)[0]}:
            if k in keys and keys[k] != f:
                dup.add(k)                                 # 同名文件在不同文件夹：这个键有歧义
            keys.setdefault(k, f)
    for k in dup:
        keys.pop(k, None)
    best = None
    tabs = [t for t in _walk_files(root, 50000) if os.path.splitext(t)[1].lower() in TAB_EXT
            and os.path.getsize(t) < 200 * 1024 ** 2][:30]
    for t in tabs:
        try:
            header, rows = read_table(t)
        except Exception:  # noqa
            continue
        if len(header) < 2 or not rows:
            continue
        for ci, h in enumerate(header):
            named = str(h).strip().lower() in FILE_COLS
            hit = {}
            for r in rows:
                k = _norm_key(r[ci]) if r[ci] is not None else ""
                if not named and _to_float(k) is not None:   # 纯数字只在列名像“文件名/ID”时才匹配
                    continue
                f = keys.get(k) or keys.get(k.rsplit("/", 1)[-1]) or keys.get(os.path.splitext(k.rsplit("/", 1)[-1])[0])
                if f is not None:
                    hit[f] = r
            score = len(hit) + (0.5 if str(h).strip().lower() in FILE_COLS else 0)
            if len(hit) >= 4 and len(hit) >= 0.3 * min(len(files), len(rows)) and (best is None or score > best[0]):
                best = (score, t, header, ci, hit)
    if best is None:
        return None
    _, t, header, fc, hit = best
    others = [h for i, h in enumerate(header) if i != fc]
    tcol = target if target in others else None
    if tcol is None:
        tcol = next((h for h in others if str(h).strip().lower() in LABEL_COLS), None)
    if tcol is None:                                   # 取第一列“像标签”的列（取值种类不多）
        for h in others:
            vals = {str(r[header.index(h)]) for r in hit.values() if not _is_missing(r[header.index(h)])}
            if 2 <= len(vals) <= max(MAX_CATEGORIES, len(hit) // 3):
                tcol = h
                break
    tcol = tcol or others[-1]
    ti = header.index(tcol)
    pairs = [(f, r[ti]) for f, r in hit.items() if r[ti] is not None and not _is_missing(r[ti])]
    return {"file": t, "columns": header, "file_col": header[fc], "target": tcol, "pairs": sorted(pairs, key=lambda p: p[0]),
            "unmatched": len(files) - len(hit)}


def _prefix_label(stem):
    m = re.match(r"^([^\W\d_](?:[^\W\d_]|[ \-][^\W\d_])*?)[\s_.\-()（）#]*\d", stem, re.U)
    if m:
        return m.group(1).strip().lower()
    m = re.match(r"^([^\W\d_]+)[_.\-]", stem, re.U)
    return m.group(1).lower() if m else None


def _label_files(root, exts, target=None) -> dict:
    """找出每个文件的标签：元数据表 > 类别子文件夹 > 文件名前缀 / where do the labels come from?"""
    files = sorted(f for f in _walk_files(root) if os.path.splitext(f)[1].lower() in exts)
    meta = _find_metadata(root, files, target)
    if meta is not None and meta["pairs"]:
        pairs = meta["pairs"]
        num = [_to_float(v) for _, v in pairs]
        numeric = all(v is not None for v in num) and len(set(num)) > MAX_CATEGORIES
        return {"pairs": pairs, "source": "metadata", "meta": meta, "numeric": numeric,
                "splits": {f: s for f, _ in pairs if (s := _split_of(root, f))},
                "desc": f"元数据表 {os.path.relpath(meta['file'], root)}：按「{meta['file_col']}」列匹配文件，"
                        f"标签取「{meta['target']}」列（匹配到 {len(pairs)} 个文件）"}
    classes = _class_dirs(root, exts)
    if len(classes) >= 2:
        pairs = [(f, c) for c, fs in classes.items() for f in fs]
        return {"pairs": pairs, "source": "folders", "meta": None, "numeric": False,
                "splits": {f: s for f, _ in pairs if (s := _split_of(root, f))},
                "desc": f"类别子文件夹（{len(classes)} 类）"}
    labeled = [(f, _prefix_label(os.path.splitext(os.path.basename(f))[0])) for f in files]
    labeled = [(f, l) for f, l in labeled if l]
    counts = {}
    for _, l in labeled:
        counts[l] = counts.get(l, 0) + 1
    good = {l for l, n in counts.items() if n >= 2}
    if len(good) >= 2 and len(labeled) >= 0.8 * len(files) and len(good) <= max(2, len(files) // 3):
        pairs = [(f, l) for f, l in labeled if l in good]
        return {"pairs": pairs, "source": "filename", "meta": None, "numeric": False,
                "splits": {f: s for f, _ in pairs if (s := _split_of(root, f))},
                "desc": f"文件名前缀（例如 {os.path.basename(pairs[0][0])} → {pairs[0][1]}），共 {len(good)} 类"}
    kind = "图片" if exts is IMG_EXT else "音频"
    one = list(classes) or sorted(good) or sorted(counts)
    if len(one) == 1:
        n1 = len(classes[one[0]]) if classes else counts.get(one[0], len(files))
        raise DataError(f"目前只有 1 个类别「{one[0]}」（{n1} 个{kind}）。分类至少需要 2 个类别——"
                        f"请继续上传另一个类别的文件夹（例如再上传 dog 文件夹），它会自动加入同一个数据集。")
    raise DataError(f"找到 {len(files)} 个{kind}文件，但没找到标签。任选一种方式告诉我每个文件属于哪一类：\n"
                    f"① 放一张元数据表（如 labels.csv，两列：filename,label）；\n"
                    f"② 按类别分子文件夹：数据/猫/1.jpg、数据/狗/2.jpg（至少 2 类）；\n"
                    f"③ 文件名带类别前缀：cat_001.jpg、dog_002.jpg。")


def _encode_labels(raw, numeric):
    if numeric:
        return np.array([_to_float(v) for v in raw], np.float32), [], "regression"
    keys = [str(v).strip() for v in raw]
    fl = [_to_float(k) for k in keys]
    if all(v is not None for v in fl):                # 数字标签按数值排序 / numeric labels sorted numerically
        uniq = sorted(set(keys), key=lambda k: float(k))
    else:
        uniq = sorted(set(keys))
    idx = {u: i for i, u in enumerate(uniq)}
    return np.array([idx[k] for k in keys], np.int64), uniq, "classification"


def _even_take(files, k):
    if len(files) <= k:
        return files
    idx = np.linspace(0, len(files) - 1, k).round().astype(int)
    return [files[i] for i in idx]


def _cap_per_class(pairs, k):
    by = {}
    for f, l in pairs:
        by.setdefault(str(l), []).append((f, l))
    return [p for v in by.values() for p in _even_take(v, k)]


def load_images(root, img_size=32, max_per_class=1000, target=None) -> Dataset:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise DataError("读取图片需要 Pillow：pip install pillow")
    img_size = int(max(8, min(128, img_size)) // 4 * 4)
    lab = _label_files(root, IMG_EXT, target)
    if lab["numeric"]:
        raise DataError(f"标签列「{lab['meta']['target']}」是连续数值；图片目前只支持分类。请在“目标列”里换一列类别标签。")
    pairs = _cap_per_class(lab["pairs"], max_per_class)
    # 判断灰度还是彩色 / grayscale vs color
    gray = True
    for f, _ in pairs[:: max(1, len(pairs) // 12)][:12]:
        try:
            with Image.open(f) as im:
                if im.mode not in ("L", "1", "LA", "I", "I;16", "F"):
                    a = np.asarray(im.convert("RGB").resize((16, 16)), np.int16)
                    if np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() > 2:
                        gray = False
                        break
        except Exception:  # noqa
            continue
    C = 1 if gray else 3
    X, raw, sp, bad, used = [], [], [], 0, []
    for f, l in pairs:
        try:
            with Image.open(f) as im:
                im = ImageOps.exif_transpose(im).convert("L" if gray else "RGB")
                im = ImageOps.fit(im, (img_size, img_size))
                a = np.asarray(im, np.float32) / 127.5 - 1.0
        except Exception:  # noqa
            bad += 1
            continue
        X.append(a[None] if gray else a.transpose(2, 0, 1))
        raw.append(l)
        used.append(f)
        sp.append(lab["splits"].get(f))
    if not X:
        raise DataError("没有成功读取任何图片（文件可能损坏或格式不受 Pillow 支持）。")
    y, names, _ = _encode_labels(raw, False)
    if len(names) < 2:
        raise DataError(f"只找到 1 个类别（{names[0]}），至少需要 2 个类别。")
    tname = lab["meta"]["target"] if lab["meta"] else ("文件夹(类别)" if lab["source"] == "folders" else "文件名前缀")
    ds = Dataset("image", "classification", np.stack(X).astype(np.float32), y,
                 [f"像素 {C}×{img_size}×{img_size}"], names, root, target_name=tname)
    ds.extra["files"] = [os.path.relpath(f, root).replace(os.sep, "/") for f in used]
    ds.notes.append("标签来源：" + lab["desc"] + "。")
    ds.notes.append(f"图片统一裁剪缩放为 {img_size}×{img_size}，{'灰度' if gray else '彩色 RGB'}，像素归一化到 [-1, 1]。")
    if lab["meta"] and lab["meta"]["unmatched"]:
        ds.notes.append(f"{lab['meta']['unmatched']} 个图片在元数据表里找不到对应行，已跳过。")
    if bad:
        ds.notes.append(f"{bad} 个文件无法读取，已跳过。")
    if len(pairs) < len(lab["pairs"]):
        ds.notes.append(f"每类最多取 {max_per_class} 张（可在选项里调整）。")
    _apply_folder_split(ds, sp)
    return ds


def _apply_folder_split(ds, sp):
    if any(s == "val" for s in sp) and any(s == "train" for s in sp):
        ds.train_idx = np.array([i for i, s in enumerate(sp) if s != "val"])
        ds.val_idx = np.array([i for i, s in enumerate(sp) if s == "val"])
        ds.notes.append("使用文件夹里已有的 train / val(test) 划分。")


# =============================================================================
# 音频 / audio
# =============================================================================
def _read_wav_riff(path, max_seconds):
    """直接解析 RIFF/WAVE：支持 PCM 8/16/24/32 位、32/64 位浮点、WAVE_FORMAT_EXTENSIBLE / tiny WAV parser."""
    import struct
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
            raise ValueError("not a RIFF/WAVE file")
        fmt = None
        while True:
            ch = f.read(8)
            if len(ch) < 8:
                raise ValueError("no data chunk")
            cid, size = ch[:4], struct.unpack("<I", ch[4:])[0]
            if cid == b"fmt ":
                b = f.read(size)
                tag, nch, sr, _, _, bits = struct.unpack("<HHIIHH", b[:16])
                if tag == 0xFFFE and len(b) >= 26:          # EXTENSIBLE：真实格式在子格式 GUID 的前两字节
                    tag = struct.unpack("<H", b[24:26])[0]
                fmt = (tag, nch, sr, bits)
            elif cid == b"data":
                if fmt is None:
                    raise ValueError("data before fmt")
                tag, nch, sr, bits = fmt
                bps = bits // 8
                if size in (0, 0xFFFFFFFF):
                    size = 1 << 62
                n = min(size // (bps * nch), int(sr * max_seconds))
                raw = f.read(n * bps * nch)
                break
            else:
                f.seek(size + (size & 1), 1)
    if tag == 3:
        a = np.frombuffer(raw, "<f4" if bits == 32 else "<f8").astype(np.float32)
    elif tag == 1:
        if bits == 8:
            a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
        elif bits == 16:
            a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
        elif bits == 24:
            b = np.frombuffer(raw[: len(raw) // 3 * 3], np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            a = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / (1 << 23)
        elif bits == 32:
            a = np.frombuffer(raw, "<i4").astype(np.float32) / (1 << 31)
        else:
            raise ValueError(f"unsupported bit depth {bits}")
    else:
        raise ValueError(f"压缩编码的 WAV（格式号 {tag}）")
    a = a[: len(a) // nch * nch]
    return a.reshape(-1, nch).mean(1), sr


def read_audio(path, max_seconds=10.0):
    """读取音频为 16 kHz 单声道 float32 / load audio as 16 kHz mono."""
    ext = os.path.splitext(path)[1].lower()
    a = sr = None
    if ext == ".wav":
        try:
            a, sr = _read_wav_riff(path, max_seconds)
        except Exception:  # noqa
            a = None
    if a is None:
        try:
            import soundfile as sf
        except ImportError:
            raise _NeedSoundfile()
        with sf.SoundFile(path) as f:
            sr = f.samplerate
            a = f.read(frames=int(sr * max_seconds), dtype="float32", always_2d=True).mean(1)
    if sr != SR and len(a) > 1:                                    # 线性插值重采样 / resample
        t_new = np.arange(0, len(a) / sr, 1 / SR)
        a = np.interp(t_new, np.arange(len(a)) / sr, a).astype(np.float32)
    return a.astype(np.float32)


class _NeedSoundfile(Exception):
    pass


def read_wav(path, max_seconds=10.0):                              # 兼容旧名字 / backward-compatible name
    return read_audio(path, max_seconds)


def _band_edges():
    lo, hi = 60.0, SR / 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(lo), mel(hi), N_BANDS + 1))
    freqs = np.fft.rfftfreq(512, 1 / SR)
    return [(np.searchsorted(freqs, pts[i]), max(np.searchsorted(freqs, pts[i]) + 1, np.searchsorted(freqs, pts[i + 1])))
            for i in range(N_BANDS)], pts


BANDS, BAND_HZ = _band_edges()
AUDIO_FEATURES = ([f"频带{i + 1:02d}({int(BAND_HZ[i])}Hz)·均值" for i in range(N_BANDS)]
                  + [f"频带{i + 1:02d}·波动" for i in range(N_BANDS)]
                  + ["响度·均值", "响度·波动", "过零率", "频谱质心·均值", "频谱质心·波动", "滚降频率", "频谱平坦度", "时长(秒)"])


def audio_features(a):
    """把一段波形变成固定长度的声学特征向量 / fixed-length acoustic features."""
    dur = len(a) / SR
    if len(a) < 512:
        a = np.pad(a, (0, 512 - len(a)))
    frames = np.lib.stride_tricks.sliding_window_view(a, 512)[::256] * np.hanning(512)
    P = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-10                       # 功率谱 (帧, 频率)
    band = np.stack([P[:, s:e].mean(1) for s, e in BANDS], 1)
    logb = np.log10(band)
    freqs = np.fft.rfftfreq(512, 1 / SR)
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    zcr = (np.abs(np.diff(np.sign(a))) > 0).mean()
    cent = (P * freqs).sum(1) / P.sum(1)
    cum = np.cumsum(P, 1)
    roll = freqs[(cum < 0.85 * cum[:, -1:]).sum(1).clip(max=len(freqs) - 1)]
    flat = np.exp(np.log(P).mean(1)) / P.mean(1)
    return np.concatenate([logb.mean(0), logb.std(0),
                           [np.log10(rms.mean()), rms.std() / (rms.mean() + 1e-9), zcr, cent.mean() / 1000, cent.std() / 1000,
                            roll.mean() / 1000, flat.mean(), dur]]).astype(np.float32)


def load_audio(root, max_per_class=1000, target=None, task_type="auto") -> Dataset:
    lab = _label_files(root, AUD_EXT, target)
    numeric = lab["numeric"] if task_type == "auto" else task_type == "regression"
    if numeric and not all(_to_float(v) is not None for _, v in lab["pairs"]):
        raise DataError("回归需要数值标签，当前标签列里有非数字。")
    pairs = lab["pairs"] if numeric else _cap_per_class(lab["pairs"], max_per_class)
    X, raw, sp, bad, need_sf, used = [], [], [], 0, 0, []
    for f, l in pairs:
        try:
            X.append(audio_features(read_audio(f)))
            raw.append(l)
            used.append(f)
            sp.append(lab["splits"].get(f))
        except _NeedSoundfile:
            need_sf += 1
        except Exception:  # noqa
            bad += 1
    if not X:
        if need_sf:
            raise DataError("这些音频不是 WAV，读取 .flac/.ogg/.mp3 需要：pip install soundfile（或先转成 .wav）。")
        raise DataError("没有成功读取任何音频文件（可能已损坏，或是压缩编码的 WAV）。")
    y, names, tt = _encode_labels(raw, numeric)
    if tt == "classification" and len(names) < 2:
        raise DataError(f"只找到 1 个类别（{names[0]}），至少需要 2 个类别。")
    Xa = np.stack(X)
    tname = lab["meta"]["target"] if lab["meta"] else ("文件夹(类别)" if lab["source"] == "folders" else "文件名前缀")
    ds = Dataset("audio", tt, Xa, y, list(AUDIO_FEATURES), names, root, target_name=tname)
    ds.notes.append("标签来源：" + lab["desc"] + "。")
    ds.notes.append(f"每段音频（最长 10 秒，重采样到 16 kHz）提取 {Xa.shape[1]} 个声学特征：{N_BANDS} 个梅尔频带能量的均值与波动、响度、过零率、频谱质心、滚降、平坦度、时长。")
    if need_sf:
        ds.notes.append(f"{need_sf} 个非 WAV 文件需要 pip install soundfile 才能读取，已跳过。")
    if bad:
        ds.notes.append(f"{bad} 个文件无法读取，已跳过。")
    if lab["meta"] and lab["meta"]["unmatched"]:
        ds.notes.append(f"{lab['meta']['unmatched']} 个音频在元数据表里找不到对应行，已跳过。")
    ds.extra["band_hz"] = [int(b) for b in BAND_HZ[:N_BANDS]]
    ds.extra["files"] = [os.path.relpath(f, root).replace(os.sep, "/") for f in used]
    _apply_folder_split(ds, sp)
    return ds


# =============================================================================
# 数组 / npz · npy
# =============================================================================
def _resize_nearest(img, s):
    h, w = img.shape[-2:]
    yi = (np.arange(s) * h / s).astype(int)
    xi = (np.arange(s) * w / s).astype(int)
    return img[..., yi[:, None], xi[None, :]]


_XKEYS = ("x", "data", "features", "inputs", "input", "images", "image", "samples", "arr_0")
_YKEYS = ("y", "labels", "label", "target", "targets", "classes", "class", "arr_1")


def _open_arrays(path) -> dict:
    """npz / npy / 含多个 .npy 的文件夹 → {名字: 数组} / collect arrays."""
    out = {}
    try:
        if os.path.isdir(path):
            for f in sorted(_walk_files(path, 5000)):
                e = os.path.splitext(f)[1].lower()
                if e == ".npy":
                    out[os.path.splitext(os.path.basename(f))[0]] = np.load(f, allow_pickle=False)
                elif e == ".npz" and not out:
                    return _open_arrays(f)
        elif path.lower().endswith(".npz"):
            with np.load(path, allow_pickle=False) as z:
                out = {k: np.asarray(z[k]) for k in z.files}
        else:
            out[os.path.splitext(os.path.basename(path))[0]] = np.load(path, allow_pickle=False)
    except ValueError as e:
        if "pickle" in str(e).lower():
            raise DataError("数组里含有 Python 对象（object 类型），出于安全不加载。请保存成数值数组：np.save(..., arr.astype('float32'))")
        raise DataError(f"无法读取数组：{e}")
    if not out:
        raise DataError("没有找到 .npy / .npz 数组。")
    return out


def _array_shapes(path):
    return {k: list(v.shape) for k, v in _open_arrays(path).items()}


def _pick_xy(arrs):
    """找出 X / y（以及可能的 train/test 划分）/ pick X, y and an optional predefined split."""
    low = {k.lower(): k for k in arrs}
    tr = [(x, y) for x in ("x_train", "train_x", "xtrain", "train_images", "train_data")
          for y in ("y_train", "train_y", "ytrain", "train_labels", "train_label") if x in low and y in low]
    te = [(x, y) for x in ("x_test", "test_x", "xtest", "test_images", "x_val", "val_x", "test_data")
          for y in ("y_test", "test_y", "ytest", "test_labels", "y_val", "val_y", "test_label") if x in low and y in low]
    if tr:
        X1, y1 = arrs[low[tr[0][0]]], arrs[low[tr[0][1]]].reshape(-1)
        if te:
            X2, y2 = arrs[low[te[0][0]]], arrs[low[te[0][1]]].reshape(-1)
            return np.concatenate([X1, X2]), np.concatenate([y1, y2]), len(X1), "x_train/y_train + x_test/y_test"
        return X1, y1, None, "x_train/y_train"
    xk = next((low[k] for k in _XKEYS if k in low), None)
    yk = next((low[k] for k in _YKEYS if k in low and low[k] != xk), None)
    if xk is not None and yk is not None:
        return arrs[xk], arrs[yk].reshape(-1), None, f"{xk} / {yk}"
    two = [k for k, v in arrs.items() if v.ndim == 2 and v.shape[1] >= 2 and v.dtype.kind in "iufb"]
    if len(arrs) == 1 and two:                        # 单个二维数组：最后一列当目标 / last column = target
        a = arrs[two[0]]
        return a[:, :-1], a[:, -1], None, f"{two[0]}（前 {a.shape[1] - 1} 列为特征，最后一列为目标）"
    if len(arrs) == 2:                                # 两个数组：样本数相同，维度高的为 X
        (k1, a1), (k2, a2) = arrs.items()
        if len(a1) == len(a2):
            if a1.ndim < a2.ndim or (a1.ndim == a2.ndim and a1.size < a2.size):
                (k1, a1), (k2, a2) = (k2, a2), (k1, a1)
            return a1, a2.reshape(len(a2), -1)[:, 0], None, f"{k1} / {k2}"
    raise DataError(f"没认出哪个是 X、哪个是 y。当前数组：{ {k: list(v.shape) for k, v in arrs.items()} }。"
                    "请命名为 X 和 y（np.savez('d.npz', X=X, y=y)），或保存成 X.npy + y.npy。")


def load_arrays(path, task_type="auto", img_size=32) -> Dataset:
    X, yraw, n_train, how = _pick_xy(_open_arrays(path))
    X = np.asarray(X)
    if len(X) != len(yraw):
        raise DataError(f"X 和 y 的样本数不一致（{len(X)} vs {len(yraw)}）。")
    if X.dtype.kind not in "iufb":
        raise DataError(f"X 需要是数值数组，当前类型是 {X.dtype}。")
    if task_type == "auto":
        task_type = "classification" if (yraw.dtype.kind in "iubUSO" or len(np.unique(yraw)) <= 20) else "regression"
    if task_type == "classification":
        y, cls, _ = _encode_labels(yraw.tolist(), False)
    else:
        y, cls = yraw.astype(np.float32), []
    ds = _arrays_to_dataset(X, y, cls, task_type, path, img_size)
    ds.notes.append(f"数组：{how}，X 形状 {list(X.shape)}。")
    if n_train:
        ds.train_idx, ds.val_idx = np.arange(n_train), np.arange(n_train, len(y))
        ds.notes.append("使用数组里已有的 train / test 划分。")
    return ds


def _arrays_to_dataset(X, y, cls, task_type, path, img_size):
    if X.ndim == 2:
        return Dataset("array", task_type, X.astype(np.float32), y, [f"x{i}" for i in range(X.shape[1])], cls, path)
    if X.ndim == 3:
        X = X[:, None]
    elif X.ndim == 4 and X.shape[-1] in (1, 3) and X.shape[1] not in (1, 3):
        X = X.transpose(0, 3, 1, 2)
    if X.ndim != 4 or task_type != "classification" or X.shape[1] not in (1, 3):
        X = X.reshape(len(X), -1)
        return Dataset("array", task_type, X.astype(np.float32), y, [f"x{i}" for i in range(X.shape[1])], cls, path)
    X = X.astype(np.float32)
    lo, hi = float(X.min()), float(X.max())
    X = (X - lo) / max(hi - lo, 1e-6) * 2 - 1
    s = int(max(8, min(128, img_size)) // 4 * 4)
    if X.shape[-1] != s or X.shape[-2] != s:
        X = _resize_nearest(X, s)
    return Dataset("image", task_type, X, y, [f"像素 {X.shape[1]}×{s}×{s}"], cls, path)


def load_npz(path, task_type="auto", img_size=32) -> Dataset:      # 兼容旧名字 / backward-compatible name
    return load_arrays(path, task_type, img_size)


# =============================================================================
# 划分与标准化 / split & scaling
# =============================================================================
def split(y, task_type, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    val_ratio = float(min(0.5, max(0.05, val_ratio)))
    if task_type == "classification":
        tr, va = [], []
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            k = int(round(len(idx) * val_ratio)) if len(idx) >= 3 else 0
            va += idx[:k].tolist(); tr += idx[k:].tolist()
        if not va:                                       # 保证验证集非空 / keep validation non-empty
            va = [tr.pop(int(rng.integers(len(tr))))]
        return np.array(sorted(tr)), np.array(sorted(va))
    idx = rng.permutation(n)
    k = max(1, int(round(n * val_ratio)))
    return np.sort(idx[k:]), np.sort(idx[:k])


def _standardize(ds: Dataset):
    if ds.is_image:
        return
    tr = ds.X[ds.train_idx]
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd < 1e-8] = 1.0
    ds.X = ((ds.X - mu) / sd).astype(np.float32)
    ds.extra["x_mean"], ds.extra["x_std"] = mu, sd
    if ds.task_type == "regression":
        ds.y_mean, ds.y_std = float(ds.y[ds.train_idx].mean()), float(ds.y[ds.train_idx].std() or 1.0)
        ds.y = ((ds.y - ds.y_mean) / ds.y_std).astype(np.float32)


# =============================================================================
# 分析 / analysis
# =============================================================================
def _r(x, k=4):
    x = float(x)
    return round(x, k) if math.isfinite(x) else None


def relevance(X, y, task_type, names, top=20):
    """每个特征与目标的相关程度：分类用相关比 η²，回归用 |皮尔逊相关| / feature relevance."""
    scores = []
    for j in range(X.shape[1]):
        x = X[:, j].astype(np.float64)
        if x.std() < 1e-12:
            scores.append(0.0); continue
        if task_type == "classification":
            m = x.mean()
            ss_b = sum(((x[y == c].mean() - m) ** 2) * (y == c).sum() for c in np.unique(y))
            scores.append(float(ss_b / (((x - m) ** 2).sum() + 1e-12)))
        else:
            scores.append(float(abs(np.corrcoef(x, y)[0, 1])))
    order = np.argsort(scores)[::-1][:top]
    return [{"name": names[j], "score": _r(scores[j])} for j in order]


def pca2(X, max_points=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(len(X), max_points), replace=False) if len(X) > max_points else np.arange(len(X))
    Z = X[idx].reshape(len(idx), -1).astype(np.float64)
    Z = Z - Z.mean(0)
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    P = Z @ Vt[:2].T
    var = (S ** 2) / max(1e-12, (S ** 2).sum())
    if P.shape[1] < 2:
        P = np.concatenate([P, np.zeros((len(P), 1))], 1)
    return idx, P, var[:2].tolist() + [0.0] * (2 - len(var[:2]))


def analyze(ds: Dataset) -> dict:
    n = len(ds.y)
    out = {"kind": ds.kind, "task_type": ds.task_type, "source": ds.source, "target": ds.target_name,
           "n": n, "n_train": int(len(ds.train_idx)), "n_val": int(len(ds.val_idx)),
           "n_features": int(np.prod(ds.X.shape[1:])), "input_shape": list(ds.X.shape[1:]),
           "classes": ds.class_names, "notes": ds.notes, "dropped": ds.dropped, "model": ds.describe_model()}
    # 目标分布 / target distribution
    if ds.task_type == "classification":
        counts = np.bincount(ds.y, minlength=len(ds.class_names))
        out["target_dist"] = {"type": "classes", "counts": counts.tolist()}
        msg = "类别数量不均衡（最多的类是最少的 5 倍以上），准确率可能偏向多数类。"
        if counts.min() * 5 < counts.max() and msg not in ds.notes:
            ds.notes.append("类别数量不均衡（最多的类是最少的 5 倍以上），准确率可能偏向多数类。")
    else:
        yv = ds.y * ds.y_std + ds.y_mean
        out["target_dist"] = {"type": "hist", **_hist(yv), "mean": _r(yv.mean()), "std": _r(yv.std())}
    # 列统计 / column profiles
    out["columns"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in c.items()} for c in ds.columns]
    # 特征相关性 / feature relevance
    if not ds.is_image:
        out["relevance"] = relevance(ds.X, ds.y, ds.task_type, ds.feature_names)
    # PCA 二维投影 / PCA projection
    Xp = ds.X if not ds.is_image else _resize_nearest(ds.X, min(16, ds.X.shape[-1]))
    idx, P, var = pca2(Xp)
    lab = ds.y[idx]
    out["pca"] = {"points": [[_r(a, 3), _r(b, 3), int(l) if ds.task_type == "classification" else _r(l, 3)]
                             for (a, b), l in zip(P, lab)], "explained": [_r(v, 3) for v in var]}
    # 样本预览 / samples
    if ds.kind == "tabular" and ds.preview_rows:
        out["samples"] = {"type": "rows", **ds.preview_rows}
    elif ds.is_image:
        items = []
        for c in range(len(ds.class_names)):
            for i in np.where(ds.y == c)[0][:3]:
                img = ds.X[i]
                px = ((img.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
                items.append({"w": img.shape[2], "h": img.shape[1], "c": img.shape[0],
                              "px": px.flatten().tolist(), "label": ds.class_names[c]})
            if len(items) >= 24:
                break
        out["samples"] = {"type": "images", "items": items}
    elif ds.kind == "audio":
        mu = ds.extra.get("x_mean"); sd = ds.extra.get("x_std")
        raw = ds.X * sd + mu if mu is not None else ds.X
        out["samples"] = {"type": "spectra", "band_hz": ds.extra.get("band_hz", []),
                          "classes": [{"name": c, "mean": [_r(v, 3) for v in raw[ds.y == k, :N_BANDS].mean(0)]}
                                      for k, c in enumerate(ds.class_names)]}
    out["notes"] = ds.notes
    return out
