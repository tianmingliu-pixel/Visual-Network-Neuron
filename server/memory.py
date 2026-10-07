"""
机器记忆库 / Neural memory store
================================
把每次训练里“机器学到了什么”存进本地数据库，下次训练可以接着用：

  data/memory/neuro_memory.db      SQLite（WAL 模式，Python 自带，无需安装），所有表见 SCHEMA
  data/memory/ckpt/<run>/*.pt      权重 + 优化器状态（只保留“最新”和“最佳”两份，节省磁盘）

记忆的组成（类似向量数据库的 collection：id + vector + metadata）：
  runs          一次训练 = 一条记录：任务、数据集、超参数、从哪个记忆继续、最佳验证成绩
  snapshots     每隔 N 步一个快照：loss、训练/验证准确率、梯度范数、权重总变化量、检查点文件
  layer_states  每个快照里每一层的权重范数、梯度范数、与上一快照相比的变化量（看哪一层在学）
  samples       样本元数据：来源文件、标签、属于训练还是验证
  embeddings    每个快照里每个样本的特征向量（倒数第二层，float32 BLOB）+ 预测 + 置信度 + 是否正确
  sample_memory 长期记忆：每个样本被看过几次、错过几次、平滑后的损失 —— “难例回放”据此多练错过的样本

下次训练怎么用记忆：
  ① 从某个快照继续：加载权重 + AdamW 的动量状态 + 步数，接着学，不从零开始
  ② 难例回放：按 sample_memory 里的错误次数 / 损失给训练样本加权抽样
  ③ 最佳快照：验证准确率最高的那一刻自动另存为 best，过拟合后也能退回去
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import threading
import time
import uuid

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from .config import DATA_DIR  # noqa: E402
MEM_DIR = os.path.join(DATA_DIR, "memory")
DB_PATH = os.path.join(MEM_DIR, "neuro_memory.db")
SCHEMA_VERSION = 1
MAX_EMBED = 400            # 每个快照最多存多少个样本的特征向量 / vectors per snapshot

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,            -- 运行编号 run_YYYYmmdd_HHMMSS_xxxx
  created REAL,                   -- 创建时间（unix 秒）
  task TEXT, title TEXT,          -- 任务编号与标题
  dataset TEXT,                   -- 数据集编号（自定义数据才有）
  model TEXT, n_params INTEGER,   -- 网络类型与参数量
  params TEXT,                    -- JSON：步数、学习率、批大小、是否回放/增强
  parent_snapshot INTEGER,        -- 从哪个快照的记忆继续（NULL = 从零开始）
  status TEXT,                    -- running | finished | stopped | error
  steps INTEGER,                  -- 已训练到第几步
  best_val REAL, best_snapshot INTEGER
);
CREATE TABLE IF NOT EXISTS snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT REFERENCES runs(id) ON DELETE CASCADE,
  step INTEGER, created REAL,
  loss REAL, train_acc REAL, val_acc REAL, val_loss REAL,
  grad_norm REAL, lr REAL,
  weight_norm REAL,               -- 全部权重的 L2 范数
  delta_norm REAL,                -- 与上一快照相比权重变化了多少（学习的“步幅”）
  ckpt TEXT,                      -- 检查点文件（相对 data/memory），只有 latest/best 保留
  is_best INTEGER DEFAULT 0,
  metrics TEXT                    -- JSON：其他指标
);
CREATE INDEX IF NOT EXISTS ix_snap_run ON snapshots(run_id, step);
CREATE TABLE IF NOT EXISTS layer_states (
  snapshot_id INTEGER REFERENCES snapshots(id) ON DELETE CASCADE,
  layer TEXT, n_params INTEGER, wnorm REAL, gnorm REAL, delta REAL,
  PRIMARY KEY (snapshot_id, layer)
);
CREATE TABLE IF NOT EXISTS samples (
  dataset TEXT, sample_id INTEGER, split TEXT, label TEXT, source TEXT, meta TEXT,
  PRIMARY KEY (dataset, sample_id)
);
CREATE TABLE IF NOT EXISTS embeddings (
  snapshot_id INTEGER REFERENCES snapshots(id) ON DELETE CASCADE,
  sample_id INTEGER, split TEXT,
  dim INTEGER, vector BLOB,       -- float32 特征向量（网络倒数第二层的输出）
  label TEXT, pred TEXT, confidence REAL, correct INTEGER, loss REAL,
  PRIMARY KEY (snapshot_id, sample_id)
);
CREATE TABLE IF NOT EXISTS sample_memory (
  dataset TEXT, sample_id INTEGER,
  seen INTEGER, wrong INTEGER,    -- 被评估过几次、错过几次（跨多次训练累计）
  ema_loss REAL,                  -- 指数平滑后的损失
  last_pred TEXT, last_conf REAL, updated REAL,
  PRIMARY KEY (dataset, sample_id)
);
"""


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def dataset_key(ds) -> str:
    """数据集编号：名称 + 来源路径/大小的短哈希 / stable id for a dataset."""
    name = os.path.basename(os.path.normpath(str(ds.source)))[:40]
    h = hashlib.sha1(f"{os.path.abspath(str(ds.source))}|{len(ds.y)}|{tuple(ds.X.shape[1:])}".encode()).hexdigest()[:8]
    return f"{name}#{h}"


class MemoryStore:
    """线程安全的小型记忆库：训练线程写，网页接口读 / tiny thread-safe store."""

    def __init__(self, path: str = DB_PATH):
        self.path = path
        self.dir = os.path.dirname(path)
        os.makedirs(os.path.join(self.dir, "ckpt"), exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA foreign_keys=ON")
        self._con.executescript(SCHEMA)
        self._con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        self._con.commit()

    # ---- 基础 / basics ------------------------------------------------------------
    def _q(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._con.execute(sql, args).fetchall()]

    def _x(self, sql, args=()):
        with self._lock:
            cur = self._con.execute(sql, args)
            self._con.commit()
            return cur.lastrowid

    def close(self):
        with self._lock:
            self._con.close()

    # ---- 写入 / writing ---------------------------------------------------------------
    def new_run(self, task_id, title, model, n_params, params, dataset=None, parent_snapshot=None) -> str:
        rid = time.strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
        self._x("INSERT INTO runs (id, created, task, title, dataset, model, n_params, params, parent_snapshot, status, steps) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,0)",
                (rid, time.time(), task_id, title, dataset, model, int(n_params), json.dumps(params, ensure_ascii=False),
                 parent_snapshot, "running"))
        return rid

    def set_status(self, run_id, status, steps=None):
        if steps is None:
            self._x("UPDATE runs SET status=? WHERE id=?", (status, run_id))
        else:
            self._x("UPDATE runs SET status=?, steps=? WHERE id=?", (status, int(steps), run_id))

    def register_samples(self, dataset, rows):
        """rows: [(sample_id, split, label, source, meta_dict)]，已存在的跳过。"""
        with self._lock:
            self._con.executemany(
                "INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?)",
                [(dataset, int(i), sp, str(lb), src, json.dumps(m or {}, ensure_ascii=False)) for i, sp, lb, src, m in rows])
            self._con.commit()

    def add_snapshot(self, run_id, step, metrics: dict, layers: list, embed: dict | None = None,
                     dataset: str | None = None, ckpt_writer=None) -> dict:
        """写入一个快照。layers: [(name, n, wnorm, gnorm, delta)]；
        embed: {ids, split, vectors(N,D), labels, preds, conf, correct, loss}；
        ckpt_writer(path) 由调用者提供（torch.save），这里决定文件名并只保留 latest / best。"""
        val = _f(metrics.get("val_acc"))
        run = self._q("SELECT best_val, best_snapshot FROM runs WHERE id=?", (run_id,))[0]
        best_val = run["best_val"]
        is_best = val is not None and (best_val is None or val > best_val + 1e-9)
        sid = self._x(
            "INSERT INTO snapshots (run_id, step, created, loss, train_acc, val_acc, val_loss, grad_norm, lr, "
            "weight_norm, delta_norm, is_best, metrics) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, int(step), time.time(), _f(metrics.get("loss")), _f(metrics.get("acc")), val,
             _f(metrics.get("val_loss")), _f(metrics.get("grad_norm")), _f(metrics.get("lr")),
             _f(metrics.get("weight_norm")), _f(metrics.get("delta_norm")), int(is_best),
             json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}, ensure_ascii=False)))
        with self._lock:
            self._con.executemany("INSERT INTO layer_states VALUES (?,?,?,?,?,?)",
                                  [(sid, n, int(k), _f(w), _f(g), _f(d)) for n, k, w, g, d in layers])
            if embed is not None and len(embed["ids"]):
                V = np.asarray(embed["vectors"], np.float32)
                self._con.executemany(
                    "INSERT INTO embeddings VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [(sid, int(i), sp, V.shape[1], V[j].tobytes(), str(lb), str(pr), _f(cf), int(ok), _f(ls))
                     for j, (i, sp, lb, pr, cf, ok, ls) in enumerate(zip(
                         embed["ids"], embed["split"], embed["labels"], embed["preds"], embed["conf"],
                         embed["correct"], embed["loss"]))])
                if dataset:
                    self._update_sample_memory(dataset, embed)
            self._con.commit()
        # 检查点：latest 每次覆盖；创下新高时另存 best / keep only latest + best weights
        if ckpt_writer is not None:
            d = os.path.join(self.dir, "ckpt", run_id)
            os.makedirs(d, exist_ok=True)
            latest = os.path.join(d, "latest.pt")
            ckpt_writer(latest)
            rel_latest = os.path.relpath(latest, self.dir).replace(os.sep, "/")
            self._x("UPDATE snapshots SET ckpt=NULL WHERE run_id=? AND ckpt=? AND id<>?", (run_id, rel_latest, sid))
            self._x("UPDATE snapshots SET ckpt=? WHERE id=?", (rel_latest, sid))
            if is_best:
                best = os.path.join(d, "best.pt")
                shutil.copyfile(latest, best)
                rel_best = os.path.relpath(best, self.dir).replace(os.sep, "/")
                self._x("UPDATE snapshots SET ckpt=NULL WHERE run_id=? AND ckpt=?", (run_id, rel_best))
                self._x("UPDATE snapshots SET ckpt=? WHERE id=?", (rel_best, sid))
        if is_best:
            self._x("UPDATE snapshots SET is_best=0 WHERE run_id=? AND id<>?", (run_id, sid))
            self._x("UPDATE runs SET best_val=?, best_snapshot=? WHERE id=?", (val, sid, run_id))
        self._x("UPDATE runs SET steps=? WHERE id=?", (int(step), run_id))
        return self.snapshot_row(sid)

    def _update_sample_memory(self, dataset, embed):
        now = time.time()
        old = {r["sample_id"]: r for r in self._q(
            "SELECT * FROM sample_memory WHERE dataset=?", (dataset,))}
        rows = []
        for i, pr, cf, ok, ls in zip(embed["ids"], embed["preds"], embed["conf"], embed["correct"], embed["loss"]):
            o = old.get(int(i))
            ls = _f(ls) or 0.0
            ema = ls if o is None or o["ema_loss"] is None else 0.7 * o["ema_loss"] + 0.3 * ls
            rows.append((dataset, int(i), (o["seen"] if o else 0) + 1, (o["wrong"] if o else 0) + (0 if ok else 1),
                         ema, str(pr), _f(cf), now))
        self._con.executemany("INSERT OR REPLACE INTO sample_memory VALUES (?,?,?,?,?,?,?,?)", rows)

    # ---- 读取 / reading ---------------------------------------------------------------
    def snapshot_row(self, sid):
        r = self._q("SELECT * FROM snapshots WHERE id=?", (sid,))
        return r[0] if r else None

    def ckpt_path(self, sid):
        r = self.snapshot_row(sid)
        if not r or not r["ckpt"]:
            return None
        p = os.path.join(self.dir, r["ckpt"])
        return p if os.path.exists(p) else None

    def resumable(self, task_id=None, dataset=None):
        """可以“从记忆继续”的快照（有检查点文件）/ snapshots that still have weights."""
        sql = ("SELECT s.id, s.run_id, s.step, s.val_acc, s.train_acc, s.loss, s.is_best, s.ckpt, r.task, r.title, r.dataset "
               "FROM snapshots s JOIN runs r ON r.id=s.run_id WHERE s.ckpt IS NOT NULL")
        args = []
        if task_id:
            sql += " AND r.task=?"
            args.append(task_id)
        if dataset:
            sql += " AND r.dataset=?"
            args.append(dataset)
        return self._q(sql + " ORDER BY s.created DESC LIMIT 50", args)

    def replay_weights(self, dataset, ids, strength=2.0):
        """难例回放的抽样权重：错得越多、损失越大，权重越高（1 ~ 1+strength）。"""
        mem = {r["sample_id"]: r for r in self._q("SELECT * FROM sample_memory WHERE dataset=?", (dataset,))}
        if not mem:
            return None
        losses = np.array([mem[i]["ema_loss"] or 0.0 for i in mem]) if mem else np.zeros(1)
        scale = float(np.percentile(losses, 90)) + 1e-6 if len(losses) else 1.0
        w = []
        for i in ids:
            m = mem.get(int(i))
            if m is None:
                w.append(1.0)
                continue
            wrong_rate = m["wrong"] / max(1, m["seen"])
            hard = 0.5 * wrong_rate + 0.5 * min(1.0, (m["ema_loss"] or 0.0) / scale)
            w.append(1.0 + strength * hard)
        return np.array(w, np.float64)

    def overview(self):
        counts = {t: self._q(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"]
                  for t in ("runs", "snapshots", "layer_states", "samples", "embeddings", "sample_memory")}
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(self.dir) for f in fs)
        runs = self._q("SELECT r.*, (SELECT COUNT(*) FROM snapshots s WHERE s.run_id=r.id) AS n_snap, "
                       "(SELECT s.val_acc FROM snapshots s WHERE s.run_id=r.id ORDER BY s.step DESC LIMIT 1) AS last_val "
                       "FROM runs r ORDER BY r.created DESC LIMIT 60")
        for r in runs:
            r["params"] = json.loads(r["params"] or "{}")
        tables = self._q("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        return {"path": self.path, "bytes": size, "counts": counts, "runs": runs, "schema_version": SCHEMA_VERSION,
                "tables": tables}

    def run_detail(self, run_id):
        run = self._q("SELECT * FROM runs WHERE id=?", (run_id,))
        if not run:
            return None
        run = run[0]
        run["params"] = json.loads(run["params"] or "{}")
        snaps = self._q("SELECT id, step, created, loss, train_acc, val_acc, val_loss, grad_norm, lr, weight_norm, "
                        "delta_norm, ckpt, is_best, (SELECT COUNT(*) FROM embeddings e WHERE e.snapshot_id=s.id) AS n_emb "
                        "FROM snapshots s WHERE run_id=? ORDER BY step", (run_id,))
        layers = self._q("SELECT l.snapshot_id, l.layer, l.n_params, l.wnorm, l.gnorm, l.delta FROM layer_states l "
                         "JOIN snapshots s ON s.id=l.snapshot_id WHERE s.run_id=? ORDER BY s.step", (run_id,))
        names = []
        for l in layers:
            if l["layer"] not in names:
                names.append(l["layer"])
        idx = {s["id"]: k for k, s in enumerate(snaps)}
        delta = [[None] * len(snaps) for _ in names]
        gnorm = [[None] * len(snaps) for _ in names]
        for l in layers:
            j = idx.get(l["snapshot_id"])
            if j is not None:
                delta[names.index(l["layer"])][j] = l["delta"]
                gnorm[names.index(l["layer"])][j] = l["gnorm"]
        lineage = []                                       # 记忆的来源链：这次从哪次继续来的
        p = run["parent_snapshot"]
        while p and len(lineage) < 20:
            r = self._q("SELECT s.id, s.step, s.val_acc, s.run_id, r.parent_snapshot FROM snapshots s "
                        "JOIN runs r ON r.id=s.run_id WHERE s.id=?", (p,))
            if not r:
                break
            lineage.append(r[0])
            p = r[0]["parent_snapshot"]
        return {"run": run, "snapshots": snaps, "layers": {"names": names, "delta": delta, "gnorm": gnorm},
                "lineage": lineage}

    def snapshot_detail(self, sid, n_records=6):
        snap = self.snapshot_row(sid)
        if not snap:
            return None
        rows = self._q("SELECT sample_id, split, dim, vector, label, pred, confidence, correct, loss "
                       "FROM embeddings WHERE snapshot_id=? ORDER BY sample_id", (sid,))
        out = {"snapshot": snap, "layers": self._q("SELECT * FROM layer_states WHERE snapshot_id=?", (sid,))}
        if rows:
            V = np.stack([np.frombuffer(r["vector"], np.float32) for r in rows])
            P, var = _pca2(V)
            labels = sorted({r["label"] for r in rows} | {r["pred"] for r in rows})
            out["projection"] = {
                "classes": labels, "explained": [round(float(v), 3) for v in var],
                "points": [[round(float(a), 4), round(float(b), 4), labels.index(r["label"]), labels.index(r["pred"]),
                            int(r["correct"]), r["split"], r["sample_id"], round(r["confidence"] or 0, 3)]
                           for (a, b), r in zip(P, rows)]}
            out["dim"] = int(V.shape[1])
            out["separation"] = _separation(V, [r["label"] for r in rows])
            src = {r["sample_id"]: r for r in self._q(
                "SELECT sample_id, source, meta FROM samples WHERE dataset=(SELECT dataset FROM runs WHERE id=?)",
                (snap["run_id"],))}
            out["records"] = [{
                "id": f"{sid}:{r['sample_id']}", "vector": [round(float(v), 4) for v in np.frombuffer(r["vector"], np.float32)[:8]],
                "dim": r["dim"],
                "metadata": {"sample_id": r["sample_id"], "split": r["split"], "label": r["label"], "pred": r["pred"],
                             "confidence": round(r["confidence"] or 0, 4), "correct": bool(r["correct"]),
                             "loss": round(r["loss"] or 0, 5), "source": (src.get(r["sample_id"]) or {}).get("source"),
                             "snapshot": sid, "step": snap["step"]}}
                for r in (sorted(rows, key=lambda r: (r["correct"], -(r["loss"] or 0)))[:n_records])]
        return out

    def sample_memory(self, dataset, limit=200):
        return self._q(
            "SELECT m.*, s.split, s.label, s.source FROM sample_memory m LEFT JOIN samples s "
            "ON s.dataset=m.dataset AND s.sample_id=m.sample_id WHERE m.dataset=? "
            "ORDER BY (1.0*m.wrong/MAX(1,m.seen)) DESC, m.ema_loss DESC LIMIT ?", (dataset, int(limit)))

    def delete_run(self, run_id):
        self._x("DELETE FROM runs WHERE id=?", (run_id,))
        shutil.rmtree(os.path.join(self.dir, "ckpt", run_id), ignore_errors=True)

    def forget_samples(self, dataset):
        self._x("DELETE FROM sample_memory WHERE dataset=?", (dataset,))

    def query(self, sql, limit=200):
        """只读 SQL 控制台：只允许 SELECT / WITH / PRAGMA table_info / read-only console."""
        s = sql.strip().rstrip(";")
        head = s.split(None, 1)[0].lower() if s else ""
        if head not in ("select", "with", "pragma", "explain") or ";" in s:
            raise ValueError("只允许单条只读查询（SELECT / WITH / PRAGMA）。")
        uri = "file:" + self.path.replace("\\", "/") + "?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=10)
        try:
            cur = con.execute(s)
            cols = [d[0] for d in cur.description or []]
            rows = []
            for r in cur.fetchmany(limit):
                rows.append([f"<{len(v)} 字节 · float32×{len(v) // 4}>" if isinstance(v, bytes) else v for v in r])
            return {"columns": cols, "rows": rows, "truncated": len(rows) == limit}
        finally:
            con.close()


def _pca2(V):
    V = V - V.mean(0, keepdims=True)
    if len(V) < 2 or not np.isfinite(V).all():
        return np.zeros((len(V), 2)), [0.0, 0.0]
    U, S, Wt = np.linalg.svd(V, full_matrices=False)
    P = V @ Wt[:2].T
    if P.shape[1] < 2:
        P = np.pad(P, ((0, 0), (0, 2 - P.shape[1])))
    tot = float((S ** 2).sum()) or 1.0
    var = [(float(s) ** 2) / tot for s in S[:2]] + [0.0] * (2 - len(S[:2]))
    return P, var


def _separation(V, labels):
    """类间距离 / 类内距离：越大说明特征空间里各类分得越开 / between-class vs within-class spread."""
    labs = sorted(set(labels))
    if len(labs) < 2:
        return None
    lab = np.array(labels)
    cents = np.stack([V[lab == c].mean(0) for c in labs])
    within = np.mean([np.linalg.norm(V[lab == c] - cents[k], axis=1).mean() for k, c in enumerate(labs)])
    d = [np.linalg.norm(cents[i] - cents[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return round(float(np.mean(d) / (within + 1e-9)), 4)


_STORE = None
_STORE_LOCK = threading.Lock()


def get_store() -> MemoryStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            os.makedirs(MEM_DIR, exist_ok=True)
            _STORE = MemoryStore(DB_PATH)
        return _STORE
