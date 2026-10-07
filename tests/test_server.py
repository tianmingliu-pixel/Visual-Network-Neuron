"""可视化后端测试：每个任务能训练、能预览、逐行追踪能覆盖网络代码 / UI backend tests."""
import json
import numpy as np
import math
import os

import pytest
import torch

from server.hub import encode
from server.tasks import TASKS
from server.tracer import LineTracer
from server.trainer import ROOT, TRACE_DIRS
from server.source import read_source


@pytest.mark.parametrize("tid", list(TASKS))
def test_task_step_preview_and_trace(tid):
    torch.manual_seed(0)
    task = TASKS[tid](torch.device("cpu"), 1e-3, 8)
    m = task.train_step()
    assert math.isfinite(m["loss"]) and math.isfinite(m["grad_norm"])
    p = task.preview()
    assert p["kind"] in ("seq", "classify", "denoise", "boundary")
    json.loads(encode("preview", p).split("data: ", 1)[1])          # JSON 合法 / valid JSON

    tr = LineTracer(TRACE_DIRS)
    with tr:
        task.train_step()
    res = tr.result(ROOT)
    files = set(res["files"])
    assert "server/tasks.py" in files
    assert any(f.startswith("neurocore/models/") for f in files)
    assert "neurocore/layers/attention.py" in files or tid == "mlp"
    assert len(res["events"]) > 20 and not res["truncated"]
    # 至少有一个事件记录了张量形状 / some event recorded a tensor
    assert any(v.get("k") == "tensor" for ev in res["events"] for v in ev.get("v", []))
    json.loads(encode("trace", res).split("data: ", 1)[1])


def test_source_reader_and_security():
    src = read_source("neurocore/models/vit.py")
    assert any(l["comment"] for l in src["lines"])
    for bad in ("../README.md", "server/app.py", "neurocore/../../x.py"):
        with pytest.raises((PermissionError, FileNotFoundError)):
            read_source(bad)


@pytest.mark.parametrize("tid", list(TASKS))
def test_graph_and_pipeline(tid):
    from server.pipeline import build_pipeline
    torch.manual_seed(0)
    task = TASKS[tid](torch.device("cpu"), 1e-3, 8)
    task.train_step()
    g = task.graph()
    assert g["kind"] in ("mlp", "arch")
    if g["kind"] == "arch":
        assert len(g["nodes"]) >= 3 and g["input"]["shape"]
        assert any(n["gnorm"] for n in g["nodes"])                 # 有梯度信息 / gradients present
    else:
        assert g["sizes"] == [7, 16, 12, 3] and len(g["W"]) == 3 and g["G"][0] is not None
    json.loads(encode("graph", g).split("data: ", 1)[1])
    steps = build_pipeline(TASKS[tid])
    assert [s["id"] for s in steps] == ["data", "forward", "loss", "backward", "update"]
    for s in steps:
        assert s["refs"], s["id"]
        for r in s["refs"]:
            src = read_source(r["file"]) if not r["file"].startswith("server/") or r["file"] == "server/tasks.py" else None
            if src:
                assert 1 <= r["line"] <= len(src["lines"])


def test_custom_task_end_to_end(tmp_path):
    """用导入的数据生成任务并训练几步 / custom task trains, previews, graphs and evaluates."""
    from server.datasets import load_dataset
    from server.tasks import make_custom_task
    from server.pipeline import build_pipeline
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 25))
    y = (X[:, 0] + X[:, 3] > 0).astype(int)
    np.savez(tmp_path / "d.npz", X=X, y=y)
    for task_type in ("classification", "regression"):
        ds = load_dataset(str(tmp_path / "d.npz"), task_type=task_type)
        cls = make_custom_task(ds)
        t = cls(torch.device("cpu"), 3e-3, 16)
        for _ in range(5):
            m = t.train_step()
        assert math.isfinite(m["loss"])
        assert t.preview()["kind"] == "eval"
        g = t.graph()
        assert g["kind"] == "mlp" and g["sizes"][0] == 18 and g["true_sizes"][0] == 25
        r = t.evaluate()
        assert r["importance"] and json.loads(encode("e", r).split("data: ", 1)[1])
        assert len(build_pipeline(cls)) == 5
    img = rng.uniform(0, 255, (40, 16, 16))
    np.savez(tmp_path / "i.npz", X=img, y=np.arange(40) % 2)
    ds = load_dataset(str(tmp_path / "i.npz"), img_size=16)
    t = make_custom_task(ds)(torch.device("cpu"), 1e-3, 8)
    t.train_step()
    assert t.graph()["kind"] == "arch" and t.preview()["acc"] is not None


def test_memory_store_training_and_resume(tmp_path, monkeypatch):
    """训练写入记忆库 → 从最佳快照继续训练 / training writes memory; a new run resumes from it."""
    import time
    import server.memory as mem
    from server.datasets import load_dataset
    from server.hub import Hub
    from server.tasks import make_custom_task
    from server.trainer import Session
    store = mem.MemoryStore(str(tmp_path / "mem.db"))
    monkeypatch.setattr(mem, "_STORE", store)
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (40, 16, 16))
    np.savez(tmp_path / "i.npz", X=img, y=np.arange(40) % 2)
    ds = load_dataset(str(tmp_path / "i.npz"), img_size=16)
    s = Session(Hub(), {"custom": make_custom_task(ds)})

    def run(steps, **opts):
        s.start("custom", steps, 3e-3, 8, opts)
        for _ in range(600):
            if s.state in ("finished", "error"):
                break
            time.sleep(0.05)
        assert s.state == "finished", s.state
        return s.run_id

    r1 = run(60, memory=True, augment=True, replay=True, mem_every=20)
    d = store.run_detail(r1)
    assert [x["step"] for x in d["snapshots"]] == [20, 40, 60]
    assert d["snapshots"][-1]["n_emb"] == 40 and d["run"]["best_val"] is not None
    assert d["layers"]["names"] and d["snapshots"][-1]["delta_norm"] > 0
    assert store.snapshot_detail(d["snapshots"][-1]["id"])["projection"]["points"]
    r2 = run(20, memory=True, resume="best", replay=True, mem_every=20)
    d2 = store.run_detail(r2)
    assert d2["run"]["parent_snapshot"] == d["run"]["best_snapshot"]
    assert d2["snapshots"][0]["step"] > 20                        # 步数接着上次的记忆往后数
