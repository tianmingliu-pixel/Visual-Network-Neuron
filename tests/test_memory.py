"""机器记忆库测试（只用 numpy + sqlite，无需 torch）/ memory store tests."""
import json
import os

import numpy as np
import pytest

from server.memory import MemoryStore


def _embed(rng, n, step):
    labels = ["cat", "dog"] * (n // 2)
    sep = step / 100                                         # 训练越久，两类特征分得越开
    V = rng.normal(size=(n, 16)) + np.array([[sep if l == "cat" else -sep] + [0] * 15 for l in labels])
    correct = [bool(rng.uniform() < 0.5 + sep / 4) for _ in range(n)]
    return {"ids": list(range(n)), "split": ["val"] * 4 + ["train"] * (n - 4), "vectors": V.astype(np.float32),
            "labels": labels, "preds": [l if c else ("dog" if l == "cat" else "cat") for l, c in zip(labels, correct)],
            "conf": rng.uniform(0.5, 1, n).tolist(), "correct": correct, "loss": rng.uniform(0, 1, n).tolist()}


def test_runs_snapshots_embeddings_and_replay(tmp_path):
    rng = np.random.default_rng(0)
    st = MemoryStore(str(tmp_path / "m.db"))
    rid = st.new_run("custom", "自定义 · 猫狗", "ViT", 1234, {"lr": 3e-3}, "猫狗#abcd")
    st.register_samples("猫狗#abcd", [(i, "val" if i < 4 else "train", "cat" if i % 2 == 0 else "dog", f"cat/{i}.jpg", {})
                                      for i in range(20)])
    vals = [0.5, 0.75, 0.6]
    for k, va in enumerate(vals):
        written = []
        row = st.add_snapshot(rid, (k + 1) * 100, {"loss": 1 / (k + 1), "acc": 0.9, "val_acc": va, "grad_norm": 0.1},
                              [("blocks.0", 100, 1.0, 0.1, 0.01 * k), ("head", 10, 0.5, 0.2, 0.02)],
                              _embed(rng, 20, (k + 1) * 100), "猫狗#abcd",
                              lambda p: (open(p, "wb").write(b"weights"), written.append(p)))
        assert row["step"] == (k + 1) * 100
    d = st.run_detail(rid)
    assert [s["step"] for s in d["snapshots"]] == [100, 200, 300]
    assert d["run"]["best_val"] == 0.75 and sum(s["is_best"] for s in d["snapshots"]) == 1
    assert d["layers"]["names"] == ["blocks.0", "head"] and len(d["layers"]["delta"][0]) == 3
    # 只保留 latest 和 best 两份权重 / only latest + best keep weights
    ck = [s["ckpt"] for s in d["snapshots"]]
    assert ck[0] is None and ck[1].endswith("best.pt") and ck[2].endswith("latest.pt")
    assert os.path.exists(st.ckpt_path(d["snapshots"][1]["id"]))
    assert len(st.resumable("custom", "猫狗#abcd")) == 2
    # 特征向量的投影与记录 / projection + vector records
    sd = st.snapshot_detail(d["snapshots"][2]["id"])
    assert len(sd["projection"]["points"]) == 20 and sd["dim"] == 16 and sd["separation"] > 0
    assert sd["records"][0]["metadata"]["source"].endswith(".jpg") and len(sd["records"][0]["vector"]) == 8
    json.dumps(sd)
    # 长期记忆：看过 3 次，难例权重 / long-term sample memory and replay weights
    mem = st.sample_memory("猫狗#abcd")
    assert len(mem) == 20 and all(m["seen"] == 3 for m in mem)
    w = st.replay_weights("猫狗#abcd", list(range(4, 20)))
    assert w.shape == (16,) and w.min() >= 1 and w.max() <= 3
    # 后续运行从最佳快照继续，能追溯来源 / lineage
    rid2 = st.new_run("custom", "x", "ViT", 1234, {}, "猫狗#abcd", parent_snapshot=d["snapshots"][1]["id"])
    assert st.run_detail(rid2)["lineage"][0]["step"] == 200
    ov = st.overview()
    assert ov["counts"]["runs"] == 2 and ov["counts"]["embeddings"] == 60 and len(ov["tables"]) >= 6


def test_readonly_query_console(tmp_path):
    st = MemoryStore(str(tmp_path / "m.db"))
    st.new_run("mlp", "t", "MLP", 10, {}, None)
    r = st.query("SELECT task, title FROM runs")
    assert r["columns"] == ["task", "title"] and r["rows"] == [["mlp", "t"]]
    for bad in ("DELETE FROM runs", "SELECT 1; DROP TABLE runs", "UPDATE runs SET task='x'"):
        with pytest.raises(ValueError):
            st.query(bad)
    st.delete_run(st.overview()["runs"][0]["id"])
    assert st.overview()["counts"]["runs"] == 0
