"""数据导入测试（只用 numpy/Pillow，无需 torch 也能跑）/ data-import tests."""
import csv
import json
import os
import wave

import numpy as np
import pytest

from server.datasets import DataError, analyze, detect, load_dataset


@pytest.fixture
def tmp(tmp_path):
    rng = np.random.default_rng(0)
    with open(tmp_path / "t.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "a", "b", "颜色", "备注", "label"])
        for i in range(120):
            k = i % 3
            w.writerow([i, k + rng.normal(0, .3), "" if i % 11 == 0 else rng.normal(), ["红", "绿"][i % 2],
                        f"some note text number {i}", ["x", "y", "z"][k]])
    with open(tmp_path / "r.tsv", "w", newline="", encoding="gbk") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["面积", "价格"])
        for i in range(80):
            a = rng.uniform(30, 100)
            w.writerow([round(a, 1), round(a * 3 + rng.normal(0, 5), 1)])
    json.dump([{"x": float(v), "y": int(v > 0)} for v in rng.normal(size=60)], open(tmp_path / "j.json", "w"))
    np.savez(tmp_path / "a.npz", X=rng.normal(size=(50, 4)), y=rng.integers(0, 2, 50))
    try:
        from PIL import Image
        for c in ("cat", "dog"):
            os.makedirs(tmp_path / "img" / c)
            for i in range(12):
                Image.fromarray((rng.uniform(0, 255, (20, 24, 3))).astype(np.uint8)).save(tmp_path / "img" / c / f"{i}.png")
    except ImportError:
        pass
    for c, fr in (("low", 300), ("high", 2000)):
        os.makedirs(tmp_path / "aud" / c)
        for i in range(6):
            t = np.arange(0, .3, 1 / 8000)
            with wave.open(str(tmp_path / "aud" / c / f"{i}.wav"), "wb") as wv:
                wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(8000)
                wv.writeframes((np.sin(2 * np.pi * fr * t) * 20000).astype("<i2").tobytes())
    return tmp_path


def test_csv_classification(tmp):
    ds = load_dataset(str(tmp / "t.csv"))
    assert ds.task_type == "classification" and ds.class_names == ["x", "y", "z"]
    assert any(d["name"] == "id" for d in ds.dropped)
    assert any(n.startswith("颜色=") for n in ds.feature_names) and any("#词" in n for n in ds.feature_names)
    a = analyze(ds)
    json.dumps(a)
    assert a["relevance"][0]["name"] == "a"                     # a 与类别强相关
    assert len(a["pca"]["points"]) == 120


def test_regression_gbk_tsv(tmp):
    ds = load_dataset(str(tmp / "r.tsv"))
    assert ds.task_type == "regression" and ds.target_name == "价格"
    assert abs(float(ds.y[ds.train_idx].mean())) < 1e-4          # 目标已标准化


def test_json_npz_audio_images(tmp):
    assert load_dataset(str(tmp / "j.json")).class_names == ["0", "1"]
    assert load_dataset(str(tmp / "a.npz")).X.shape == (50, 4)
    au = load_dataset(str(tmp / "aud"))
    assert au.kind == "audio" and au.X.shape[0] == 12
    if (tmp / "img").exists():
        im = load_dataset(str(tmp / "img"), img_size=16)
        assert im.X.shape[1:] == (3, 16, 16) and len(im.val_idx) > 0
        json.dumps(analyze(im))


def test_errors(tmp):
    with pytest.raises(DataError):
        detect(str(tmp / "missing.csv"))
    if (tmp / "img").exists():
        with pytest.raises(DataError):
            load_dataset(str(tmp / "img" / "cat"))                 # 没有类别子文件夹


# ---- 上传文件夹 / 元数据 / 数组 / Excel 的更多格式 -------------------------------------------
def _png(path, rng, color):
    from PIL import Image
    a = np.zeros((12, 12, 3), np.uint8) + np.array(color, np.uint8)
    Image.fromarray((a + rng.integers(0, 30, a.shape)).astype(np.uint8)).save(path)


def _wav(path, fr, sr=8000, fmt="pcm16"):
    t = np.arange(0, .25, 1 / sr)
    s = np.sin(2 * np.pi * fr * t)
    if fmt == "pcm16":
        with wave.open(str(path), "wb") as wv:
            wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr)
            wv.writeframes((s * 20000).astype("<i2").tobytes())
    else:                                               # 32 位浮点 WAV（wave 模块不支持）
        import struct
        data = s.astype("<f4").tobytes()
        fmtc = struct.pack("<HHIIHH", 3, 1, sr, sr * 4, 4, 32)
        with open(path, "wb") as f:
            f.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmtc) + 8 + len(data)) + b"WAVE")
            f.write(b"fmt " + struct.pack("<I", len(fmtc)) + fmtc + b"data" + struct.pack("<I", len(data)) + data)


def test_images_with_metadata_table(tmp_path):
    pytest.importorskip("PIL")
    rng = np.random.default_rng(1)
    root = tmp_path / "upload" / "pets"
    (root / "images").mkdir(parents=True)
    with open(root / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["文件名", "品种"])
        for i in range(24):
            name = f"IMG_{i:03d}.png"
            _png(root / "images" / name, rng, (200, 20, 20) if i % 2 else (20, 20, 200))
            w.writerow([f"images/{name}", "红" if i % 2 else "蓝"])
    info = detect(str(tmp_path / "upload"))
    assert info["kind"] == "image" and info["label_source"] == "metadata"
    assert info["file_column"] == "文件名" and info["suggested_target"] == "品种" and info["classes"] == {"红": 12, "蓝": 12}
    ds = load_dataset(str(tmp_path / "upload"), img_size=8)
    assert ds.class_names == ["红", "蓝"] and ds.X.shape == (24, 3, 8, 8)
    json.dumps(analyze(ds))


def test_images_by_filename_prefix_and_split(tmp_path):
    pytest.importorskip("PIL")
    rng = np.random.default_rng(2)
    for sp in ("train", "test"):
        (tmp_path / sp).mkdir()
        for i in range(8):
            _png(tmp_path / sp / f"cat.{i}.png", rng, (250, 250, 250))
            _png(tmp_path / sp / f"dog_{i}.png", rng, (0, 0, 0))
    info = detect(str(tmp_path))
    assert info["label_source"] == "filename" and info["classes"] == {"cat": 16, "dog": 16} and info["has_split"]
    ds = load_dataset(str(tmp_path), img_size=8)
    assert len(ds.val_idx) == 16 and sorted(ds.class_names) == ["cat", "dog"]


def test_audio_metadata_regression_and_float_wav(tmp_path):
    with open(tmp_path / "meta.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "pitch"])
        for i in range(40):
            fr = 200 + i * 40
            _wav(tmp_path / f"{i}.wav", fr, fmt="float" if i % 2 else "pcm16")
            w.writerow([f"{i}.wav", fr])
    ds = load_dataset(str(tmp_path))
    assert ds.kind == "audio" and ds.task_type == "regression" and len(ds.y) == 40 and ds.target_name == "pitch"
    json.dumps(analyze(ds))


def test_npy_variants(tmp_path):
    rng = np.random.default_rng(3)
    a = np.c_[rng.normal(size=(60, 3)), rng.integers(0, 2, 60)]
    np.save(tmp_path / "table.npy", a)
    ds = load_dataset(str(tmp_path / "table.npy"))
    assert ds.X.shape == (60, 3) and ds.task_type == "classification"
    d = tmp_path / "pair"; d.mkdir()
    np.save(d / "X.npy", rng.normal(size=(30, 5))); np.save(d / "y.npy", rng.normal(size=30))
    ds = load_dataset(str(d))
    assert ds.task_type == "regression" and ds.X.shape == (30, 5)
    np.savez(tmp_path / "mnist.npz", x_train=rng.integers(0, 255, (40, 28, 28)), y_train=np.arange(40) % 4,
             x_test=rng.integers(0, 255, (12, 28, 28)), y_test=np.arange(12) % 4)
    ds = load_dataset(str(tmp_path / "mnist.npz"), img_size=16)
    assert ds.is_image and len(ds.val_idx) == 12 and ds.class_names == ["0", "1", "2", "3"]
    np.save(tmp_path / "obj.npy", np.array([{"a": 1}] * 12, dtype=object), allow_pickle=True)
    with pytest.raises(DataError):
        load_dataset(str(tmp_path / "obj.npy"))


def test_xlsx_without_pandas(tmp_path, monkeypatch):
    import sys, zipfile
    monkeypatch.setitem(sys.modules, "pandas", None)             # 模拟没装 pandas
    rows = [["面积", "城市", "价格"]] + [[30 + i, ["北京", "上海"][i % 2], 100 + 3 * i] for i in range(30)]
    strings = sorted({v for r in rows for v in r if isinstance(v, str)})
    def cell(c, r, v):
        ref = f"{'ABC'[c]}{r + 1}"
        if isinstance(v, str):
            return f'<c r="{ref}" t="s"><v>{strings.index(v)}</v></c>'
        return f'<c r="{ref}"><v>{v}</v></c>'
    sheet = "".join(f'<row r="{r + 1}">' + "".join(cell(c, r, v) for c, v in enumerate(row)) + "</row>" for r, row in enumerate(rows))
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    with zipfile.ZipFile(tmp_path / "t.xlsx", "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", f"<worksheet {ns}><sheetData>{sheet}</sheetData></worksheet>")
        z.writestr("xl/sharedStrings.xml", f"<sst {ns}>" + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>")
    ds = load_dataset(str(tmp_path / "t.xlsx"))
    assert ds.task_type == "regression" and ds.target_name == "价格" and "面积" in ds.feature_names


def test_safe_relpath():
    from server.datasets import safe_relpath
    assert safe_relpath("../../etc/passwd").replace(os.sep, "/") == "etc/passwd"
    assert safe_relpath("数据\\猫\\1.jpg").replace(os.sep, "/") == "数据/猫/1.jpg"


def test_single_class_folder_is_not_ready(tmp_path):
    pytest.importorskip("PIL")
    rng = np.random.default_rng(4)
    (tmp_path / "cat").mkdir()
    for i in range(6):
        _png(tmp_path / "cat" / f"cat_{i:03d}.jpg", rng, (200, 200, 200))
    info = detect(str(tmp_path))
    assert info["ready"] is False and "cat" in info["problem"] and info["tree"] == {"cat": 6}
    with pytest.raises(DataError):
        load_dataset(str(tmp_path))
    (tmp_path / "dog").mkdir()                                   # 再上传一个类别 → 可以训练
    for i in range(6):
        _png(tmp_path / "dog" / f"dog_{i:03d}.jpg", rng, (10, 10, 10))
    assert detect(str(tmp_path))["classes"] == {"cat": 6, "dog": 6}
