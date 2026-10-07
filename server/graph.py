"""
网络结构图数据 / Architecture graph capture
==========================================
对任意网络：在顶层子模块上挂前向钩子，跑一小批数据，按 *实际执行顺序* 记录每层输出：
形状、均值/标准差、若干“代表性神经元”（抽样的通道/特征值）、参数量、权重范数、梯度范数，
以及该层类在源码中的位置（前端点击可跳转到代码）。
Works for any nn.Module: hooks on top-level children, recorded in execution order.
"""
from __future__ import annotations

import inspect
import math
import os

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = 10          # 每层展示的代表性神经元个数 / units shown per layer
MAX_NODES = 28      # 最多展示多少层 / max columns


def _f(x):
    x = float(x)
    return round(x, 4) if math.isfinite(x) else None


def source_of(obj):
    """返回 (相对路径, 行号)；PyTorch 内置层返回 None / project source location."""
    try:
        target = obj if inspect.isclass(obj) or inspect.isfunction(obj) else type(obj)
        path = os.path.abspath(inspect.getsourcefile(target))
        _, line = inspect.getsourcelines(target)
    except (TypeError, OSError):
        return None, None
    try:
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:                    # Windows：torch 在 C 盘、项目在 D 盘，跨盘无法算相对路径
        return None, None
    if rel.startswith(".."):
        return None, None
    return rel, line


def _units(t: torch.Tensor, as_input=False):
    """从张量中取第一个样本，抽出 UNITS 个代表值 / pick representative unit values."""
    t = t.detach().float()
    if t.dim() == 0:
        v = t.view(1)
    else:
        x = t[0]
        if as_input or x.dim() <= 1:
            v = x.flatten()
        elif x.dim() >= 3:                 # (C, H, W) -> 每个通道的平均 / per-channel mean
            v = x.flatten(1).mean(1)
        else:                              # (N, D) -> 对 token 取平均 / mean over tokens
            v = x.mean(0)
    n = v.numel()
    idx = torch.linspace(0, n - 1, min(UNITS, n)).round().long()
    return [_f(a) for a in v[idx]], n


def _norms(mod):
    w2 = g2 = 0.0
    n = 0
    for p in mod.parameters():
        n += p.numel()
        w2 += float(p.detach().float().pow(2).sum())
        if p.grad is not None:
            g2 += float(p.grad.detach().float().pow(2).sum())
    return n, _f(math.sqrt(w2)), _f(math.sqrt(g2))


def _targets(model):
    for name, m in model.named_children():
        if isinstance(m, nn.ModuleList):
            for i, sub in enumerate(m):
                yield f"{name}.{i}", sub
        elif not isinstance(m, (nn.Dropout, nn.Identity)):
            yield name, m


@torch.no_grad()
def capture_arch(task) -> dict:
    model = task.model
    records, seen, hooks = [], set(), []
    inp = {}

    def pre_hook(mod, args):
        if not inp and args and torch.is_tensor(args[0]):
            t = args[0]
            units, width = _units(t, as_input=True)
            inp.update(shape=list(t.shape), units=units, width=width, dtype=str(t.dtype).replace("torch.", ""),
                       mean=_f(t.float().mean()), std=_f(t.float().std()) if t.numel() > 1 else 0.0)

    def make_hook(name):
        def hook(mod, args, out):
            if name in seen:
                return
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t):
                return
            seen.add(name)
            units, width = _units(t)
            n, wn, gn = _norms(mod)
            file, line = source_of(mod)
            tf = t.detach().float()
            records.append(dict(name=name, cls=type(mod).__name__, shape=list(t.shape), units=units, width=width,
                                mean=_f(tf.mean()), std=_f(tf.std()) if tf.numel() > 1 else 0.0,
                                params=n, wnorm=wn, gnorm=gn, file=file, line=line))
        return hook

    hooks.append(model.register_forward_pre_hook(pre_hook))
    for name, m in _targets(model):
        hooks.append(m.register_forward_hook(make_hook(name)))
    was_training, bs = model.training, task.batch_size
    try:
        model.eval()
        task.batch_size = min(bs, 8)
        loss, _ = task.compute_loss()                 # 跑一小批，钩子按执行顺序记录 / run one small batch
    finally:
        task.batch_size = bs
        model.train(was_training)
        for h in hooks:
            h.remove()
    if len(records) > MAX_NODES:                      # 层太多时均匀抽取 / thin out very deep nets
        k = len(records) / MAX_NODES
        records = [records[int(i * k)] for i in range(MAX_NODES - 1)] + [records[-1]]
    mfile, mline = source_of(model)
    return {"kind": "arch", "model": type(model).__name__, "file": mfile, "line": mline,
            "input": inp, "nodes": records, "loss": _f(loss), "total_params": sum(p.numel() for p in model.parameters())}
