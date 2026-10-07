"""
训练全过程的 5 个阶段（数据 → 前向 → 损失 → 反向 → 更新），每个阶段附公式与对应代码行
The five training stages, each with a formula and the exact source lines that implement it.
"""
from __future__ import annotations

import inspect
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ref_in(func, needle: str, label: str = ""):
    """在函数源码里找包含 needle 的那一行 / locate the line containing `needle` inside `func`."""
    try:
        lines, start = inspect.getsourcelines(func)
        path = os.path.abspath(inspect.getsourcefile(func))
    except (TypeError, OSError):
        return None
    try:
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:                    # Windows 跨盘（如 torch 在 C 盘、项目在 D 盘）
        return None
    if rel.startswith(".."):
        return None
    for i, ln in enumerate(lines):
        if needle in ln:
            return {"file": rel, "line": start + i, "label": label or ln.strip()[:60]}
    return {"file": rel, "line": start, "label": label or func.__name__}


def def_ref(obj, label=""):
    """对象（类/函数）定义所在行 / where a class or function is defined."""
    from .graph import source_of
    f, l = source_of(obj)
    return {"file": f, "line": l, "label": label or getattr(obj, "__name__", "")} if f else None


def build_pipeline(task_cls) -> list:
    from .tasks import BaseTask
    from neurocore.diffusion.ddpm import GaussianDiffusion
    import neurocore as nc

    cl = task_cls.compute_loss
    is_diff = getattr(task_cls, "IS_DIFFUSION", False)
    loss_ref = ref_in(GaussianDiffusion.training_loss, "F.mse_loss(", "MSE 损失") if is_diff \
        else ref_in(cl, task_cls.LOSS_NEEDLE, "交叉熵损失")
    data_ref = ref_in(cl, task_cls.DATA_NEEDLE, "取一批数据")
    fwd_ref = ref_in(cl, task_cls.FORWARD_NEEDLE, "调用网络")
    model_ref = None
    try:
        model_ref = def_ref(nc.MODEL_REGISTRY[task_cls.id].builder, "网络定义")
    except KeyError:
        pass
    steps = [
        {"id": "data", "title": "① 数据", "text": task_cls.DATA_TEXT, "formula": "(x, y) ~ 训练集，取一个小批量 B",
         "refs": [r for r in (data_ref,) if r],
         "fns": [task_cls.DATA_NEEDLE.replace("self.", "").rstrip("(")]},
        {"id": "forward", "title": "② 前向传播", "text": task_cls.FORWARD_TEXT,
         "formula": "a⁽ˡ⁾ = σ(W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)，逐层计算直到输出",
         "refs": [r for r in (fwd_ref, model_ref) if r]},
        {"id": "loss", "title": "③ 损失", "text": task_cls.LOSS_TEXT, "formula": task_cls.LOSS_FORMULA,
         "refs": [r for r in (loss_ref,) if r]},
        {"id": "backward", "title": "④ 反向传播",
         "text": "从损失出发按链式法则把误差一层层往回传，算出每个权重对损失的影响（梯度）。"
                 "梯度为正说明把该权重调小能降低损失。总梯度范数超过 1 时按比例缩小（梯度裁剪）。",
         "formula": "∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ · a⁽ˡ⁻¹⁾ᵀ，  δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾) ⊙ σ'(z⁽ˡ⁾)",
         "refs": [ref_in(BaseTask.train_step, "loss.backward()", "loss.backward()"),
                  ref_in(BaseTask.train_step, "clip_grad_norm_", "梯度裁剪")]},
        {"id": "update", "title": "⑤ 更新权重",
         "text": "AdamW 用梯度的滑动平均（动量 m）和平方的滑动平均（v）给每个参数自适应步长，"
                 "沿“让损失下降”的方向挪一小步，并做权重衰减。然后清空梯度，进入下一次迭代。",
         "formula": "m←β₁m+(1−β₁)g， v←β₂v+(1−β₂)g²， θ ← θ − lr·m̂/(√v̂+ε) − lr·λθ",
         "refs": [ref_in(BaseTask.train_step, "self.opt.step()", "opt.step()"),
                  ref_in(BaseTask.train_step, "zero_grad", "清空梯度")]},
    ]
    for s in steps:
        s["refs"] = [r for r in s["refs"] if r]
    return steps
