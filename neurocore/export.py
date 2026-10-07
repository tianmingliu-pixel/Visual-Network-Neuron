"""
ONNX 导出 / ONNX export
=======================
把训练好的网络导出为 .onnx，可在网页（onnxruntime-web）、C++、移动端等运行推理。
Export a trained network to ONNX for browser (onnxruntime-web), C++ or mobile inference.

    from neurocore.export import export_onnx
    export_onnx(model, (torch.randn(1, 3, 32, 32),), "vit.onnx", input_names=["image"])
"""
from __future__ import annotations

import copy
import os
from typing import Sequence

import torch


def export_onnx(model: torch.nn.Module, example_inputs: Sequence[torch.Tensor], path: str,
                input_names=None, output_names=("output",), opset: int = 18) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    m = copy.deepcopy(model).cpu().eval()
    args = tuple(t.cpu() for t in example_inputs)
    kw = dict(input_names=list(input_names) if input_names else None,
              output_names=list(output_names), opset_version=opset)
    try:
        # 新版导出器（需要 onnx + onnxscript）/ dynamo-based exporter
        torch.onnx.export(m, args, path, dynamo=True, **kw)
    except Exception as e_new:  # noqa
        try:
            torch.onnx.export(m, args, path, dynamo=False, **kw)    # 旧版导出器回退 / legacy fallback
        except Exception as e_old:
            raise RuntimeError(
                f"ONNX export failed. Install exporters with: pip install onnx onnxscript\n"
                f"dynamo exporter: {e_new}\nlegacy exporter: {e_old}") from e_old
    return path


def export_task_onnx(task, out_dir: str) -> str:
    """按任务类型构造示例输入并导出 / build example inputs for a UI task and export."""
    dev = next(task.model.parameters()).device
    tid = task.id
    if tid == "transformer":
        src = torch.randint(3, task.V, (1, task.L), device=dev)
        tgt = torch.full((1, 1), task.BOS, device=dev)
        inputs, names = (src, tgt), ["src_tokens", "tgt_tokens"]
    elif tid == "custom":                                    # 自定义数据：取一条真实样本作示例
        inputs, names = (task.X[:1].to(dev),), ["image" if task.dataset.is_image else "features"]
    elif tid == "vit":
        inputs, names = (torch.randn(1, 1, 32, 32, device=dev),), ["image"]
    else:  # unet / dit：噪声图 + 时间步（+ 类别）
        x = torch.randn(1, 1, task.size, task.size, device=dev)
        t = torch.tensor([50], device=dev)
        if getattr(task, "conditional", False):
            inputs, names = (x, t, torch.tensor([0], device=dev)), ["x_t", "timestep", "label"]
        else:
            inputs, names = (x, t), ["x_t", "timestep"]
    return export_onnx(task.model, inputs, os.path.join(out_dir, f"{tid}.onnx"), input_names=names)
