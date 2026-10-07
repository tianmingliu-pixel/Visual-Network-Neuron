"""
被 PowerShell 脚本调用，输出 JSON 描述 PyTorch 状态。
Called by the PowerShell tools; prints one JSON line describing the PyTorch setup.
"""
import json
import platform
import sys

info = {"python": platform.python_version(), "executable": sys.executable,
        "torch": None, "cuda_available": False, "cuda_build": None, "gpus": [],
        "mps": False, "matmul_ok": False, "error": None}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_build"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        info["gpus"] = [{"name": torch.cuda.get_device_name(i),
                         "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                         "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1)}
                        for i in range(torch.cuda.device_count())]
    mps = getattr(torch.backends, "mps", None)
    info["mps"] = bool(mps and mps.is_available())
    dev = "cuda" if info["cuda_available"] else "cpu"
    a = torch.randn(256, 256, device=dev)
    info["matmul_ok"] = bool(torch.isfinite(a @ a).all().item())
except Exception as e:  # noqa
    info["error"] = f"{type(e).__name__}: {e}"
print(json.dumps(info))
