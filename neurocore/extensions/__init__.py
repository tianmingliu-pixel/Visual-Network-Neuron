"""
扩展模块目录 / Extension modules
================================
未来任何 IT 领域的网络（GNN、音频、时序、推荐、安全检测、强化学习……）
只需在本目录新建一个 .py 文件，用 `@register_model(..., domain=...)` 注册，
即会被自动发现并加载 —— 无需改动核心代码。以下划线开头的文件不会被加载。

Drop a new .py file in this folder, register your network with
`@register_model(..., domain=...)`, and it is auto-discovered on import.
Files starting with "_" (e.g. `_template.py`) are skipped.
"""
import importlib
import pkgutil
import warnings

loaded = []

for _m in pkgutil.iter_modules(__path__):
    if _m.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{_m.name}")
        loaded.append(_m.name)
    except Exception as e:  # 单个扩展出错不影响核心 / one bad extension won't break the core
        warnings.warn(f"[neurocore] failed to load extension '{_m.name}': {e}")
