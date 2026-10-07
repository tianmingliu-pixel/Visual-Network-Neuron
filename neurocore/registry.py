"""
模型注册中心 / Model Registry
================================
所有网络（当前：Transformer / ViT / U-Net / DiT，未来：任意 IT 领域的网络）
都通过 `@register_model` 注册，然后统一用 `build_model(name, **kwargs)` 构建。

All networks register themselves with `@register_model` and are built with
`build_model(name, **kwargs)`. New domains (graph, audio, time-series, security,
recommendation, ...) plug in without touching the core code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# 预留的领域标签（可自由增加）/ Domain tags (free to extend)
KNOWN_DOMAINS = (
    "basic", "nlp", "vision", "generative",   # 已实现 / implemented
    "graph", "audio", "multimodal",           # 规划中 / planned
    "timeseries", "recsys", "security", "rl", "other",
)


@dataclass
class ModelSpec:
    name: str
    builder: Callable
    domain: str = "other"
    description: str = ""
    tags: List[str] = field(default_factory=list)


MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(name: Optional[str] = None, *, domain: str = "other",
                   description: str = "", tags: Optional[List[str]] = None):
    """装饰器：注册一个模型类或工厂函数 / Decorator: register a class or factory."""
    def deco(obj):
        key = (name or obj.__name__).lower()
        if key in MODEL_REGISTRY:
            raise KeyError(f"Model '{key}' already registered")
        MODEL_REGISTRY[key] = ModelSpec(key, obj, domain, description or (obj.__doc__ or "").strip().split("\n")[0], tags or [])
        return obj
    return deco


def build_model(name: str, **kwargs):
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key].builder(**kwargs)


def list_models(domain: Optional[str] = None) -> List[ModelSpec]:
    specs = sorted(MODEL_REGISTRY.values(), key=lambda s: (s.domain, s.name))
    return [s for s in specs if domain is None or s.domain == domain]


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
