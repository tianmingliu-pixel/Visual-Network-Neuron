"""
NeuroCore —— 可扩展的神经网络核心库 / An extensible neural-network core library
当前模块 / Current: Transformer, ViT, U-Net, DiT (+ DDPM diffusion)
"""
__version__ = "0.1.0"

from .registry import register_model, build_model, list_models, count_parameters, MODEL_REGISTRY
from . import layers, models, diffusion
from . import extensions  # 自动加载扩展 / auto-load extensions

__all__ = ["register_model", "build_model", "list_models", "count_parameters", "MODEL_REGISTRY",
           "layers", "models", "diffusion", "extensions"]
