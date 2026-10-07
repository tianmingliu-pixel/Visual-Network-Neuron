"""
ViT —— Vision Transformer
=========================
核心思想："一张图 = 16×16 个词"。把图像切成 patch，当作 token 序列，
加上 [CLS] 与位置编码，送入标准 Transformer 编码器，用 [CLS] 输出做分类。
Core idea: an image is a sequence of patches. Patch -> token, prepend [CLS],
add position embeddings, run a Transformer encoder, classify from [CLS].
"""
from __future__ import annotations  # 延迟解析类型注解

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块

from ..layers import PatchEmbed, TransformerBlock  # 导入公共零件：图像分块嵌入、Transformer 块
from ..registry import register_model  # 导入模型注册装饰器


@register_model("vit", domain="vision", description="Vision Transformer image classifier")  # 以名字 "vit" 注册，领域为 vision
class ViT(nn.Module):  # 定义 Vision Transformer
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,  # 输入图像边长、每个小块边长、输入通道数（RGB=3）
                 num_classes: int = 1000, dim: int = 768, depth: int = 12, num_heads: int = 12,  # 类别数、隐藏维度、层数、注意力头数
                 mlp_ratio: float = 4.0, dropout: float = 0.0, use_sdpa: bool = False):  # FFN 放大倍数、丢弃率、是否用融合注意力内核
        super().__init__()  # 父类初始化
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, dim)  # 把图像切成 patch 并线性投影成 dim 维 token
        n = self.patch_embed.num_patches  # patch 总数 = (img_size / patch_size)²，例如 224/16 → 196
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))  # 可学习的 [CLS] token，用于汇总整张图的信息
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))  # 可学习的位置编码，n 个 patch + 1 个 [CLS]
        self.pos_drop = nn.Dropout(dropout)  # 加完位置编码后的 dropout
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_ratio, dropout, use_sdpa=use_sdpa)  # depth 个 Transformer 编码器块
                                     for _ in range(depth)])  # 循环生成多层
        self.norm = nn.LayerNorm(dim)  # 最后的层归一化
        self.head = nn.Linear(dim, num_classes)  # 分类头：dim -> 类别数
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # 位置编码用截断正态分布初始化（std=0.02，与原论文一致）
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # [CLS] 同样初始化
        self.apply(self._init)  # 对所有子模块递归应用下面的初始化函数

    @staticmethod  # 静态方法：不需要 self
    def _init(m):  # 权重初始化规则
        if isinstance(m, nn.Linear):  # 如果是全连接层
            nn.init.trunc_normal_(m.weight, std=0.02)  # 权重用截断正态分布初始化
            if m.bias is not None:  # 如果有偏置
                nn.init.zeros_(m.bias)  # 偏置初始化为 0
        elif isinstance(m, nn.LayerNorm):  # 如果是层归一化
            nn.init.ones_(m.weight)  # 缩放系数初始化为 1
            nn.init.zeros_(m.bias)  # 平移系数初始化为 0

    def forward_features(self, x):  # 提取特征（不含分类头），x 形状 (B, C, H, W)
        x = self.patch_embed(x)                                   # 图像 → patch token 序列，形状 (B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # 把 [CLS] 复制 B 份，形状 (B, 1, D)
        x = self.pos_drop(torch.cat([cls, x], dim=1) + self.pos_embed)  # [CLS] 拼到最前面 → 加位置编码 → dropout，形状 (B, N+1, D)
        for blk in self.blocks:  # 逐层经过 Transformer 块
            x = blk(x)  # 每个 patch 通过自注意力与所有其他 patch 交换信息
        return self.norm(x)  # 返回归一化后的全部 token 特征

    def forward(self, x):  # 完整前向传播
        return self.head(self.forward_features(x)[:, 0])        # 取第 0 个位置（[CLS]）的特征送入分类头，输出 logits


# 常用预设 / Common presets
@register_model("vit_tiny", domain="vision", description="ViT-Ti/16 (dim 192, depth 12)")  # 注册 ViT-Tiny 预设
def vit_tiny(**kw):  # 工厂函数，**kw 可覆盖任意参数
    return ViT(**{**dict(dim=192, depth=12, num_heads=3), **kw})  # 默认配置 dim=192、12 层、3 头，用户传入的参数优先


@register_model("vit_small", domain="vision", description="ViT-S/16 (dim 384, depth 12)")  # 注册 ViT-Small 预设
def vit_small(**kw):  # 工厂函数
    return ViT(**{**dict(dim=384, depth=12, num_heads=6), **kw})  # dim=384、12 层、6 头


@register_model("vit_base", domain="vision", description="ViT-B/16 (dim 768, depth 12)")  # 注册 ViT-Base 预设
def vit_base(**kw):  # 工厂函数
    return ViT(**{**dict(dim=768, depth=12, num_heads=12), **kw})  # dim=768、12 层、12 头（原论文 ViT-B/16）
