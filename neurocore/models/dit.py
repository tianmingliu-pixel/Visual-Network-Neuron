"""
DiT —— Diffusion Transformer (Peebles & Xie, 2023)
==================================================
核心思想：用 Transformer 替代 U-Net 作为扩散去噪网络。
  1) 噪声图(或 VAE latent) 切 patch -> token
  2) 条件 c = 时间步嵌入 + 类别嵌入
  3) adaLN-Zero：由 c 回归出每个块的 shift/scale/gate，gate 初始化为 0，
     让每个残差块初始为恒等映射，训练更稳定
  4) 线性解码回 patch -> unpatchify 还原成图，预测噪声 ε (可选同时预测方差)
Core idea: replace the U-Net denoiser with a Transformer over patches, and
inject conditioning (timestep + class) via adaptive LayerNorm with
zero-initialised gates (adaLN-Zero).
"""
from __future__ import annotations  # 延迟解析类型注解

from typing import Optional  # 类型提示：可选值

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块

from ..layers import MultiHeadAttention, FeedForward, PatchEmbed, TimestepEmbedder, get_2d_sincos_pos_embed  # 导入公共零件：注意力、前馈网络、分块嵌入、时间步嵌入、2D 位置编码
from ..registry import register_model  # 导入模型注册装饰器


def modulate(x, shift, scale):  # 调制函数：用条件向量对归一化后的特征做仿射变换
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # x·(1+scale)+shift；unsqueeze(1) 让 (B,D) 广播到 (B,N,D) 的每个 token


class LabelEmbedder(nn.Module):  # 类别嵌入器
    """类别嵌入 + 训练时随机丢弃标签(用于 CFG) / label embedding with label dropout for CFG."""

    def __init__(self, num_classes: int, dim: int, dropout_prob: float = 0.1):  # 类别数、嵌入维度、标签丢弃概率
        super().__init__()  # 父类初始化
        self.num_classes, self.dropout_prob = num_classes, dropout_prob  # 保存类别数和丢弃概率
        self.table = nn.Embedding(num_classes + 1, dim)  # 最后一个 = 空类 / last = null；嵌入表多一行表示"无条件"

    def forward(self, y, train: bool):  # y 为类别编号 (B,)，train 表示是否处于训练
        if train and self.dropout_prob > 0:  # 训练时按概率丢弃标签
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout_prob  # 为每个样本随机决定是否丢弃
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)  # 被丢弃的样本换成"空类" —— 让同一个模型同时学会有条件和无条件生成
        return self.table(y)  # 查表返回类别嵌入 (B, dim)


class DiTBlock(nn.Module):  # DiT 的核心块：带 adaLN-Zero 条件注入的 Transformer 块
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, use_sdpa: bool = False):  # 维度、头数、FFN 倍数、是否用融合内核
        super().__init__()  # 父类初始化
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)  # 注意力前的层归一化；不带自身可学习参数，缩放/平移由条件提供
        self.attn = MultiHeadAttention(dim, num_heads, use_sdpa=use_sdpa)  # 多头自注意力
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)  # FFN 前的层归一化（同样无自身参数）
        self.mlp = FeedForward(dim, int(dim * mlp_ratio))  # 前馈网络
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))  # 从条件 c 回归出 6 组参数：两组 shift、scale、gate
        nn.init.zeros_(self.adaLN[-1].weight)      # adaLN-Zero：权重初始化为 0
        nn.init.zeros_(self.adaLN[-1].bias)  # 偏置也为 0 → 初始 gate=0，整个块一开始就是恒等映射，训练更稳

    def forward(self, x, c):  # x 为 token 序列 (B,N,D)，c 为条件向量 (B,D)
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)  # 把 6·dim 的输出切成 6 份
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), sh1, sc1))  # 归一化 → 用条件调制 → 自注意力 → 乘门控 g1 → 残差相加
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh2, sc2))  # 归一化 → 用条件调制 → FFN → 乘门控 g2 → 残差相加
        return x  # 返回更新后的 token


class FinalLayer(nn.Module):  # 输出层：把 token 映射回 patch 像素
    def __init__(self, dim: int, patch_size: int, out_channels: int):  # 维度、patch 边长、输出通道数
        super().__init__()  # 父类初始化
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)  # 最后的层归一化（无自身参数）
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)  # 每个 token → 一个 patch 的全部像素值（p·p·C 个数）
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))  # 由条件回归出 shift、scale 两组参数
        for m in (self.linear, self.adaLN[-1]):  # 对输出线性层和 adaLN 线性层
            nn.init.zeros_(m.weight)  # 权重初始化为 0
            nn.init.zeros_(m.bias)  # 偏置初始化为 0 → 模型初始输出全为 0

    def forward(self, x, c):  # 前向传播
        shift, scale = self.adaLN(c).chunk(2, dim=-1)  # 得到平移和缩放参数
        return self.linear(modulate(self.norm(x), shift, scale))  # 归一化 → 调制 → 线性映射到 patch 像素


@register_model("dit", domain="generative", description="Diffusion Transformer (adaLN-Zero)")  # 以名字 "dit" 注册，领域为 generative
class DiT(nn.Module):  # 定义 Diffusion Transformer
    def __init__(self, img_size: int = 32, patch_size: int = 2, in_channels: int = 4, dim: int = 384,  # 输入边长（常为 VAE latent 的 32）、patch 边长、输入通道（latent 为 4）、隐藏维度
                 depth: int = 12, num_heads: int = 6, mlp_ratio: float = 4.0,  # 层数、头数、FFN 倍数
                 num_classes: Optional[int] = 1000, class_dropout_prob: float = 0.1,  # 类别数（None 为无条件）、训练时标签丢弃概率
                 learn_sigma: bool = False, use_sdpa: bool = False):  # 是否同时预测方差、是否用融合内核
        super().__init__()  # 父类初始化
        self.in_channels = in_channels  # 保存输入通道数
        self.out_channels = in_channels * 2 if learn_sigma else in_channels  # 预测方差时输出通道翻倍（噪声 ε + 方差）
        self.patch_size = patch_size  # 保存 patch 边长
        self.x_embed = PatchEmbed(img_size, patch_size, in_channels, dim)  # 噪声图 → patch token
        self.t_embed = TimestepEmbedder(dim)  # 时间步 t → 嵌入向量
        self.y_embed = LabelEmbedder(num_classes, dim, class_dropout_prob) if num_classes else None  # 类别 y → 嵌入向量（可选）
        self.num_classes = num_classes  # 保存类别数（CFG 采样时需要）
        pos = get_2d_sincos_pos_embed(dim, self.x_embed.grid).unsqueeze(0)  # 计算固定的 2D 正余弦位置编码，形状 (1, N, dim)
        self.register_buffer("pos_embed", pos, persistent=False)       # 注册为缓冲区：随模型移动设备，但不参与训练
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, mlp_ratio, use_sdpa) for _ in range(depth)])  # depth 个 DiT 块
        self.final = FinalLayer(dim, patch_size, self.out_channels)  # 输出层

    def unpatchify(self, x):  # 把 patch 序列拼回完整图像
        """(B, N, p*p*C) -> (B, C, H, W)"""
        p, c, g = self.patch_size, self.out_channels, self.x_embed.grid  # patch 边长、输出通道、每边 patch 个数
        x = x.reshape(x.shape[0], g, g, p, p, c)  # 拆成 (B, 网格行, 网格列, patch 内行, patch 内列, 通道)
        x = torch.einsum("nhwpqc->nchpwq", x)  # 重排维度为 (B, 通道, 网格行, patch 内行, 网格列, patch 内列)
        return x.reshape(x.shape[0], c, g * p, g * p)  # 合并行列，得到 (B, C, H, W) 的图像

    def forward(self, x, t, y: Optional[torch.Tensor] = None):  # 前向传播：x 噪声图、t 时间步、y 类别
        h = self.x_embed(x) + self.pos_embed                         # 分块嵌入 + 位置编码，形状 (B, N, D)
        c = self.t_embed(t)  # 时间步嵌入作为条件
        if self.y_embed is not None:  # 如果使用类别条件
            if y is None:  # 没传类别
                y = torch.full((x.size(0),), self.num_classes, dtype=torch.long, device=x.device)  # 视为"空类"（无条件）
            c = c + self.y_embed(y, self.training)  # 条件 = 时间嵌入 + 类别嵌入（训练时会随机丢弃标签）
        for blk in self.blocks:  # 逐层经过 DiT 块
            h = blk(h, c)  # 每个块都通过 adaLN 接收条件 c
        return self.unpatchify(self.final(h, c))  # 输出层映射回 patch → 拼回整图，得到预测的噪声 ε


# DiT 论文预设 / presets from the paper
@register_model("dit_s_2", domain="generative", description="DiT-S/2 (dim 384, depth 12)")  # 注册 DiT-S/2 预设
def dit_s_2(**kw):  # 工厂函数，**kw 可覆盖参数
    return DiT(**{**dict(dim=384, depth=12, num_heads=6, patch_size=2), **kw})  # Small：dim=384、12 层、6 头、patch=2


@register_model("dit_b_2", domain="generative", description="DiT-B/2 (dim 768, depth 12)")  # 注册 DiT-B/2 预设
def dit_b_2(**kw):  # 工厂函数
    return DiT(**{**dict(dim=768, depth=12, num_heads=12, patch_size=2), **kw})  # Base：dim=768、12 层、12 头


@register_model("dit_xl_2", domain="generative", description="DiT-XL/2 (dim 1152, depth 28)")  # 注册 DiT-XL/2 预设
def dit_xl_2(**kw):  # 工厂函数
    return DiT(**{**dict(dim=1152, depth=28, num_heads=16, patch_size=2), **kw})  # XL：dim=1152、28 层、16 头（论文最强配置）
