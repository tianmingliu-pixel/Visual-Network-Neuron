"""位置编码 / 时间步编码 / 图像分块 —— Positional, timestep and patch embeddings."""
from __future__ import annotations  # 延迟解析类型注解

import math  # 数学库：log

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块


class SinusoidalPositionalEncoding(nn.Module):  # 正余弦位置编码
    """Transformer 原论文的正余弦位置编码 / Sinusoidal PE from 'Attention Is All You Need'."""

    def __init__(self, dim: int, max_len: int = 5000):  # 维度、支持的最大长度
        super().__init__()  # 父类初始化
        pos = torch.arange(max_len).unsqueeze(1)  # 位置编号 0..max_len-1，形状 (max_len, 1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))  # 各维度的频率 1/10000^(2i/d)：低维变化快、高维变化慢
        pe = torch.zeros(max_len, dim)  # 编码表
        pe[:, 0::2] = torch.sin(pos * div)  # 偶数维用 sin
        pe[:, 1::2] = torch.cos(pos * div)  # 奇数维用 cos
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # 存为缓冲区（不训练），形状 (1, max_len, dim)

    def forward(self, x):                       # x: (B, N, D)
        return x + self.pe[:, : x.size(1)]  # 取前 N 个位置的编码加到输入上


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:  # 扩散时间步 → 向量
    """扩散时间步 t -> 正余弦向量 / Map diffusion timesteps to sinusoidal features. t: (B,)"""
    half = dim // 2  # 一半维度放 cos，一半放 sin
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)  # 一组从高到低的频率
    args = t.float()[:, None] * freqs[None]  # 每个时间步乘以每个频率，形状 (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # 拼接 cos 与 sin，形状 (B, dim)
    if dim % 2:  # 维度为奇数时
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)  # 补一列 0
    return emb  # 返回时间步编码


class TimestepEmbedder(nn.Module):  # 时间步嵌入器（DiT 用）
    """t -> sinusoidal -> MLP"""

    def __init__(self, dim: int, freq_dim: int = 256):  # 输出维度、正余弦编码维度
        super().__init__()  # 父类初始化
        self.freq_dim = freq_dim  # 保存正余弦编码维度
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))  # 两层 MLP 把编码变成可学习的嵌入

    def forward(self, t):  # t：时间步 (B,)
        return self.mlp(timestep_embedding(t, self.freq_dim))  # 先正余弦编码，再过 MLP


class PatchEmbed(nn.Module):  # 图像分块嵌入（ViT / DiT 用）
    """
    把图像切成 p×p 小块并线性投影 —— 等价于 kernel=stride=p 的卷积。
    Split an image into p×p patches and project them (a strided conv).
    (B, C, H, W) -> (B, N, D),  N = (H/p)·(W/p)
    """

    def __init__(self, img_size: int, patch_size: int, in_channels: int, dim: int):  # 图像边长、块边长、输入通道、输出维度
        super().__init__()  # 父类初始化
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"  # 图像边长必须能被块边长整除
        self.img_size, self.patch_size = img_size, patch_size  # 保存尺寸
        self.grid = img_size // patch_size  # 每边有多少个块
        self.num_patches = self.grid ** 2  # 总块数
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)  # 卷积核=步长=p：每个块恰好被投影一次

    def forward(self, x):  # x：(B, C, H, W)
        return self.proj(x).flatten(2).transpose(1, 2)  # 卷积 → (B,D,grid,grid) → 展平 → 转置为 (B, N, D) 的 token 序列


def get_2d_sincos_pos_embed(dim: int, grid: int) -> torch.Tensor:  # 二维正余弦位置编码
    """固定 2D 正余弦位置编码（DiT/MAE 用法）/ Fixed 2D sin-cos PE. Returns (grid*grid, dim)."""
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin-cos PE"  # 需要能被 4 整除（行/列各一半，每半再分 sin/cos）

    def _1d(d, pos):  # 一维正余弦编码
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / (d / 2.0))  # 频率
        out = pos.reshape(-1)[:, None].double() * omega[None]  # 位置 × 频率
        return torch.cat([out.sin(), out.cos()], dim=1)  # 拼接 sin 与 cos

    ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")  # 每个块的行号与列号
    emb = torch.cat([_1d(dim // 2, ys), _1d(dim // 2, xs)], dim=1)  # 一半维度编码行，一半编码列
    return emb.float()  # 转为 float32 返回
