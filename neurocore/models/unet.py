"""
U-Net
=====
核心思想：编码器逐级下采样提取语义，解码器逐级上采样恢复分辨率，
同级之间用 **跳跃连接(skip connection)** 拼接特征，保留细节。
- time_conditioned=False：经典 U-Net（如图像分割）
- time_conditioned=True ：扩散模型去噪 U-Net（DDPM），输入噪声图 + 时间步 t(+类别 y)
Core idea: contracting path (down) + expanding path (up) joined by skip
connections that concatenate same-resolution features. Optionally conditioned
on a diffusion timestep (and class label) for DDPM-style denoising.
"""
from __future__ import annotations  # 延迟解析类型注解

import math  # 数学库，用于 gcd 最大公约数
from typing import Optional, Sequence  # 类型提示：可选值、序列

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口（插值等）

from ..layers import MultiHeadAttention, timestep_embedding  # 导入多头注意力、时间步正余弦编码
from ..registry import register_model  # 导入模型注册装饰器


def norm(ch: int) -> nn.GroupNorm:  # 归一化层工厂函数
    return nn.GroupNorm(math.gcd(32, ch), ch)  # 组归一化；组数取 32 与通道数的最大公约数，保证能整除


class ResBlock(nn.Module):  # 残差块：U-Net 的基本卷积单元
    """GN-SiLU-Conv ×2 + 残差；注入时间嵌入 / residual block with time-embedding injection."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: Optional[int], dropout: float = 0.0):  # 输入通道、输出通道、条件嵌入维度（None 表示无条件）、丢弃率
        super().__init__()  # 父类初始化
        self.block1 = nn.Sequential(norm(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))  # 第一段：归一化 → SiLU 激活 → 3×3 卷积（改变通道数）
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch)) if emb_dim else None  # 把时间/类别嵌入投影到 out_ch 维，用于注入特征图
        self.block2 = nn.Sequential(norm(out_ch), nn.SiLU(), nn.Dropout(dropout),  # 第二段：归一化 → SiLU → dropout
                                    nn.Conv2d(out_ch, out_ch, 3, padding=1))  # → 3×3 卷积
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()  # 残差捷径：通道数不同时用 1×1 卷积对齐，否则直接恒等

    def forward(self, x, emb=None):  # 前向传播，x 为特征图，emb 为条件嵌入
        h = self.block1(x)  # 经过第一段卷积
        if self.emb_proj is not None and emb is not None:  # 如果有条件嵌入
            h = h + self.emb_proj(emb)[:, :, None, None]  # 投影成 (B, C, 1, 1) 后加到每个像素上 —— 告诉网络"现在是第几步去噪"
        return self.block2(h) + self.skip(x)  # 第二段卷积结果 + 残差捷径


class SpatialSelfAttention(nn.Module):  # 空间自注意力：让图像中任意两个像素直接交互
    """把 H×W 展平为序列做自注意力 / self-attention over flattened pixels."""

    def __init__(self, ch: int, num_heads: int = 4):  # 通道数、注意力头数
        super().__init__()  # 父类初始化
        heads = num_heads if ch % num_heads == 0 else 1  # 通道数能被头数整除才用多头，否则退化为单头
        self.norm = norm(ch)  # 注意力前的组归一化
        self.attn = MultiHeadAttention(ch, heads)  # 多头自注意力（公共零件）

    def forward(self, x, emb=None):  # 前向传播（emb 参数不使用，只为与其他层接口统一）
        B, C, H, W = x.shape  # 批大小、通道、高、宽
        h = self.norm(x).flatten(2).transpose(1, 2)               # 归一化 → 把 H×W 展平 → 转置为序列 (B, HW, C)，每个像素是一个 token
        h = self.attn(h).transpose(1, 2).reshape(B, C, H, W)  # 做自注意力 → 转置并还原为特征图形状
        return x + h  # 残差连接


class Downsample(nn.Module):  # 下采样层：分辨率减半
    def __init__(self, ch):  # 通道数
        super().__init__()  # 父类初始化
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)  # 步长为 2 的 3×3 卷积，高宽各缩小一半

    def forward(self, x, emb=None):  # 前向传播（emb 不使用）
        return self.conv(x)  # 执行下采样卷积


class Upsample(nn.Module):  # 上采样层：分辨率翻倍
    def __init__(self, ch):  # 通道数
        super().__init__()  # 父类初始化
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)  # 上采样后接一个 3×3 卷积，平滑放大带来的块状伪影

    def forward(self, x, emb=None):  # 前向传播（emb 不使用）
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))  # 最近邻插值放大 2 倍，再卷积


class EmbedSequential(nn.Sequential):  # 顺序容器，与 nn.Sequential 的区别是会把 emb 一并传给每层
    """顺序容器，把 emb 传给每一层 / Sequential that passes `emb` to every layer."""

    def forward(self, x, emb=None):  # 前向传播
        for layer in self:  # 依次遍历容器内的每一层
            x = layer(x, emb)  # 每层都接收特征图和条件嵌入
        return x  # 返回最终输出


@register_model("unet", domain="vision", description="U-Net (segmentation or diffusion denoiser)")  # 以名字 "unet" 注册
class UNet(nn.Module):  # 定义 U-Net 主体
    def __init__(self, in_channels: int = 3, out_channels: Optional[int] = None, base_channels: int = 64,  # 输入通道、输出通道（默认同输入）、基础通道数
                 channel_mults: Sequence[int] = (1, 2, 4), num_res_blocks: int = 2,  # 每一级的通道倍数（级数 = 长度）、每级残差块个数
                 attn_levels: Sequence[int] = (2,), dropout: float = 0.0,  # 哪几级加入自注意力、丢弃率
                 time_conditioned: bool = True, num_classes: Optional[int] = None):  # 是否以扩散时间步为条件、类别数（None 表示不用类别条件）
        super().__init__()  # 父类初始化
        out_channels = out_channels or in_channels  # 没指定输出通道时，输出通道 = 输入通道（扩散模型预测同形状的噪声）
        self.num_classes = num_classes  # 保存类别数（CFG 采样时需要知道"空类"编号）
        emb_dim = base_channels * 4 if (time_conditioned or num_classes) else None  # 条件嵌入维度取基础通道的 4 倍；完全无条件时为 None
        self.time_embed = (nn.Sequential(nn.Linear(base_channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))  # 时间步嵌入 MLP：正余弦编码 → 线性 → SiLU → 线性
                           if time_conditioned else None)  # 不需要时间条件时为 None
        self.base = base_channels  # 保存基础通道数（也是正余弦时间编码的维度）
        # 类别条件：多 1 个"空类"用于 classifier-free guidance / +1 null class for CFG
        self.label_embed = nn.Embedding(num_classes + 1, emb_dim) if num_classes else None  # 类别嵌入表，最后一个编号代表"无类别"

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)  # 输入卷积：把图像通道变为 base_channels
        self.downs = nn.ModuleList()  # 编码器（下采样路径）的所有层
        skip_chs, ch = [base_channels], base_channels  # skip_chs 记录每个跳跃连接的通道数；ch 为当前通道数
        for lvl, mult in enumerate(channel_mults):  # 遍历每一级（lvl=级别编号，mult=通道倍数）
            for _ in range(num_res_blocks):  # 每级放 num_res_blocks 个残差块
                layers = [ResBlock(ch, base_channels * mult, emb_dim, dropout)]  # 残差块：通道 ch → base*mult
                ch = base_channels * mult  # 更新当前通道数
                if lvl in attn_levels:  # 如果这一级需要注意力
                    layers.append(SpatialSelfAttention(ch))  # 在残差块后加入空间自注意力
                self.downs.append(EmbedSequential(*layers))  # 打包成一个整体加入编码器
                skip_chs.append(ch)  # 记录这个输出的通道数，供解码器跳跃连接使用
            if lvl != len(channel_mults) - 1:  # 除最后一级外
                self.downs.append(EmbedSequential(Downsample(ch)))  # 加一个下采样层，分辨率减半
                skip_chs.append(ch)  # 下采样输出也作为一个跳跃连接

        self.mid = EmbedSequential(ResBlock(ch, ch, emb_dim, dropout), SpatialSelfAttention(ch),  # 瓶颈层（最低分辨率）：残差块 → 自注意力
                                   ResBlock(ch, ch, emb_dim, dropout))  # → 残差块

        self.ups = nn.ModuleList()  # 解码器（上采样路径）的所有层
        for lvl, mult in reversed(list(enumerate(channel_mults))):  # 从最深一级倒序往回走
            for i in range(num_res_blocks + 1):  # 每级比编码器多一个块，用来消化下采样层产生的那个跳跃连接
                layers = [ResBlock(ch + skip_chs.pop(), base_channels * mult, emb_dim, dropout)]  # 输入通道 = 当前通道 + 拼接进来的跳跃连接通道
                ch = base_channels * mult  # 更新当前通道数
                if lvl in attn_levels:  # 如果这一级需要注意力
                    layers.append(SpatialSelfAttention(ch))  # 加入空间自注意力
                if lvl != 0 and i == num_res_blocks:  # 除最浅一级外，每级最后一个块之后
                    layers.append(Upsample(ch))  # 加上采样层，分辨率翻倍
                self.ups.append(EmbedSequential(*layers))  # 打包加入解码器

        self.out = nn.Sequential(norm(ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, padding=1))  # 输出层：归一化 → SiLU → 卷积到输出通道数
        self.downsample_factor = 2 ** (len(channel_mults) - 1)  # 总下采样倍数（输入边长需能被它整除）

    def forward(self, x, t: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None):  # 前向传播：x 图像、t 时间步、y 类别
        emb = None  # 条件嵌入，默认没有
        if self.time_embed is not None:  # 如果模型以时间步为条件
            if t is None:  # 但没有传入 t
                raise ValueError("time_conditioned UNet requires timestep t")  # 报错提示
            emb = self.time_embed(timestep_embedding(t, self.base))  # t → 正余弦编码 → MLP，得到时间嵌入
        if self.label_embed is not None:  # 如果模型以类别为条件
            if y is None:   # 无条件 = 空类 / unconditional = null class
                y = torch.full((x.size(0),), self.num_classes, dtype=torch.long, device=x.device)  # 用"空类"编号填充
            le = self.label_embed(y)  # 查表得到类别嵌入
            emb = le if emb is None else emb + le  # 与时间嵌入相加（或单独使用）

        h = self.in_conv(x)  # 输入卷积
        skips = [h]  # 跳跃连接栈，先压入输入卷积的输出
        for m in self.downs:                 # 编码路径 / contracting path
            h = m(h, emb)  # 经过残差块 / 注意力 / 下采样
            skips.append(h)  # 每一步的输出都压栈，留给解码器
        h = self.mid(h, emb)                 # 瓶颈 / bottleneck
        for m in self.ups:                   # 解码路径 + 跳跃连接 / expanding path + skips
            h = m(torch.cat([h, skips.pop()], dim=1), emb)  # 从栈顶取出同分辨率的编码特征，在通道维拼接后再处理（U-Net 的灵魂）
        return self.out(h)  # 输出层：分割时为每像素类别得分，扩散时为预测的噪声


@register_model("unet_seg", domain="vision", description="Classic U-Net for segmentation (no time conditioning)")  # 注册分割用 U-Net
def unet_seg(in_channels: int = 3, num_classes: int = 2, **kw):  # 工厂函数：输入通道、分割类别数、其他参数
    """分割：输出 num_classes 个通道的 logits / outputs per-pixel class logits."""
    kw.setdefault("attn_levels", ())  # 默认不加注意力（经典 U-Net 只有卷积）
    return UNet(in_channels=in_channels, out_channels=num_classes, time_conditioned=False, **kw)  # 输出通道 = 类别数，关闭时间条件
