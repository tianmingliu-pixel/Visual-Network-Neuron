"""
核心构件：自注意力 / Core building blocks: (self-)attention
=========================================================
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

这里刻意用显式矩阵运算写出注意力，便于学习核心思想；
设 `use_sdpa=True` 可切换为 PyTorch 融合内核（更快、更省显存）。
Attention is written out explicitly to show the idea; `use_sdpa=True`
switches to PyTorch's fused kernel for speed.
"""
from __future__ import annotations  # 延迟解析类型注解

import math  # 数学库，用于 sqrt
from typing import Optional  # 类型提示：可选值

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口（融合注意力内核）


class MultiHeadAttention(nn.Module):  # 多头注意力：整个项目最核心的计算
    """多头注意力（自注意力 / 交叉注意力）/ Multi-head self- or cross-attention."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0,  # 特征维度、头数、注意力权重丢弃率
                 bias: bool = True, use_sdpa: bool = False):  # 线性层是否带偏置、是否使用融合内核
        super().__init__()  # 父类初始化
        assert dim % num_heads == 0, "dim must be divisible by num_heads"  # 维度必须能被头数整除，才能平均分给每个头
        self.num_heads = num_heads  # 保存头数
        self.head_dim = dim // num_heads  # 每个头分到的维度 d_k
        self.use_sdpa = use_sdpa  # 保存是否用融合内核
        self.q_proj = nn.Linear(dim, dim, bias=bias)  # 查询投影 W_Q：每个 token 生成“我要找什么”
        self.k_proj = nn.Linear(dim, dim, bias=bias)  # 键投影 W_K：每个 token 生成“我有什么特征”
        self.v_proj = nn.Linear(dim, dim, bias=bias)  # 值投影 W_V：每个 token 生成“我能提供的内容”
        self.out_proj = nn.Linear(dim, dim, bias=bias)  # 输出投影 W_O：把多个头的结果融合回 dim 维
        self.attn_drop = nn.Dropout(dropout)  # 对注意力权重做 dropout
        self.dropout_p = dropout  # 保存丢弃率（融合内核需要数值）

    def _split(self, x: torch.Tensor) -> torch.Tensor:          # 拆成多头：(B,N,D) -> (B,h,N,d)
        B, N, _ = x.shape  # 批大小 B、序列长度 N
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # 把最后一维切成 h 份，再把“头”维移到前面

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,  # x：查询序列；context：键值序列
                key_padding_mask: Optional[torch.Tensor] = None,  # 填充掩码
                is_causal: bool = False) -> torch.Tensor:  # 是否因果（不能看未来）
        """
        x:       (B, N, D) 查询序列 / queries
        context: (B, M, D) 键值序列；None 表示自注意力 / keys & values (None = self-attn)
        key_padding_mask: (B, M) bool，True = 有效 token / True = keep
        is_causal: 因果遮罩（解码器）/ causal mask for decoders
        """
        context = x if context is None else context  # 没给 context 就是自注意力：Q、K、V 都来自 x
        B, N, D = x.shape  # 查询的批大小、长度、维度
        M = context.shape[1]  # 键值序列长度
        q, k, v = self._split(self.q_proj(x)), self._split(self.k_proj(context)), self._split(self.v_proj(context))  # 线性投影得到 Q、K、V 并拆成多头

        mask = None  # bool, True = attend; broadcast to (B,h,N,M)  # 掩码：True 表示允许注意
        if key_padding_mask is not None:  # 如果有填充掩码
            mask = key_padding_mask[:, None, None, :].bool()  # 扩展成 (B,1,1,M)，可广播到所有头和所有查询
        if is_causal:  # 如果需要因果掩码
            causal = torch.ones(N, M, dtype=torch.bool, device=x.device).tril(diagonal=M - N)  # 下三角矩阵：第 i 个位置只能看 ≤ i 的位置
            mask = causal if mask is None else (mask & causal)  # 与填充掩码合并

        if self.use_sdpa:  # 使用 PyTorch 融合内核
            out = F.scaled_dot_product_attention(  # 一次调用完成整套注意力计算（更快）
                q, k, v, attn_mask=mask, dropout_p=self.dropout_p if self.training else 0.0)  # 训练时才 dropout
        else:  # 显式写出公式（便于学习）
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # 相似度 QKᵀ/√d_k，形状 (B,h,N,M)；除以 √d 防止 softmax 过于尖锐
            if mask is not None:  # 有掩码时
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)  # 不允许注意的位置填极小值，softmax 后≈0
            attn = self.attn_drop(scores.softmax(dim=-1))  # softmax 归一化成注意力权重（每行和为 1），再 dropout
            out = attn @ v                                                  # 按权重对 V 加权求和，形状 (B,h,N,d)

        out = out.transpose(1, 2).reshape(B, N, D)  # 合并多头：(B,h,N,d) → (B,N,h·d=D)
        return self.out_proj(out)  # 输出投影，融合各头信息


class FeedForward(nn.Module):  # 前馈网络：对每个 token 独立做非线性变换
    """逐位置前馈网络 / Position-wise MLP: Linear -> GELU -> Linear."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):  # 输入维度、隐藏维度（通常 4×dim）、丢弃率
        super().__init__()  # 父类初始化
        self.net = nn.Sequential(  # 顺序网络
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),  # 升维 → GELU 激活 → dropout
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),  # 降回原维度 → dropout
        )  # 结束 Sequential

    def forward(self, x):  # 前向传播
        return self.net(x)  # 依次经过上面各层


class TransformerBlock(nn.Module):  # 一个标准 Transformer 块
    """
    Pre-LN Transformer 块 / Pre-LN block
      x = x + SelfAttn(LN(x))
      x = x + CrossAttn(LN(x), memory)   (可选 / optional, decoder)
      x = x + FFN(LN(x))
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,  # 维度、头数、FFN 放大倍数
                 dropout: float = 0.0, cross_attention: bool = False, use_sdpa: bool = False):  # 丢弃率、是否带交叉注意力、是否用融合内核
        super().__init__()  # 父类初始化
        self.norm1 = nn.LayerNorm(dim)  # 自注意力前的层归一化
        self.attn = MultiHeadAttention(dim, num_heads, dropout, use_sdpa=use_sdpa)  # 自注意力
        self.cross = None  # 默认没有交叉注意力
        if cross_attention:  # 解码器需要交叉注意力
            self.norm_c = nn.LayerNorm(dim)  # 交叉注意力前的层归一化
            self.cross = MultiHeadAttention(dim, num_heads, dropout, use_sdpa=use_sdpa)  # 交叉注意力：Q 来自自己，K/V 来自编码器
        self.norm2 = nn.LayerNorm(dim)  # FFN 前的层归一化
        self.ff = FeedForward(dim, int(dim * mlp_ratio), dropout)  # 前馈网络
        self.drop = nn.Dropout(dropout)  # 残差分支上的 dropout

    def forward(self, x, memory=None, self_mask=None, memory_mask=None, is_causal=False):  # x 输入、memory 编码器输出、各种掩码
        x = x + self.drop(self.attn(self.norm1(x), key_padding_mask=self_mask, is_causal=is_causal))  # 残差 + 自注意力（先归一化）
        if self.cross is not None and memory is not None:  # 如果有交叉注意力且提供了 memory
            x = x + self.drop(self.cross(self.norm_c(x), context=memory, key_padding_mask=memory_mask))  # 残差 + 交叉注意力
        x = x + self.ff(self.norm2(x))  # 残差 + 前馈网络
        return x  # 返回块输出
