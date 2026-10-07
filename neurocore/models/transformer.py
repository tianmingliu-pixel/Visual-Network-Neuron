"""
Transformer（编码器-解码器）/ Encoder-Decoder Transformer
======================================================
核心思想：完全用注意力替代循环/卷积。
  Encoder: 双向自注意力理解源序列
  Decoder: 因果自注意力 + 交叉注意力(看编码器输出) 逐 token 生成
Core idea: attention only. The encoder reads the source with bidirectional
self-attention; the decoder generates autoregressively with causal
self-attention plus cross-attention to the encoder memory.
"""
from __future__ import annotations  # 允许在类型注解中使用尚未定义的类型（延迟解析注解）

import math  # 导入数学库，用于 sqrt 开平方

import torch  # 导入 PyTorch 主库（张量运算）
import torch.nn as nn  # 导入神经网络模块，简写为 nn

from ..layers import TransformerBlock, SinusoidalPositionalEncoding  # 导入公共零件：Transformer 块、正余弦位置编码
from ..registry import register_model  # 导入模型注册装饰器


@register_model("transformer", domain="nlp", description="Encoder-decoder Transformer (seq2seq)")  # 以名字 "transformer" 注册到模型库，领域为 nlp
class Transformer(nn.Module):  # 定义编码器-解码器 Transformer，继承 nn.Module
    def __init__(self, src_vocab_size: int = 1000, tgt_vocab_size: int = 1000, dim: int = 256,  # 源词表大小、目标词表大小、隐藏维度
                 depth: int = 4, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1,  # 层数、注意力头数、FFN 放大倍数、丢弃率
                 max_len: int = 512, pad_id: int = 0, use_sdpa: bool = False):  # 最大序列长度、填充符 id、是否用 PyTorch 融合注意力内核
        super().__init__()  # 调用父类 nn.Module 的初始化
        self.dim, self.pad_id = dim, pad_id  # 保存隐藏维度和填充符 id，后面要用
        self.src_emb = nn.Embedding(src_vocab_size, dim, padding_idx=pad_id)  # 源语言词嵌入表：token id -> dim 维向量（填充符向量恒为 0）
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dim, padding_idx=pad_id)  # 目标语言词嵌入表
        self.pos = SinusoidalPositionalEncoding(dim, max_len)  # 正余弦位置编码，告诉模型每个 token 的位置
        self.drop = nn.Dropout(dropout)  # 嵌入之后的随机丢弃，防止过拟合
        self.encoder = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_ratio, dropout, use_sdpa=use_sdpa)  # 编码器：depth 个标准 Transformer 块
                                      for _ in range(depth)])  # 循环 depth 次生成多层
        self.decoder = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_ratio, dropout,  # 解码器：depth 个 Transformer 块
                                                       cross_attention=True, use_sdpa=use_sdpa)  # 解码器块额外开启交叉注意力（去看编码器输出）
                                      for _ in range(depth)])  # 循环 depth 次生成多层
        self.enc_norm = nn.LayerNorm(dim)  # 编码器最后的层归一化（Pre-LN 结构需要在末尾再归一化一次）
        self.dec_norm = nn.LayerNorm(dim)  # 解码器最后的层归一化
        self.lm_head = nn.Linear(dim, tgt_vocab_size)  # 输出头：把 dim 维向量映射成目标词表上每个词的得分（logits）

    def encode(self, src: torch.Tensor):  # 编码：读入源序列，输出"记忆"（memory）
        src_mask = src != self.pad_id                                     # 有效位置掩码 (B, S)：非填充为 True
        x = self.drop(self.pos(self.src_emb(src) * math.sqrt(self.dim)))  # 词嵌入 × √dim（放大到与位置编码同量级）→ 加位置编码 → dropout
        for blk in self.encoder:  # 逐层经过编码器块
            x = blk(x, self_mask=src_mask)  # 双向自注意力 + FFN；填充位置不被注意
        return self.enc_norm(x), src_mask  # 返回归一化后的编码结果，以及源掩码（解码器交叉注意力要用）

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor):  # 解码：根据已生成的目标序列 + 编码记忆预测下一个词
        tgt_mask = tgt != self.pad_id  # 目标序列的有效位置掩码
        y = self.drop(self.pos(self.tgt_emb(tgt) * math.sqrt(self.dim)))  # 目标词嵌入 × √dim → 加位置编码 → dropout
        for blk in self.decoder:  # 逐层经过解码器块
            y = blk(y, memory=memory, self_mask=tgt_mask, memory_mask=src_mask, is_causal=True)  # 因果自注意力（不能偷看未来）+ 交叉注意力（看源句）+ FFN
        return self.lm_head(self.dec_norm(y))                             # 归一化后映射到词表，得到 (B, T, V) 的 logits

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:  # 训练时的前向传播（teacher forcing：一次性喂入整个目标序列）
        memory, src_mask = self.encode(src)  # 先编码源序列
        return self.decode(tgt, memory, src_mask)  # 再解码，返回每个位置的下一个词预测

    @torch.no_grad()  # 推理时不计算梯度，省显存、更快
    def greedy_decode(self, src: torch.Tensor, bos_id: int, eos_id: int, max_new_tokens: int = 50):  # 贪心解码：每步选概率最大的词
        """贪心解码 / Greedy autoregressive generation."""
        self.eval()  # 切换到评估模式（关闭 dropout）
        memory, src_mask = self.encode(src)  # 源序列只需编码一次
        out = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)  # 输出序列初始化为 [BOS]，形状 (B, 1)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)  # 记录每个样本是否已生成 EOS
        for _ in range(max_new_tokens):  # 最多生成 max_new_tokens 个词
            nxt = self.decode(out, memory, src_mask)[:, -1].argmax(-1)  # 取最后一个位置的 logits，选得分最高的词作为下一个词
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)  # 已结束的样本后面只补填充符
            out = torch.cat([out, nxt[:, None]], dim=1)  # 把新词拼接到输出序列末尾
            done |= nxt == eos_id  # 生成了 EOS 的样本标记为结束
            if done.all():  # 如果所有样本都结束了
                break  # 提前停止
        return out  # 返回生成的序列（以 BOS 开头）


@register_model("transformer_encoder", domain="nlp", description="Encoder-only Transformer classifier (BERT-style)")  # 注册仅编码器分类模型
class TransformerEncoderClassifier(nn.Module):  # 仅编码器 + [CLS] 分类头（类似 BERT）
    """仅编码器 + [CLS] 分类头 / Encoder-only with a [CLS] classification head."""

    def __init__(self, vocab_size: int = 1000, num_classes: int = 2, dim: int = 256, depth: int = 4,  # 词表大小、类别数、隐藏维度、层数
                 num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1, max_len: int = 512,  # 头数、FFN 倍数、丢弃率、最大长度
                 pad_id: int = 0):  # 填充符 id
        super().__init__()  # 父类初始化
        self.pad_id = pad_id  # 保存填充符 id
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)  # 词嵌入表
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))  # 可学习的 [CLS] 向量，放在序列最前面，用来汇总整句信息
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, dim))  # 可学习的位置编码（+1 是给 [CLS] 留的位置）
        nn.init.trunc_normal_(self.pos, std=0.02)  # 用截断正态分布初始化位置编码
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])  # depth 层编码器块
        self.norm = nn.LayerNorm(dim)  # 最后的层归一化
        self.head = nn.Linear(dim, num_classes)  # 分类头：dim -> 类别数

    def forward(self, tokens):  # 前向传播，tokens 形状 (B, N)
        B, N = tokens.shape  # 批大小 B，序列长度 N
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=tokens.device), tokens != self.pad_id], 1)  # 掩码：[CLS] 永远有效 + 非填充 token 有效
        x = torch.cat([self.cls.expand(B, -1, -1), self.emb(tokens)], 1) + self.pos[:, : N + 1]  # 在最前面拼上 [CLS]，再加位置编码
        for blk in self.blocks:  # 逐层编码
            x = blk(x, self_mask=mask)  # 双向自注意力 + FFN
        return self.head(self.norm(x[:, 0]))  # 只取 [CLS] 位置的输出做分类
