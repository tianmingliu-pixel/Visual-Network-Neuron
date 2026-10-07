"""
MLP —— 多层感知机（全连接神经网络）
==================================
核心思想：每一层都是 “线性变换 + 非线性激活”：
    z = W·a + b      （每个神经元 = 上一层所有神经元的加权和 + 偏置）
    a = σ(z)          （激活函数让网络能拟合弯曲的边界）
最后一层输出每个类别的得分（logits），softmax 后得到概率。
这是理解所有深度网络的起点：Transformer 里的 FFN、分类头本质上都是 MLP。
"""
from __future__ import annotations  # 延迟解析类型注解

from typing import Sequence  # 类型提示：序列

import torch.nn as nn  # 神经网络模块

from ..registry import register_model  # 模型注册装饰器

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "sigmoid": nn.Sigmoid}  # 可选的激活函数


@register_model("mlp", domain="basic", description="多层感知机（全连接网络）/ Multi-layer perceptron")  # 注册为 "mlp"
class MLP(nn.Module):  # 定义多层感知机
    def __init__(self, in_dim: int = 7, hidden: Sequence[int] = (16, 12), out_dim: int = 3, act: str = "tanh"):  # 输入维度、各隐藏层宽度、输出维度、激活函数
        super().__init__()  # 父类初始化
        dims = [in_dim, *hidden, out_dim]  # 每层神经元个数，例如 [7, 16, 12, 3]
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])  # 相邻两层之间一个全连接层：权重矩阵 W 形状 (后一层, 前一层)
        self.act = ACTIVATIONS[act]()  # 隐藏层使用的激活函数

    def forward(self, x, return_activations: bool = False):  # 前向传播；可同时返回每层激活值（可视化用）
        acts = [x]  # 第 0 层就是输入特征
        for i, layer in enumerate(self.layers):  # 逐层计算
            x = layer(x)  # 线性变换 z = W·a + b
            if i < len(self.layers) - 1:  # 除最后一层外
                x = self.act(x)  # 非线性激活 a = σ(z)
            acts.append(x)  # 记录这一层的输出
        return (x, acts) if return_activations else x  # 最后一层不激活，直接输出 logits
