"""
DDPM 高斯扩散过程 / Gaussian diffusion (DDPM)
============================================
前向加噪 q(x_t | x_0) = N( sqrt(ᾱ_t)·x_0 , (1-ᾱ_t)·I )
训练目标：网络 ε_θ(x_t, t, y) 预测所加噪声 ε，损失 = MSE(ε, ε_θ)
反向采样：从纯噪声 x_T 逐步去噪到 x_0（支持 classifier-free guidance）
Works with any denoiser with signature `model(x, t, y)` — both UNet and DiT.
"""
from __future__ import annotations  # 延迟解析类型注解

import math  # 数学库：cos、pi
from typing import Optional  # 类型提示

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口（MSE 损失）


def make_beta_schedule(kind: str, T: int) -> torch.Tensor:  # 生成噪声日程表 β_1..β_T（每一步加多少噪声）
    if kind == "linear":  # 线性日程（DDPM 原论文）
        scale = 1000 / T  # 步数不是 1000 时按比例缩放
        return torch.linspace(scale * 1e-4, scale * 0.02, T, dtype=torch.float64)  # β 从 1e-4 线性增长到 0.02
    if kind == "cosine":                                   # 余弦日程 Nichol & Dhariwal 2021
        s = 0.008  # 小偏移，避免 t=0 附近 β 过小
        steps = torch.arange(T + 1, dtype=torch.float64) / T  # 归一化时间 0..1
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2  # ᾱ(t) 的余弦形状
        return (1 - f[1:] / f[:-1]).clamp(max=0.999)  # 由相邻 ᾱ 的比值得到 β，并限制上限
    raise ValueError(kind)  # 未知日程类型


class GaussianDiffusion(nn.Module):  # 高斯扩散过程（本身没有可训练参数）
    def __init__(self, timesteps: int = 1000, schedule: str = "linear"):  # 总步数 T、日程类型
        super().__init__()  # 父类初始化
        self.T = timesteps  # 保存总步数
        betas = make_beta_schedule(schedule, timesteps)  # β_t
        alphas = 1.0 - betas  # α_t = 1 - β_t
        ac = torch.cumprod(alphas, 0)  # ᾱ_t = α_1·α_2·…·α_t（累计保留的信号比例）
        ac_prev = torch.cat([torch.ones(1, dtype=torch.float64), ac[:-1]])  # ᾱ_{t-1}（t=0 时为 1）
        reg = lambda n, v: self.register_buffer(n, v.float(), persistent=False)  # 小工具：注册为 float32 缓冲区
        reg("betas", betas)  # β_t
        reg("sqrt_ac", ac.sqrt())  # √ᾱ_t：x0 的系数
        reg("sqrt_1m_ac", (1 - ac).sqrt())  # √(1-ᾱ_t)：噪声的系数
        reg("sqrt_recip_alphas", (1.0 / alphas).sqrt())  # 1/√α_t：反向采样用
        reg("posterior_var", betas * (1 - ac_prev) / (1 - ac))  # 后验方差 β̃_t：反向每步要加的随机性

    @staticmethod  # 静态方法
    def _extract(a, t, x):  # 从系数表 a 中取出第 t 步的值，并调整形状以便和图像 x 相乘
        return a.gather(0, t).view(-1, *([1] * (x.dim() - 1)))  # (B,) → (B,1,1,1)

    def q_sample(self, x0, t, noise=None):  # 前向加噪：一步直接得到第 t 步的噪声图
        """前向加噪 / forward noising."""
        noise = torch.randn_like(x0) if noise is None else noise  # 没给噪声就随机采样 ε ~ N(0, I)
        return self._extract(self.sqrt_ac, t, x0) * x0 + self._extract(self.sqrt_1m_ac, t, x0) * noise  # x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε

    def training_loss(self, model, x0, y: Optional[torch.Tensor] = None):  # 训练损失（DDPM 的简化目标）
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)  # 为每张图随机挑一个时间步
        noise = torch.randn_like(x0)  # 随机噪声 ε
        pred = model(self.q_sample(x0, t, noise), t, y)  # 加噪后交给网络，让它猜加了什么噪声
        pred = pred[:, : x0.size(1)]                     # learn_sigma 时只取 ε 部分
        return F.mse_loss(pred, noise)  # 损失 = 预测噪声与真实噪声的均方误差

    def _eps(self, model, x, t, y, cfg_scale):  # 采样时预测噪声（可选 CFG）
        if y is None or cfg_scale == 1.0:  # 无条件或不做引导
            return model(x, t, y)[:, : x.size(1)]  # 直接预测
        null = torch.full_like(y, model.num_classes)     # 空类 / null class
        eps = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]))[:, : x.size(1)]  # 一次前向同时算“有条件”和“无条件”
        cond, uncond = eps.chunk(2)  # 拆成两半
        return uncond + cfg_scale * (cond - uncond)  # CFG：沿“条件方向”放大，生成更符合类别的图

    @torch.no_grad()  # 采样不需要梯度
    def sample(self, model, shape, y: Optional[torch.Tensor] = None, cfg_scale: float = 1.0,  # 模型、输出形状、类别、引导强度
               device=None, progress: bool = False):  # 设备、是否打印进度
        """祖先采样 / ancestral sampling  x_T -> x_0"""
        model.eval()  # 评估模式
        device = device or next(model.parameters()).device  # 默认用模型所在设备
        x = torch.randn(shape, device=device)  # 从纯噪声 x_T 开始
        steps = range(self.T - 1, -1, -1)  # 时间步倒序：T-1, …, 0
        for i in steps:  # 逐步去噪
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)  # 当前时间步
            eps = self._eps(model, x, t, y, cfg_scale)  # 网络预测噪声
            coef = self._extract(self.betas, t, x) / self._extract(self.sqrt_1m_ac, t, x)  # β_t / √(1-ᾱ_t)
            mean = self._extract(self.sqrt_recip_alphas, t, x) * (x - coef * eps)  # 反向均值 μ = (x_t - β_t/√(1-ᾱ_t)·ε) / √α_t
            if i > 0:  # 不是最后一步
                x = mean + self._extract(self.posterior_var, t, x).sqrt() * torch.randn_like(x)  # 加一点随机噪声得到 x_{t-1}
            else:  # 最后一步
                x = mean  # 直接取均值作为 x_0
            if progress and i % max(1, self.T // 10) == 0:  # 每 10% 打印一次进度
                print(f"  sampling step {self.T - i}/{self.T}")  # 打印
        return x  # 返回生成的图像
