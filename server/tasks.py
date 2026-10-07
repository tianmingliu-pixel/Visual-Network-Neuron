"""
训练任务 / Training tasks shown in the UI
=========================================
每个任务 = 一个网络 + 一份合成数据 + 一个损失函数 + 一个“预览”（给前端看学到了什么）。
本文件也会被逐行追踪，所以每一行都带中文注释，前端讲解面板会直接显示这些注释。
"""
from __future__ import annotations  # 延迟解析类型注解

import torch  # PyTorch 主库
import torch.nn.functional as F  # 函数式接口（交叉熵、均方误差等）

import neurocore as nc  # 我们自己的网络库（Transformer / ViT / U-Net / DiT）
from neurocore.diffusion import GaussianDiffusion  # DDPM 扩散过程


def to_pixels(img: torch.Tensor) -> list:  # 把 [-1,1] 的单通道图像转成 0~255 的整数列表，发给前端画图
    x = img.detach().float().clamp(-1, 1)  # 脱离计算图、转 float、裁剪到 [-1,1]
    return ((x + 1) * 127.5).round().to(torch.uint8).flatten().tolist()  # 线性映射到 0~255 并展平成列表


class BaseTask:  # 所有任务的基类：封装“一次训练迭代”的通用流程
    id = "base"  # 任务编号（前端下拉框用）
    title = ""  # 中文标题
    description = ""  # 一句话说明
    files = ["server/tasks.py"]  # 代码面板默认展示的文件
    defaults = {"steps": 500, "lr": 1e-3, "batch_size": 32}  # 默认训练参数
    preview_every = 1.0  # 预览/结构图的刷新间隔（秒）
    DATA_TEXT = ""  # 讲解：喂给网络的是什么数据
    FORWARD_TEXT = ""  # 讲解：前向传播经过哪些层
    LOSS_TEXT = ""  # 讲解：损失函数的含义
    LOSS_FORMULA = ""  # 损失函数公式
    DATA_NEEDLE = ""  # compute_loss 中“取数据”那一行的关键字（用于定位代码行）
    FORWARD_NEEDLE = "self.model("  # compute_loss 中“前向传播”那一行的关键字
    LOSS_NEEDLE = "F.cross_entropy("  # 计算损失那一行的关键字
    IS_DIFFUSION = False  # 是否为扩散任务（损失写在 ddpm.py 里）

    def __init__(self, device: torch.device, lr: float, batch_size: int):  # 构造任务：设备、学习率、批大小
        self.device, self.batch_size = device, batch_size  # 保存设备与批大小
        self.model = self.build().to(device)  # 构建网络并搬到 CPU/GPU
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)  # AdamW 优化器：负责根据梯度更新参数

    def build(self):  # 子类实现：返回网络
        raise NotImplementedError  # 基类不实现

    def compute_loss(self):  # 子类实现：生成一批数据 → 前向传播 → 返回 (损失, 额外指标)
        raise NotImplementedError  # 基类不实现

    def train_step(self) -> dict:  # 一次完整的训练迭代（前向 → 反向 → 更新）
        self.model.train()  # 切换到训练模式（启用 dropout 等）
        loss, info = self.compute_loss()  # 前向传播：数据流过网络，算出损失
        self.opt.zero_grad(set_to_none=True)  # 清空上一步残留的梯度
        loss.backward()  # 反向传播：autograd 在 C++ 中自动求出每个参数的梯度
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪：总范数超过 1 就等比缩小，防止训练爆炸
        self.opt.step()  # 参数更新：参数 ← 参数 − 学习率 × (Adam 调整后的梯度)
        return {"loss": loss.item(), "grad_norm": float(grad_norm), **info}  # 返回本步指标给前端画曲线

    @torch.no_grad()  # 预览时不需要梯度
    def preview(self):  # 子类实现：返回给前端展示的“当前学习效果”
        return None  # 默认没有预览

    def graph(self):  # 网络结构图数据：默认用前向钩子自动抓取每一层的输出
        from .graph import capture_arch  # 延迟导入，避免循环引用
        return capture_arch(self)  # 返回各层形状、代表性神经元数值、权重/梯度范数


# ---------------------------------------------------------------------------
class TransformerReverseTask(BaseTask):  # Transformer：学习把序列倒过来（例如 3 7 5 → 5 7 3）
    id = "transformer"  # 任务编号
    title = "Transformer · 序列反转"  # 标题
    description = "编码器读入数字序列，解码器逐个生成它的倒序；观察注意力如何学会“对齐”位置。"  # 说明
    files = ["server/tasks.py", "neurocore/models/transformer.py", "neurocore/layers/attention.py",  # 相关代码文件（代码面板的标签页）
             "neurocore/layers/embeddings.py"]  # 相关代码文件
    defaults = {"steps": 600, "lr": 1e-3, "batch_size": 64}  # 默认参数
    V, PAD, BOS, EOS, L = 16, 0, 1, 2, 6  # 词表大小、填充符、开始符、结束符、序列长度
    DATA_TEXT = "随机生成 6 个数字（3~15）组成的序列作为输入；标签是它的倒序，并在开头/结尾加上 BOS/EOS 特殊符号。"  # 数据说明
    FORWARD_TEXT = "数字 → 词嵌入(64维)+位置编码 → 2 层编码器(自注意力) → 2 层解码器(因果自注意力+交叉注意力) → 线性层得到 16 个词的得分。"  # 前向说明
    LOSS_TEXT = "逐个位置的交叉熵：模型对“正确下一个数字”给出的概率越低，损失越大；对 B×T 个位置取平均。"  # 损失说明
    LOSS_FORMULA = "L = −(1/BT) Σ log softmax(z_t)[y_t]"  # 损失公式
    DATA_NEEDLE = "self.batch("  # 取数据的那一行

    def build(self):  # 构建一个小型编码器-解码器 Transformer
        return nc.build_model("transformer", src_vocab_size=self.V, tgt_vocab_size=self.V, dim=64,  # 构建一个小型编码器-解码器 Transformer
                              depth=2, num_heads=4, dropout=0.0)  # 64 维、2 层、4 头

    def batch(self, n):  # 生成一批训练数据
        body = torch.randint(3, self.V, (n, self.L), device=self.device)  # 随机数字序列（3 以上，避开特殊符号）
        rev = body.flip(1)  # 目标：倒序
        bos = torch.full((n, 1), self.BOS, device=self.device)  # 开始符列
        eos = torch.full((n, 1), self.EOS, device=self.device)  # 结束符列
        return body, torch.cat([bos, rev], 1), torch.cat([rev, eos], 1)  # 源序列、解码器输入（BOS+目标）、解码器标签（目标+EOS）

    def compute_loss(self):  # 前向 + 损失
        src, tgt_in, tgt_out = self.batch(self.batch_size)  # 取一批数据
        logits = self.model(src, tgt_in)  # 前向传播：得到每个位置对下一个词的打分 (B, T, V)
        loss = F.cross_entropy(logits.reshape(-1, self.V), tgt_out.reshape(-1))  # 交叉熵：预测分布与真实下一个词之间的差距
        acc = (logits.argmax(-1) == tgt_out).float().mean().item()  # 逐 token 准确率
        return loss, {"acc": acc}  # 返回损失与准确率

    @torch.no_grad()  # 不求梯度
    def preview(self):  # 展示：贪心解码几个样例
        src, _, tgt_out = self.batch(4)  # 取 4 个样例
        gen = self.model.greedy_decode(src, self.BOS, self.EOS, max_new_tokens=self.L + 1)  # 模型自己逐词生成
        rows = [{"src": src[i].tolist(), "target": tgt_out[i, :self.L].tolist(),  # 源序列与正确答案
                 "pred": gen[i, 1:self.L + 1].tolist()} for i in range(4)]  # 模型输出（去掉 BOS）
        return {"kind": "seq", "rows": rows}  # 预览类型：序列


# ---------------------------------------------------------------------------
def make_shapes(n, size, device):  # 合成 4 类图形：横条、竖条、方块、十字
    y = torch.randint(0, 4, (n,), device=device)  # 随机类别
    x = torch.randn(n, 1, size, size, device=device) * 0.3 - 1.0  # 暗背景 + 噪声
    for i in range(n):  # 逐张画图
        r, c = torch.randint(4, size - 12, (2,)).tolist()  # 随机位置
        k = int(y[i])  # 当前类别
        if k == 0: x[i, 0, r + 3:r + 6, c:c + 10] = 1.0  # 类别 0：横条
        elif k == 1: x[i, 0, r:r + 10, c + 3:c + 6] = 1.0  # 类别 1：竖条
        elif k == 2: x[i, 0, r + 1:r + 9, c + 1:c + 9] = 1.0  # 类别 2：方块
        else: x[i, 0, r + 4:r + 6, c:c + 10] = 1.0; x[i, 0, r:r + 10, c + 4:c + 6] = 1.0  # 类别 3：十字
    return x, y  # 返回图像与标签


class ViTShapesTask(BaseTask):  # ViT：给图形分类
    id = "vit"  # 任务编号
    title = "ViT · 图形分类"  # 标题
    description = "把 32×32 图像切成 4×4 的小块当作“词”，用 Transformer 判断是横条/竖条/方块/十字。"  # 说明
    files = ["server/tasks.py", "neurocore/models/vit.py", "neurocore/layers/attention.py",  # 相关代码文件（代码面板的标签页）
             "neurocore/layers/embeddings.py"]  # 相关代码文件
    defaults = {"steps": 400, "lr": 1e-3, "batch_size": 64}  # 默认参数
    CLASSES = ["横条", "竖条", "方块", "十字"]  # 类别名
    DATA_TEXT = "合成 32×32 灰度图：暗背景+噪声上随机位置画一个横条/竖条/方块/十字；标签是图形类别（0~3）。"  # 数据说明
    FORWARD_TEXT = "图像切成 64 个 4×4 小块 → 每块投影成 64 维 token，前面加 [CLS] 和位置编码 → 3 层 Transformer → 取 [CLS] 经线性层得 4 类得分。"  # 前向说明
    LOSS_TEXT = "交叉熵：softmax 把 4 个得分变成概率，损失 = −log(正确类别的概率)，对批内取平均。"  # 损失说明
    LOSS_FORMULA = "L = −(1/B) Σ log softmax(z)[y]"  # 损失公式
    DATA_NEEDLE = "make_shapes("  # 取数据的那一行

    def build(self):  # 构建小型 ViT
        return nc.build_model("vit", img_size=32, patch_size=4, in_channels=1, num_classes=4,  # 构建小型 ViT：1 通道 32×32 输入、4 类输出
                              dim=64, depth=3, num_heads=4)  # 64 个 patch、64 维、3 层

    def compute_loss(self):  # 前向 + 损失
        x, y = make_shapes(self.batch_size, 32, self.device)  # 生成一批图像和标签
        logits = self.model(x)  # 前向传播：每张图对 4 个类别的打分
        loss = F.cross_entropy(logits, y)  # 交叉熵损失
        acc = (logits.argmax(-1) == y).float().mean().item()  # 准确率
        return loss, {"acc": acc}  # 返回损失与准确率

    @torch.no_grad()  # 不求梯度
    def preview(self):  # 展示：8 张图及预测
        self.model.eval()  # 评估模式
        x, y = make_shapes(8, 32, self.device)  # 8 张新图
        p = self.model(x).softmax(-1)  # 预测概率
        items = [{"w": 32, "h": 32, "px": to_pixels(x[i, 0]), "label": self.CLASSES[int(y[i])],  # 图像像素与真实类别
                  "pred": self.CLASSES[int(p[i].argmax())], "conf": float(p[i].max())} for i in range(8)]  # 预测类别与置信度
        return {"kind": "classify", "items": items}  # 预览类型：分类


# ---------------------------------------------------------------------------
def make_blobs(n, size, device, y=None):  # 扩散用的合成图：类别 0 = 方块，类别 1 = 横条
    y = torch.randint(0, 2, (n,), device=device) if y is None else y  # 随机或指定类别
    x = -torch.ones(n, 1, size, size, device=device)  # 黑色背景（值为 -1）
    for i in range(n):  # 逐张画
        r, c = torch.randint(1, size - 7, (2,)).tolist()  # 随机位置
        if int(y[i]) == 0: x[i, 0, r:r + 6, c:c + 6] = 1.0  # 方块
        else: x[i, 0, r + 2:r + 4, c - 1:c + 7] = 1.0  # 横条
    return x, y  # 返回图像与类别


class DiffusionTask(BaseTask):  # 扩散任务基类：U-Net 与 DiT 共用
    size, T = 16, 100  # 图像边长、扩散总步数
    defaults = {"steps": 800, "lr": 2e-3, "batch_size": 32}  # 默认参数
    conditional = False  # 是否使用类别条件
    IS_DIFFUSION = True  # 损失在 ddpm.py 的 training_loss 中计算
    DATA_TEXT = "合成 16×16 图（方块或横条，像素 −1/1）作为干净图 x₀；每张随机抽一个时间步 t 和高斯噪声 ε，得到加噪图 xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε。"  # 数据说明
    LOSS_TEXT = "网络看到加噪图 xₜ 和 t，去猜加进去的噪声 ε；损失是预测噪声与真实噪声的均方误差。学会去噪后就能从纯噪声一步步生成图像。"  # 损失说明
    LOSS_FORMULA = "L = ‖ε − ε_θ(xₜ, t)‖²  (MSE)"  # 损失公式
    DATA_NEEDLE = "make_blobs("  # 取数据的那一行
    FORWARD_NEEDLE = "training_loss("  # 前向发生在 training_loss 内部
    LOSS_NEEDLE = "F.mse_loss("  # 损失在 ddpm.py 的 training_loss 中

    def __init__(self, device, lr, batch_size):  # 构造
        self.diffusion = GaussianDiffusion(timesteps=self.T).to(device)  # 创建扩散过程（噪声日程表）
        super().__init__(device, lr, batch_size)  # 调用基类构造（建模型、优化器）

    def compute_loss(self):  # 前向 + 损失
        x0, y = make_blobs(self.batch_size, self.size, self.device)  # 干净图像 x0 与类别
        loss = self.diffusion.training_loss(self.model, x0, y if self.conditional else None)  # 随机加噪 → 网络预测噪声 → MSE
        return loss, {}  # 无额外指标

    @torch.no_grad()  # 不求梯度
    def preview(self):  # 展示：不同噪声强度下，网络“看穿噪声”还原出的图
        self.model.eval()  # 评估模式
        ts = [10, 35, 60, 90]  # 四个时间步：越大噪声越强
        y = torch.tensor([0, 1, 0, 1], device=self.device)  # 类别
        x0, _ = make_blobs(4, self.size, self.device, y)  # 干净图
        t = torch.tensor(ts, device=self.device)  # 时间步张量
        noise = torch.randn_like(x0)  # 随机噪声
        xt = self.diffusion.q_sample(x0, t, noise)  # 加噪后的图 x_t
        eps = self.model(xt, t, y if self.conditional else None)[:, :1]  # 网络预测的噪声
        a = self.diffusion.sqrt_ac[t].view(-1, 1, 1, 1)  # √ᾱ_t
        b = self.diffusion.sqrt_1m_ac[t].view(-1, 1, 1, 1)  # √(1-ᾱ_t)
        x0_hat = (xt - b * eps) / a  # 由 x_t 与预测噪声反推出干净图 x̂0
        rows = [{"t": ts[i], "w": self.size, "h": self.size, "clean": to_pixels(x0[i, 0]),  # 整理成前端需要的格式：原图 / 加噪图 / 网络还原图
                 "noisy": to_pixels(xt[i, 0]), "pred": to_pixels(x0_hat[i, 0])} for i in range(4)]  # 原图 / 加噪图 / 还原图
        return {"kind": "denoise", "rows": rows}  # 预览类型：去噪


class UNetDiffusionTask(DiffusionTask):  # U-Net 扩散
    id = "unet"  # 任务编号
    title = "U-Net · 扩散去噪"  # 标题
    description = "给方块/横条图加不同程度的噪声，U-Net 学习预测噪声；观察还原图从模糊到清晰。"  # 说明
    FORWARD_TEXT = "加噪图 → 输入卷积 → 下采样路径(残差块,分辨率减半) → 瓶颈(残差+注意力) → 上采样路径(拼接跳跃连接) → 输出卷积，得到与输入同尺寸的噪声预测；时间步 t 编码后注入每个残差块。"  # 前向说明
    files = ["server/tasks.py", "neurocore/models/unet.py", "neurocore/diffusion/ddpm.py",  # 相关代码文件（代码面板的标签页）
             "neurocore/layers/attention.py", "neurocore/layers/embeddings.py"]  # 相关代码文件

    def build(self):  # 构建小型 U-Net
        return nc.build_model("unet", in_channels=1, base_channels=32, channel_mults=(1, 2),  # 构建小型 U-Net：1 通道输入、基础通道 32
                              num_res_blocks=1, attn_levels=(1,))  # 2 级、每级 1 个残差块、第 2 级带注意力


class DiTDiffusionTask(DiffusionTask):  # DiT 扩散（带类别条件）
    id = "dit"  # 任务编号
    title = "DiT · 条件扩散"  # 标题
    description = "Transformer 作为去噪网络，并通过 adaLN-Zero 接收“时间步 + 类别”条件。"  # 说明
    FORWARD_TEXT = "加噪图切成 2×2 小块 → token + 2D 位置编码 → 3 个 DiT 块（自注意力+FFN，由“时间步+类别”经 adaLN-Zero 调制）→ 线性层还原成图像形状的噪声预测。"  # 前向说明
    files = ["server/tasks.py", "neurocore/models/dit.py", "neurocore/diffusion/ddpm.py",  # 相关代码文件（代码面板的标签页）
             "neurocore/layers/attention.py", "neurocore/layers/embeddings.py"]  # 相关代码文件
    conditional = True  # 使用类别条件

    def build(self):  # 构建小型 DiT
        return nc.build_model("dit", img_size=self.size, patch_size=2, in_channels=1, dim=64,  # 构建小型 DiT：16×16 输入、patch=2、带类别条件
                              depth=3, num_heads=4, num_classes=2)  # 64 个 patch、64 维、3 层、2 个类别

# ---------------------------------------------------------------------------
SPIRAL_FEATURES = ["x", "y", "x²", "y²", "x·y", "sin 3x", "sin 3y"]  # 输入层的 7 个特征名


def make_spiral(n_per_class, generator):  # 生成三臂螺旋数据：二维平面上三条交错的螺旋线
    pts, labels = [], []  # 坐标与标签
    for k in range(3):  # 三个类别
        r = torch.linspace(0.05, 1.0, n_per_class)  # 半径从中心向外
        theta = r * 4.0 + k * 2.0944 + torch.randn(n_per_class, generator=generator) * 0.2  # 角度随半径增加；每类相差 120°；加一点噪声
        pts.append(torch.stack([r * torch.cos(theta), r * torch.sin(theta)], 1))  # 极坐标转直角坐标
        labels.append(torch.full((n_per_class,), k, dtype=torch.long))  # 类别标签
    return torch.cat(pts), torch.cat(labels)  # 合并三类


def spiral_features(xy):  # 特征工程：由 (x, y) 构造 7 个输入特征
    x, y = xy[:, 0], xy[:, 1]  # 拆出横纵坐标
    return torch.stack([x, y, x * x, y * y, x * y, torch.sin(3 * x), torch.sin(3 * y)], 1)  # 拼成 (N, 7) 的特征矩阵


class MLPSpiralTask(BaseTask):  # 最基础的全连接网络：可以画出每一个神经元和每一条权重
    id = "mlp"  # 任务编号
    title = "MLP · 螺旋分类（神经元级）"  # 标题
    description = "平面上三条交错的螺旋，MLP 根据 7 个坐标特征判断点属于哪一条；可以看到每个神经元、每条权重和决策边界如何随训练变化。"  # 说明
    files = ["server/tasks.py", "neurocore/models/mlp.py"]  # 相关代码文件
    defaults = {"steps": 3000, "lr": 0.01, "batch_size": 64}  # 默认参数
    preview_every = 0.25  # 结构图刷新更快，动画更连贯
    CLASSES = ["红臂", "绿臂", "蓝臂"]  # 类别名
    DATA_TEXT = "三条螺旋上的 600 个点（每条 200 个）。每个点的坐标 (x, y) 被扩展成 7 个特征 [x, y, x², y², xy, sin3x, sin3y]，标签是它属于哪条螺旋。"  # 数据说明
    FORWARD_TEXT = "7 个输入特征 → 隐藏层1：16 个神经元，z=W¹x+b¹，a=tanh(z) → 隐藏层2：12 个神经元 → 输出层：3 个得分，softmax 后是 3 类概率。"  # 前向说明
    LOSS_TEXT = "交叉熵：softmax 把 3 个得分变成概率，损失 = −log(正确螺旋的概率)；越自信地答对损失越小。"  # 损失说明
    LOSS_FORMULA = "L = −(1/B) Σ log softmax(W³a² + b³)[y]"  # 损失公式
    DATA_NEEDLE = "spiral_features("  # 取数据的那一行

    def __init__(self, device, lr, batch_size):  # 构造：先生成固定的数据集
        g = torch.Generator().manual_seed(0)  # 固定随机种子，数据集每次相同
        self.xy, self.y = make_spiral(200, g)  # 训练集 600 个点
        self.val_xy, self.val_y = make_spiral(60, g)  # 验证集 180 个点
        self.sample_i = 0  # 结构图轮流展示的样本编号
        super().__init__(device, lr, batch_size)  # 基类构造（建模型、优化器）

    def build(self):  # 构建 MLP：7 → 16 → 12 → 3
        return nc.build_model("mlp", in_dim=7, hidden=(16, 12), out_dim=3, act="tanh")  # tanh 激活

    def compute_loss(self):  # 前向 + 损失
        idx = torch.randint(0, len(self.y), (self.batch_size,))  # 随机抽一个小批量的下标
        x = spiral_features(self.xy[idx]).to(self.device)  # 取出这些点并转成 7 维特征
        y = self.y[idx].to(self.device)  # 对应标签
        logits = self.model(x)  # 前向传播：得到每个点对 3 类的得分
        loss = F.cross_entropy(logits, y)  # 交叉熵损失
        acc = (logits.argmax(-1) == y).float().mean().item()  # 准确率
        return loss, {"acc": acc}  # 返回损失与准确率

    @torch.no_grad()  # 不求梯度
    def preview(self):  # 展示：决策边界 + 验证集散点
        self.model.eval()  # 评估模式
        n = 48  # 网格分辨率
        g = torch.linspace(-1.1, 1.1, n)  # 网格坐标
        gy, gx = torch.meshgrid(g, g, indexing="ij")  # 二维网格
        grid = torch.stack([gx.flatten(), -gy.flatten()], 1)  # 每个格点的 (x, y)，y 轴向上
        p = self.model(spiral_features(grid).to(self.device)).softmax(-1).cpu()  # 每个格点属于 3 类的概率
        val_p = self.model(spiral_features(self.val_xy).to(self.device)).argmax(-1).cpu()  # 验证集预测
        acc = float((val_p == self.val_y).float().mean())  # 验证准确率
        pts = [[round(float(a), 3), round(float(b), 3), int(c)] for (a, b), c in zip(self.val_xy, self.val_y)]  # 验证点坐标与标签
        return {"kind": "boundary", "n": n, "cls": p.argmax(-1).tolist(), "conf": [round(float(v), 2) for v in p.max(-1).values],  # 每格的预测类别与置信度
                "points": pts, "val_acc": acc, "classes": self.CLASSES}  # 散点、验证准确率、类别名

    @torch.no_grad()  # 不求梯度
    def graph(self):  # 神经元级结构图：每个神经元的激活值、每条连线的权重与梯度
        self.model.eval()  # 评估模式
        i = self.sample_i % len(self.val_y)  # 轮流选一个验证样本
        self.sample_i += 7  # 下次换一个
        xy = self.val_xy[i:i + 1]  # 该样本坐标
        logits, acts = self.model(spiral_features(xy).to(self.device), return_activations=True)  # 前向并取出每层激活
        r = lambda t: [round(float(v), 4) for v in t.flatten()]  # 小工具：张量 → 保留 4 位小数的列表
        layers = self.model.layers  # 所有全连接层
        return {"kind": "mlp", "sizes": [layers[0].in_features] + [l.out_features for l in layers],  # 每层神经元个数
                "in_names": SPIRAL_FEATURES, "out_names": self.CLASSES,  # 输入特征名与输出类别名
                "W": [[r(row) for row in l.weight] for l in layers],  # 每层权重矩阵（行 = 后一层神经元）
                "b": [r(l.bias) for l in layers],  # 每层偏置
                "G": [[r(row) for row in l.weight.grad] if l.weight.grad is not None else None for l in layers],  # 上一步的权重梯度
                "acts": [r(a[0]) for a in acts],  # 该样本在每层的激活值
                "probs": r(logits.softmax(-1)[0]), "label": int(self.val_y[i]), "point": r(xy[0]),  # 输出概率、真实标签、样本坐标
                "refs": mlp_layer_refs()}  # 各层对应的代码行


def mlp_layer_refs():  # 结构图上每一层 → 对应的代码行（点击可跳转）
    from .pipeline import ref_in  # 延迟导入
    from neurocore.models.mlp import MLP  # MLP 类
    return {"input": ref_in(spiral_features, "torch.stack"),  # 输入层：特征工程
            "linear": ref_in(MLP.forward, "x = layer(x)"),  # 隐藏层：线性变换
            "act": ref_in(MLP.forward, "self.act(x)"),  # 隐藏层：激活函数
            "output": ref_in(MLP.forward, "return (x, acts)"),  # 输出层：logits
            "loss": ref_in(MLPSpiralTask.compute_loss, "F.cross_entropy(")}  # 损失


# ---------------------------------------------------------------------------
class CustomTask(BaseTask):  # 你自己的数据：由“数据导入”模块动态生成（见 make_custom_task）
    id = "custom"  # 任务编号
    title = "自定义数据"  # 标题（导入数据后会改成数据集名称）
    files = ["server/tasks.py", "neurocore/models/mlp.py"]  # 相关代码文件
    defaults = {"steps": 2000, "lr": 3e-3, "batch_size": 64}  # 默认参数
    preview_every = 0.5  # 刷新间隔
    DATA_NEEDLE = "self.X[idx]"  # 取数据的那一行
    dataset = None  # 由 make_custom_task 填入的数据集（server/datasets.py 的 Dataset）

    def __init__(self, device, lr, batch_size):  # 构造：把 numpy 数据转成张量
        ds = self.dataset  # 导入好的数据集
        self.regression = ds.task_type == "regression"  # 是回归还是分类
        self.X = torch.from_numpy(ds.X)  # 全部输入（已标准化），形状 (N, D) 或 (N, C, H, W)
        self.Y = torch.from_numpy(ds.y)  # 全部标签：类别编号或标准化后的数值
        self.tr = torch.from_numpy(ds.train_idx).long()  # 训练集下标
        self.va = torch.from_numpy(ds.val_idx).long()  # 验证集下标
        self.sample_i = 0  # 结构图轮流展示的样本
        self.replay_w = None  # 难例回放权重（来自记忆库；None = 每个训练样本机会均等）
        self.augment = False  # 是否做图片数据增强（由“记忆与增强”选项打开）
        super().__init__(device, lr, batch_size)  # 基类构造（建模型、优化器）

    def build(self):  # 按数据类型自动选网络
        ds = self.dataset  # 数据集
        if ds.is_image:  # 图片 → ViT
            C, H = ds.X.shape[1], ds.X.shape[2]  # 通道数与边长
            return nc.build_model("vit", img_size=H, patch_size=4 if H <= 32 else 8, in_channels=C,  # 小图用 4×4 块，大图用 8×8 块
                                  num_classes=len(ds.class_names), dim=64, depth=3, num_heads=4)  # 64 维、3 层
        from .datasets import hidden_sizes  # 隐藏层宽度随输入维度自动调整
        h1, h2 = hidden_sizes(ds.X.shape[1])  # 例如 40 个特征 → 64、32
        out = 1 if self.regression else len(ds.class_names)  # 回归输出 1 个数，分类输出 K 个得分
        return nc.build_model("mlp", in_dim=ds.X.shape[1], hidden=(h1, h2), out_dim=out, act="tanh")  # 向量数据 → MLP

    def compute_loss(self):  # 前向 + 损失
        idx = self._draw()  # 从训练集抽一个小批量（开启难例回放时，以前错过的样本更容易被抽到）
        x = self.X[idx].to(self.device)  # 这一批的输入
        if self.augment and self.model.training:  # 图片数据增强：只在训练时做
            x = augment_images(x)  # 随机左右翻转 + 随机平移，小数据集不容易“死记硬背”
        y = self.Y[idx].to(self.device)  # 这一批的标签
        out = self.model(x)  # 前向传播
        if self.regression:  # 回归任务
            loss = F.mse_loss(out.squeeze(-1), y)  # 均方误差：预测值与真实值之差的平方的平均
            return loss, {}  # 回归没有准确率
        loss = F.cross_entropy(out, y)  # 分类：交叉熵
        acc = (out.argmax(-1) == y).float().mean().item()  # 准确率
        return loss, {"acc": acc}  # 返回损失与准确率

    def _draw(self):  # 抽样：均匀抽，或按记忆库给的权重抽
        if self.replay_w is None:  # 没有记忆：每个训练样本机会均等
            return self.tr[torch.randint(0, len(self.tr), (self.batch_size,))]  # 有放回均匀抽样
        pick = torch.multinomial(self.replay_w, self.batch_size, replacement=True)  # 按权重抽：难例权重大
        return self.tr[pick]  # 换算成样本编号

    def set_replay(self, weights):  # 记忆库更新难例权重（顺序与 self.tr 一致）
        self.replay_w = None if weights is None else torch.as_tensor(weights, dtype=torch.float)  # 转成张量

    @torch.no_grad()  # 不求梯度
    def memory_embed(self, max_n=400):  # 给记忆库：每个样本的特征向量 + 预测，以及验证集成绩
        ds = self.dataset  # 数据集
        ids = torch.cat([self.va, self.tr])[:max_n]  # 先放验证集，再放训练集
        n_va = min(len(self.va), max_n)  # 其中验证集的个数
        was = self.model.training  # 记下原来的模式
        self.model.eval()  # 评估模式
        feats, outs = [], []  # 特征向量与输出
        for k in range(0, len(ids), 256):  # 分块计算，省内存
            x = self.X[ids[k:k + 256]].to(self.device)  # 一块输入
            if ds.is_image:  # ViT：[CLS] 位置的特征就是整张图的向量
                f = self.model.forward_features(x)[:, 0]  # (B, dim)
                o = self.model.head(f)  # 分类得分
            else:  # MLP：最后一个隐藏层的激活就是特征向量
                o, acts = self.model(x, return_activations=True)  # 前向并取出各层激活
                f = acts[-2]  # 倒数第二层
            feats.append(f.float().cpu())  # 收集特征
            outs.append(o.float().cpu())  # 收集输出
        self.model.train(was)  # 恢复模式
        F_, O = torch.cat(feats), torch.cat(outs)  # 拼起来
        y = self.Y[ids]  # 真实标签
        split = ["val"] * n_va + ["train"] * (len(ids) - n_va)  # 每个样本属于哪一部分
        if self.regression:  # 回归：误差小于 0.5 个标准差算“对”
            p = O.squeeze(-1)  # 预测（标准化单位）
            err = (p - y.float()) ** 2  # 平方误差
            conv = lambda v: round(float(v) * ds.y_std + ds.y_mean, 4)  # 换回原始单位
            labels, preds = [conv(v) for v in y], [conv(v) for v in p]  # 原始单位的真实值与预测值
            correct = (err.sqrt() < 0.5).tolist()  # 是否足够接近
            conf, loss = [None] * len(ids), err.tolist()  # 回归没有置信度
            yv, pv = y[:n_va].float(), p[:n_va]  # 验证部分
            ss = float(((yv - yv.mean()) ** 2).sum()) or 1.0  # 总平方和
            metrics = {"val_acc": 1 - float(((yv - pv) ** 2).sum()) / ss, "val_loss": float(err[:n_va].mean())}  # R² 与 MSE
        else:  # 分类：softmax 概率、预测类别、交叉熵
            prob = O.softmax(-1)  # 概率
            pc = prob.argmax(-1)  # 预测类别
            labels = [ds.class_names[int(v)] for v in y]  # 真实类别名
            preds = [ds.class_names[int(v)] for v in pc]  # 预测类别名
            correct = (pc == y).tolist()  # 是否正确
            conf = prob.max(-1).values.tolist()  # 置信度
            loss = F.cross_entropy(O, y, reduction="none").tolist()  # 每个样本的损失
            metrics = {"val_acc": float((pc[:n_va] == y[:n_va]).float().mean()) if n_va else None,  # 验证准确率
                       "val_loss": float(torch.tensor(loss[:n_va]).mean()) if n_va else None}  # 验证损失
        embed = {"ids": ids.tolist(), "split": split, "vectors": F_.numpy(), "labels": labels, "preds": preds,  # 交给记忆库
                 "conf": conf, "correct": correct, "loss": loss}  # 交给记忆库
        return embed, metrics  # 返回

    def memory_samples(self):  # 样本元数据：编号、划分、标签、来源文件
        ds = self.dataset  # 数据集
        files = ds.extra.get("files")  # 图片/音频的来源文件（表格是行号）
        va = set(self.va.tolist())  # 验证集编号
        rows = []  # 结果
        for i in range(len(ds.y)):  # 每个样本
            lab = ds.class_names[int(ds.y[i])] if ds.task_type == "classification" else round(float(ds.y[i]) * ds.y_std + ds.y_mean, 4)  # 标签
            src = files[i] if files else (f"第 {i + 1} 行" if ds.kind == "tabular" else f"索引 {i}")  # 来源
            rows.append((i, "val" if i in va else "train", lab, src, {"kind": ds.kind}))  # 一行元数据
        return rows  # 返回

    @torch.no_grad()  # 不求梯度
    def preview(self):  # 每次刷新：在验证集上评估（混淆矩阵 / 回归散点）
        from .evaluate import quick_eval  # 评估工具
        return quick_eval(self)  # 返回给前端

    def evaluate(self):  # “分析模型”按钮：完整评估 + 特征重要性
        from .evaluate import full_eval  # 评估工具
        return full_eval(self)  # 返回给前端

    @torch.no_grad()  # 不求梯度
    def graph(self):  # 结构图：图片走层流程图；向量数据画神经元级图（神经元太多时只画最重要的）
        if self.dataset.is_image:  # 图片任务
            return super().graph()  # 用通用的层流程图
        self.model.eval()  # 评估模式
        ds = self.dataset  # 数据集
        i = int(self.va[self.sample_i % len(self.va)])  # 轮流取一个验证样本
        self.sample_i += 1  # 下次换一个
        out, acts = self.model(self.X[i:i + 1].to(self.device), return_activations=True)  # 前向并取出每层激活
        L = self.model.layers  # 全连接层
        keep = [torch.arange(L[0].in_features)]  # 每层要画出的神经元下标，先放输入层
        if L[0].in_features > 18:  # 输入特征太多：只画第一层权重最大的 18 个
            keep[0] = L[0].weight.abs().sum(0).topk(18).indices.sort().values  # 按权重绝对值之和挑选
        for l in L[:-1]:  # 隐藏层
            keep.append(torch.arange(l.out_features) if l.out_features <= 16 else l.weight.abs().sum(1).topk(16).indices.sort().values)  # 超过 16 个就挑权重最大的
        n_out = L[-1].out_features  # 输出个数
        label = None if self.regression else int(self.Y[i])  # 真实类别
        if n_out <= 10:  # 类别不多：全部画出
            keep.append(torch.arange(n_out))  # 所有输出
        else:  # 类别太多：画概率最高的 9 个 + 真实类别
            top = out[0].topk(9).indices.tolist()  # 概率最高的类别
            keep.append(torch.tensor(sorted(set(top + [label]))))  # 加上真实类别
        r = lambda t: [round(float(v), 4) for v in t.flatten()]  # 小工具：张量 → 列表
        W = [L[k].weight[keep[k + 1]][:, keep[k]] for k in range(len(L))]  # 只保留要画的连线
        G = [L[k].weight.grad[keep[k + 1]][:, keep[k]] if L[k].weight.grad is not None else None for k in range(len(L))]  # 对应梯度
        probs = out[0].softmax(-1) if not self.regression else torch.tensor([0.5])  # 输出概率（回归用占位）
        names = ds.class_names if not self.regression else [ds.target_name]  # 输出名
        g = {"kind": "mlp", "sizes": [len(k) for k in keep], "true_sizes": [L[0].in_features] + [l.out_features for l in L],  # 画出的/真实的神经元个数
             "in_names": [ds.feature_names[j][:16] for j in keep[0].tolist()], "out_names": [names[j][:10] for j in keep[-1].tolist()],  # 特征名与输出名
             "W": [[r(row) for row in w] for w in W], "G": [[r(row) for row in g_] if g_ is not None else None for g_ in G],  # 权重与梯度
             "acts": [r(a[0][k]) for a, k in zip(acts, keep)], "probs": r(probs[keep[-1]] if not self.regression else probs),  # 激活与概率
             "label": keep[-1].tolist().index(label) if label is not None else -1, "point": [float(i), 0.0],  # 真实类别在画出列表中的位置
             "sample_index": i,  # 这是验证集里的第几条样本
             "refs": custom_refs()}  # 代码跳转
        if self.regression:  # 回归：附上预测值与真实值（原始单位）
            g["regression"] = {"pred": round(float(out[0, 0]) * ds.y_std + ds.y_mean, 4),  # 预测值
                               "target": round(float(self.Y[i]) * ds.y_std + ds.y_mean, 4)}  # 真实值
        return g  # 返回给前端


def augment_images(x):  # 小数据集的图片增强（训练时每一批都随机变一点）
    flip = torch.rand(x.size(0), device=x.device) < 0.5  # 一半的图片左右翻转
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)  # 翻转
    pad = max(1, x.size(-1) // 8)  # 平移幅度：边长的 1/8
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")  # 四周补边
    i, j = torch.randint(0, 2 * pad + 1, (2,)).tolist()  # 随机平移量
    return xp[..., i:i + x.size(-2), j:j + x.size(-1)]  # 裁回原大小


def custom_refs():  # 自定义任务结构图各层对应的代码行
    from .pipeline import ref_in  # 延迟导入
    from neurocore.models.mlp import MLP  # MLP 类
    return {"input": ref_in(CustomTask.compute_loss, "self.X[idx]"),  # 输入层：取一批数据
            "linear": ref_in(MLP.forward, "x = layer(x)"), "act": ref_in(MLP.forward, "self.act(x)"),  # 隐藏层
            "output": ref_in(MLP.forward, "return (x, acts)"), "loss": ref_in(CustomTask.compute_loss, "loss = F.")}  # 输出层与损失


def make_custom_task(ds):  # 根据导入的数据集生成一个任务类（每次导入都重新生成）
    import os  # 路径工具
    name = os.path.basename(os.path.normpath(ds.source))  # 数据集名称
    is_img = ds.is_image  # 是否图片
    kind_cn = {"tabular": "表格", "image": "图片", "audio": "音频", "array": "数组"}[ds.kind]  # 中文类型名
    reg = ds.task_type == "regression"  # 是否回归
    attrs = {"title": f"自定义 · {name[:24]}",  # 标题
             "description": f"{kind_cn}数据 {len(ds.y)} 条，{'回归预测「' + ds.target_name + '」' if reg else str(len(ds.class_names)) + ' 类分类'}；模型：{ds.describe_model()}",  # 说明
             "dataset": ds,  # 数据集本身
             "files": ["server/tasks.py", "neurocore/models/vit.py" if is_img else "neurocore/models/mlp.py",  # 代码面板文件
                       "neurocore/layers/attention.py"] if is_img else ["server/tasks.py", "neurocore/models/mlp.py"],  # 代码面板文件
             "DATA_TEXT": f"你导入的{kind_cn}数据：{len(ds.train_idx)} 条训练、{len(ds.val_idx)} 条验证。"  # 数据说明
                          + ("每张图统一缩放并归一化到 [-1,1]。" if is_img else f"共 {ds.X.shape[1]} 个特征（已按训练集均值/标准差标准化）。"),  # 数据说明
             "FORWARD_TEXT": ("图片切成小块 → token + 位置编码 → 3 层 Transformer → [CLS] → 线性层得到各类得分。" if is_img  # 前向说明
                              else f"{ds.X.shape[1]} 个特征 → 两个 tanh 隐藏层 → 输出层（{'1 个预测值' if reg else str(len(ds.class_names)) + ' 个类别得分'}）。"),  # 前向说明
             "LOSS_TEXT": ("均方误差：预测值与真实值之差的平方取平均（目标已标准化）。" if reg  # 损失说明
                           else "交叉熵：softmax 得到各类概率，损失 = −log(正确类别的概率)。"),  # 损失说明
             "LOSS_FORMULA": "L = (1/B) Σ (ŷ − y)²" if reg else "L = −(1/B) Σ log softmax(z)[y]",  # 损失公式
             "LOSS_NEEDLE": "F.mse_loss(" if reg else "F.cross_entropy("}  # 定位损失代码行
    return type("CustomTask", (CustomTask,), attrs)  # 动态创建子类


TASKS = {t.id: t for t in (MLPSpiralTask, TransformerReverseTask, ViTShapesTask, UNetDiffusionTask, DiTDiffusionTask)}  # 任务注册表
