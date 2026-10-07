# NeuroCore

可扩展的 PyTorch 神经网络核心库 · An extensible PyTorch neural-network core

第一阶段实现四个基础网络，并用 **注册表 + 扩展目录** 为以后任何 IT 领域的网络（GNN、音频、时序、推荐、安全、强化学习……）预留接口。
Phase 1 implements four foundational networks; a **registry + extensions folder** lets future networks from any IT domain plug in without touching the core.

| 模块 Module | 文件 File | 核心思想 Core idea |
|---|---|---|
| **Transformer** | `neurocore/models/transformer.py` | 全注意力编码器-解码器：`softmax(QKᵀ/√d)V`，因果掩码 + 交叉注意力 · attention-only encoder–decoder |
| **ViT** | `neurocore/models/vit.py` | 图像切 patch → token，[CLS] + 位置编码 → Transformer 编码器 · an image is a sequence of patches |
| **U-Net** | `neurocore/models/unet.py` | 下采样/上采样 + 同级跳跃连接；可选时间步/类别条件用于扩散 · down/up paths joined by skip connections |
| **DiT** | `neurocore/models/dit.py` | Transformer 作为扩散去噪器，adaLN-Zero 注入 t 与类别 · Transformer denoiser with adaLN-Zero conditioning |
| DDPM | `neurocore/diffusion/ddpm.py` | 前向加噪 / 预测噪声 ε / 祖先采样 + CFG，U-Net 与 DiT 通用 · shared by U-Net & DiT |
| 共享构件 Blocks | `neurocore/layers/` | 多头注意力、FFN、Pre-LN 块、正余弦/时间步/Patch 嵌入 |

## 快速开始 Quick start (Windows PowerShell)

```powershell
cd NeuroCore
# 一条命令：检查环境 -> 创建 .venv -> 按显卡自动装 PyTorch -> 测试 -> 演示
powershell -ExecutionPolicy Bypass -File .\deploy.ps1
```

分步执行 Step by step:

```powershell
.\scripts\check_env.ps1                 # 只检查，不安装 / check only (PASS/WARN/FAIL + fixes)
.\scripts\check_env.ps1 -ReportPath env_report.json
.\scripts\setup_env.ps1                 # 部署 / deploy (auto GPU detection)
.\scripts\setup_env.ps1 -Cuda cpu       # 强制 CPU / force CPU build
.\scripts\setup_env.ps1 -Cuda cu130 -TorchVersion 2.14.0 -Recreate
.\scripts\run.ps1 -Task demo            # 四个网络的玩具训练演示 / toy training demos
.\scripts\run.ps1 -Task demo -Model dit -Steps 300
.\scripts\run.ps1 -Task test            # pytest
.\scripts\run.ps1 -Task list            # 已注册模型 / registered models
.\scripts\run.ps1 -Task info            # PyTorch / GPU 状态
```

### `check_env.ps1` 检查项 / what is checked
64 位系统、PowerShell ≥ 5.1、Python 3.10–3.14（优先 `py` 启动器）、pip、venv、Git、NVIDIA 显卡 + 驱动支持的 CUDA 版本、内存、磁盘、Windows 长路径、执行策略、已安装的 PyTorch 是否能用 GPU。每个非 PASS 项都附带修复命令。

### PyTorch 安装源自动选择 / wheel selection
`setup_env.ps1` 读取 `nvidia-smi` 报告的驱动 CUDA 版本，按顺序尝试：

| 驱动 CUDA | 尝试顺序 Tried in order |
|---|---|
| ≥ 13.2 | cu132 → cu130 → cu126 → cpu |
| 13.0–13.1 | cu130 → cu126 → cpu |
| 12.6–12.9 | cu126 → cpu（注意：PyTorch 2.14 是最后一个提供 CUDA 12.x 的版本）|
| 无 NVIDIA / 更旧 | cpu |

**已装好 PyTorch？** 默认会自动找到本机已有的 PyTorch（例如全局 `pip` 装的 2.14.1），用 `--system-site-packages` 建 `.venv` 直接复用，无需重新下载；若它是 CPU 版而你有 NVIDIA 显卡，脚本会提示，可用 `-FreshTorch -Recreate` 单独安装 CUDA 版。

装完会实际调用 `torch.cuda.is_available()` 验证；某个源失败会自动退回下一个。所有输出记录在 `deploy.log`。macOS/Linux 上也可用 PowerShell 7 (`pwsh`) 运行同样的脚本。

## 训练可视化界面 / Visual training UI

```powershell
.\scripts\run.ps1 -Task ui        # 打开 http://127.0.0.1:8765（已附带构建好的前端，无需 Node.js）
```

- **神经网络结构图**：默认任务 **MLP · 螺旋分类** 画出每个神经元与每条权重（橙=正权重、蓝=负权重、粗细=|w|），节点颜色是当前样本在该层的激活值，
  青色脉冲 = 前向传播、品红脉冲 = 反向传播（沿梯度最大的连线），右侧是输出概率条（白线 = 真实类别），下方“学习效果”显示决策边界逐渐弯成螺旋。
  ViT / Transformer / U-Net / DiT 显示**层流程图**：按真实执行顺序每列一层，列中的点是抽样的代表性神经元，顶部品红条 = 该层梯度范数；悬停看形状/参数，点击跳到代码。
- **一次训练迭代的全过程**：① 数据 → ② 前向传播 → ③ 损失 → ④ 反向传播 → ⑤ 更新权重，每步给出含义、公式（交叉熵 / MSE / 链式法则 / AdamW）和对应代码行（点击跳转）；
  单步讲解回放时会自动高亮当前处在哪一步，结构图的脉冲方向也随之切换。
- **左侧**：实时 loss / 准确率 / 梯度范数曲线、迭代速度、"学习效果预览"（分类结果、序列反转、扩散去噪的 原图→加噪→还原）。
- **右侧**：点 **● 单步讲解**，后台把下一次迭代（前向 → 损失 → 反向 → 更新）逐行录下来，然后像慢动作一样回放：
  代码面板高亮当前行、右列显示该行的中文注释，下方显示这一行产生的张量（形状、均值±标准差、范围、分布条）和调用栈。
  空格 = 播放/暂停，← → = 单步；时间轴颜色代表文件、高度代表调用深度。
- **不卡的原因**：训练在后台线程；指标每 100ms 合并推送一次（≤10 次/秒，超量自动抽稀）；图表用 Canvas 按像素抽稀绘制；
  逐行追踪只在"单步讲解"那一步开启，录好后一次性发到浏览器本地回放，全速训练不受影响。
- **导出 ONNX**：界面右上角按钮，保存到 `exports/<任务>.onnx`（需 `pip install -r requirements-export.txt`）。

前端开发（React + Vite，需要 Node.js）：

```powershell
.\scripts\run.ps1 -Task ui-dev     # 后端 :8765 + Vite 热更新 :5173
.\scripts\run.ps1 -Task build-ui   # 重新构建 web\dist
```

## 数据导入与分析 / Bring your own data

界面顶部切换到 **数据导入与分析**，输入本地路径、点「上传文件」/「上传文件夹」，或把文件/文件夹直接拖进框里：

| 数据 | 格式 | 自动处理 |
|---|---|---|
| 表格 | CSV / TSV / TXT / JSON / JSONL / Excel(.xlsx，无需 pandas) / .xls* / Parquet* | 识别每列类型：数值（缺失补均值）、类别（独热）、文本（哈希词袋）、ID/常数列（丢弃）；默认最后一列为目标，可改选 |
| 图片 | 整个文件夹或 zip | 裁剪缩放到 16–64 像素，判断灰度/彩色 → ViT |
| 音频 | `.wav`（PCM / 浮点）；`.flac .ogg .mp3` 需 `pip install soundfile` | 提取 56 个声学特征 → MLP；元数据里的数值标签可做回归 |
| 数组 | `.npz` / `.npy`：`X`+`y`、`x_train/y_train/x_test/y_test`、文件夹里的 `X.npy`+`y.npy`，或单个二维数组（最后一列为目标） | 二维 → MLP；图像形状 → ViT |

**图片 / 音频的标签（元数据）从哪里来**，按优先级自动识别，识别结果会显示在界面上：

1. **元数据表**：文件夹里任意 CSV / Excel / JSON，有一列是文件名（`img001.jpg`、`images/img001.jpg` 或不带扩展名都行），
   其他列是标签，例如 `labels.csv`：`filename,breed,weight`。可在界面「标签列」里切换用哪一列。
2. **类别子文件夹**：`数据\猫\*.jpg`、`数据\狗\*.jpg`，也支持 `train\ val\ test\` 再分类别。
3. **文件名前缀**：`cat_001.jpg`、`dog.12.jpg` → `cat` / `dog`。

上传文件夹时会保留子文件夹结构，保存到 `data/uploads/<批次号>/`。

\* .xls / Parquet 需要 `pip install -r requirements-data.txt`。

> 页面顶部出现“后端是旧版本”的红色提示时：关闭正在运行后端的 PowerShell 窗口（或 Ctrl+C），重新运行 `.\scripts\run.ps1 -Task ui` 再刷新页面。

**分析内容**：样本数与训练/验证划分、目标分布（类别数量或数值直方图）、类别不均衡提示、每列类型/缺失/统计/分布、
特征与目标的相关程度（分类用相关比 η²，回归用 |皮尔逊相关|）、PCA 二维投影、样本预览（表格行 / 图片 / 每类平均频谱）。

**训练**：点「用这份数据训练」自动生成「自定义」任务（向量数据 → MLP，图片 → ViT；分类用交叉熵，回归用 MSE），
结构图、单步讲解、全过程五步都照常可用；输入特征太多时结构图只画权重最大的那些神经元。

**模型表现分析**：准确率 / R²·MAE·RMSE、混淆矩阵、每类精确率/召回率/F1、置换法特征重要性、最自信的错误或误差最大的样本。

所有数据只在本机处理（后端只监听 127.0.0.1）；上传的文件保存在 `data/uploads/`。

## 机器记忆库 / Neural memory store

训练页右下角「🧠 机器记忆库」。训练时每隔 N 步（默认本次步数的 1/40）把机器此刻的状态写进本地数据库
`data/memory/neuro_memory.db`（SQLite，Python 自带），权重文件在 `data/memory/ckpt/<运行>/latest.pt / best.pt`。

| 表 | 内容 |
|---|---|
| `runs` | 每次训练：任务、数据集、超参数、从哪个快照继续、最佳验证成绩 |
| `snapshots` | 每个快照：loss、训练/验证准确率、梯度范数、学习率、权重总变化量、检查点 |
| `layer_states` | 每层的权重范数、梯度范数、与上一快照相比的变化量 |
| `samples` | 样本元数据：来源文件、标签、训练/验证 |
| `embeddings` | 每个样本的特征向量（倒数第二层，float32）+ 预测 + 置信度 + 是否正确 |
| `sample_memory` | 跨多次训练累计的“错题本”：看过几次、错过几次、平滑损失 |

**下次训练怎么用这些记忆**（面板上的选项）：
- **起点 = 从最佳记忆继续 / 某个快照**：加载权重 + AdamW 动量 + 步数，接着学，不从零开始；
- **难例回放**：按错题本给训练样本加权抽样，以前错过的样本多练；
- **图片增强**：随机左右翻转 + 平移，样本少时减少“死记硬背”（训练 100%、验证低就是这个问题）；
- 自动保存**验证准确率最高**的快照为 best，过拟合后也能退回去。

面板里可以看：成长曲线（历次训练对比 + 记忆来源链）、特征向量空间（每个快照的样本向量 PCA 投影 + 分离度 + 原始记录）、
每层变化热力图、样本错题本、表结构和只读 SQL 控制台。

## 在线部署 / Deploy

网页放 **Vercel**、PyTorch 训练后端放 **Hugging Face Spaces**（Docker），推送到 GitHub 后两边自动更新。步骤见 [DEPLOY.md](DEPLOY.md)：

```powershell
.\scripts\publish_github.ps1 -User 你的GitHub用户名     # 推送到 github.com/你的用户名/Visual-Network-Neuron
```

## Python 用法 / Usage

```python
import torch, neurocore as nc
from neurocore.diffusion import GaussianDiffusion

vit = nc.build_model("vit_small", img_size=224, num_classes=10)
logits = vit(torch.randn(2, 3, 224, 224))

seq2seq = nc.build_model("transformer", src_vocab_size=8000, tgt_vocab_size=8000)

diffusion = GaussianDiffusion(timesteps=1000, schedule="cosine")
dit = nc.build_model("dit_s_2", img_size=32, in_channels=4, num_classes=1000)
loss = diffusion.training_loss(dit, torch.randn(8, 4, 32, 32), torch.randint(0, 1000, (8,)))
samples = diffusion.sample(dit, (4, 4, 32, 32), y=torch.tensor([1, 2, 3, 4]), cfg_scale=4.0)

unet = nc.build_model("unet", in_channels=3, base_channels=64)      # 扩散去噪 / diffusion
seg = nc.build_model("unet_seg", in_channels=3, num_classes=21)     # 分割 / segmentation
```

已注册 Registered: `transformer`, `transformer_encoder`, `vit`, `vit_tiny/small/base`, `unet`, `unet_seg`, `dit`, `dit_s_2/b_2/xl_2`.

注意力默认用显式矩阵运算写出以便学习；传 `use_sdpa=True` 切换到 PyTorch 融合内核（测试中验证两者数值一致）。
Attention is written out explicitly for clarity; pass `use_sdpa=True` for PyTorch's fused kernel (tests verify they match).

## 扩展到新领域 / Adding a new domain

1. 复制 `neurocore/extensions/_template.py` 为例如 `gnn.py`（去掉下划线）。
2. 用 `@register_model("my_net", domain="graph")` 注册。
3. `import neurocore` 时自动加载，`nc.build_model("my_net")` 即可使用；某个扩展出错只会警告，不影响核心。

预留领域标签 Domain tags: `nlp, vision, generative, graph, audio, multimodal, timeseries, recsys, security, rl, other`。

## 目录 / Layout

```
NeuroCore/
├─ deploy.ps1               一键 check -> setup -> demo
├─ scripts/
│  ├─ common.ps1            共享函数（查找 Python、读取 GPU、选 wheel 源）
│  ├─ check_env.ps1         环境检查
│  ├─ setup_env.ps1         部署（.venv + PyTorch + 依赖 + 测试）
│  ├─ run.ps1               运行 demo / test / list / info
│  └─ verify_torch.py       PyTorch/GPU 状态（JSON）
├─ neurocore/
│  ├─ registry.py           模型注册中心
│  ├─ layers/               注意力、嵌入
│  ├─ models/               transformer / vit / unet / dit
│  ├─ diffusion/            DDPM
│  └─ extensions/           未来领域模块（自动发现）
├─ server/                 可视化后端（Starlette + SSE）
│  ├─ tasks.py              四个训练任务（逐行中文注释，会被追踪讲解）
│  ├─ trainer.py            后台训练线程、暂停/继续/单步追踪
│  ├─ tracer.py             sys.settrace 逐行追踪器
│  ├─ hub.py                节流推送（≤10 次/秒）
│  └─ app.py                HTTP 接口 + 提供前端页面
├─ web/                     React + Vite 前端（web/dist 为构建结果）
├─ tests/                   test_models.py · test_server.py
└─ demo.py
```
