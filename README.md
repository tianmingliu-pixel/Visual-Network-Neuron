# NeuroCore · Visual Network Neuron

**中文** | [English](README.en.md) | [한국어](README.ko.md) | [Deutsch](README.de.md)

可扩展的 PyTorch 神经网络核心库，附带能“看着网络学习”的可视化训练界面。

第一阶段实现四个基础网络，并用 **注册表 + 扩展目录** 为以后任何 IT 领域的网络（GNN、音频、时序、推荐、安全、强化学习……）预留接口。

| 模块 | 文件 | 核心思想 |
|---|---|---|
| **Transformer** | `neurocore/models/transformer.py` | 全注意力编码器-解码器：`softmax(QKᵀ/√d)V`，因果掩码 + 交叉注意力 |
| **ViT** | `neurocore/models/vit.py` | 图像切成 patch → token，[CLS] + 位置编码 → Transformer 编码器 |
| **U-Net** | `neurocore/models/unet.py` | 下采样 / 上采样 + 同级跳跃连接；可选时间步 / 类别条件用于扩散 |
| **DiT** | `neurocore/models/dit.py` | Transformer 作为扩散去噪器，adaLN-Zero 注入时间步与类别 |
| DDPM | `neurocore/diffusion/ddpm.py` | 前向加噪 / 预测噪声 ε / 祖先采样 + CFG，U-Net 与 DiT 通用 |
| MLP | `neurocore/models/mlp.py` | 最基础的全连接网络，结构图里能看到每个神经元和每条权重 |
| 共享构件 | `neurocore/layers/` | 多头注意力、FFN、Pre-LN 块、正余弦 / 时间步 / Patch 嵌入 |

## 在线网站

**<https://visual-network-neuron.vercel.app>** —— 打开即用。网页会自动连接后端：你自己电脑上运行着 `.\scripts\run.ps1 -Task ui` 时用你自己的电脑训练，否则连作者电脑上的后端观看。口令、隧道、常见问题见 [DEPLOY.md](DEPLOY.md)。

👉 **第一次使用？** 按 [新手指南 GETTING_STARTED.md](GETTING_STARTED.md) 一步一步在你自己的电脑上安装并运行（不需要密码）。

## 快速开始（Windows PowerShell）

```powershell
cd NeuroCore
# 一条命令：检查环境 → 创建 .venv → 按显卡自动安装 PyTorch → 测试 → 演示
powershell -ExecutionPolicy Bypass -File .\deploy.ps1
```

分步执行：

```powershell
.\scripts\check_env.ps1                 # 只检查，不安装（PASS / WARN / FAIL + 修复命令）
.\scripts\check_env.ps1 -ReportPath env_report.json
.\scripts\setup_env.ps1                 # 部署（自动识别显卡）
.\scripts\setup_env.ps1 -Cuda cpu       # 强制 CPU 版
.\scripts\setup_env.ps1 -Cuda cu130 -TorchVersion 2.14.0 -Recreate
.\scripts\run.ps1 -Task demo            # 四个网络的小型训练演示
.\scripts\run.ps1 -Task demo -Model dit -Steps 300
.\scripts\run.ps1 -Task test            # 运行测试（pytest）
.\scripts\run.ps1 -Task list            # 已注册的模型
.\scripts\run.ps1 -Task info            # PyTorch / GPU 状态
```

### `check_env.ps1` 检查项
64 位系统、PowerShell ≥ 5.1、Python 3.10–3.14（优先 `py` 启动器）、pip、venv、Git、NVIDIA 显卡与驱动支持的 CUDA 版本、内存、磁盘、Windows 长路径、执行策略、已安装的 PyTorch 能否用 GPU。每个非 PASS 项都附带修复命令。

### PyTorch 安装源自动选择
`setup_env.ps1` 读取 `nvidia-smi` 报告的驱动 CUDA 版本，按顺序尝试：

| 驱动 CUDA | 尝试顺序 |
|---|---|
| ≥ 13.2 | cu132 → cu130 → cu126 → cpu |
| 13.0–13.1 | cu130 → cu126 → cpu |
| 12.6–12.9 | cu126 → cpu（PyTorch 2.14 是最后一个提供 CUDA 12.x 的版本） |
| 无 NVIDIA / 更旧 | cpu |

**已经装好 PyTorch？** 脚本会自动找到本机已有的 PyTorch（例如全局 `pip` 安装的 2.14.1），用 `--system-site-packages` 建 `.venv` 直接复用，无需重新下载；如果它是 CPU 版而你有 NVIDIA 显卡，脚本会提示，可用 `-FreshTorch -Recreate` 单独安装 CUDA 版。

安装后会实际调用 `torch.cuda.is_available()` 验证；某个源失败会自动退回下一个。全部输出记录在 `deploy.log`。macOS / Linux 上也可以用 PowerShell 7（`pwsh`）运行同样的脚本。

## 训练可视化界面

```powershell
.\scripts\run.ps1 -Task ui        # 打开 http://127.0.0.1:8765（已附带构建好的前端，无需 Node.js）
```

- **神经网络结构图**：默认任务 **MLP · 螺旋分类** 画出每个神经元与每条权重（橙 = 正权重、蓝 = 负权重、粗细 = |w|），节点颜色是当前样本在该层的激活值；青色脉冲 = 前向传播，品红脉冲 = 反向传播（沿梯度最大的连线）。ViT / Transformer / U-Net / DiT 显示**层流程图**：按真实执行顺序每列一层，悬停看形状和参数，点击跳到代码。
- **一次训练迭代的全过程**：① 数据 → ② 前向传播 → ③ 损失 → ④ 反向传播 → ⑤ 更新权重，每步给出含义、公式（交叉熵 / MSE / 链式法则 / AdamW）和对应代码行。
- **左侧**：实时 loss / 准确率 / 梯度范数曲线、迭代速度、学习效果预览（分类结果、序列反转、扩散去噪的 原图 → 加噪 → 还原）。
- **右侧**：点 **● 单步讲解**，后台把下一次迭代逐行录下来，像慢动作一样回放：代码面板高亮当前行并显示中文注释，下方显示这一行产生的张量（形状、均值 ± 标准差、范围、分布）和调用栈。空格 = 播放 / 暂停，← → = 单步。
- **为什么不卡**：训练在后台线程；指标每 100 ms 合并推送一次（≤ 10 次/秒，超量自动抽稀）；图表用 Canvas 绘制；逐行追踪只在“单步讲解”那一步开启。
- **界面语言与布局**：右上角「中 / EN」切换中英文（后端提示、日志、任务说明也一起翻译；代码注释保持中文）。左右两栏之间、代码与下方面板之间的分隔条可以拖动；左栏每张卡片拖底边可以改高度，双击恢复。「机器记忆库」和「逐行讲解」可以标签页切换、**并排**显示、**浮动**成可拖动可缩放的窗口，或**弹出**到单独的浏览器窗口（可以放到第二块屏幕）。布局会自动记住，右上角 ⟲ 恢复默认。
- **导出 ONNX**：右上角按钮，保存到 `exports/<任务>.onnx`（需要 `pip install -r requirements-export.txt`）。

前端开发（React + Vite，需要 Node.js）：

```powershell
.\scripts\run.ps1 -Task ui-dev     # 后端 :8765 + Vite 热更新 :5173
.\scripts\run.ps1 -Task build-ui   # 重新构建 web\dist
```

## 数据导入与分析

界面顶部切换到 **数据导入与分析**，输入本地路径、点「上传文件」/「上传文件夹」，或把文件 / 文件夹直接拖进框里：

| 数据 | 格式 | 自动处理 |
|---|---|---|
| 表格 | CSV / TSV / TXT / JSON / JSONL / Excel（.xlsx，无需 pandas）/ .xls\* / Parquet\* | 识别每列类型：数值（缺失补均值）、类别（独热）、文本（哈希词袋）、ID / 常数列（丢弃）；默认最后一列为目标，可改选 |
| 图片 | 整个文件夹或 zip | 裁剪缩放到 16–64 像素，判断灰度 / 彩色 → ViT |
| 音频 | `.wav`（PCM / 浮点）；`.flac .ogg .mp3` 需要 `pip install soundfile` | 提取 56 个声学特征 → MLP；元数据里的数值标签可做回归 |
| 数组 | `.npz` / `.npy`：`X`+`y`、`x_train/y_train/x_test/y_test`、文件夹里的 `X.npy`+`y.npy`，或单个二维数组（最后一列为目标） | 二维 → MLP；图像形状 → ViT |

\* .xls / Parquet 需要 `pip install -r requirements-data.txt`。

**图片 / 音频的标签从哪里来**（按优先级自动识别，结果显示在界面上）：

1. **元数据表**：文件夹里任意 CSV / Excel / JSON，有一列是文件名（`img001.jpg`、`images/img001.jpg` 或不带扩展名都可以），其他列是标签，例如 `labels.csv`：`filename,breed,weight`。可以在「标签列」里切换。
2. **类别子文件夹**：`数据\猫\*.jpg`、`数据\狗\*.jpg`，也支持 `train\ val\ test\` 下再分类别。
3. **文件名前缀**：`cat_001.jpg`、`dog.12.jpg` → `cat` / `dog`。

多次上传的文件夹会合并进「当前数据集」：先传 cat 文件夹，再传 dog 文件夹 = 2 个类别。上传的文件保存在 `data/uploads/<批次号>/`。

**分析内容**：样本数与训练 / 验证划分、目标分布、类别不均衡提示、每列类型 / 缺失 / 统计 / 分布、特征与目标的相关程度（分类用相关比 η²，回归用 |皮尔逊相关|）、PCA 二维投影、样本预览。

**训练**：点「用这份数据训练」自动生成「自定义」任务（向量数据 → MLP，图片 → ViT；分类用交叉熵，回归用 MSE），结构图、单步讲解、五步全过程照常可用。

**模型表现分析**：准确率 / R²·MAE·RMSE、混淆矩阵、每类精确率 / 召回率 / F1、置换法特征重要性、错得最离谱的样本。

> 页面顶部出现“后端是旧版本”的提示时：关闭正在运行后端的 PowerShell 窗口（或 Ctrl+C），重新运行 `.\scripts\run.ps1 -Task ui` 再刷新页面。

## 机器记忆库

训练页右下角「🧠 机器记忆库」。训练时每隔 N 步（默认本次步数的 1/40）把网络此刻的状态写进本地数据库 `data/memory/neuro_memory.db`（SQLite，Python 自带），权重文件在 `data/memory/ckpt/<运行>/latest.pt` 和 `best.pt`。

| 表 | 内容 |
|---|---|
| `runs` | 每次训练：任务、数据集、超参数、从哪个快照继续、最佳验证成绩 |
| `snapshots` | 每个快照：loss、训练 / 验证准确率、梯度范数、学习率、权重总变化量、检查点 |
| `layer_states` | 每层的权重范数、梯度范数、与上一快照相比的变化量 |
| `samples` | 样本元数据：来源文件、标签、训练 / 验证 |
| `embeddings` | 每个样本的特征向量（倒数第二层，float32）+ 预测 + 置信度 + 是否正确 |
| `sample_memory` | 跨多次训练累计的“错题本”：评估过几次、错过几次、平滑损失 |

**下次训练怎么用这些记忆**（面板上的选项）：

- **起点 = 从最佳记忆 / 某个快照继续**：加载权重 + AdamW 动量 + 步数，接着学，不从零开始；
- **难例回放**：按错题本给训练样本加权抽样，以前错过的样本多练；
- **图片增强**：随机左右翻转 + 平移，样本少时减少“死记硬背”（训练 100%、验证偏低就是这个问题）；
- 自动把**验证准确率最高**的快照另存为 best，过拟合后也能退回去。

面板里可以看：成长曲线（历次训练对比 + 记忆来源链）、特征向量空间（每个快照的样本向量 PCA 投影 + 分离度 + 原始记录）、每层变化热力图、样本错题本、表结构和只读 SQL 控制台。

## 在线部署

网页放 **Vercel**，PyTorch 训练后端跑在**你自己的电脑**上，通过免费的 Cloudflare 隧道接到网页（`.\scripts\serve_public.ps1 -Publish`）。详细步骤见 [DEPLOY.md](DEPLOY.md)。

```powershell
.\scripts\publish_github.ps1 -User 你的GitHub用户名     # 推送到 github.com/你的用户名/Visual-Network-Neuron
```

## Python 用法

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

unet = nc.build_model("unet", in_channels=3, base_channels=64)      # 扩散去噪
seg = nc.build_model("unet_seg", in_channels=3, num_classes=21)     # 图像分割
```

已注册的模型：`transformer`、`transformer_encoder`、`vit`、`vit_tiny/small/base`、`unet`、`unet_seg`、`dit`、`dit_s_2/b_2/xl_2`、`mlp`。

注意力默认用显式矩阵运算写出，便于学习；传 `use_sdpa=True` 切换到 PyTorch 融合内核（测试中验证两者数值一致）。

## 扩展到新领域

1. 复制 `neurocore/extensions/_template.py`，例如改名为 `gnn.py`（去掉下划线）。
2. 用 `@register_model("my_net", domain="graph")` 注册。
3. `import neurocore` 时自动加载，`nc.build_model("my_net")` 即可使用；某个扩展出错只会警告，不影响核心。

预留的领域标签：`nlp, vision, generative, graph, audio, multimodal, timeseries, recsys, security, rl, basic, other`。

## 目录

```
NeuroCore/
├─ deploy.ps1                  一键 检查 → 部署 → 演示
├─ DEPLOY.md                   在线部署（Vercel + 本机后端隧道）
├─ Dockerfile · vercel.json    云端部署配置
├─ scripts/
│  ├─ common.ps1               共享函数（查找 Python、读取 GPU、选择安装源）
│  ├─ check_env.ps1            环境检查
│  ├─ setup_env.ps1            部署（.venv + PyTorch + 依赖 + 测试）
│  ├─ run.ps1                  demo / test / list / info / ui / build-ui
│  ├─ publish_github.ps1       一键推送到 GitHub
│  └─ vercel_build.mjs         Vercel 构建步骤
├─ neurocore/
│  ├─ registry.py              模型注册中心
│  ├─ layers/                  注意力、嵌入
│  ├─ models/                  transformer / vit / unet / dit / mlp
│  ├─ diffusion/               DDPM
│  ├─ export.py                ONNX 导出
│  └─ extensions/              未来领域模块（自动发现）
├─ server/                     可视化后端（Starlette + SSE）
│  ├─ tasks.py                 训练任务（逐行中文注释，会被追踪讲解）
│  ├─ trainer.py               后台训练线程、暂停 / 继续 / 单步追踪 / 写入记忆
│  ├─ tracer.py                sys.settrace 逐行追踪器
│  ├─ graph.py · pipeline.py   结构图与五步全过程
│  ├─ datasets.py · evaluate.py 数据导入分析与模型评估
│  ├─ memory.py                机器记忆库（SQLite）
│  ├─ config.py                云端配置（口令、跨域、数据目录）
│  ├─ hub.py                   节流推送（≤ 10 次/秒）
│  └─ app.py                   HTTP 接口 + 前端页面
├─ web/                        React 前端（web/dist 为构建结果）
├─ tests/                      模型 / 后端 / 数据 / 记忆库测试
└─ demo.py
```
