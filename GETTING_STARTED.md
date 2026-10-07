# 新手指南：在你自己的电脑上运行 NeuroCore

**中文** | [English](GETTING_STARTED.en.md) | [한국어](GETTING_STARTED.ko.md) | [Deutsch](GETTING_STARTED.de.md)

这份指南写给**第一次来到这个仓库的人**。按顺序做完，大约 15–30 分钟（主要是下载 PyTorch 的时间），你就能在自己的电脑上训练神经网络，并在网页里看着它学习。

> 下面所有路径都以 `C:\NeuroCore` 为例。你可以换成别的位置，但**建议用英文、不带空格的短路径**，最省事。

---

## 先弄清楚：网页和后端是什么关系

```
网页（界面）  https://visual-network-neuron.vercel.app   或   http://127.0.0.1:8765
     │  只负责显示：按钮、曲线、网络结构图
     ▼
后端（计算）  你电脑上的 python -m server（127.0.0.1:8765）
        真正训练神经网络、分析数据、保存“机器记忆库”
```

- **网页只是界面**，自己不会计算。
- **后端才是干活的**。本指南就是教你把后端装在**你自己的电脑**上。
- `127.0.0.1` 的意思是“这台电脑自己”，别人从网上访问不到它。

---

## 第 0 步：准备（只做一次）

| 需要 | 怎么检查 | 没有的话 |
|---|---|---|
| Windows 10 / 11 | — | macOS / Linux 见文末「不是 Windows？」 |
| Python 3.10 – 3.14 | 打开 PowerShell，输入 `python --version` | 到 <https://www.python.org/downloads/> 安装，**安装时勾选「Add python.exe to PATH」**；或运行 `winget install Python.Python.3.12` |
| Git（可选） | `git --version` | 没有也行，第 2 步可以下载 ZIP |
| NVIDIA 显卡（可选） | — | 没有显卡也能用 CPU 训练，只是慢一点 |

**怎么打开 PowerShell：** 按 `Win` 键 → 输入 `PowerShell` → 回车。

---

## 第 1 步：创建一个文件夹

在 PowerShell 里输入（每行回车一次）：

```powershell
mkdir C:\NeuroCore
cd C:\NeuroCore
```

现在你在 `C:\NeuroCore` 里。之后所有命令都**在这个文件夹里**运行。

---

## 第 2 步：把代码下载到这个文件夹

**方法 A：用 Git（推荐，以后更新方便）**

```powershell
cd C:\
git clone https://github.com/tianmingliu-pixel/Visual-Network-Neuron.git NeuroCore
cd C:\NeuroCore
```

> 如果第 1 步已经建了空的 `C:\NeuroCore`，`git clone` 也能直接放进去（文件夹是空的就行）。

**方法 B：下载 ZIP**

1. 打开 <https://github.com/tianmingliu-pixel/Visual-Network-Neuron>
2. 点绿色的 **Code** 按钮 → **Download ZIP**
3. 解压后会得到一个叫 `Visual-Network-Neuron-main` 的文件夹
4. 把**它里面的所有文件**复制到 `C:\NeuroCore`

**检查：** 运行 `dir`，应该能看到 `README.md`、`scripts`、`server`、`web` 等。如果只看到一个 `Visual-Network-Neuron-main` 文件夹，说明多套了一层，进去把文件移出来。

---

## 第 3 步：允许运行脚本（每次打开新 PowerShell 窗口时做）

Windows 默认禁止运行 `.ps1` 脚本。下面这条**只对当前窗口有效**，关掉窗口就恢复，不会改动系统设置：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

如果用 ZIP 下载，再运行一次（解除“从网上下载”的标记）：

```powershell
Get-ChildItem C:\NeuroCore -Recurse | Unblock-File
```

---

## 第 4 步：一键安装环境（只做一次）

```powershell
cd C:\NeuroCore
.\scripts\check_env.ps1      # 先检查：每一项显示 PASS / WARN / FAIL，FAIL 后面会告诉你怎么修
.\scripts\setup_env.ps1      # 再安装：创建 C:\NeuroCore\.venv，并按你的显卡自动安装 PyTorch
```

- 安装会下载 PyTorch（几百 MB 到 2 GB+），请耐心等待。
- 所有东西都装在 `C:\NeuroCore\.venv` 里，**不会影响你电脑上的其他 Python**。
- 看到最后的自检全部通过就成功了。

想强制用 CPU 版（例如显卡驱动有问题）：`.\scripts\setup_env.ps1 -Cuda cpu`

---

## 第 5 步：启动后端（每次想用的时候做）

```powershell
cd C:\NeuroCore
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run.ps1 -Task ui
```

- 看到 `NeuroCore UI -> http://127.0.0.1:8765` 就说明后端启动了。
- 浏览器会自动打开 `http://127.0.0.1:8765`。
- **这个 PowerShell 窗口不要关**——关掉 = 后端停止。用完按 `Ctrl + C` 或直接关窗口。

---

## 第 6 步：打开网页，确认用的是你自己的电脑

两个网址都可以，**效果一样**：

| 网址 | 说明 |
|---|---|
| `http://127.0.0.1:8765` | 你电脑上的后端自带的网页，第 5 步会自动打开，**断网也能用** |
| `https://visual-network-neuron.vercel.app` | 在线网页，会**自动发现**你电脑上正在运行的后端 |

打开后，**先看右上角**：

| 右上角显示 | 意思 | 要不要处理 |
|---|---|---|
| `● 后端 · 本页面` | 用的是你自己的电脑（从 127.0.0.1:8765 打开时） | ✅ 不用 |
| `● 后端 · 本机` | 用的是你自己的电脑（从在线网页打开时） | ✅ 不用 |
| `● 后端 · 网站后端 · 只读观看` 和「👀 观看模式」 | 连到的是**作者的电脑**，只能看 | ❌ 说明你的后端没在运行，回到第 5 步 |
| `○ … · 未连接` | 哪个后端都没连上 | ❌ 回到第 5 步，确认窗口还开着 |

> 界面是英文的话，点右上角 **中 / EN** 切换语言。英文界面里 “本机” 显示为 `local`，“本页面” 为 `this page`，“只读观看” 为 `view only`。

### 后端面板里每个选项是什么意思（一般不用动）

点右上角的 **后端** 按钮会弹出「后端连接」面板：

| 选项（中文 / English） | 意思 | 什么时候用 |
|---|---|---|
| 后端地址 / Backend URL | 网页去哪台电脑找后端 | 一般**留空**，交给“自动选择” |
| 本页面同源 / Same as this page | 后端就是打开这个网页的那台机器 | 从 `127.0.0.1:8765` 打开时 |
| 本机 / local `http://127.0.0.1:8765` | **你自己的电脑** | 在线网页没自动发现你的后端时，手动点它 |
| 网站默认后端 / Site default backend | 作者的电脑（通过隧道） | 只想**观看**作者的训练时 |
| 口令 / Access token | 作者后端的密码 | **用自己的电脑时留空，不需要密码** |
| 自动选择 / Auto-select | 清除手动设置，恢复自动 | 设乱了的时候 |
| 保存并重新连接 / Save & reconnect | 保存上面的设置并刷新 | 改完之后 |

**自动选择的顺序：** ① 你手动设过的地址 → ② 网页自己所在的地址 → ③ 你电脑上的 `127.0.0.1:8765` → ④ 作者的电脑（只读观看）。所以只要第 5 步的窗口开着，打开网页就会自动用你自己的电脑。

---

## 第 7 步：开始第一次训练

1. 顶部选 **训练 / Training** 标签。
2. 下拉框选一个任务，新手推荐 **MLP · 螺旋分类（神经元级）**。
3. 步数、学习率、批大小保持默认，点 **▶ 开始训练 / Start training**。
4. 你会看到：网络结构图里每个神经元和权重在变化、损失曲线下降、准确率上升，右侧「逐行讲解」高亮正在运行的代码行，右下角「机器记忆库」记录每次快照。

**用自己的数据：** 点顶部 **数据 / Data** 标签 → 上传文件夹（例如 `cat\`、`dog\` 两个子文件夹的图片，或带标签列的表格）→ 加载并分析 → 去训练。数据只保存在 `C:\NeuroCore\data\` 里。

---

## 为什么要用自己的电脑当后端？

| | 用自己的电脑 | 连作者的电脑（观看） |
|---|---|---|
| 需要密码吗 | **不需要** | 需要作者的口令，否则只能看 |
| 能训练 / 上传 / 导出 | ✅ 全部可以 | ❌ 只能看 |
| 你的数据去哪 | **只在你的电脑上**（`C:\NeuroCore\data`） | 会传到别人电脑上 |
| 速度 | 用你的 CPU / GPU，不经过网络 | 取决于作者的电脑和网络 |
| 什么时候能用 | 随时，断网也行 | 只有作者开着电脑和隧道时 |
| 机器记忆库 | 你自己的，越练越多，可续训 | 看不到 |

---

## 以后每次使用：只需要这三行

```powershell
cd C:\NeuroCore
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run.ps1 -Task ui
```

**更新到最新版本**（用 Git 下载的）：`cd C:\NeuroCore` → `git pull` → 再启动。用 ZIP 的话重新下载并覆盖即可（`data\` 文件夹里是你的数据和记忆库，不要删）。

---

## 遇到问题

| 现象 | 解决 |
|---|---|
| `无法加载文件 …ps1，因为在此系统上禁止运行脚本` | 做第 3 步 |
| 运行后**卡住、一个字都不输出**（连 `=== NeuroCore environment check ===` 都没有） | 你在 cmd 里，脚本根本没启动。按 `Ctrl + C`，改用 `powershell -ExecutionPolicy Bypass -File .\scripts\check_env.ps1`，或打开真正的 PowerShell（提示符以 `PS` 开头） |
| `无法将“.\scripts\…ps1”项识别为 cmdlet…` | 不在项目文件夹里：运行 `Test-Path .\scripts\setup_env.ps1`，`False` 就 `cd C:\NeuroCore`；ZIP 多套了一层就进去那一层 |
| `'.\scripts\…ps1' 不是内部或外部命令` | 打开的是 cmd，不是 PowerShell；或改用万能写法 `powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1` |
| `…未经数字签名` | ZIP 下载的文件被标记：`Get-ChildItem . -Recurse \| Unblock-File` |
| `python` 不是内部或外部命令 | 重新安装 Python 并勾选「Add to PATH」，然后**重新打开** PowerShell |
| `setup_env.ps1` 报 Python 版本不对 | 需要 3.10–3.14：`winget install Python.Python.3.12` |
| 端口 8765 被占用 | 关掉之前开的 `run.ps1` 窗口；或用 `.\scripts\run.ps1 -Task ui -Port 8766`，然后在后端面板里手动填 `http://127.0.0.1:8766` |
| 在线网页一直显示「只读观看」或「未连接」 | 确认第 5 步窗口还开着；在后端面板点「本机」→「保存并重新连接」；仍不行就直接打开 `http://127.0.0.1:8765`（部分浏览器如 Safari 会阻止在线网页访问本机） |
| `UnicodeEncodeError: 'charmap' codec` | 旧版本的问题，更新到最新代码即可 |
| 点「开始训练」没反应 | 看页面最下方「日志」的红字；如果写着“需要口令”，说明你连的是作者的电脑，回到第 6 步 |

---

## 不是 Windows？（macOS / Linux）

脚本是 PowerShell 写的，在 macOS / Linux 上手动执行等价步骤：

```bash
mkdir -p ~/NeuroCore && cd ~/NeuroCore
git clone https://github.com/tianmingliu-pixel/Visual-Network-Neuron.git .
python3 -m venv .venv
source .venv/bin/activate
pip install torch            # 有 NVIDIA 显卡请按 https://pytorch.org 选择对应的安装命令
pip install -r requirements.txt
python -m server --open      # 启动后端并打开 http://127.0.0.1:8765
```

以后每次：`cd ~/NeuroCore && source .venv/bin/activate && python -m server --open`

---

更多：项目介绍见 [README.md](README.md)；仓库主人如何把自己的电脑开放给网站访问者，见 [DEPLOY.md](DEPLOY.md)（普通使用者不需要看）。
