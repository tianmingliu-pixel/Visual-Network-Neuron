# 在线部署：Vercel（网页）+ 你自己的电脑（训练后端，经免费 Cloudflare 隧道）

**中文** | [English](DEPLOY.en.md) | [한국어](DEPLOY.ko.md) | [Deutsch](DEPLOY.de.md)

```
浏览器 ──► https://visual-network-neuron.vercel.app        网页界面（Vercel，静态文件）
              │  /api/…（跨域 + 口令）
              ▼
        https://xxxx.trycloudflare.com                      免费 Cloudflare 隧道
              ▼
        你电脑上的 python -m server（127.0.0.1:8765）        PyTorch 训练后端：你的 CPU / GPU，数据留在你电脑上
```

> 这份文档写给**仓库主人**（把自己的电脑开放给网站访问者）。只想在自己电脑上使用，请看 [GETTING_STARTED.md](GETTING_STARTED.md)。下面的 `C:\NeuroCore` 换成你的项目文件夹。

Vercel 只能放网页，跑不了 PyTorch；Hugging Face 的 Docker Space 现在需要付费。所以训练后端就放在你自己的电脑上，
用 Cloudflare 的免费隧道给它一个 https 公网地址。不用付费、不用信用卡，还能用你自己的显卡。

---

## ① 推送到 GitHub（每次更新都用这一步）

```powershell
cd C:\NeuroCore
.\scripts\publish_github.ps1 -User tianmingliu-pixel
```

`data\`（上传的数据、记忆库、口令）、`exports\`、`.venv\` 不会上传。改了前端代码（`web\src`）要先 `.\scripts\run.ps1 -Task build-ui` 再推送。

## ② 网页：Vercel（只需设置一次）

1. 登录 <https://vercel.com>（用 GitHub 账户）→ **Add New… → Project** → 选 `Visual-Network-Neuron` → **Import**。
2. Framework Preset 选 **Other**，其他保持默认（`vercel.json` 已写好：不安装依赖，直接发布 `web/dist`）→ **Deploy**。
3. 得到网址，例如 `https://visual-network-neuron.vercel.app`。

以后每次推送到 GitHub，Vercel 自动重新发布。

## ③ 后端：在你电脑上开放（每次要用网站时运行）

```powershell
.\scripts\serve_public.ps1 -Publish
```

脚本会：

1. 第一次运行时让你设置一个**口令**（保存在 `data\.public_token`，不会上传）；
2. 自动安装 Cloudflare 的 `cloudflared`（winget）；
3. 在后台启动后端，并建立隧道，得到 `https://xxxx.trycloudflare.com`（复制到剪贴板）；
4. `-Publish`：把这个地址写进 `deploy/backend_url.txt` 并推送，Vercel 约 1 分钟后更新，**别人打开网站也会自动连上**；
5. 打开网站（固定网址 `https://visual-network-neuron.vercel.app`）。你在自己电脑上打开时，网页会自动连本机后端；在右上角 **后端** 里输入一次口令即可训练、上传（以后浏览器会记住）。

关掉这个 PowerShell 窗口（或 Ctrl+C）= 停止后端和隧道，网站就连不上了。

**地址会变吗？** 网站地址 `visual-network-neuron.vercel.app` 永远不变；隧道地址 `xxxx.trycloudflare.com` 每次启动都会随机变化，但它只在幕后用——`-Publish` 会自动把新地址告诉网站。所以分享给别人时只给 Vercel 网址，自己也直接打开 Vercel 网址（不要打开隧道地址：口令是按网址保存的，换了地址就要重新输入）。

## 打开网站后：自动选择后端

网页本身只是界面，训练和数据分析都在后端运行。打开网站时网页会**自动**按顺序找后端：

| 情况 | 自动连到 | 右上角显示 | 能做什么 |
|---|---|---|---|
| 访问者自己电脑上开着后端（`run.ps1 -Task ui`） | 他自己的电脑 `127.0.0.1:8765` | 后端 · 本机 | 全部功能，用他自己的 CPU/GPU，数据不出他的电脑 |
| 没有本机后端，你的隧道开着 | 你的电脑（经隧道） | 后端 · 网站后端 · 只读观看 | 只能观看；输入口令后才能训练 / 上传 |
| 都没有 | — | 未连接 | 只有界面，什么都跑不了 |

手动指定：点右上角 **后端** → 填后端地址 →「保存并重新连接」；点「自动选择」恢复自动。

### 在哪里输入口令

1. 点网页最右上角的 **后端** 按钮（在 中/EN 切换的右边），弹出「后端连接」面板；
2. **后端地址**留空（= 自动 / 与本页同一个地址）；
3. **口令**框填 `serve_public.ps1` 第一次让你设的口令（忘了：`Get-Content data\.public_token`）；
4. 点「保存并重新连接」，右上角的「只读观看」消失后就能训练了。

> 点「开始训练」没反应、日志里写着「需要口令」= 还没输入口令，不是少上传了文件。

### 别人想用自己的电脑训练

右上角会显示「👀 观看模式 · 自己训练？」，点开有步骤：

1. 下载代码：<https://github.com/tianmingliu-pixel/Visual-Network-Neuron>
2. 在代码文件夹运行 `.\scripts\setup_env.ps1`，再运行 `.\scripts\run.ps1 -Task ui`
3. 回到网站刷新：网页自动改用他自己电脑上的后端（不需要口令，数据也不会传到你的电脑）

## 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| `UnicodeEncodeError: 'charmap' codec can't encode` | 旧版本把中文写进日志时用了 Windows 默认编码；已修复（后端强制 UTF-8，脚本设置 `PYTHONUTF8=1`）。拉取最新代码即可 |
| 「后端没有启动成功」 | 看 `data\backend.err.log`；端口 8765 被占用就先关掉其他 `run.ps1 -Task ui` 窗口 |
| 网站显示「未连接」 | 你的 `serve_public.ps1` 没在运行，或刚用 `-Publish` 推送、Vercel 还在更新（等约 1 分钟） |
| 点训练没反应 | 没输入口令（见上面「在哪里输入口令」） |

## 安全

| 谁 | 能做什么 |
|---|---|
| 你（有口令） | 训练、上传数据、查看 / 删除记忆库、使用本机路径 |
| 别人（没有口令） | 只能观看训练过程（曲线、结构图、逐行讲解）；看不到你的数据、上传记录和记忆库 |

后端只监听 127.0.0.1，只有隧道能从外面访问；不运行 `serve_public.ps1` 时什么都不会对外开放。

## 不想每次换地址？

- **固定地址**：在 Cloudflare 添加你自己的域名，用 named tunnel（`cloudflared tunnel create`），地址就固定了。
- **云服务器**：仓库里的 `Dockerfile` 可以直接部署到任何 Docker 主机（Hugging Face PRO、Google Cloud Run、Render 付费档等，内存至少 2 GB）；在 Vercel 设置环境变量 `NEUROCORE_API_BASE` 指向它即可。

## 相关文件

| 文件 | 作用 |
|---|---|
| `scripts/publish_github.ps1` | 一键提交并推送到 GitHub |
| `scripts/serve_public.ps1` | 启动后端 + Cloudflare 隧道，并把地址告诉网站 |
| `vercel.json`、`scripts/vercel_build.mjs` | Vercel 发布 `web/dist`，后端地址来自 `NEUROCORE_API_BASE` 或 `deploy/backend_url.txt` |
| `server/config.py` | 口令、跨域、只读保护、数据目录等设置 |
| `Dockerfile` | 以后想放到云服务器时用 |
