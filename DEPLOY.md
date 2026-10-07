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

Vercel 只能放网页，跑不了 PyTorch；Hugging Face 的 Docker Space 现在需要付费。所以训练后端就放在你自己的电脑上，
用 Cloudflare 的免费隧道给它一个 https 公网地址。不用付费、不用信用卡，还能用你自己的显卡。

---

## ① 推送到 GitHub（每次更新都用这一步）

```powershell
cd D:\网络神经测试\核心\NeuroCore
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
5. 打开 `你的网站/?api=隧道地址`；在网页右上角 **后端** 里输入口令即可训练、上传。

关掉这个 PowerShell 窗口（或 Ctrl+C）= 停止后端和隧道，网站就连不上了。免费隧道每次启动地址都会变，所以每次用 `-Publish` 更新即可。

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
