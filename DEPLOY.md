# 在线部署：GitHub → Vercel（网页）+ Hugging Face Spaces（训练后端）

**中文** | [English](DEPLOY.en.md) | [한국어](DEPLOY.ko.md) | [Deutsch](DEPLOY.de.md)

```
浏览器 ──► https://visual-network-neuron.vercel.app            网页界面（Vercel，静态文件）
              │  /api/…（跨域 + 口令）
              ▼
        https://<HF用户名>-visual-network-neuron.hf.space       PyTorch 训练后端（Docker，免费 CPU）
```

Vercel 只能放网页，跑不了 PyTorch（体积太大、不能长时间运行、没有硬盘），所以训练后端放在 Hugging Face Spaces 的免费 Docker 空间里（2 核 CPU、16 GB 内存）。推送到 GitHub 后，两边都会**自动重新部署**。

---

## ① 推送到 GitHub（每次更新都用这一步）

```powershell
cd D:\网络神经测试\核心\NeuroCore
.\scripts\publish_github.ps1 -User tianmingliu-pixel
```

- 推送到 `https://github.com/tianmingliu-pixel/Visual-Network-Neuron`；第一次会弹出 GitHub 登录授权。
- 脚本会自动把 `deploy/deploy-backend-hf.yml` 放进 `.github/workflows/`（自动部署后端用）。
- 仓库里已有的 README 等会自动合并。`data\`（上传的数据、记忆库）、`exports\`、`.venv\` 不会上传。
- 改了前端代码（`web\src`）要先 `.\scripts\run.ps1 -Task build-ui`（需要 Node.js）再推送；Vercel 直接使用仓库里的 `web\dist`。

## ② 后端：Hugging Face Space（只需设置一次）

1. 注册 / 登录 <https://huggingface.co> → 右上角头像 **New Space**：
   名字 `visual-network-neuron`，**SDK 选 Docker → Blank**，硬件 **CPU basic（免费）**，Public 或 Private 都行。
2. Space 页面 → **Settings → Variables and secrets**：
   - **New secret** `NEUROCORE_TOKEN` = 你自己定的口令（有口令才能训练 / 上传 / 删除记忆；没有口令的访客只能观看）
   - （可选）**New variable** `NEUROCORE_ALLOWED_ORIGINS` = 第③步得到的 Vercel 网址，例如 `https://visual-network-neuron.vercel.app`
3. 头像 → **Settings → Access Tokens → Create new token**，类型 **Write**，复制下来。
4. GitHub 仓库 → **Settings → Secrets and variables → Actions**：
   - **Secrets** 标签：New repository secret `HF_TOKEN` = 上一步的 token
   - **Variables** 标签：New repository variable `HF_SPACE` = `你的HF用户名/visual-network-neuron`
5. GitHub 仓库 → **Actions → Deploy backend to Hugging Face Space → Run workflow**。
   Space 开始构建（第一次约 5–10 分钟，要下载 PyTorch）。完成后打开
   `https://你的HF用户名-visual-network-neuron.hf.space/api/version`，看到 `{"version": 4, "cloud": true, …}` 就成功了。

以后每次推送到 GitHub，后端都会自动同步并重建。

## ③ 网页：Vercel（只需设置一次）

1. 登录 <https://vercel.com>（用 GitHub 账户登录）→ **Add New… → Project** → 选 `Visual-Network-Neuron` → **Import**。
2. Framework Preset 选 **Other**，其他保持默认（仓库里的 `vercel.json` 已经写好：不安装依赖，直接发布 `web/dist`）。
3. 展开 **Environment Variables**，添加
   `NEUROCORE_API_BASE` = `https://你的HF用户名-visual-network-neuron.hf.space`
4. **Deploy**。得到网址，例如 `https://visual-network-neuron.vercel.app`。
5. 打开网址 → 右上角 **后端** → 填入口令 → 保存。显示「云端 ☁」和「口令正确」即可开始训练、上传数据。

以后每次推送到 GitHub，Vercel 会自动重新发布。

## 同一个网页也能连你自己的电脑

在网页右上角 **后端** 里点「本机 http://127.0.0.1:8765」并保存，网页就改用你电脑上的后端（先运行 `.\scripts\run.ps1 -Task ui`）：训练用自己的 CPU / GPU，数据不离开电脑。点「恢复默认」切回云端。

## 免费云端的限制

| 限制 | 说明 |
|---|---|
| 速度 | 免费 CPU，没有 GPU。演示任务和几百张小图没问题；大数据集建议用本机后端 |
| 休眠 | 48 小时没人访问会休眠，再次打开要等约 1 分钟启动 |
| 数据不持久 | Space 重启后，上传的数据和记忆库会清空。需要保留：在 Space Settings 购买 Persistent storage，并加变量 `NEUROCORE_DATA_DIR=/data` |
| 共享会话 | 所有访客看到的是同一个训练过程；口令保护写操作，没有口令只能观看 |
| 安全 | 云端模式只能分析上传到服务器的数据，不能读取服务器上的任意路径；单个上传文件上限 200 MB（`NEUROCORE_MAX_UPLOAD_MB` 可改） |

## 相关文件

| 文件 | 作用 |
|---|---|
| `scripts/publish_github.ps1` | 一键提交并推送到 GitHub |
| `vercel.json`、`scripts/vercel_build.mjs` | Vercel 发布 `web/dist`，并把后端地址写进 `config.js` |
| `Dockerfile`、`.dockerignore` | 后端镜像（CPU 版 PyTorch，端口 7860） |
| `deploy/deploy-backend-hf.yml` | GitHub Actions 工作流模板；推送前复制到 `.github/workflows/`，推送后自动同步到 Hugging Face Space |
| `server/config.py` | 云端设置：口令、跨域、数据目录、上传上限 |
