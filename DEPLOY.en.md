# Online deployment: GitHub → Vercel (UI) + Hugging Face Spaces (training backend)

[中文](DEPLOY.md) | **English** | [한국어](DEPLOY.ko.md) | [Deutsch](DEPLOY.de.md)

```
Browser ──► https://visual-network-neuron.vercel.app           UI (Vercel, static files)
              │  /api/… (CORS + access token)
              ▼
        https://<hf-user>-visual-network-neuron.hf.space        PyTorch training backend (Docker, free CPU)
```

Vercel can only host the web page — it cannot run PyTorch (too large, no long-running processes, no disk). The training backend therefore runs in a free Docker Space on Hugging Face (2 vCPU, 16 GB RAM). After every push to GitHub, **both sides redeploy automatically**.

---

## ① Push to GitHub (use this step for every update)

```powershell
cd D:\网络神经测试\核心\NeuroCore
.\scripts\publish_github.ps1 -User tianmingliu-pixel
```

- Pushes to `https://github.com/tianmingliu-pixel/Visual-Network-Neuron`; the first time, a GitHub sign-in window opens.
- The script copies `deploy/deploy-backend-hf.yml` into `.github/workflows/` (used to deploy the backend automatically).
- Anything already in the repository (e.g. a README) is merged. `data\` (uploads, memory store), `exports\` and `.venv\` are never uploaded.
- If you changed frontend code (`web\src`), run `.\scripts\run.ps1 -Task build-ui` first (needs Node.js); Vercel serves the committed `web\dist`.

## ② Backend: Hugging Face Space (one-time setup)

1. Sign up / sign in at <https://huggingface.co> → avatar → **New Space**:
   name `visual-network-neuron`, **SDK: Docker → Blank**, hardware **CPU basic (free)**, public or private.
2. Space → **Settings → Variables and secrets**:
   - **New secret** `NEUROCORE_TOKEN` = an access token you choose (required to train / upload / delete memory; visitors without it can only watch)
   - (optional) **New variable** `NEUROCORE_ALLOWED_ORIGINS` = your Vercel URL from step ③, e.g. `https://visual-network-neuron.vercel.app`
3. Avatar → **Settings → Access Tokens → Create new token**, type **Write**, copy it.
4. GitHub repository → **Settings → Secrets and variables → Actions**:
   - **Secrets** tab: New repository secret `HF_TOKEN` = the token from the previous step
   - **Variables** tab: New repository variable `HF_SPACE` = `your-hf-user/visual-network-neuron`
5. GitHub repository → **Actions → Deploy backend to Hugging Face Space → Run workflow**.
   The Space starts building (about 5–10 minutes the first time, it downloads PyTorch). Then open
   `https://your-hf-user-visual-network-neuron.hf.space/api/version`; `{"version": 4, "cloud": true, …}` means it works.

From now on every push to GitHub syncs and rebuilds the backend.

## ③ UI: Vercel (one-time setup)

1. Sign in at <https://vercel.com> (with your GitHub account) → **Add New… → Project** → choose `Visual-Network-Neuron` → **Import**.
2. Framework Preset: **Other**; leave the rest as is (`vercel.json` in the repository installs nothing and publishes `web/dist`).
3. Expand **Environment Variables** and add
   `NEUROCORE_API_BASE` = `https://your-hf-user-visual-network-neuron.hf.space`
4. **Deploy**. You get a URL such as `https://visual-network-neuron.vercel.app`.
5. Open it → **后端 (Backend)** at the top right → enter the access token → save. When it shows "cloud ☁" and "token correct", you can train and upload data.

From now on every push to GitHub republishes the site.

## The same page can also use your own computer

In **后端 (Backend)** choose "本机 (local) http://127.0.0.1:8765" and save; the page then talks to the backend on your computer (start it with `.\scripts\run.ps1 -Task ui`): training uses your own CPU / GPU and your data never leaves the machine. "Reset to default" switches back to the cloud.

## Limits of the free cloud

| Limit | Details |
|---|---|
| Speed | Free CPU, no GPU. Fine for the demos and a few hundred small images; use the local backend for large datasets |
| Sleep | After 48 hours without visitors the Space sleeps; the next visit takes about a minute to wake it |
| No persistence | Uploads and the memory store are cleared when the Space restarts. To keep them, buy Persistent storage in the Space settings and add the variable `NEUROCORE_DATA_DIR=/data` |
| Shared session | All visitors see the same training run; the token protects write actions, without it you can only watch |
| Security | Cloud mode only analyzes data uploaded to the server and cannot read arbitrary server paths; upload limit 200 MB per file (`NEUROCORE_MAX_UPLOAD_MB`) |

## Related files

| File | Purpose |
|---|---|
| `scripts/publish_github.ps1` | Commit and push to GitHub in one step |
| `vercel.json`, `scripts/vercel_build.mjs` | Vercel publishes `web/dist` and writes the backend URL into `config.js` |
| `Dockerfile`, `.dockerignore` | Backend image (CPU PyTorch, port 7860) |
| `deploy/deploy-backend-hf.yml` | GitHub Actions workflow template; copied to `.github/workflows/` before pushing, then syncs the Space on every push |
| `server/config.py` | Cloud settings: token, CORS, data directory, upload limit |
