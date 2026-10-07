# Online deployment: Vercel (UI) + your own computer (training backend, via a free Cloudflare tunnel)

[中文](DEPLOY.md) | **English** | [한국어](DEPLOY.ko.md) | [Deutsch](DEPLOY.de.md)

```
Browser ──► https://visual-network-neuron.vercel.app        UI (Vercel, static files)
              │  /api/… (CORS + access token)
              ▼
        https://xxxx.trycloudflare.com                      free Cloudflare tunnel
              ▼
        python -m server on your computer (127.0.0.1:8765)   PyTorch backend: your CPU / GPU, data stays on your machine
```

Vercel can only host the web page, not PyTorch, and Hugging Face Docker Spaces now require a paid plan. So the training backend runs on your own computer and a free Cloudflare tunnel gives it a public https address — no payment, no credit card, and your own GPU.

---

## ① Push to GitHub (for every update)

```powershell
cd D:\网络神经测试\核心\NeuroCore
.\scripts\publish_github.ps1 -User tianmingliu-pixel
```

`data\` (uploads, memory store, token), `exports\` and `.venv\` are never uploaded. If you changed frontend code (`web\src`), run `.\scripts\run.ps1 -Task build-ui` before pushing.

## ② UI: Vercel (one-time setup)

1. Sign in at <https://vercel.com> (with GitHub) → **Add New… → Project** → choose `Visual-Network-Neuron` → **Import**.
2. Framework Preset: **Other**, leave the rest (`vercel.json` installs nothing and publishes `web/dist`) → **Deploy**.
3. You get a URL such as `https://visual-network-neuron.vercel.app`.

Every push to GitHub republishes the site.

## ③ Backend: open it up from your computer (whenever you want to use the site)

```powershell
.\scripts\serve_public.ps1 -Publish
```

The script:

1. asks for an **access token** on the first run (stored in `data\.public_token`, never uploaded);
2. installs Cloudflare's `cloudflared` (winget);
3. starts the backend in the background and opens a tunnel, giving `https://xxxx.trycloudflare.com` (copied to the clipboard);
4. with `-Publish`, writes that address to `deploy/backend_url.txt` and pushes it — Vercel updates within about a minute and **other visitors connect automatically**;
5. opens `your-site/?api=tunnel-url`; enter the token under **Backend** (top right) to train and upload.

Closing that PowerShell window (or Ctrl+C) stops the backend and the tunnel. The free tunnel address changes every time, so just use `-Publish` each time.

## Security

| Who | Can do |
|---|---|
| You (with the token) | train, upload data, view / delete the memory store, use local paths |
| Others (no token) | only watch training (charts, network graph, step-through); cannot see your data, uploads or memory store |

The backend only listens on 127.0.0.1; only the tunnel reaches it from outside, and nothing is exposed while `serve_public.ps1` is not running.

## Want a fixed address?

- **Fixed URL**: add your own domain to Cloudflare and use a named tunnel (`cloudflared tunnel create`).
- **Cloud server**: the repository's `Dockerfile` runs on any Docker host (Hugging Face PRO, Google Cloud Run, a paid Render plan, … at least 2 GB RAM); set `NEUROCORE_API_BASE` in Vercel to point at it.

## Related files

| File | Purpose |
|---|---|
| `scripts/publish_github.ps1` | Commit and push to GitHub in one step |
| `scripts/serve_public.ps1` | Start backend + Cloudflare tunnel and tell the site its address |
| `vercel.json`, `scripts/vercel_build.mjs` | Vercel publishes `web/dist`; backend URL from `NEUROCORE_API_BASE` or `deploy/backend_url.txt` |
| `server/config.py` | Token, CORS, read protection, data directory |
| `Dockerfile` | For a cloud server later |
