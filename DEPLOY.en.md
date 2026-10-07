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

> This document is for the **repository owner** (opening their computer to site visitors). To simply use NeuroCore on your own computer, see [GETTING_STARTED.en.md](GETTING_STARTED.en.md). Replace `C:\NeuroCore` below with your project folder.

Vercel can only host the web page, not PyTorch, and Hugging Face Docker Spaces now require a paid plan. So the training backend runs on your own computer and a free Cloudflare tunnel gives it a public https address — no payment, no credit card, and your own GPU.

---

## ① Push to GitHub (for every update)

```powershell
cd C:\NeuroCore
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
5. opens the site (fixed address `https://visual-network-neuron.vercel.app`). On your own computer the page connects to the local backend automatically; enter the token once under **Backend** (top right) to train and upload (the browser remembers it).

Closing that PowerShell window (or Ctrl+C) stops the backend and the tunnel.

**Does the address change?** The site address `visual-network-neuron.vercel.app` never changes; the tunnel address `xxxx.trycloudflare.com` is random on every start, but it only works behind the scenes — `-Publish` tells the site the new one. Share only the Vercel address and open it yourself too (not the tunnel address: the token is stored per address, so a new address means entering it again).

## After opening the site: automatic backend selection

The page is only the interface; training and data analysis run in a backend. On load the page **automatically** looks for one, in this order:

| Situation | Connects to | Top right shows | What works |
|---|---|---|---|
| The visitor runs a backend on their own computer (`run.ps1 -Task ui`) | their computer `127.0.0.1:8765` | Backend · local | everything, on their own CPU/GPU; data stays on their machine |
| No local backend, your tunnel is running | your computer (via the tunnel) | Backend · site backend · view only | watch only; train / upload after entering the token |
| Neither | — | disconnected | interface only, nothing runs |

To choose manually: click **Backend** (top right) → enter a backend URL → "Save & reconnect"; "Auto-select" restores the automatic choice.

### Where to enter the token

1. Click the **Backend** button at the far top right (right of the 中/EN switch) to open the "Backend connection" panel;
2. leave **Backend URL** empty (= automatic / same address as this page);
3. put the token you set the first time you ran `serve_public.ps1` into **Access token** (forgot it: `Get-Content data\.public_token`);
4. click "Save & reconnect". Once "view only" disappears you can train.

> "Start training" does nothing and the log says "Access token required" = the token has not been entered; no files are missing.

### Visitors who want to train on their own computer

The top right shows "👀 Watching · train yourself?"; clicking it shows the steps:

1. Download the code: <https://github.com/tianmingliu-pixel/Visual-Network-Neuron>
2. In the code folder run `.\scripts\setup_env.ps1`, then `.\scripts\run.ps1 -Task ui`
3. Reload the site: it switches to the backend on their own computer automatically (no token needed, and their data never reaches yours)

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `UnicodeEncodeError: 'charmap' codec can't encode` | Older versions wrote Chinese text to the log with the Windows default encoding; fixed (the backend forces UTF-8 and the scripts set `PYTHONUTF8=1`). Pull the latest code |
| "Backend did not start" | Check `data\backend.err.log`; if port 8765 is busy, close other `run.ps1 -Task ui` windows first |
| Site shows "disconnected" | `serve_public.ps1` is not running, or you just pushed with `-Publish` and Vercel is still updating (about 1 minute) |
| Training button does nothing | Token not entered (see "Where to enter the token") |

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
