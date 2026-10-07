# Beginner's guide: run NeuroCore on your own computer

[中文](GETTING_STARTED.md) | **English** | [한국어](GETTING_STARTED.ko.md) | [Deutsch](GETTING_STARTED.de.md)

This guide is for **people visiting this repository for the first time**. Follow it in order — about 15–30 minutes (mostly downloading PyTorch) — and you will train neural networks on your own computer and watch them learn in the browser.

> All paths below use `C:\NeuroCore` as the example. You can choose another location, but **a short path in English letters without spaces** causes the fewest problems.

---

## First: how the web page and the backend relate

```
Web page (interface)  https://visual-network-neuron.vercel.app   or   http://127.0.0.1:8765
     │  only displays: buttons, curves, the network graph
     ▼
Backend (computation)  python -m server on your computer (127.0.0.1:8765)
        actually trains the networks, analyses data, keeps the "machine memory store"
```

- **The web page is only the interface**; it computes nothing itself.
- **The backend does the work.** This guide installs the backend on **your own computer**.
- `127.0.0.1` means "this computer itself"; nobody on the internet can reach it.

---

## Step 0: Prerequisites (once)

| You need | How to check | If missing |
|---|---|---|
| Windows 10 / 11 | — | macOS / Linux: see "Not on Windows?" at the end |
| Python 3.10 – 3.14 | open PowerShell, type `python --version` | install from <https://www.python.org/downloads/> and **tick "Add python.exe to PATH"**; or run `winget install Python.Python.3.12` |
| Git (optional) | `git --version` | not required — Step 2 can use a ZIP |
| NVIDIA GPU (optional) | — | without one, training runs on the CPU, just slower |

**Opening PowerShell:** press `Win` → type `PowerShell` → Enter.

---

## Step 1: Create a folder

In PowerShell (press Enter after each line):

```powershell
mkdir C:\NeuroCore
cd C:\NeuroCore
```

You are now in `C:\NeuroCore`. Run every later command **inside this folder**.

---

## Step 2: Download the code into the folder

**Option A: Git (recommended, easy to update later)**

```powershell
cd C:\
git clone https://github.com/tianmingliu-pixel/Visual-Network-Neuron.git NeuroCore
cd C:\NeuroCore
```

> If you already created an empty `C:\NeuroCore` in Step 1, `git clone` can still use it (it only has to be empty).

**Option B: Download ZIP**

1. Open <https://github.com/tianmingliu-pixel/Visual-Network-Neuron>
2. Click the green **Code** button → **Download ZIP**
3. Unzipping gives a folder called `Visual-Network-Neuron-main`
4. Copy **everything inside it** into `C:\NeuroCore`

**Check:** run `dir`; you should see `README.md`, `scripts`, `server`, `web` and so on. If you only see a `Visual-Network-Neuron-main` folder, there is one level too many — move its contents up.

---

## Step 3: Allow scripts (in every new PowerShell window)

Windows blocks `.ps1` scripts by default. This command **only affects the current window** and changes no system settings:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If you used the ZIP, also run this once (removes the "downloaded from the internet" mark):

```powershell
Get-ChildItem C:\NeuroCore -Recurse | Unblock-File
```

---

## Step 4: Install the environment (once)

```powershell
cd C:\NeuroCore
.\scripts\check_env.ps1      # check first: each item shows PASS / WARN / FAIL, and every FAIL tells you how to fix it
.\scripts\setup_env.ps1      # then install: creates C:\NeuroCore\.venv and installs the PyTorch build for your GPU
```

- This downloads PyTorch (a few hundred MB to 2 GB+); be patient.
- Everything goes into `C:\NeuroCore\.venv` and **does not affect any other Python on your computer**.
- When the final self-test passes, you are done.

To force the CPU build (e.g. GPU driver problems): `.\scripts\setup_env.ps1 -Cuda cpu`

---

## Step 5: Start the backend (each time you want to use it)

```powershell
cd C:\NeuroCore
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run.ps1 -Task ui
```

- `NeuroCore UI -> http://127.0.0.1:8765` means the backend is running.
- The browser opens `http://127.0.0.1:8765` automatically.
- **Keep this PowerShell window open** — closing it stops the backend. When finished, press `Ctrl + C` or close the window.

---

## Step 6: Open the page and confirm it uses your computer

Either address works **the same way**:

| Address | Notes |
|---|---|
| `http://127.0.0.1:8765` | the page served by your own backend; opened automatically in Step 5; **works offline** |
| `https://visual-network-neuron.vercel.app` | the online page; it **automatically finds** the backend running on your computer |

Once it loads, **look at the top right first**:

| Top right shows | Meaning | Action |
|---|---|---|
| `● Backend · this page` | your own computer (opened from 127.0.0.1:8765) | ✅ nothing |
| `● Backend · local` | your own computer (opened from the online page) | ✅ nothing |
| `● Backend · site backend · view only` and "👀 Watching" | connected to **the author's computer**, watch only | ❌ your backend is not running — back to Step 5 |
| `○ … · disconnected` | no backend reachable | ❌ back to Step 5, make sure the window is still open |

> Switch the interface language with **中 / EN** at the top right.

### What each option in the Backend panel means (usually leave it alone)

Clicking **Backend** at the top right opens the "Backend connection" panel:

| Option | Meaning | When to use |
|---|---|---|
| Backend URL | where the page looks for a backend | normally **leave empty** and let auto-select decide |
| Same as this page | the machine that served this page | when opened from `127.0.0.1:8765` |
| local `http://127.0.0.1:8765` | **your own computer** | if the online page did not detect your backend, click it manually |
| Site default backend | the author's computer (via a tunnel) | only to **watch** the author's training |
| Access token | the author backend's password | **leave empty when using your own computer — no password needed** |
| Auto-select | clears manual settings, back to automatic | if things got mixed up |
| Save & reconnect | saves the settings above and reloads | after changing anything |

**Auto-select order:** ① an address you set manually → ② the page's own address → ③ `127.0.0.1:8765` on your computer → ④ the author's computer (view only). So as long as the Step 5 window is open, the page uses your own computer automatically.

---

## Step 7: Your first training run

1. Choose the **Training** tab at the top.
2. Pick a task from the drop-down; for beginners: **MLP · spiral classification (neuron level)**.
3. Keep the default steps, learning rate and batch size; click **▶ Start training**.
4. You will see every neuron and weight change in the network graph, the loss fall, the accuracy rise, the "Line-by-line" panel highlight the running code, and the "Memory store" (bottom right) record snapshots.

**Your own data:** click the **Data** tab → upload a folder (e.g. images in `cat\` and `dog\` subfolders, or a table with a label column) → Load & analyse → train. Data stays in `C:\NeuroCore\data\`.

---

## Why use your own computer as the backend?

| | Your own computer | The author's computer (watching) |
|---|---|---|
| Password | **not needed** | the author's token, otherwise watch only |
| Train / upload / export | ✅ everything | ❌ watch only |
| Where your data goes | **only your computer** (`C:\NeuroCore\data`) | someone else's computer |
| Speed | your CPU / GPU, no network in between | depends on the author's machine and network |
| Availability | any time, even offline | only while the author runs the computer and tunnel |
| Machine memory store | your own; grows with every run; resumable | not visible |

---

## Every later session: just three lines

```powershell
cd C:\NeuroCore
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run.ps1 -Task ui
```

**Updating** (Git): `cd C:\NeuroCore` → `git pull` → start again. With a ZIP, download again and overwrite (the `data\` folder holds your data and memory store — do not delete it).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `…ps1 cannot be loaded because running scripts is disabled on this system` | do Step 3 |
| It **hangs with no output at all** (not even `=== NeuroCore environment check ===`) | you are in cmd and the script never started. Press `Ctrl + C`, then use `powershell -ExecutionPolicy Bypass -File .\scripts\check_env.ps1`, or open real PowerShell (the prompt starts with `PS`) |
| `The term '.\scripts\…ps1' is not recognized…` | you are not in the project folder: run `Test-Path .\scripts\setup_env.ps1`; if `False`, `cd C:\NeuroCore` (with a ZIP, step into the extra nested folder) |
| `'.\scripts\…ps1' is not recognized as an internal or external command` | that is cmd, not PowerShell; or use the universal form `powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1` |
| `…is not digitally signed` | files from the ZIP are marked: `Get-ChildItem . -Recurse \| Unblock-File` |
| `python` is not recognized | reinstall Python with "Add to PATH" ticked, then **reopen** PowerShell |
| `setup_env.ps1` complains about the Python version | 3.10–3.14 needed: `winget install Python.Python.3.12` |
| Port 8765 in use | close the earlier `run.ps1` window; or use `.\scripts\run.ps1 -Task ui -Port 8766` and enter `http://127.0.0.1:8766` in the Backend panel |
| Online page keeps showing "view only" or "disconnected" | check the Step 5 window is open; in the Backend panel click "local" → "Save & reconnect"; if it still fails, open `http://127.0.0.1:8765` directly (some browsers, e.g. Safari, block online pages from reaching your own computer) |
| `UnicodeEncodeError: 'charmap' codec` | an old-version issue — update to the latest code |
| "Start training" does nothing | read the red text in "Log" at the bottom; "Access token required" means you are connected to the author's computer — back to Step 6 |

---

## Not on Windows? (macOS / Linux)

The scripts are PowerShell; on macOS / Linux run the equivalent steps by hand:

```bash
mkdir -p ~/NeuroCore && cd ~/NeuroCore
git clone https://github.com/tianmingliu-pixel/Visual-Network-Neuron.git .
python3 -m venv .venv
source .venv/bin/activate
pip install torch            # with an NVIDIA GPU, use the command from https://pytorch.org
pip install -r requirements.txt
python -m server --open      # starts the backend and opens http://127.0.0.1:8765
```

Every later time: `cd ~/NeuroCore && source .venv/bin/activate && python -m server --open`

---

More: project overview in [README.en.md](README.en.md); how the repository owner opens their own computer to site visitors is in [DEPLOY.en.md](DEPLOY.en.md) (regular users do not need it).
