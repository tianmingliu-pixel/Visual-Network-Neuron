# NeuroCore · Visual Network Neuron

[中文](README.md) | **English** | [한국어](README.ko.md) | [Deutsch](README.de.md)

An extensible PyTorch neural-network core, with a visual training UI that lets you watch a network learn.

Phase 1 implements four foundational networks. A **registry + extensions folder** lets future networks from any IT domain (GNN, audio, time series, recommender systems, security, reinforcement learning …) plug in without touching the core.

| Module | File | Core idea |
|---|---|---|
| **Transformer** | `neurocore/models/transformer.py` | Attention-only encoder–decoder: `softmax(QKᵀ/√d)V`, causal mask + cross-attention |
| **ViT** | `neurocore/models/vit.py` | An image becomes a sequence of patches → tokens, [CLS] + positional encoding → Transformer encoder |
| **U-Net** | `neurocore/models/unet.py` | Down/up paths joined by skip connections; optional timestep / class conditioning for diffusion |
| **DiT** | `neurocore/models/dit.py` | Transformer as a diffusion denoiser, timestep and class injected via adaLN-Zero |
| DDPM | `neurocore/diffusion/ddpm.py` | Forward noising / ε-prediction / ancestral sampling + CFG, shared by U-Net and DiT |
| MLP | `neurocore/models/mlp.py` | The simplest fully connected network — every neuron and weight is drawn in the graph |
| Shared blocks | `neurocore/layers/` | Multi-head attention, FFN, Pre-LN block, sinusoidal / timestep / patch embeddings |

## Quick start (Windows PowerShell)

```powershell
cd NeuroCore
# One command: check environment → create .venv → install PyTorch for your GPU → test → demo
powershell -ExecutionPolicy Bypass -File .\deploy.ps1
```

Step by step:

```powershell
.\scripts\check_env.ps1                 # check only, install nothing (PASS / WARN / FAIL + fix commands)
.\scripts\check_env.ps1 -ReportPath env_report.json
.\scripts\setup_env.ps1                 # deploy (GPU auto-detection)
.\scripts\setup_env.ps1 -Cuda cpu       # force the CPU build
.\scripts\setup_env.ps1 -Cuda cu130 -TorchVersion 2.14.0 -Recreate
.\scripts\run.ps1 -Task demo            # small training demos for the four networks
.\scripts\run.ps1 -Task demo -Model dit -Steps 300
.\scripts\run.ps1 -Task test            # run the tests (pytest)
.\scripts\run.ps1 -Task list            # registered models
.\scripts\run.ps1 -Task info            # PyTorch / GPU status
```

### What `check_env.ps1` checks
64-bit OS, PowerShell ≥ 5.1, Python 3.10–3.14 (prefers the `py` launcher), pip, venv, Git, NVIDIA GPU and the CUDA version the driver supports, memory, disk, Windows long paths, execution policy, and whether an installed PyTorch can use the GPU. Every non-PASS item comes with a fix command.

### Automatic PyTorch wheel selection
`setup_env.ps1` reads the driver's CUDA version from `nvidia-smi` and tries, in order:

| Driver CUDA | Tried in order |
|---|---|
| ≥ 13.2 | cu132 → cu130 → cu126 → cpu |
| 13.0–13.1 | cu130 → cu126 → cpu |
| 12.6–12.9 | cu126 → cpu (PyTorch 2.14 is the last release with CUDA 12.x) |
| No NVIDIA / older | cpu |

**PyTorch already installed?** The script finds it (e.g. a global `pip` install of 2.14.1) and creates `.venv` with `--system-site-packages` to reuse it — no re-download. If it is a CPU build and you have an NVIDIA GPU, the script tells you; use `-FreshTorch -Recreate` to install a CUDA build separately.

After installing, the script calls `torch.cuda.is_available()` to verify; if one index fails it falls back to the next. All output is logged to `deploy.log`. On macOS / Linux the same scripts run with PowerShell 7 (`pwsh`).

## Visual training UI

```powershell
.\scripts\run.ps1 -Task ui        # opens http://127.0.0.1:8765 (prebuilt UI included, no Node.js needed)
```

- **Network graph**: the default task **MLP · spiral classification** draws every neuron and weight (orange = positive, blue = negative, thickness = |w|); node color is the activation for the current sample; cyan pulses = forward pass, magenta pulses = backward pass (along the largest gradients). ViT / Transformer / U-Net / DiT show a **layer flow**: one column per layer in real execution order; hover for shapes and parameters, click to jump to the code.
- **One training iteration end to end**: ① data → ② forward pass → ③ loss → ④ backpropagation → ⑤ weight update, each with its meaning, formula (cross-entropy / MSE / chain rule / AdamW) and the exact source lines.
- **Left**: live loss / accuracy / gradient-norm charts, iterations per second, and a learning preview (classification results, sequence reversal, diffusion: original → noised → restored).
- **Right**: click **● Step-through** and the backend records the next iteration line by line, then replays it in slow motion: the code panel highlights the current line with its comment, and below it shows the tensors that line produced (shape, mean ± std, range, distribution) plus the call stack. Space = play / pause, ← → = step.
- **Why it stays smooth**: training runs in a background thread; metrics are batched every 100 ms (≤ 10 pushes/s, automatically thinned); charts are drawn on Canvas; line tracing is only on for the step being explained.
- **Export ONNX**: top-right button, saved to `exports/<task>.onnx` (needs `pip install -r requirements-export.txt`).

Frontend development (React + Vite, needs Node.js):

```powershell
.\scripts\run.ps1 -Task ui-dev     # backend :8765 + Vite hot reload :5173
.\scripts\run.ps1 -Task build-ui   # rebuild web\dist
```

## Bring your own data

Switch to **数据导入与分析 (Data import & analysis)** at the top. Type a local path, click *Upload files* / *Upload folder*, or drag files or folders into the box:

| Data | Formats | Automatic processing |
|---|---|---|
| Tables | CSV / TSV / TXT / JSON / JSONL / Excel (.xlsx, no pandas needed) / .xls\* / Parquet\* | Column types detected: numeric (missing → mean), categorical (one-hot), text (hashed bag of words), ID / constant (dropped); last column is the target by default, selectable |
| Images | A whole folder or a zip | Center-cropped and resized to 16–64 px, grayscale / color detected → ViT |
| Audio | `.wav` (PCM / float); `.flac .ogg .mp3` need `pip install soundfile` | 56 acoustic features → MLP; numeric labels in metadata enable regression |
| Arrays | `.npz` / `.npy`: `X`+`y`, `x_train/y_train/x_test/y_test`, `X.npy`+`y.npy` in a folder, or one 2-D array (last column = target) | 2-D → MLP; image-shaped → ViT |

\* .xls / Parquet need `pip install -r requirements-data.txt`.

**Where image / audio labels come from** (detected automatically in this order and shown in the UI):

1. **Metadata table**: any CSV / Excel / JSON in the folder with a file-name column (`img001.jpg`, `images/img001.jpg` or without extension) and label columns, e.g. `labels.csv`: `filename,breed,weight`. Switch the label column in the UI.
2. **Class sub-folders**: `data\cat\*.jpg`, `data\dog\*.jpg`; `train\ val\ test\` with classes inside also work.
3. **File-name prefix**: `cat_001.jpg`, `dog.12.jpg` → `cat` / `dog`.

Folders uploaded one after another are merged into the *current dataset*: upload the cat folder, then the dog folder = 2 classes. Uploads are stored in `data/uploads/<batch>/`.

**Analysis**: sample count and train / validation split, target distribution, class-imbalance warning, per-column type / missing values / statistics / distribution, feature–target relevance (correlation ratio η² for classification, |Pearson r| for regression), 2-D PCA projection, sample preview.

**Training**: *Train with this data* creates a *custom* task automatically (vectors → MLP, images → ViT; cross-entropy for classification, MSE for regression). The network graph, step-through and five-stage pipeline all work as usual.

**Model evaluation**: accuracy / R²·MAE·RMSE, confusion matrix, per-class precision / recall / F1, permutation feature importance, and the worst mistakes.

> If the page says the backend is outdated: close the PowerShell window running the backend (or Ctrl+C), run `.\scripts\run.ps1 -Task ui` again and refresh the page.

## Neural memory store

Bottom-right of the training page: **🧠 机器记忆库 (memory store)**. Every N steps (default: 1/40 of the run) the network's current state is written to a local database, `data/memory/neuro_memory.db` (SQLite, built into Python). Weights are stored in `data/memory/ckpt/<run>/latest.pt` and `best.pt`.

| Table | Contents |
|---|---|
| `runs` | One row per training run: task, dataset, hyper-parameters, which snapshot it continued from, best validation score |
| `snapshots` | Per snapshot: loss, train / validation accuracy, gradient norm, learning rate, total weight change, checkpoint |
| `layer_states` | Per layer: weight norm, gradient norm, change since the previous snapshot |
| `samples` | Sample metadata: source file, label, train / validation |
| `embeddings` | Per-sample feature vector (penultimate layer, float32) + prediction + confidence + correct or not |
| `sample_memory` | A "mistake notebook" accumulated across runs: times evaluated, times wrong, smoothed loss |

**How the next run uses this memory** (options in the panel):

- **Start from the best memory / a chosen snapshot**: loads weights + AdamW moments + step count and keeps learning instead of starting from zero;
- **Hard-example replay**: samples are drawn with weights from the mistake notebook, so past mistakes get more practice;
- **Image augmentation**: random horizontal flips + shifts reduce memorization on small datasets (100 % train accuracy with low validation accuracy is exactly this problem);
- The snapshot with the **highest validation accuracy** is kept as *best*, so you can roll back after overfitting.

The panel shows growth curves (runs compared + memory lineage), the feature-vector space (PCA projection per snapshot + separation score + raw records), a per-layer change heat map, the mistake notebook, the table schema and a read-only SQL console.

## Online deployment

The UI runs on **Vercel**, the PyTorch training backend on **Hugging Face Spaces** (Docker); both redeploy automatically after a push to GitHub. See [DEPLOY.en.md](DEPLOY.en.md).

```powershell
.\scripts\publish_github.ps1 -User your-github-name     # pushes to github.com/your-github-name/Visual-Network-Neuron
```

## Python usage

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

unet = nc.build_model("unet", in_channels=3, base_channels=64)      # diffusion denoiser
seg = nc.build_model("unet_seg", in_channels=3, num_classes=21)     # segmentation
```

Registered models: `transformer`, `transformer_encoder`, `vit`, `vit_tiny/small/base`, `unet`, `unet_seg`, `dit`, `dit_s_2/b_2/xl_2`, `mlp`.

Attention is written out explicitly for clarity; pass `use_sdpa=True` for PyTorch's fused kernel (the tests verify both give the same result).

## Adding a new domain

1. Copy `neurocore/extensions/_template.py`, e.g. to `gnn.py` (drop the underscore).
2. Register it with `@register_model("my_net", domain="graph")`.
3. It loads automatically on `import neurocore`; use `nc.build_model("my_net")`. A broken extension only produces a warning and never breaks the core.

Reserved domain tags: `nlp, vision, generative, graph, audio, multimodal, timeseries, recsys, security, rl, basic, other`.

## Layout

```
NeuroCore/
├─ deploy.ps1                  one command: check → setup → demo
├─ DEPLOY.md                   online deployment (Vercel + Hugging Face)
├─ Dockerfile · vercel.json    cloud deployment config
├─ scripts/
│  ├─ common.ps1               shared helpers (find Python, read GPU, pick wheel index)
│  ├─ check_env.ps1            environment check
│  ├─ setup_env.ps1            setup (.venv + PyTorch + dependencies + tests)
│  ├─ run.ps1                  demo / test / list / info / ui / build-ui
│  ├─ publish_github.ps1       push to GitHub in one step
│  └─ vercel_build.mjs         Vercel build step
├─ neurocore/
│  ├─ registry.py              model registry
│  ├─ layers/                  attention, embeddings
│  ├─ models/                  transformer / vit / unet / dit / mlp
│  ├─ diffusion/               DDPM
│  ├─ export.py                ONNX export
│  └─ extensions/              future domain modules (auto-discovered)
├─ server/                     visualization backend (Starlette + SSE)
│  ├─ tasks.py                 training tasks (commented line by line, traced in the UI)
│  ├─ trainer.py               background training thread, pause / resume / tracing / memory
│  ├─ tracer.py                sys.settrace line tracer
│  ├─ graph.py · pipeline.py   network graph and five-stage pipeline
│  ├─ datasets.py · evaluate.py data import & analysis, model evaluation
│  ├─ memory.py                neural memory store (SQLite)
│  ├─ config.py                cloud settings (token, CORS, data directory)
│  ├─ hub.py                   throttled push (≤ 10/s)
│  └─ app.py                   HTTP API + UI
├─ web/                        React UI (web/dist is the build output)
├─ tests/                      model / backend / data / memory tests
└─ demo.py
```
