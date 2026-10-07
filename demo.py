"""
NeuroCore 演示 / demo
=====================
对每个网络做：构建 -> 前向形状检查 -> 若干步玩具训练（loss 应下降）-> 推理
For every network: build -> shape check -> a few toy training steps -> inference.

  python demo.py                 # 全部 / all
  python demo.py --model vit     # 单个 / one of: transformer vit unet dit
  python demo.py --list          # 列出已注册模型 / list registered models
"""
import argparse
import time

import torch
import torch.nn.functional as F

import neurocore as nc
from neurocore.diffusion import GaussianDiffusion


def pick_device(pref: str):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(model, step_fn, steps, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    first = last = None
    for i in range(steps):
        loss = step_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    return first, last


def demo_transformer(dev, steps):
    """复制任务：输出 = 输入 / copy task"""
    V, PAD, BOS, EOS, L = 20, 0, 1, 2, 8
    m = nc.build_model("transformer", src_vocab_size=V, tgt_vocab_size=V, dim=64, depth=2,
                       num_heads=4, dropout=0.0).to(dev)

    def batch():
        body = torch.randint(3, V, (32, L), device=dev)
        src = body
        tgt_in = torch.cat([torch.full((32, 1), BOS, device=dev), body], 1)
        tgt_out = torch.cat([body, torch.full((32, 1), EOS, device=dev)], 1)
        return src, tgt_in, tgt_out

    def step():
        s, ti, to = batch()
        logits = m(s, ti)
        return F.cross_entropy(logits.reshape(-1, V), to.reshape(-1))

    out = m(*batch()[:2])
    assert out.shape == (32, L + 1, V), out.shape
    f, l = train(m, step, steps)
    src = batch()[0][:2]
    gen = m.greedy_decode(src, BOS, EOS, max_new_tokens=L + 1)
    print(f"  src : {src[0].tolist()}\n  gen : {gen[0, 1:L + 1].tolist()}")
    return m, f, l


def demo_vit(dev, steps):
    """区分两类合成图像 / two synthetic classes: bright-left vs bright-right"""
    m = nc.build_model("vit", img_size=32, patch_size=4, num_classes=2, dim=64, depth=2, num_heads=4).to(dev)

    def batch():
        y = torch.randint(0, 2, (32,), device=dev)
        x = torch.randn(32, 3, 32, 32, device=dev) * 0.5
        x[y == 0, :, :, :16] += 1.0
        x[y == 1, :, :, 16:] += 1.0
        return x, y

    assert m(batch()[0]).shape == (32, 2)
    f, l = train(m, lambda: _ce(m, batch), steps)
    x, y = batch()
    m.eval()
    acc = (m(x).argmax(-1) == y).float().mean().item()
    print(f"  toy accuracy: {acc:.2%}")
    return m, f, l


def _ce(m, batch):
    x, y = batch()
    return F.cross_entropy(m(x), y)


def _toy_images(dev, n=16, size=16):
    """合成数据：随机位置的方块，归一化到 [-1,1] / random squares in [-1,1]"""
    x = -torch.ones(n, 1, size, size, device=dev)
    for i in range(n):
        r, c = torch.randint(0, size - 6, (2,)).tolist()
        x[i, :, r:r + 6, c:c + 6] = 1.0
    return x


def demo_unet(dev, steps):
    diff = GaussianDiffusion(timesteps=100).to(dev)
    m = nc.build_model("unet", in_channels=1, base_channels=32, channel_mults=(1, 2),
                       num_res_blocks=1, attn_levels=(1,)).to(dev)
    x = _toy_images(dev)
    t = torch.randint(0, 100, (x.size(0),), device=dev)
    assert m(x, t).shape == x.shape
    f, l = train(m, lambda: diff.training_loss(m, _toy_images(dev)), steps, lr=2e-3)
    s = diff.sample(m, (2, 1, 16, 16))
    print(f"  sample range: [{s.min():.2f}, {s.max():.2f}]")
    seg = nc.build_model("unet_seg", in_channels=3, num_classes=5, base_channels=16, channel_mults=(1, 2)).to(dev)
    assert seg(torch.randn(2, 3, 32, 32, device=dev)).shape == (2, 5, 32, 32)
    print("  unet_seg output OK: (2, 5, 32, 32)")
    return m, f, l


def demo_dit(dev, steps):
    diff = GaussianDiffusion(timesteps=100).to(dev)
    m = nc.build_model("dit", img_size=16, patch_size=2, in_channels=1, dim=64, depth=2,
                       num_heads=4, num_classes=2).to(dev)

    def loss():
        x = _toy_images(dev)
        y = torch.randint(0, 2, (x.size(0),), device=dev)
        return diff.training_loss(m, x, y)

    x = _toy_images(dev, 4)
    assert m(x, torch.zeros(4, dtype=torch.long, device=dev), torch.zeros(4, dtype=torch.long, device=dev)).shape == x.shape
    f, l = train(m, loss, steps, lr=2e-3)
    s = diff.sample(m, (2, 1, 16, 16), y=torch.tensor([0, 1], device=dev), cfg_scale=2.0)
    print(f"  CFG sample range: [{s.min():.2f}, {s.max():.2f}]")
    return m, f, l


DEMOS = {"transformer": demo_transformer, "vit": demo_vit, "unet": demo_unet, "dit": demo_dit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", *DEMOS])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for s in nc.list_models():
            print(f"  [{s.domain:<10}] {s.name:<20} {s.description}")
        return

    dev = pick_device(a.device)
    torch.manual_seed(0)
    print(f"NeuroCore {nc.__version__} | torch {torch.__version__} | device: {dev}")
    if dev.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    names = list(DEMOS) if a.model == "all" else [a.model]
    ok = True
    for n in names:
        print(f"\n=== {n.upper()} ===")
        t0 = time.time()
        m, f, l = DEMOS[n](dev, a.steps)
        status = "PASS" if l < f else "WARN (loss did not drop)"
        ok &= l < f
        print(f"  params: {nc.count_parameters(m):,} | loss {f:.4f} -> {l:.4f} | {time.time() - t0:.1f}s | {status}")
    print("\nALL DEMOS PASSED" if ok else "\nDONE (with warnings)")


if __name__ == "__main__":
    main()
