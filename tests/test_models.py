"""pytest -q  —— 形状、梯度、注册表、注意力正确性 / shapes, grads, registry, attention correctness"""
import torch
import pytest

import neurocore as nc
from neurocore.layers import MultiHeadAttention
from neurocore.diffusion import GaussianDiffusion

torch.manual_seed(0)


def _grad_ok(model, loss):
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad), "some params got no grad"


def test_registry_lists_core_models():
    names = {s.name for s in nc.list_models()}
    assert {"transformer", "transformer_encoder", "vit", "unet", "unet_seg", "dit"} <= names


def test_attention_matches_sdpa():
    a = MultiHeadAttention(32, 4)
    b = MultiHeadAttention(32, 4, use_sdpa=True)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 7, 32)
    mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3]).bool()
    for causal in (False, True):
        torch.testing.assert_close(a(x, key_padding_mask=mask, is_causal=causal),
                                   b(x, key_padding_mask=mask, is_causal=causal), atol=1e-5, rtol=1e-4)


def test_causal_mask_no_future_leak():
    a = MultiHeadAttention(16, 2).eval()
    x = torch.randn(1, 5, 16)
    y1 = a(x, is_causal=True)
    x2 = x.clone()
    x2[:, -1] += 10.0                               # 改变最后一个 token / perturb last token
    y2 = a(x2, is_causal=True)
    torch.testing.assert_close(y1[:, :-1], y2[:, :-1])


def test_transformer():
    m = nc.build_model("transformer", src_vocab_size=50, tgt_vocab_size=60, dim=32, depth=2, num_heads=4)
    src, tgt = torch.randint(1, 50, (3, 9)), torch.randint(1, 60, (3, 7))
    src[0, -3:] = 0                                  # padding
    out = m(src, tgt)
    assert out.shape == (3, 7, 60)
    _grad_ok(m, out.mean())
    gen = m.greedy_decode(src, bos_id=1, eos_id=2, max_new_tokens=5)
    assert gen.shape[0] == 3 and gen.shape[1] <= 6


def test_encoder_classifier():
    m = nc.build_model("transformer_encoder", vocab_size=50, num_classes=3, dim=32, depth=1, num_heads=4)
    assert m(torch.randint(1, 50, (2, 10))).shape == (2, 3)


def test_vit():
    m = nc.build_model("vit", img_size=32, patch_size=8, num_classes=10, dim=48, depth=2, num_heads=4)
    out = m(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
    _grad_ok(m, out.sum())


@pytest.mark.parametrize("mults,attn", [((1, 2), (1,)), ((1, 2, 2), (2,))])
def test_unet_diffusion(mults, attn):
    m = nc.build_model("unet", in_channels=3, base_channels=16, channel_mults=mults,
                       num_res_blocks=1, attn_levels=attn, num_classes=4)
    x = torch.randn(2, 3, 16, 16)
    out = m(x, torch.tensor([1, 50]), torch.tensor([0, 3]))
    assert out.shape == x.shape
    _grad_ok(m, out.pow(2).mean())


def test_unet_seg():
    m = nc.build_model("unet_seg", in_channels=1, num_classes=3, base_channels=16, channel_mults=(1, 2, 4))
    assert m(torch.randn(1, 1, 32, 32)).shape == (1, 3, 32, 32)


def test_dit_and_zero_init():
    m = nc.build_model("dit", img_size=8, patch_size=2, in_channels=4, dim=32, depth=2, num_heads=4,
                       num_classes=5, learn_sigma=True)
    x = torch.randn(2, 4, 8, 8)
    out = m(x, torch.tensor([3, 9]), torch.tensor([1, 4]))
    assert out.shape == (2, 8, 8, 8)
    assert torch.count_nonzero(out) == 0             # adaLN-Zero: 初始输出为 0 / zero output at init


def test_diffusion_loop_with_dit_and_unet():
    diff = GaussianDiffusion(timesteps=10, schedule="cosine")
    dit = nc.build_model("dit", img_size=8, patch_size=2, in_channels=1, dim=32, depth=1, num_heads=4, num_classes=2)
    unet = nc.build_model("unet", in_channels=1, base_channels=16, channel_mults=(1, 2), num_res_blocks=1)
    x0 = torch.randn(2, 1, 8, 8)
    for m in (dit, unet):
        loss = diff.training_loss(m, x0, torch.tensor([0, 1]) if m is dit else None)
        assert torch.isfinite(loss)
    s = diff.sample(dit, (2, 1, 8, 8), y=torch.tensor([0, 1]), cfg_scale=3.0)
    assert s.shape == (2, 1, 8, 8) and torch.isfinite(s).all()


def test_custom_extension_registration():
    @nc.register_model("tmp_test_net", domain="timeseries")
    class Net(torch.nn.Module):
        def __init__(self, d=4):
            super().__init__()
            self.l = torch.nn.Linear(d, 1)

        def forward(self, x):
            return self.l(x)

    assert nc.build_model("tmp_test_net", d=3)(torch.randn(2, 3)).shape == (2, 1)
    assert any(s.name == "tmp_test_net" for s in nc.list_models("timeseries"))
    del nc.MODEL_REGISTRY["tmp_test_net"]


def test_mlp_activations():
    m = nc.build_model("mlp", in_dim=7, hidden=(16, 12), out_dim=3)
    x = torch.randn(5, 7)
    out, acts = m(x, return_activations=True)
    assert out.shape == (5, 3) and [a.shape[-1] for a in acts] == [7, 16, 12, 3]
