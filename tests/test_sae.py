import torch

from percinterp.sae import TopKSAE


def test_topk_keeps_k_positive_per_token():
    torch.manual_seed(0)
    sae = TopKSAE(d=8, n_latents=32, k=4)
    x = torch.randn(50, 8)
    sae.set_scale(x)
    a = sae.encode(x)
    assert ((a > 0).sum(-1) <= 4).all()
    assert (a >= 0).all()


def test_scale_roundtrip_and_learning():
    torch.manual_seed(0)
    # data on a few sparse directions: an SAE should reconstruct it well
    dirs = torch.randn(16, 12)
    codes = (torch.rand(4000, 16) < 0.1).float() * torch.rand(4000, 16) * 3
    x = codes @ dirs * 5.0
    sae = TopKSAE(d=12, n_latents=64, k=4, dead_tokens=10_000)
    sae.set_scale(x)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    first = None
    for step in range(400):
        loss, stats = sae.loss(x)
        first = stats["fvu"] if first is None else first
        opt.zero_grad()
        loss.backward()
        sae.remove_parallel_grad()
        opt.step()
        sae.normalize_decoder()
    assert stats["fvu"] < 0.5 * first
    # decode returns the original scale
    recon, _, _ = sae(x)
    assert (recon - x).pow(2).sum(-1).mean() < x.pow(2).sum(-1).mean()
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-5)


def test_state_roundtrip():
    sae = TopKSAE(d=6, n_latents=10, k=2)
    sae.set_scale(torch.randn(20, 6))
    clone = TopKSAE.from_state(sae.state())
    x = torch.randn(5, 6)
    assert torch.allclose(sae.encode(x), clone.encode(x))
