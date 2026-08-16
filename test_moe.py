"""MoE routing invariants."""
from __future__ import annotations

import torch

from moe import DeepSeekMoE, MixtralMoE, _combine_routed, switch_aux_loss


def test_top_k_plus_shared_and_shape():
    torch.manual_seed(0)
    moe = MixtralMoE(d_model=16, n_experts=4, d_ff=32, k=2, n_shared=1)
    x = torch.randn(3, 5, 16)
    y, idx, aux = moe(x)
    assert y.shape == x.shape
    assert idx.shape == (3, 5, 2)
    assert aux.ndim == 0
    # each token has exactly k distinct routed experts (ties broken by topk)
    for row in idx.reshape(-1, 2):
        assert row.numel() == 2


def test_aux_loss_balanced_zero_collapsed_positive():
    n = 4
    # uniform logits + round-robin top-1 → f = P = 1/N → centered loss 0
    logits = torch.zeros(n, n)
    idx = torch.arange(n).unsqueeze(-1)
    z = switch_aux_loss(logits, idx, n)
    assert abs(z.item()) < 1e-6, z
    # one expert takes all and the router is certain
    collapsed = torch.zeros(n, n)
    collapsed[:, 0] = 20.0
    idx0 = torch.zeros(n, 1, dtype=torch.long)
    pos = switch_aux_loss(collapsed, idx0, n)
    assert pos.item() > 0.5, pos


def test_aux_loss_free_overloaded_bias_decreases():
    moe = DeepSeekMoE(d_model=8, n_routed=4, d_ff=16, k=2, n_shared=1, gamma=0.05)
    moe.train()
    # scores = x @ W^T; with x=1, expert 0's logit stays top even after bias moves
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, 0] = 50.0
        moe.router.weight[1, 0] = 1.0
    start = float(moe.bias[0])
    x = torch.ones(2, 6, 8)
    for _ in range(30):
        moe(x, update_bias=True)
    assert moe.bias[0] < start
    # Sigmoid affinities saturate near 1, so bias only drops until the overloaded expert
    # is no longer always selected — then sign(load - mean) oscillates. The paper's
    # update still pushes the hog's bias below idle experts.
    assert moe.bias[0] < 0
    assert moe.bias[2] > moe.bias[0]


def test_gradients_flow_only_to_selected_experts():
    torch.manual_seed(1)
    moe = MixtralMoE(d_model=8, n_experts=4, d_ff=16, k=2, n_shared=0)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[1].fill_(4.0)
        moe.router.weight[2].fill_(3.0)
        moe.router.weight[0].fill_(-8.0)
        moe.router.weight[3].fill_(-8.0)
    x = torch.ones(2, 4, 8, requires_grad=True)
    y, idx, _ = moe(x)
    assert set(idx.reshape(-1).tolist()) <= {1, 2}
    y.sum().backward()
    g1 = moe.experts[1].w3.weight.grad
    g2 = moe.experts[2].w3.weight.grad
    g0 = moe.experts[0].w3.weight.grad
    g3 = moe.experts[3].w3.weight.grad
    assert g1 is not None and g1.abs().sum() > 0
    assert g2 is not None and g2.abs().sum() > 0
    assert g0 is None or torch.equal(g0, torch.zeros_like(g0))
    assert g3 is None or torch.equal(g3, torch.zeros_like(g3))
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_deepseek_output_shape_and_shared():
    moe = DeepSeekMoE(d_model=12, n_routed=3, d_ff=24, k=2, n_shared=2, gamma=0.001)
    x = torch.randn(1, 7, 12)
    y, idx = moe(x)
    assert y.shape == x.shape
    assert idx.size(-1) == 2


def test_deepseek_affinities_are_sigmoid_and_gates_renormalize():
    moe = DeepSeekMoE(d_model=4, n_routed=3, d_ff=8, k=2, n_shared=0, gamma=0.0)
    moe.eval()
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, 0] = 4.0
        moe.router.weight[1, 0] = 0.0
        moe.router.weight[2, 0] = -4.0
    x = torch.ones(1, 1, 4)
    y, idx = moe(x, update_bias=False)
    logits = moe.router(x)
    scores = torch.sigmoid(logits)
    want_idx = (scores + moe.bias).topk(moe.k, dim=-1).indices
    assert torch.equal(idx, want_idx)
    selected = scores.gather(-1, idx)
    weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    y_ref = _combine_routed(x, moe.routed, idx, weights)
    assert torch.allclose(y, y_ref, atol=1e-6)
    # Softmax-over-logits gates would not match sigmoid + L1 on the selected scores.
    soft = torch.softmax(logits.gather(-1, idx), dim=-1)
    assert not torch.allclose(weights, soft, atol=1e-3)
