"""DeepSeek-V3 MoE (default) + Mixtral token-choice top-k variant.

Default (`MoE`): sigmoid affinities, L1-normalized gates on the selected experts,
shared + routed experts, aux-loss-free bias used **only** for top-k selection
(DeepSeek-V3 §2.1.2). Bias update: b_i -= γ * sign(load_i - mean).

Named variant (`MixtralMoE`): softmax over the selected top-k logits + Switch /
Fedus auxiliary load-balancing loss.

Neither algorithm drops tokens (DeepSeek-V3 §"No Token-Dropping"; Mixtral has no
capacity factor). Experts are SwiGLU FFNs; dispatch subsets tokens so unused
experts get no forward / no gradient.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(u) * v)


def switch_aux_loss(router_logits: torch.Tensor, dispatch_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Centered Switch / Fedus aux loss: N * sum_i f_i P_i - 1.

    Raw Switch is 1 when f = P = 1/N and N when one expert takes all with one-hot P.
    Subtracting the balanced floor makes the perfectly-balanced case 0, as the tests require.
    f_i is the fraction of (token, slot) assignments; P_i is mean softmax probability.
    """
    probs = torch.softmax(router_logits, dim=-1)
    p = probs.reshape(-1, n_experts).mean(dim=0)
    flat = dispatch_idx.reshape(-1)
    f = torch.bincount(flat, minlength=n_experts).float() / flat.numel()
    return n_experts * (f * p).sum() - 1.0


def _combine_routed(
    x: torch.Tensor,
    experts: nn.ModuleList,
    idx: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """idx, weights: (..., k). Only tokens assigned to expert e go through expert e."""
    prefix = x.shape[:-1]
    d = x.size(-1)
    k = idx.size(-1)
    x_flat = x.reshape(-1, d)
    idx_flat = idx.reshape(-1, k)
    w_flat = weights.reshape(-1, k)
    out = torch.zeros_like(x_flat)
    for e, expert in enumerate(experts):
        slot_hit = idx_flat == e
        if not slot_hit.any():
            continue
        token_mask = slot_hit.any(dim=-1)
        y = expert(x_flat[token_mask])
        w = (w_flat * slot_hit.float()).sum(dim=-1)[token_mask].unsqueeze(-1)
        out[token_mask] = out[token_mask] + y * w
    return out.view(*prefix, d)


class MoE(nn.Module):
    """DeepSeek-V3 MoE: sigmoid affinity, bias-only-for-topk, L1 gates, shared+routed.

    Paper (eqs. 12–16):
      s_{i,t} = sigmoid(u_t^T e_i)
      select top-k on (s_{i,t} + b_i); gates from unbiased s, L1-renormalized
      h' = u + sum_shared FFN^s(u) + sum_i g_i FFN^r_i(u)
      after step: b_i -= γ * sign(load_i - mean_load)
    """

    def __init__(
        self,
        d_model: int,
        n_routed: int,
        d_ff: int,
        k: int = 2,
        n_shared: int = 1,
        gamma: float = 0.001,
    ):
        super().__init__()
        self.n_routed = n_routed
        self.k = k
        self.gamma = gamma
        self.router = nn.Linear(d_model, n_routed, bias=False)
        self.register_buffer("bias", torch.zeros(n_routed))
        self.routed = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_routed)])
        self.shared = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_shared)])

    def forward(self, x: torch.Tensor, update_bias: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(self.router(x))
        _, idx = (scores + self.bias).topk(self.k, dim=-1)
        selected = scores.gather(-1, idx)
        weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        routed = _combine_routed(x, self.routed, idx, weights)
        shared = sum((ex(x) for ex in self.shared), start=torch.zeros_like(x))
        if update_bias and self.training:
            self.update_bias(idx)
        # eq. (12): residual + shared + gated routed
        return x + shared + routed, idx

    @torch.no_grad()
    def update_bias(self, idx: torch.Tensor) -> None:
        load = torch.bincount(idx.reshape(-1), minlength=self.n_routed).float()
        mean = load.mean()
        self.bias -= self.gamma * torch.sign(load - mean)


class MixtralMoE(nn.Module):
    """Named variant: Mixtral token-choice top-k + Switch aux loss.

    Softmax over the *selected* expert logits only (not full N). No capacity
    dropping (Mixtral does not use a capacity factor). Optional shared experts
    for A/B with DeepSeek; Mixtral-8x7B itself has no shared experts.
    """

    def __init__(self, d_model: int, n_experts: int, d_ff: int, k: int = 2, n_shared: int = 0):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_experts)])
        self.shared = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_shared)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        top_logits, idx = logits.topk(self.k, dim=-1)
        weights = torch.softmax(top_logits, dim=-1)
        routed = _combine_routed(x, self.experts, idx, weights)
        shared = sum((ex(x) for ex in self.shared), start=torch.zeros_like(x))
        aux = switch_aux_loss(logits, idx, self.n_experts)
        return routed + shared, idx, aux
