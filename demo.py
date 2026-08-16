"""Token-choice MoE vs aux-loss-free DeepSeek routing on random tokens."""
from __future__ import annotations

import torch

from moe import DeepSeekMoE, MixtralMoE


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)
    mix = MixtralMoE(16, n_experts=4, d_ff=32, k=2, n_shared=1)
    y, idx, aux = mix(x)
    print("mixtral", tuple(y.shape), "idx", idx[0, 0].tolist(), "aux", float(aux))
    ds = DeepSeekMoE(16, n_routed=4, d_ff=32, k=2, n_shared=1, gamma=0.01)
    ds.train()
    for _ in range(20):
        ds(torch.randn_like(x))
    print("deepseek bias", ds.bias.tolist())


if __name__ == "__main__":
    main()
