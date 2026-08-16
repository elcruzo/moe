"""DeepSeek-V3 MoE (default) vs Mixtral aux-loss variant on toy tokens."""
from __future__ import annotations

import torch

from moe import MixtralMoE, MoE


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    torch.manual_seed(0)
    device = _device()
    x = torch.randn(2, 8, 16, device=device)

    # Default = DeepSeek-V3 bias-only-for-topk (sigmoid + L1 gates).
    ds = MoE(16, n_routed=4, d_ff=32, k=2, n_shared=1, gamma=0.01).to(device)
    ds.train()
    y, idx = ds(x)
    print("default DeepSeekMoE", tuple(y.shape), "idx", idx[0, 0].tolist(), "device", device.type)
    for _ in range(20):
        ds(torch.randn_like(x))
    print("deepseek bias after load updates", [round(float(b), 4) for b in ds.bias.tolist()])

    # Named variant: Mixtral top-k + Switch aux loss.
    mix = MixtralMoE(16, n_experts=4, d_ff=32, k=2, n_shared=0).to(device)
    ym, idm, aux = mix(x)
    print("variant MixtralMoE", tuple(ym.shape), "idx", idm[0, 0].tolist(), "aux", float(aux.detach()))


if __name__ == "__main__":
    main()
