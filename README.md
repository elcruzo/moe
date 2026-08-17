# 09-moe — DeepSeek-V3 Mixture of Experts

Implements **DeepSeek-V3** MoE routing (default: `MoE`).

Affinity $s_{i,t}=\sigma(u_t^\top e_i)$. Bias $b_i$ is added **only** for top-k selection, never for the gate values (eq. 16). Gates are the selected sigmoid scores L1-renormalized (eqs. 13–15). Output is residual + shared experts + gated routed experts (eq. 12). After each training step $b_i \leftarrow b_i - \gamma\,\mathrm{sign}(\mathrm{load}_i-\overline{\mathrm{load}})$. No token dropping (DeepSeek-V3 keeps load balance without a capacity factor).

## Named variant: Mixtral + aux loss

`MixtralMoE`: linear router, softmax **only over the selected** top-k logits, Switch / Fedus auxiliary loss $N\sum_i f_i P_i$ centered by $-1$ (balanced → 0, collapsed → $>0$). Mixtral has no capacity factor / token drop either.

Experts are SwiGLU: $W_3(\mathrm{SiLU}(xW_1)\odot xW_2)$. Tokens are indexed into chosen experts (unused experts are not called → `None`/zero grad).

## Papers on disk

- [`papers/deepseek-v3-2024.pdf`](papers/deepseek-v3-2024.pdf) — DeepSeek-AI. DeepSeek-V3 Technical Report (2024) ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
- [`papers/jiang-mixtral-2024.pdf`](papers/jiang-mixtral-2024.pdf) — Jiang et al. Mixtral of Experts (2024) ([arXiv:2401.04088](https://arxiv.org/abs/2401.04088))
- [`papers/fedus-switch-transformers-2021.pdf`](papers/fedus-switch-transformers-2021.pdf) — Fedus et al. Switch Transformers (2021) ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961))

## Compared to DeepSeek-V3 MoE

**What you learn here:**
- Sigmoid affinity + bias **only** for top-k (gates from unbiased scores)
- Aux-loss-free load balance via $b_i \leftarrow b_i - \gamma\,\mathrm{sign}(\mathrm{load}_i-\overline{\mathrm{load}})$
- Named Mixtral variant with Switch aux loss (no token drop)

| | This repo | DeepSeek-V3 |
|---|---|---|
| Scale | Toy `n_routed=8`, `k=2`, `d=16` | 256 routed, top-8, 671B tot / 37B act |
| Experts | SwiGLU | DeepSeekMoE + MLA |
| Balance | Bias update | Same aux-loss-free idea |

### Numbers (2026-08-16, Darwin 25.5.0 arm64 / Apple M5)

| Metric | This repo | Baseline | Source |
|---|---|---|---|
| Activated param frac | ~0.34 (k=2/8 + shared) | 37B/671B ≈ 0.055 | DeepSeek-V3 report; `MoE` param count |
| Mixtral aux (demo) | 0.042 | 0 when balanced | `python main.py` |
| Bias after load | non-zero ±γ | — | same |

```bash
python main.py
```

## Run

```bash
python main.py
python -m pytest 09-moe -q
```
