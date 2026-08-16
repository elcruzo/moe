# 09-moe — Mixture of Experts routing

Two routers, one file.

1. **Mixtral-style token-choice top-k** (k=2): linear router, softmax **only over the selected experts**, weighted sum of expert FFNs. Switch / Fedus et al. auxiliary load-balancing loss \(N\sum_i f_i P_i\), centered by \(-1\) so a perfectly balanced uniform router scores 0 and a collapsed router scores \(>0\).
2. **DeepSeek-V3 auxiliary-loss-free balancing**: affinity \(s_{i,t}=\sigma(u_t^\top e_i)\). Bias \(b_i\) is added **only** for top-k selection, never for the gate values. Gates are the selected sigmoid scores renormalized to sum to 1. After each training step \(b_i \leftarrow b_i - \gamma\,\mathrm{sign}(\mathrm{load}_i-\overline{\mathrm{load}})\). Shared expert(s) always run; routed experts are top-k.

Experts are SwiGLU: \(W_3(\mathrm{SiLU}(xW_1)\odot xW_2)\). Tokens are actually indexed into the chosen experts (unused experts are not called, so they get `None`/zero grad).

Papers: Fedus et al. *Switch Transformers* (2021); Jiang et al. Mixtral (2024); DeepSeek-V3 technical report (2024), §2.1.1 / eqs. for \(g'_{i,t}\).

## Papers on disk

- [`papers/fedus-switch-transformers-2021.pdf`](papers/fedus-switch-transformers-2021.pdf) — Fedus et al. Switch Transformers (2021) ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961))
- [`papers/jiang-mixtral-2024.pdf`](papers/jiang-mixtral-2024.pdf) — Jiang et al. Mixtral of Experts (2024) ([arXiv:2401.04088](https://arxiv.org/abs/2401.04088))
- [`papers/deepseek-v3-2024.pdf`](papers/deepseek-v3-2024.pdf) — DeepSeek-AI. DeepSeek-V3 Technical Report (2024) ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))

## Run

```bash
python demo.py
python -m pytest 09-moe -q
```
