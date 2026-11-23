# Missing-pattern-shift experiment implementation

The official source repositories are pinned under `baselines/`:

| baseline | repository revision | experiment adapter |
|---|---|---|
| S4M | `a718823addd3606e763dfc261174e0135b2535f4` | official `model/S4M.py`; CUDA S4M environment |
| BiTGraph | `4f1d05bcc20bb3f5084bd1facc995478f84a40ed` | official `models/BiaTCGNet/BiaTCGNet.py` |
| ChannelTokenFormer | `b1c100effb9c27cd25de07e91de7f9005d18d5e7` | official `models/ChannelTokenFormer_missing.py`, `keep_prob=1` |

For CTF-strict, `keep_prob=1.0` is deliberately the *no additional patch
masking* setting: the official `layers/CTF_Embed.py` skips a training patch
only when `torch.rand() > keep_prob`, which is impossible at one.  The sole
training corruption is therefore this experiment's point mask.

`scripts/protocol.py` is model-free: it fixes time splits, training-only scaling,
exact 30% P/B masks on history only, complete future targets, fixed mask seeds,
and atomic artifacts.  Per-split mask audits record actual missing rate,
continuous-run length and count, and the affected-variable fraction.
`scripts/run_official_baseline.py` imports the official
model objects without changing their architectures. The only compatibility edit
inside the S4M clone makes two unused imports optional; its Bio.Cluster
memory-bank algorithm is unchanged.

Results are atomically saved beneath `results/primary/`; checkpoints are in
`checkpoints/primary/`; worker and hourly progress are in `logs/primary/`.
Run `scripts/aggregate_primary.py` to regenerate the per-mask results,
condition-level summary, matched P→B-versus-B→B pairs, and the compact
`summary/primary_p_to_b_delta_summary.csv` (mean ΔMAE/ΔMSE plus bootstrap
95% CIs and paired Wilcoxon signed-rank statistics).  The complete three-effect
2×2 decomposition is also saved as `summary/primary_2x2_effect_pairs.csv` and
`summary/primary_2x2_effect_summary.csv`: P→B penalty, block-intrinsic
difficulty, and reverse B→P penalty.  The hourly monitor separately records completed test evaluations
and completed training runs, so its progress count is not conflated with
checkpoint/progress metadata; its aggregation cycle also refreshes these
figures after saving the new snapshot.  `summary/primary_runtime_per_run.csv`
and `summary/primary_runtime_summary.csv` separately retain training/evaluation
elapsed time and, for newly launched processes, peak allocated/reserved CUDA
memory.  New evaluation artifacts also retain raw-unit MAE/MSE at each forecast
step; their condition summaries are written to `summary/primary_per_step.csv`
and `summary/primary_per_step_summary.csv`.  Results written before this
instrumentation remain valid for aggregate metrics and leave these optional
tables empty rather than being silently recomputed.

`summary/primary_completion.csv` compares registered and completed test
evaluations for each baseline/dataset/horizon condition; it is the authoritative
progress table for determining whether a reported condition is final.
