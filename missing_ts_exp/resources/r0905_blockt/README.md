# R0905 Block-T Experiment Resources

This directory contains fixed resources for the current MR-MSGF experiment round.

## Scope

- Main missing pattern: `block_t`
- Auxiliary missing pattern: `point`
- Excluded from this round: `block_st`

The experiment targets the setting where temporal observations at a node are missing, while spatial neighbors and regional context remain available for compensation.

## Files

- `manifest.csv`: canonical mask manifest.
- `manifest.json`: JSON copy of the same manifest.
- `masks/METR-LA/*.npy`: fixed observed masks for METR-LA.
- `masks/AQI36/*.npy`: fixed observed masks for AQI36 original and +Point settings.

Mask convention:

```text
1 = observed
0 = artificially missing
```

Dataset natural missingness is still handled by each dataset's validity mask. The effective input mask is:

```text
valid_mask * observed_mask
```

## Prepared Grid

| Dataset | Modes | Rates | Seeds |
|---|---|---|---|
| METR-LA | `point`, `block_t` | 30%, 50%, 70% | 2024, 2025, 2026, 2027, 2028 |
| AQI36 | `original`, `point` | original, 25% point, 50% point | 2024, 2025, 2026, 2027, 2028 |

## Commands

Generated command files:

```text
missing_ts_exp/scripts/r0905_blockt_smoke_cmds.txt
missing_ts_exp/scripts/r0905_blockt_main_cmds.txt
missing_ts_exp/scripts/r0905_blockt_ablation_cmds.txt
```

Launcher:

```bash
bash missing_ts_exp/scripts/r0905_run_blockt.sh missing_ts_exp/scripts/r0905_blockt_smoke_cmds.txt 0 1 missing_ts_exp/logs/r0905_blockt_smoke_ok
```

The smoke test has passed for:

```text
METR-LA / block_t / 50% / seed 2024
official_base
official_reliable
```
