# Reproducibility

## Environment

```bash
pip install -e ".[dev]"        # Python >= 3.9, PyTorch >= 2.1
```

## Seeds and splits

- Training seeds: `0 1 2 3 4` (five seeds).
- Data split seed: `42` (fixed; group-aware split by top-level process unit, so
  no process unit appears in more than one of train/val/test).

## Train / evaluate

```bash
python scripts/train.py
python scripts/evaluate.py
python scripts/run_ablations.py           # ablation + baseline matrix
python scripts/run_ablations.py --smoke   # fast CPU smoke run (~1 min)
pytest
```

## Certificate protocol (frozen paper settings)

See `docs/certificate_protocol.md`. The family-level certificate uses these
pre-evaluation frozen parameters:

| parameter | value | flag |
|---|---|---|
| non-inferiority margin `m` | 0.03 | `--f1-equivalence-margin` |
| direct-superiority margin `m_sup` | 0.03 | `--direct-gap-margin` |
| CHSIC equivalence width `delta` | 5e-5 | `--equivalence-delta` |
| min family size `n_min` | 40 | `--min-family-n` |
| paired bootstrap `B` | 10000 | `--n-bootstrap 10000` |
| CHSIC permutations | 1000 | `--n-permutation` |

> The published certificates were produced on a multi-GPU server with
> `--n-bootstrap 10000`. The script's CLI default is lower for quick local
> runs — pass `--n-bootstrap 10000` to reproduce the paper numbers.
