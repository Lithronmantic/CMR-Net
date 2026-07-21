# Reproducibility

## Environment

```bash
pip install -r requirements.txt
```

## Configuration

Use `configs/welding_inference.yaml` for the robotic-welding evaluation.

```bash
python scripts/validate_checkpoint.py \
  --config configs/welding_inference.yaml \
  --checkpoint /path/to/checkpoint.pt
```

## Evaluation protocol

The certificate procedure uses five training seeds, a fixed group-aware data split, 10000 paired-bootstrap replicates, 1000 CHSIC permutations, and leave-one-seed-out stability analysis.

| Parameter | Value |
|---|---:|
| Predictive Non-Inferiority margin `m` | 0.03 |
| Residual-Pathway Superiority margin `m_sup` | 0.03 |
| CHSIC practical-equivalence width `delta` | `5e-5` |
| Minimum family size `n_min` | 40 |
| Paired-bootstrap replicates `B` | 10000 |
| CHSIC permutations | 1000 |

## Command

```bash
python scripts/certificate_diagnostics.py \
  --config configs/welding_inference.yaml \
  --checkpoint \
    /path/to/seed0.pt \
    /path/to/seed1.pt \
    /path/to/seed2.pt \
    /path/to/seed3.pt \
    /path/to/seed4.pt \
  --split test \
  --n-bootstrap 10000 \
  --n-permutation 1000 \
  --min-family-n 40 \
  --f1-equivalence-margin 0.03 \
  --direct-gap-margin 0.03 \
  --equivalence-delta 5e-5 \
  --output-dir results/certificate
```

The generated files are `family_certificates.json` and `family_certificates.md`.
