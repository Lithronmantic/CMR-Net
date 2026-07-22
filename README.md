# CMR-Net

Certificate-guided Multimodal Residual Network (CMR-Net) provides frozen-checkpoint inference and family-level process-sufficiency certificate evaluation for industrial multimodal defect recognition.

The implementation exposes three prediction pathways:

- Process-Mediated Pathway: `logits_mediated`
- Residual Direct Pathway: `logits_direct`
- Additive Full Pathway: `logits_pathway`

The certificate states are Process-Sufficient (PS), Process-Sufficient with Residual Modality Dependence (PS-RMD), Process-Insufficient (PI), and Indeterminate (IND).

## Repository contents

```text
configs/welding_inference.yaml
scripts/validate_checkpoint.py
scripts/certificate_diagnostics.py
docs/data.md
docs/reproducibility.md
docs/resource_accounting.md
docs/code_map.md
docs/certificate_protocol.md
```

The repository provides inference and certificate evaluation. Model training is not included.

## Installation

```bash
pip install -r requirements.txt
```

Extract the supplied model package so that `src/cmr_net` is available from the repository root. Set the Intel Robotic Welding Multimodal Dataset location:

```bash
export CMR_DATA_ROOT=/path/to/intel_robotic_welding_dataset
```

The validation command checks the configuration dimensions, dual-pathway settings, residual gate, direct-head input dimension, and checkpoint contents.

## Certificate evaluation

Five checkpoints are required to reproduce the cross-seed certificate release procedure:

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

The evaluation uses process-residualized modality embeddings, a two-level paired bootstrap, majority aggregation of CHSIC indicators, leave-one-seed-out stability analysis, and threshold-sensitivity evaluation.

## Data

The Intel Robotic Welding Multimodal Dataset is not redistributed. The expected input interface and defect-family mapping are documented in `docs/data.md`.

## Citation

Please cite the associated article and dataset sources when using this repository.
