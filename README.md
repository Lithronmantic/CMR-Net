# CMR-Net

**Certificate-guided Multimodal Residual Network (CMR-Net)** — a research
codebase for *family-level process-sufficiency certificates* that evaluate,
before deployment, whether audio/video modalities can be reduced for each defect
family in industrial multimodal welding-defect modeling.

CMR-Net decomposes a multimodal prediction into a **Process-Mediated Pathway**
and a **Residual Direct Pathway**, then emits, per defect family, one of four
certificate states — **PS**, **PS-RMD**, **PI**, **IND** — from Predictive
Non-Inferiority, Residual Modality Dependence (CHSIC), paired-bootstrap intervals,
and leave-one-seed-out stability. A deployment-resource accounting maps each state
to a candidate online modality-usage mode (a pre-deployment quantitative
reference, not a production deployment).

## Repository structure

```text
cmr-net/
├── src/cmr_net/        # model, modules, losses, training, evaluation, baselines
├── scripts/            # training, evaluation, certificate, bootstrap, perturbation, ablation, figures
├── configs/            # Hydra configs (default + baselines)
├── tests/              # unit / smoke tests
├── docs/               # design, certificate protocol, data, reproducibility, resource accounting
├── pyproject.toml
└── README.md
```

## Install

```bash
pip install -e ".[dev]"     # Python >= 3.9, PyTorch >= 2.1
```

## Data

The Intel robotic welding dataset is not bundled. See
**[docs/data.md](docs/data.md)** for the expected layout and how to set
`CMR_DATA_ROOT`. 12 fine-grained defect classes aggregate into 6 engineering
families.

## Usage

```bash
python scripts/train.py
python scripts/evaluate.py
python scripts/run_ablations.py            # ablation + baseline matrix
python scripts/run_ablations.py --smoke    # fast CPU smoke run
pytest
```

Certificate, sensitivity, perturbation and resource scripts live in `scripts/`
(e.g. `family_mediation_diagnostics.py`, `reassign_family_certificates_ci.py`,
`certificate_sensitivity_sweep.py`, `paired_bootstrap_significance.py`,
`perturbation_robustness.py`).

## Documentation

- **[docs/certificate_protocol.md](docs/certificate_protocol.md)** — four-value certificate, two-layer protocol, p/q values as diagnostics.
- **[docs/resource_accounting.md](docs/resource_accounting.md)** — pre-deployment resource reference.
- **[docs/reproducibility.md](docs/reproducibility.md)** — seeds, splits, frozen certificate parameters.
- **[docs/data.md](docs/data.md)** — dataset layout and configuration.
- `docs/cmr_net_design.md` — method / design notes.

## Evaluation metric conventions

All metrics live in `src/cmr_net/evaluation/diagnostics.py` and are shared by
`scripts/evaluate.py` and `scripts/run_ablations.py`.

- **AUROC / PR-AUC**: macro one-vs-rest over classes present in the split;
  zero-sup
