# Core Code Map

The publishable core of this repository, grouped by role and mapped to the
paper. Items marked **(internal)** are server / round-specific and are excluded
from the public repo via `.gitignore` (or are candidates for `scripts/legacy/`).

## Package `src/cmr_net/` (core — keep)

| file | role | paper component |
|---|---|---|
| `model.py` | CMR-Net assembly | CMR-Net |
| `modules/treatment_encoder.py` | process-summary encoder `phi_T` (t -> z_T) | Process-Mediated Pathway |
| `modules/process_encoder.py` | process encoder `phi_P` (-> Z_P) | Process-Mediated Pathway |
| `modules/audio_encoder.py`, `video_encoder.py` | A/V encoders `phi_A`, `phi_V` | residual A/V branches |
| `modules/intervention_head.py` | residual direct head `D_raw` + scalar gate | Residual Direct Pathway |
| `modules/dual_pathway.py` | dual-pathway composition | Process + Residual + Additive Full |
| `modules/causal_fusion.py` | additive full-pathway logits | Additive Full-Pathway |
| `modules/modality_t_adv.py` | audio/video-to-t adversary (GRL) | process-conditioned residualization |
| `modules/trunks.py` | shared trunk blocks | — |
| `losses/hsic.py` | conditional HSIC (train + diagnostic) | CHSIC / Residual Modality Dependence |
| `losses/pathway.py` | pathway orth / balance / pred / gate | dual-pathway separability |
| `losses/mechanism.py` | reconstruction / anchor / KL / InfoNCE | process-representation regularizers |
| `losses/intervention.py` | residual / direct losses | Residual Direct Pathway |
| `losses/prediction.py` | classification / focal / auxiliary heads | core prediction |
| `losses/total_loss.py` | objective aggregation | training objective |
| `evaluation/diagnostics.py` | metrics + certificate diagnostics | certificate evaluation |
| `training/trainer.py` | single-model 4-stage curriculum trainer | training |
| `data/dataset.py` | loading, group-aware split, family mapping | data / families |
| `baselines/*.py` | comparison models (TARNet/DragonNet/DECI, process/AV baselines) | baselines |

> **Naming note.** Some modules keep earlier causal-inference names:
> `treatment_encoder` = process-summary encoder, `intervention_head` = residual
> direct head, `causal_fusion` = additive full-pathway. These are exactly the
> components the paper calls Process-Mediated / Residual Direct / Additive
> Full-Pathway. They are intentionally **not renamed** to avoid breaking imports.

## Scripts (core — keep)

- **Train / evaluate:** `train.py`, `evaluate.py`, `run_ablations.py`, `train_ddp.py`, `train_baseline.py`, `train_baseline_ddp.py`, `ddp_smoke.py`
- **Certificate:** `family_mediation_diagnostics.py`, `reassign_family_certificates_ci.py`, `certificate_rule_ablation.py`, `certificate_sensitivity_sweep.py`, `certificate_sensitivity_aggregate.py`, `calibrated_family_router.py`, `selective_deployment_routing.py`
- **Statistics:** `paired_bootstrap_significance.py`, `aggregate_seeds.py`, `summarize_delta_stability_interval.py`
- **Perturbation:** `perturbation_robustness.py`
- **Baselines:** `external_baseline_eval.py`, `external_baseline_aggregate.py`, `external_baseline_table.py`, `run_external_baseline_eval.sh`
- **Probes / diagnostics:** `physical_concordance_probe.py`, `diagnose_zp_capacity.py`, `audit_treatment_source.py`, `step0_alpha_diagnostics.py`, `step0_loss_audit.py`, `step0_perclass_gap.py`, `parse_training_logs.py`, `read_lf_gaps.py`, `_causal_audit_common.py`

> `audit_certificate_consistency.py` embeds an **outdated** "paper wording error"
> note that points at an old `sections/03_method.tex`; it no longer matches the
> current manuscript. Update or drop that embedded text before publishing.

## Figures / tables (keep — candidate for `scripts/figures/`)

`plot_ablation_figures.py`, `plot_figure5_sensitivity_dense.py`, `plot_paper_zh_figures.py`, `plot_sensitivity_grid.py`, `plot_sufficiency_figures.py`, `plot_training_curves.py`, `render_dataset_appendix.py`, `render_loss_catalogue.py`, `render_related_work_table.py`, `make_track1_variance_table.py`, `export_review_artifacts.py`

## Internal / server (**internal** — `scripts/legacy/` or `.gitignore`)

`run_round19*.sh`, `run_round20*.sh`, `run_round21*.sh`, `run_repair_track1-4.sh`, `emit_guard_values.sh`, `update_paper_ready_round21.py`

## Configs (core — keep)

- `configs/default.yaml` — main config (dataset path via `${CMR_DATA_ROOT}`); the
  `synthetic` block uses a 3-class mock for smoke runs.
- `configs/baselines/*.yaml` — baseline recipes.
- `configs/cmr_net_round21_process_teacher_router.yaml` — production run config
  (real **12 fine classes / 6 families**, stratified split, process teacher).
- `configs/repair_*.yaml` — **(internal)** repair sweeps; candidate for legacy.
