# Code Map

| File | Function |
|---|---|
| `configs/welding_inference.yaml` | Robotic-welding inference configuration |
| `scripts/validate_checkpoint.py` | Checkpoint and configuration validation |
| `scripts/certificate_diagnostics.py` | Family-level candidate diagnostics, pooled inference, LOSO analysis, and certificate release |
| `docs/data.md` | Dataset interface and defect-family mapping |
| `docs/certificate_protocol.md` | Certificate decision rules |
| `docs/reproducibility.md` | Environment and execution commands |
| `docs/resource_accounting.md` | Deployment-resource accounting |

## Output names

| Implementation name | Method term |
|---|---|
| `logits_mediated` | Process-Mediated Pathway logits |
| `logits_direct` | Residual Direct Pathway logits |
| `logits_pathway` | Additive Full Pathway logits |
| `Z_A_perp` | Process-residualized audio representation |
| `Z_V_perp` | Process-residualized video representation |

The direct prediction head receives the concatenated 32-dimensional residual representations. The resulting input dimension is 64. The context vector is not concatenated with the direct-head input.
