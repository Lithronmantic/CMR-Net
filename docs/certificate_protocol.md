# Family-Level Process-Sufficiency Certificate

For each defect family the protocol emits one of four states. The repository's
internal labels map to the paper's four-value scheme as follows:

| paper | meaning | internal label |
|---|---|---|
| PS     | process path predictively non-inferior; no residual A/V dependence | `process_sufficient` |
| PS-RMD | process path non-inferior, but residual A/V dependence remains | `process_sufficient` + residual-dependence flag |
| PI     | process path does not meet the modality-reduction criteria | `non_sufficient` |
| IND    | insufficient evidence: `n < n_min`, missing inputs, or seed instability | `undetermined` |

> The PS / PS-RMD split is assembled in the paper-facing reassignment step; see
> `scripts/reassign_family_certificates_ci.py` for the exact mapping.

## Two layers

1. **Candidate (per seed)** — CHSIC practical-equivalence on `y / A / V`,
   Predictive Non-Inferiority (paired-bootstrap CI upper bound `<= m`), and
   residual-pathway superiority (CI lower bound `> m_sup`).
2. **Release** — two-level paired bootstrap over seeds and samples (pool),
   majority CHSIC, then leave-one-seed-out (LOSO) downgrade to IND on
   instability, plus a `delta`-sweep boundary-sensitivity flag `b_f`.

## p / q values are diagnostics only

BH-FDR `q` values and permutation `p` values are reported as
conditional-independence diagnostics; they do **not** assign the certificate.
Assignment uses the frozen practical-equivalence width (`delta`), predictive
non-inferiority margin (`m`), and direct-superiority margin (`m_sup`).

## Implementation

- `scripts/family_mediation_diagnostics.py` — per-seed candidate diagnostics.
- `scripts/reassign_family_certificates_ci.py` — CI-based release assignment.
- `scripts/certificate_sensitivity_sweep.py` — `delta` / margin sensitivity.
- `scripts/paired_bootstrap_significance.py` — paired bootstrap intervals.
