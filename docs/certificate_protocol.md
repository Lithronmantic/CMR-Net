# Certificate Protocol

For each defect family, the evaluation produces Process-Sufficient (PS), Process-Sufficient with Residual Modality Dependence (PS-RMD), Process-Insufficient (PI), or Indeterminate (IND).

## Candidate diagnostics

Each checkpoint provides:

- Predictive Non-Inferiority from the upper endpoint of the paired-bootstrap interval for Full minus Process macro-F1;
- Residual-Pathway Superiority from the lower endpoint of the paired-bootstrap interval for Direct minus Process macro-F1;
- CHSIC practical-equivalence indicators for the label, audio residual, and video residual diagnostics.

## Pooled certificate

The pooled procedure resamples training seeds and family instances. CHSIC indicators are aggregated by majority voting.

- PS: Predictive Non-Inferiority passes, the label and residual-modality CHSIC indicators pass, and Residual-Pathway Superiority does not pass.
- PS-RMD: Predictive Non-Inferiority and the label criterion pass, while residual-modality dependence or Residual-Pathway Superiority remains.
- PI: the modality-reduction criteria are not satisfied.
- IND: family support is below the minimum size or the released state is unstable under leave-one-seed-out analysis.

## Released certificate

The pooled procedure is repeated after removing each training seed. Any disagreement between a leave-one-seed-out state and the pooled state produces IND. Otherwise, the released certificate equals the pooled certificate.

## Parameters

| Parameter | Value |
|---|---:|
| `m` | 0.03 |
| `m_sup` | 0.03 |
| `delta` | `5e-5` |
| `n_min` | 40 |
| `B` | 10000 |
| CHSIC permutations | 1000 |
