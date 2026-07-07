# Deployment-Resource Accounting

This accounting is a **pre-deployment quantitative reference**, not a
production-line deployment and not a safety guarantee. It maps each family-level
certificate to a candidate online modality-usage mode and reports input burden,
activated parameters, and single-sample forward latency.

## Input burden

- Process-main path: `32 x 6 = 192` elements.
- Full multimodal path: `32 x 6 + 48 x 64 + 8 x 3 x 32 x 32 = 27,840` elements.

## Certificate -> candidate mode

| certificate | candidate mode |
|---|---|
| PS      | process-main path (audio/video reduced to low-frequency review) |
| PS-RMD  | process-main path **plus** residual audio/video monitor at duty cycle rho |
| PI      | keep the full multimodal path |
| IND     | defer the modality-reduction decision |

`PS-RMD` does **not** mean removing audio/video; it keeps a residual monitor.
`IND` is "insufficient evidence / defer", **not** a failure class.
