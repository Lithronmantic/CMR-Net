"""Family-level process-sufficiency diagnostics with FDR-aware HSIC summaries.

The conditional-independence routine is permutation-based conditional HSIC
with Gaussian Gram matrices and kernel-ridge residualization. It is not an
asymptotic KCIT implementation and does not use an RFF estimator.

Class-level claims are underpowered on this dataset. This script uses the
coarser suffix12 family labels as the confirmatory unit and leaves class-level
analyses for supplementary heatmaps.

Terminology note:
    The training code still exposes legacy names such as ``logits_mediated``.
    In this report they are interpreted as the process-only pathway, not as
    Pearl-style mediation or causal identification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmr_net.data.dataset import REAL_FAMILY_NAMES
from cmr_net.evaluation.diagnostics import _analytical_chsic_permutation

from _causal_audit_common import (
    accuracy_from_logits,
    load_config,
    load_model,
    macro_f1_from_logits,
    select_loader,
    write_json,
    write_markdown,
)

CERTIFICATE_ASSIGNMENT_VERSION = "ci_noninferiority_v1"
PREDICTIVE_NONINFERIORITY_RULE = "bootstrap_ci95_upper_bound"
CONDITIONAL_INDEPENDENCE_TEST = "permutation_based_conditional_hsic"
CONDITIONAL_INDEPENDENCE_METADATA = {
    "conditional_independence_test": CONDITIONAL_INDEPENDENCE_TEST,
    "kernel": "gaussian",
    "conditioning": "kernel_ridge_residualization",
    "null_distribution": "permutation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "causal_feasibility"))
    parser.add_argument("--n-permutation", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--min-family-n", type=int, default=40)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--equivalence-delta", type=float, default=5.0e-5)
    parser.add_argument(
        "--f1-equivalence-margin",
        type=float,
        default=0.03,
        help="Maximum full-vs-process macro-F1 improvement still treated as practically equivalent.",
    )
    parser.add_argument(
        "--direct-gap-margin",
        type=float,
        default=0.03,
        help="Maximum direct-vs-process macro-F1 advantage still treated as practically equivalent.",
    )
    parser.add_argument("--debug-lenient", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    loader = select_loader(cfg, args.split)
    model, meta = load_model(cfg, args.checkpoint, device=args.device, debug_lenient=args.debug_lenient)
    data = collect(model, loader)
    result = diagnose(
        data,
        n_permutation=args.n_permutation,
        ridge=args.ridge,
        min_family_n=args.min_family_n,
        equivalence_delta=args.equivalence_delta,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        f1_equivalence_margin=args.f1_equivalence_margin,
        direct_gap_margin=args.direct_gap_margin,
    )
    result["checkpoint"] = meta
    result["split"] = args.split
    out_dir = Path(args.output_dir)
    # Historical filenames are kept for compatibility. The sufficiency aliases
    # are the paper-facing names going forward.
    rendered = render_markdown(result)
    write_json(out_dir / "family_mediation_diagnostics.json", result)
    write_markdown(out_dir / "family_mediation_diagnostics.md", rendered)
    write_json(out_dir / "family_sufficiency_diagnostics.json", result)
    write_markdown(out_dir / "family_sufficiency_diagnostics.md", rendered)
    print(f"Wrote {out_dir / 'family_mediation_diagnostics.json'}")
    print(f"Wrote {out_dir / 'family_mediation_diagnostics.md'}")
    print(f"Wrote {out_dir / 'family_sufficiency_diagnostics.json'}")
    print(f"Wrote {out_dir / 'family_sufficiency_diagnostics.md'}")


def collect(model: torch.nn.Module, loader: Any) -> dict[str, Any]:
    device = next(model.parameters()).device
    keys = [
        "logits",
        "logits_mediated",
        "logits_direct",
        "logits_pathway",
        "Z_P",
        "Z_A",
        "Z_V",
        "T",
        "C",
        "Y",
        "Y_family",
    ]
    store: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    pathway_available = True
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            logits = outputs["logits"]
            batch_pathway_available = all(
                key in outputs for key in ("logits_mediated", "logits_direct", "logits_pathway")
            )
            pathway_available = pathway_available and batch_pathway_available
            store["logits"].append(logits.detach().cpu())
            store["logits_mediated"].append(
                outputs.get("logits_mediated", torch.full_like(logits, float("nan"))).detach().cpu()
            )
            store["logits_direct"].append(
                outputs.get("logits_direct", torch.full_like(logits, float("nan"))).detach().cpu()
            )
            store["logits_pathway"].append(outputs.get("logits_pathway", logits).detach().cpu())
            for key in ("Z_P", "Z_A", "Z_V"):
                store[key].append(outputs[key].detach().cpu())
            for key in ("T", "C", "Y", "Y_family"):
                store[key].append(batch[key].detach().cpu())
    out: dict[str, Any] = {key: torch.cat(parts, dim=0) for key, parts in store.items()}
    out["pathway_available"] = bool(pathway_available)
    return out


def diagnose(
    data: dict[str, Any],
    *,
    n_permutation: int,
    ridge: float,
    min_family_n: int,
    equivalence_delta: float,
    n_bootstrap: int,
    bootstrap_seed: int,
    f1_equivalence_margin: float,
    direct_gap_margin: float,
) -> dict[str, Any]:
    families = sorted(int(x) for x in data["Y_family"].unique().tolist())
    num_classes = int(data["logits"].shape[-1])
    pathway_available = bool(data.get("pathway_available", True))
    rows = []
    p_records = []
    for fam in families:
        mask = data["Y_family"] == fam
        n = int(mask.sum().item())
        row: dict[str, Any] = {
            "family": fam,
            "family_name": REAL_FAMILY_NAMES[fam] if 0 <= fam < len(REAL_FAMILY_NAMES) else str(fam),
            "n": n,
            "classes_present": sorted(int(x) for x in data["Y"][mask].unique().tolist()),
        }
        if n > 0:
            labels = data["Y"][mask]
            full_f1 = macro_f1_from_logits(data["logits_pathway"][mask], labels)
            if not pathway_available:
                row.update(
                    {
                        "pathway_available": False,
                        "full_additive_macro_f1": full_f1,
                        "full_additive_accuracy": accuracy_from_logits(data["logits_pathway"][mask], labels),
                        "sufficiency_certificate": "undetermined",
                        "sufficiency_reason": (
                            "dual-pathway outputs unavailable; process-sufficiency "
                            "certificate is not applicable"
                        ),
                        "deployment_recommendation": "not_applicable",
                    }
                )
                row.update(
                    _predictive_noninferiority_fields(
                        row,
                        f1_equivalence_margin=f1_equivalence_margin,
                    )
                )
                rows.append(row)
                continue
            mediated_f1 = macro_f1_from_logits(data["logits_mediated"][mask], labels)
            direct_f1 = macro_f1_from_logits(data["logits_direct"][mask], labels)
            row.update(
                {
                    # Legacy field names kept for compatibility with existing
                    # downstream notebooks/results. New papers should use the
                    # process/direct/full aliases below.
                    "mediated_macro_f1": mediated_f1,
                    "process_macro_f1": mediated_f1,
                    "direct_macro_f1": direct_f1,
                    "full_additive_macro_f1": full_f1,
                    "mediated_accuracy": accuracy_from_logits(data["logits_mediated"][mask], labels),
                    "process_accuracy": accuracy_from_logits(data["logits_mediated"][mask], labels),
                    "direct_accuracy": accuracy_from_logits(data["logits_direct"][mask], labels),
                    "full_additive_accuracy": accuracy_from_logits(data["logits_pathway"][mask], labels),
                    "mediation_gap_direct_minus_mediated": direct_f1 - mediated_f1,
                    "direct_minus_process_macro_f1": direct_f1 - mediated_f1,
                    "full_minus_process": full_f1 - mediated_f1,
                    "full_minus_process_macro_f1": full_f1 - mediated_f1,
                }
            )
            if n >= 2 and n_bootstrap > 0:
                row.update(
                    _bootstrap_gap_ci(
                        mediated_logits=data["logits_mediated"][mask],
                        direct_logits=data["logits_direct"][mask],
                        labels=labels,
                        n_bootstrap=n_bootstrap,
                        seed=bootstrap_seed + fam,
                    )
                )
                row["direct_minus_process_bootstrap_mean"] = row.get("mediation_gap_bootstrap_mean")
                row["direct_minus_process_ci95_low"] = row.get("mediation_gap_ci95_low")
                row["direct_minus_process_ci95_high"] = row.get("mediation_gap_ci95_high")
                row["direct_minus_process_ci95_excludes_zero"] = row.get("mediation_gap_ci95_excludes_zero")
                row.update(
                    _bootstrap_gap_ci(
                        mediated_logits=data["logits_mediated"][mask],
                        direct_logits=data["logits_pathway"][mask],
                        labels=labels,
                        n_bootstrap=n_bootstrap,
                        seed=bootstrap_seed + 1009 + fam,
                        prefix="full_minus_process",
                    )
                )
        if n >= min_family_n:
            cond_zp_c = torch.cat((data["Z_P"][mask], data["C"][mask]), dim=-1)
            classes_in_family = int(data["Y"][mask].unique().numel())
            tests: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
                # "audio"/"video": does the modality embedding carry process
                # summary information beyond [Z_P, C]? Small + practically
                # equivalent values support process-only deployment.
                "audio": (data["Z_A"][mask], data["T"][mask], cond_zp_c),
                "video": (data["Z_V"][mask], data["T"][mask], cond_zp_c),
            }
            if classes_in_family >= 2:
                # Process-summary sufficiency check: after conditioning on the
                # process representation Z_P and context C, does the active
                # window process summary T still carry residual information
                # about the true label Y?
                #   - HSIC practically small -> Z_P preserves useful process
                #     summary information for this family.
                #   - HSIC large -> Z_P is not a sufficient process summary for
                #     this family.
                # NOTE: we deliberately do NOT use softmax(logits_mediated) as X
                # because logits_mediated is by construction a function of
                # (Z_P, C); conditioning on those would make the test trivially
                # ~ 0 regardless of the data.
                y_onehot = torch.nn.functional.one_hot(
                    data["Y"][mask].to(torch.long), num_classes=num_classes
                ).to(torch.float32)
                tests["y"] = (y_onehot, data["T"][mask], cond_zp_c)
            else:
                row["hsic_y_note"] = f"skipped: only {classes_in_family} class present in family"
            for offset, (name, (X, Y, Z)) in enumerate(tests.items()):
                stats = _analytical_chsic_permutation(
                    X=X,
                    Y=Y,
                    Z=Z,
                    ridge=ridge,
                    n_permutation=n_permutation,
                    seed=3100 + 17 * fam + offset,
                )
                obs = float(stats["observed"])
                p_value = float(stats["p_value"])
                row[f"hsic_{name}_observed"] = obs
                row[f"hsic_{name}_p_value"] = p_value
                row[f"hsic_{name}_equivalent_delta"] = bool(obs <= equivalence_delta)
                # TOST-style practical-equivalence flag. CHSIC is
                # non-negative, so practical equivalence is one-sided:
                # dependence must lie below a pre-declared equivalence margin.
                # This is intentionally separate from the vanilla permutation
                # p-value, which tests exact independence rather than
                # practical equivalence.
                margin = float(equivalence_delta)
                row[f"hsic_{name}_tost_margin"] = margin
                row[f"hsic_{name}_tost_equivalent"] = bool(obs <= margin)
                p_records.append((len(rows), f"hsic_{name}", p_value))
        else:
            row["hsic_note"] = f"skipped: n < min_family_n ({min_family_n})"
        rows.append(row)

    q_values = _bh_q_values([p for _, _, p in p_records])
    for (row_idx, key, _), q in zip(p_records, q_values, strict=False):
        rows[row_idx][f"{key}_q_value_bh"] = float(q)
        rows[row_idx][f"{key}_reject_bh_05"] = bool(q < 0.05)

    # Certificates must be assigned *after* BH correction. Earlier versions
    # assigned labels before ``*_reject_bh_05`` existed, which let families
    # with strong A/V residual evidence (notably geometry_profile) be labelled
    # as clean ``process_sufficient`` on some seeds.
    for row in rows:
        row.update(
            _assign_certificate(
                row,
                min_family_n=min_family_n,
                f1_equivalence_margin=f1_equivalence_margin,
                direct_gap_margin=direct_gap_margin,
            )
        )

    return {
        "available": True,
        "pathway_available": pathway_available,
        "num_samples": int(data["Y"].numel()),
        "min_family_n": int(min_family_n),
        "n_permutation": int(n_permutation),
        "n_bootstrap": int(n_bootstrap),
        "ridge": float(ridge),
        "equivalence_delta": float(equivalence_delta),
        "f1_equivalence_margin": float(f1_equivalence_margin),
        "direct_gap_margin": float(direct_gap_margin),
        "certificate_assignment_version": CERTIFICATE_ASSIGNMENT_VERSION,
        "predictive_noninferiority_rule": PREDICTIVE_NONINFERIORITY_RULE,
        **CONDITIONAL_INDEPENDENCE_METADATA,
        "tost_note": (
            "TOST-style equivalence is reported as observed CHSIC below the "
            "pre-declared equivalence delta. This is separate from the "
            "permutation p-value for exact independence; treat it as an "
            "approximate process-summary sufficiency flag, not exact independence."
        ),
        "multiple_testing": "Benjamini-Hochberg over all reported family-level HSIC p-values",
        "families": rows,
        "test_semantics": {
            "audio": "HSIC(Z_A, T_summary | Z_P, C): does the audio embedding carry process-summary information beyond Z_P and context C?",
            "video": "HSIC(Z_V, T | Z_P, C): same as audio, for the video embedding.",
            "y": "HSIC(one_hot(Y), T_summary | Z_P, C): process-summary sufficiency check. Practically small values support that Z_P preserves the label-relevant process summary.",
        },
        "note": (
            "Family-level rows are the confirmatory unit. The 'y' test uses the "
            "true label Y, NOT softmax(logits_mediated), so the test is not "
            "trivialized by the process head being a function of (Z_P, C). "
            "Class-level sufficiency should be treated as exploratory unless "
            "pre-registered and sufficiently powered. This script does not "
            "claim causal identification. If pathway_available is false, the "
            "model does not expose process/direct pathway logits and all "
            "sufficiency certificates are intentionally marked undetermined."
        ),
    }


def _bootstrap_gap_ci(
    *,
    mediated_logits: torch.Tensor,
    direct_logits: torch.Tensor,
    labels: torch.Tensor,
    n_bootstrap: int,
    seed: int,
    prefix: str = "mediation_gap",
) -> dict[str, Any]:
    n = int(labels.numel())
    num_classes = int(max(mediated_logits.shape[-1], direct_logits.shape[-1]))
    generator = torch.Generator().manual_seed(int(seed))
    values = []
    for _ in range(int(n_bootstrap)):
        idx = torch.randint(0, n, (n,), generator=generator)
        med = macro_f1_from_logits(mediated_logits[idx], labels[idx], num_classes)
        direct = macro_f1_from_logits(direct_logits[idx], labels[idx], num_classes)
        values.append(float(direct - med))
    values_sorted = sorted(values)
    low = values_sorted[int(0.025 * (len(values_sorted) - 1))]
    high = values_sorted[int(0.975 * (len(values_sorted) - 1))]
    return {
        f"{prefix}_bootstrap_mean": float(sum(values) / len(values)),
        f"{prefix}_ci95_low": float(low),
        f"{prefix}_ci95_high": float(high),
        f"{prefix}_ci95_excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def _assign_certificate(
    row: dict[str, Any],
    *,
    min_family_n: int,
    f1_equivalence_margin: float,
    direct_gap_margin: float,
) -> dict[str, Any]:
    """Assign a deployment-oriented process-sufficiency label.

    The rule is deliberately conservative and combines three signals:
      1. full additive should not materially beat process-only,
      2. direct-only should not have a significant positive advantage over
         process-only beyond a practical margin,
      3. the process-summary y-test should be practically equivalent when it
         is estimable.

    Audio/video leakage does not automatically make a family non-sufficient
    for classification. It is surfaced as a subtype because a family can be
    classification-sufficient while still carrying modality-specific process
    information.
    """

    predictive_fields = _predictive_noninferiority_fields(
        row,
        f1_equivalence_margin=f1_equivalence_margin,
    )
    if int(row.get("n", 0)) < int(min_family_n):
        return {
            **predictive_fields,
            "sufficiency_certificate": "undetermined",
            "sufficiency_reason": f"n < min_family_n ({min_family_n})",
            "deployment_recommendation": "abstain",
        }
    process_f1 = row.get("process_macro_f1", row.get("mediated_macro_f1"))
    full_f1 = row.get("full_additive_macro_f1")
    if process_f1 is None or full_f1 is None:
        return {
            **predictive_fields,
            "sufficiency_certificate": "undetermined",
            "sufficiency_reason": "missing process/full pathway logits",
            "deployment_recommendation": "abstain",
        }
    full_minus_process = float(predictive_fields["full_minus_process"])
    if not predictive_fields["predictive_noninferiority_available"]:
        return {
            **predictive_fields,
            "sufficiency_certificate": "undetermined",
            "sufficiency_reason": predictive_fields["predictive_noninferiority_reason"],
            "deployment_recommendation": "abstain",
        }
    direct_ci_low = row.get("direct_minus_process_ci95_low", row.get("mediation_gap_ci95_low"))
    direct_ci_high = row.get("direct_minus_process_ci95_high", row.get("mediation_gap_ci95_high"))
    direct_superior = False
    if direct_ci_low is not None and direct_ci_high is not None:
        direct_superior = float(direct_ci_low) > float(direct_gap_margin)
    y_equiv = row.get("hsic_y_tost_equivalent")
    y_ok = True if y_equiv is None else bool(y_equiv)
    predictive_ok = bool(predictive_fields["predictive_ok_ci"])
    # Deployment labels should reflect *practical* residual information.
    # A vanilla permutation q-value can reject exact independence even when
    # the observed CHSIC is below the pre-declared equivalence margin. We
    # therefore expose BH q-values as diagnostics, but only TOST failure (or
    # a practically superior direct branch) triggers the modality-leaky label.
    modality_leaky = direct_superior or any(
        row.get(f"hsic_{name}_tost_equivalent") is False
        for name in ("audio", "video")
    )
    if predictive_ok and y_ok:
        if modality_leaky:
            return {
                **predictive_fields,
                "sufficiency_certificate": "classification_sufficient_modality_leaky",
                "sufficiency_reason": (
                    "process-only is predictively sufficient, but direct/A/V "
                    "diagnostics still reveal residual information beyond Z_P"
                ),
                "deployment_recommendation": "process_only_with_modality_monitoring",
            }
        return {
            **predictive_fields,
            "sufficiency_certificate": "process_sufficient",
            "sufficiency_reason": (
                "process-only is non-inferior to full additive under the "
                "bootstrap CI upper-bound rule"
            ),
            "deployment_recommendation": "process_only",
        }
    return {
        **predictive_fields,
        "sufficiency_certificate": "non_sufficient",
        "sufficiency_reason": (
            f"full_minus_process={full_minus_process:.4g}, "
            f"full_minus_process_ci_high={_safe_fmt(predictive_fields.get('full_minus_process_ci95_high'))}, "
            f"direct_ci=[{_safe_fmt(direct_ci_low)}, {_safe_fmt(direct_ci_high)}], "
            f"y_tost={y_equiv}"
        ),
        "deployment_recommendation": "full_additive",
    }


def _predictive_noninferiority_fields(
    row: dict[str, Any],
    *,
    f1_equivalence_margin: float,
) -> dict[str, Any]:
    """Return CI-based predictive non-inferiority fields for one family row."""

    margin = float(f1_equivalence_margin)
    process_f1 = row.get("process_macro_f1", row.get("mediated_macro_f1"))
    full_f1 = row.get("full_additive_macro_f1")
    out: dict[str, Any] = {
        "f1_equivalence_margin": margin,
        "predictive_noninferiority_rule": PREDICTIVE_NONINFERIORITY_RULE,
    }
    if process_f1 is None or full_f1 is None:
        out.update(
            {
                "full_minus_process": None,
                "full_minus_process_ci95_low": row.get("full_minus_process_ci95_low"),
                "full_minus_process_ci95_high": row.get("full_minus_process_ci95_high"),
                "predictive_noninferiority_available": False,
                "predictive_noninferiority_reason": "missing_process_or_full_f1",
                "predictive_ok_point_estimate": False,
                "predictive_ok_ci": False,
                "predictive_ok": False,
            }
        )
        return out

    full_minus_process = float(full_f1) - float(process_f1)
    ci_low = row.get("full_minus_process_ci95_low")
    ci_high = row.get("full_minus_process_ci95_high")
    predictive_ok_point = full_minus_process <= margin
    if ci_high is None:
        out.update(
            {
                "full_minus_process": full_minus_process,
                "full_minus_process_ci95_low": ci_low,
                "full_minus_process_ci95_high": ci_high,
                "predictive_noninferiority_available": False,
                "predictive_noninferiority_reason": "missing_bootstrap_ci",
                "predictive_ok_point_estimate": bool(predictive_ok_point),
                "predictive_ok_ci": False,
                "predictive_ok": False,
            }
        )
        return out

    predictive_ok_ci = float(ci_high) <= margin
    out.update(
        {
            "full_minus_process": full_minus_process,
            "full_minus_process_ci95_low": ci_low,
            "full_minus_process_ci95_high": ci_high,
            "predictive_noninferiority_available": True,
            "predictive_noninferiority_reason": None,
            "predictive_ok_point_estimate": bool(predictive_ok_point),
            "predictive_ok_ci": bool(predictive_ok_ci),
            "predictive_ok": bool(predictive_ok_ci),
        }
    )
    return out


def _bh_q_values(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    q = [1.0] * m
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        val = min(running, p_values[idx] * m / rank)
        running = val
        q[idx] = min(val, 1.0)
    return q


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Family-Level Process-Sufficiency Diagnostics",
        "",
        f"- `split`: {result['split']}",
        f"- `num_samples`: {result['num_samples']}",
        f"- `min_family_n`: {result['min_family_n']}",
        f"- `n_permutation`: {result['n_permutation']}",
        f"- `n_bootstrap`: {result['n_bootstrap']}",
        f"- `equivalence_delta`: {result['equivalence_delta']}",
        f"- `f1_equivalence_margin`: {result['f1_equivalence_margin']}",
        f"- `direct_gap_margin`: {result['direct_gap_margin']}",
        "",
        "## Family Summary",
        "",
        "| family | n | classes | process F1 | direct F1 | full F1 | direct-process gap | gap 95% CI | full-process | audio q | video q | y q | certificate | deployment |",
        "|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in result["families"]:
        lines.append(
            "| {name} | {n} | {classes} | {process:.4g} | {direct:.4g} | {full:.4g} | {gap:.4g} | {ci} | {fp:.4g} | {aq} | {vq} | {yq} | {cert} | {deploy} |".format(
                name=row["family_name"],
                n=row["n"],
                classes=",".join(str(x) for x in row["classes_present"]),
                process=row.get("process_macro_f1", row.get("mediated_macro_f1", float("nan"))),
                direct=row.get("direct_macro_f1", float("nan")),
                gap=row.get("direct_minus_process_macro_f1", row.get("mediation_gap_direct_minus_mediated", float("nan"))),
                ci=_ci(row),
                full=row.get("full_additive_macro_f1", float("nan")),
                fp=row.get("full_minus_process_macro_f1", float("nan")),
                aq=_fmt(row.get("hsic_audio_q_value_bh")),
                vq=_fmt(row.get("hsic_video_q_value_bh")),
                yq=_fmt(row.get("hsic_y_q_value_bh")),
                cert=row.get("sufficiency_certificate", "NA"),
                deploy=row.get("deployment_recommendation", "NA"),
            )
        )
    lines.extend(
        [
            "",
            "## Certificate Rule",
            "",
            "- `predictive_ok`: `full_minus_process_ci95_high <= f1_equivalence_margin`; the older point-estimate check is retained as `predictive_ok_point_estimate` for audit only.",
            "- `process_sufficient`: process-only passes CI-based non-inferiority, direct-only is not significantly superior beyond the declared margin, and the process-summary y-test is practically small when estimable.",
            "- `classification_sufficient_modality_leaky`: classification is process-sufficient under the CI rule, but audio/video still carry process-summary information beyond `Z_P`; use process-only deployment with modality monitoring.",
            "- `non_sufficient`: full/direct pathways add material predictive value under the CI rule or the process-summary y-test is not practically small.",
            "- `undetermined`: insufficient family support, missing diagnostics, or missing full-vs-process bootstrap CI.",
            "",
            "## TOST Note",
            "",
            result["tost_note"],
            "",
            "## Note",
            "",
            result["note"],
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4g}"


def _safe_fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4g}"


def _ci(row: dict[str, Any]) -> str:
    low = row.get("direct_minus_process_ci95_low", row.get("mediation_gap_ci95_low"))
    high = row.get("direct_minus_process_ci95_high", row.get("mediation_gap_ci95_high"))
    if low is None or high is None:
        return "NA"
    return f"[{float(low):.3g}, {float(high):.3g}]"


def _approx(row: dict[str, Any]) -> str:
    vals = [
        row.get("hsic_audio_tost_equivalent"),
        row.get("hsic_video_tost_equivalent"),
        row.get("hsic_y_tost_equivalent"),
    ]
    vals = [v for v in vals if v is not None]
    if not vals:
        return "NA"
    return "yes" if all(bool(v) for v in vals) else "no"


if __name__ == "__main__":
    main()
