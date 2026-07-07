"""Aggregate diagnostics across seeds for the paper main table.

Each input directory is expected to contain the JSON outputs of one seed-run
of the audit pipeline:

* ``family_sufficiency_diagnostics.json`` (from family_mediation_diagnostics.py)
* ``selective_deployment_routing.json``   (from selective_deployment_routing.py)
* ``perturbation_robustness.json``        (from perturbation_robustness.py)

The aggregator emits

* ``aggregated_metrics.json`` — every collected scalar with mean / std / 95% CI
  / per-seed values, suitable for the main table.
* ``aggregated_metrics.md``   — paper-ready markdown with the headline rows.
* ``certificate_stability.md``— per-family sufficiency-certificate vote across
  seeds; flips here are a falsifier for our pre-registered claims.

The aggregator is deliberately stdlib-only. No torch / numpy / scipy import,
because seed aggregation should never block on a heavyweight env.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Files we know how to consume. Missing files are reported, not fatal.
EXPECTED_FILES = (
    "family_sufficiency_diagnostics.json",
    "selective_deployment_routing.json",
    "perturbation_robustness.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more per-seed result directories.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional human-readable label per --results entry (defaults to dir basename).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "aggregated"),
        help="Where to write aggregated_metrics.{json,md} and certificate_stability.md.",
    )
    parser.add_argument(
        "--ci",
        choices=("none", "normal"),
        default="normal",
        help="95% CI estimator. 'normal' uses ±1.96·SE assuming a roughly Gaussian "
        "sampling distribution across seeds; 'none' omits CIs.",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=2,
        help="Skip aggregation for metrics with fewer than this many valid seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = [Path(p) for p in args.results]
    if args.labels and len(args.labels) != len(dirs):
        raise ValueError("--labels must have the same length as --results")
    labels = args.labels or [d.name for d in dirs]

    per_seed: list[dict[str, Any]] = []
    for label, d in zip(labels, dirs):
        per_seed.append(_load_seed(label, d))

    aggregated = _aggregate(per_seed, min_seeds=args.min_seeds, ci=args.ci)
    cert_table = _certificate_stability(per_seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregated_metrics.json").write_text(
        json.dumps(
            {
                "seeds": [{"label": s["label"], "dir": str(s["dir"]), "files": s["files"]} for s in per_seed],
                "min_seeds": args.min_seeds,
                "ci": args.ci,
                "metrics": aggregated,
                "certificate_stability": cert_table,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "aggregated_metrics.md").write_text(
        _render_metrics_markdown(aggregated, per_seed=per_seed, min_seeds=args.min_seeds),
        encoding="utf-8",
    )
    (out_dir / "certificate_stability.md").write_text(
        _render_cert_markdown(cert_table, per_seed),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / 'aggregated_metrics.json'}")
    print(f"Wrote {out_dir / 'aggregated_metrics.md'}")
    print(f"Wrote {out_dir / 'certificate_stability.md'}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_seed(label: str, d: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"label": label, "dir": d, "files": {}}
    for name in EXPECTED_FILES:
        path = d / name
        if path.exists():
            try:
                payload["files"][name] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                payload["files"][name] = {"_error": f"JSON decode error: {exc}"}
        else:
            payload["files"][name] = None
    payload["metrics"] = _extract_metrics(payload["files"])
    payload["certificates"] = _extract_certificates(payload["files"])
    return payload


def _extract_metrics(files: dict[str, Any]) -> dict[str, float]:
    """Flatten the per-file JSON into a flat metric_name -> scalar mapping.

    Names are deliberately stable so the aggregated table reads the same across
    paper revisions. Missing files contribute no metrics (they show up as gaps
    in `n_seeds` rather than zeros).
    """
    out: dict[str, float] = {}

    routing = files.get("selective_deployment_routing.json") or {}
    baseline = routing.get("baseline") or {}
    for k in (
        "process_macro_f1",
        "process_accuracy",
        "direct_macro_f1",
        "direct_accuracy",
        "full_additive_macro_f1",
        "full_additive_accuracy",
    ):
        v = baseline.get(k)
        if _is_finite(v):
            out[f"baseline/{k}"] = float(v)
    fpa = routing.get("family_prediction_accuracy")
    if _is_finite(fpa):
        out["routing/family_prediction_accuracy"] = float(fpa)

    for variant in routing.get("policy_variants") or []:
        ns = variant.get("non_sufficient_policy", "?")
        oracle = variant.get("oracle_family_routing") or {}
        for k in ("coverage", "macro_f1", "accuracy"):
            v = oracle.get(k)
            if _is_finite(v):
                out[f"routing/{ns}/oracle/{k}"] = float(v)
        for row in variant.get("predicted_family_routing") or []:
            thr = row.get("threshold")
            for k in ("coverage", "macro_f1", "accuracy"):
                v = row.get(k)
                if _is_finite(v) and thr is not None:
                    out[f"routing/{ns}/predicted/thr={thr:.3g}/{k}"] = float(v)

    pert = files.get("perturbation_robustness.json") or {}
    for row in pert.get("rows") or []:
        name = row.get("perturbation")
        strength = row.get("strength")
        if name is None or strength is None:
            continue
        tag = f"{name}@{float(strength):.3g}"
        for k in (
            "process_macro_f1",
            "direct_macro_f1",
            "full_additive_macro_f1",
            "process_macro_f1_drop",
            "direct_macro_f1_drop",
            "full_additive_macro_f1_drop",
        ):
            v = row.get(k)
            if _is_finite(v):
                out[f"perturbation/{tag}/{k}"] = float(v)

    fam = files.get("family_sufficiency_diagnostics.json") or {}
    for row in fam.get("families") or []:
        fam_name = str(row.get("family_name") or row.get("family"))
        for k in ("process_macro_f1", "direct_macro_f1", "full_additive_macro_f1"):
            v = row.get(k)
            if _is_finite(v):
                out[f"family/{fam_name}/{k}"] = float(v)
        # Derived per-seed Δ = full_additive − process, emitted at the SEED level
        # so the downstream aggregator's std is the paired pstdev across seeds
        # (the statistically correct seed-variance of the gap — never reconstruct
        # it from two marginal stds, which drops the full/process covariance).
        proc = row.get("process_macro_f1")
        full = row.get("full_additive_macro_f1")
        if _is_finite(proc) and _is_finite(full):
            out[f"family/{fam_name}/delta_full_minus_process_macro_f1"] = float(full) - float(proc)
        gap = row.get("direct_process_gap")
        if _is_finite(gap):
            out[f"family/{fam_name}/direct_process_gap"] = float(gap)
        for q in ("audio_q", "video_q", "y_q"):
            v = row.get(q)
            if _is_finite(v):
                out[f"family/{fam_name}/{q}"] = float(v)

    return out


def _extract_certificates(files: dict[str, Any]) -> dict[str, str]:
    fam = files.get("family_sufficiency_diagnostics.json") or {}
    out: dict[str, str] = {}
    for row in fam.get("families") or []:
        name = str(row.get("family_name") or row.get("family"))
        cert = row.get("sufficiency_certificate")
        if cert is not None:
            out[name] = str(cert)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(
    per_seed: list[dict[str, Any]],
    *,
    min_seeds: int,
    ci: str,
) -> dict[str, dict[str, Any]]:
    keys: set[str] = set()
    for s in per_seed:
        keys.update(s["metrics"].keys())
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        per_seed_values: list[tuple[str, float | None]] = []
        for s in per_seed:
            v = s["metrics"].get(key)
            per_seed_values.append((s["label"], float(v) if _is_finite(v) else None))
        finite_values = [v for _, v in per_seed_values if v is not None]
        n = len(finite_values)
        entry: dict[str, Any] = {
            "n_seeds": n,
            "per_seed": [{"label": lbl, "value": val} for lbl, val in per_seed_values],
        }
        if n >= min_seeds:
            entry["mean"] = float(statistics.fmean(finite_values))
            entry["std"] = float(statistics.pstdev(finite_values)) if n >= 2 else 0.0
            entry["sample_std"] = float(statistics.stdev(finite_values)) if n >= 2 else 0.0
            if ci == "normal" and n >= 2:
                se = entry["sample_std"] / math.sqrt(n)
                entry["ci95_low"] = entry["mean"] - 1.96 * se
                entry["ci95_high"] = entry["mean"] + 1.96 * se
        out[key] = entry
    return out


def _certificate_stability(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    family_names: set[str] = set()
    for s in per_seed:
        family_names.update(s["certificates"].keys())
    rows: list[dict[str, Any]] = []
    for name in sorted(family_names):
        votes: list[str] = []
        for s in per_seed:
            votes.append(s["certificates"].get(name, "missing"))
        unique = sorted(set(votes))
        # A family is stable iff every (non-missing) vote is identical.
        non_missing = [v for v in votes if v != "missing"]
        stable = bool(non_missing) and len(set(non_missing)) == 1
        rows.append(
            {
                "family": name,
                "votes": votes,
                "unique": unique,
                "stable": stable,
                "majority": _majority(votes),
            }
        )
    return {"rows": rows, "labels": [s["label"] for s in per_seed]}


def _majority(votes: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return "missing"
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


HEADLINE_KEYS = (
    "baseline/process_macro_f1",
    "baseline/direct_macro_f1",
    "baseline/full_additive_macro_f1",
    "baseline/process_accuracy",
    "baseline/direct_accuracy",
    "baseline/full_additive_accuracy",
    "routing/family_prediction_accuracy",
    "routing/full/oracle/macro_f1",
    "routing/full/oracle/coverage",
    "routing/direct/oracle/macro_f1",
    "routing/direct/oracle/coverage",
)


def _render_metrics_markdown(
    aggregated: dict[str, dict[str, Any]],
    *,
    per_seed: list[dict[str, Any]],
    min_seeds: int,
) -> str:
    lines = [
        "# Aggregated Metrics Across Seeds",
        "",
        f"- seeds: {[s['label'] for s in per_seed]}",
        f"- min_seeds for aggregation: {min_seeds}",
        "",
        "## Headline (paper main table)",
        "",
        "| metric | mean | std | 95% CI | n |",
        "|---|---:|---:|---|---:|",
    ]
    for key in HEADLINE_KEYS:
        lines.append(_metric_row(key, aggregated.get(key)))
    lines.extend(["", "## All Aggregated Metrics", "",
                  "| metric | mean | std | 95% CI | n |",
                  "|---|---:|---:|---|---:|"])
    for key in sorted(aggregated.keys()):
        lines.append(_metric_row(key, aggregated[key]))
    return "\n".join(lines) + "\n"


def _metric_row(key: str, entry: dict[str, Any] | None) -> str:
    if not entry or entry.get("mean") is None:
        n = entry.get("n_seeds", 0) if entry else 0
        return f"| {key} | NA | NA | NA | {n} |"
    mean = entry["mean"]
    std = entry.get("std", float("nan"))
    n = entry["n_seeds"]
    if "ci95_low" in entry and "ci95_high" in entry:
        ci = f"[{entry['ci95_low']:.4g}, {entry['ci95_high']:.4g}]"
    else:
        ci = "NA"
    return f"| {key} | {mean:.4g} | {std:.4g} | {ci} | {n} |"


def _render_cert_markdown(cert_table: dict[str, Any], per_seed: list[dict[str, Any]]) -> str:
    labels = cert_table.get("labels", [s["label"] for s in per_seed])
    lines = [
        "# Sufficiency-Certificate Stability Across Seeds",
        "",
        "A pre-registered claim is falsified if a family's certificate flips across",
        "seeds — for example, `process_sufficient` on one seed and `non_sufficient`",
        "on another. The `stable` column is True iff all non-missing votes agree.",
        "",
        "| family | " + " | ".join(labels) + " | majority | stable |",
        "|---|" + "|".join(["---"] * len(labels)) + "|---|---|",
    ]
    for row in cert_table.get("rows", []):
        votes_str = " | ".join(row["votes"])
        lines.append(
            f"| {row['family']} | {votes_str} | {row['majority']} | {'yes' if row['stable'] else 'NO'} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _is_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


if __name__ == "__main__":
    main()
