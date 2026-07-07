"""Selective process-sufficiency deployment routing.

The family-level sufficiency certificate is diagnostic unless we also show how
it can be used without peeking at the true family label. This script reports
both:

* oracle-family routing: uses true ``Y_family`` and is an upper-bound analysis;
* predicted-family routing: uses the process-only class logits to infer a coarse
  family, then routes with an optional confidence threshold.

Terminology:
    ``logits_mediated`` is treated as the process-only pathway. It is a legacy
    code name and should not be interpreted as Pearl-style mediation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmr_net.data.dataset import REAL_FAMILY_NAMES, REAL_LABEL_TO_FAMILY

from _causal_audit_common import (
    accuracy_from_logits,
    load_config,
    load_model,
    macro_f1_from_logits,
    select_loader,
    write_json,
    write_markdown,
)

ROUTE_CODE = {"abstain": -1, "process": 0, "full": 1, "direct": 2}
FAMILY_ROUTER_METADATA = {
    "family_router_source": "process_logit_aggregation",
    "uses_auxiliary_family_head_for_routing": False,
    "routing_confidence_definition": "max aggregated family probability from process-head class softmax",
    "class_to_family_mapping": {
        str(cls_idx): REAL_FAMILY_NAMES[int(fam_idx)]
        if 0 <= int(fam_idx) < len(REAL_FAMILY_NAMES)
        else str(fam_idx)
        for cls_idx, fam_idx in enumerate(REAL_LABEL_TO_FAMILY)
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--certificate", required=True, help="family_mediation/sufficiency diagnostics JSON")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "causal_feasibility"))
    parser.add_argument("--thresholds", default="0,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument(
        "--modality-leaky-policy",
        choices=("process", "full", "direct", "abstain"),
        default="process",
        help="How to route classification_sufficient_modality_leaky families.",
    )
    parser.add_argument(
        "--non-sufficient-policies",
        default="full,direct",
        help=(
            "Comma-separated routing policies to evaluate for non_sufficient families. "
            "Choices: full, direct, abstain. Each is reported as a separate variant in the "
            "output JSON. Default 'full,direct' makes the head-to-head comparison the "
            "paper relies on (full-additive fusion vs. direct-only routing)."
        ),
    )
    parser.add_argument(
        "--baseline-family-policies",
        default="random,majority",
        help=(
            "Comma-separated set of routing baselines that ignore the process logits "
            "when picking a family. Each entry is run through the same routing "
            "pipeline (and same family_policy variants) as the predicted-family "
            "routing, to bound how much of the deployable performance comes from "
            "the routing target itself vs. the per-family certificate. "
            "Choices: random, majority, ''. Default 'random,majority'."
        ),
    )
    parser.add_argument(
        "--baseline-random-seed", type=int, default=0,
        help="RNG seed for the random-family baseline (per-sample uniform over family ids).",
    )
    parser.add_argument("--save-per-sample", dest="save_per_sample", action="store_true", default=True)
    parser.add_argument("--no-save-per-sample", dest="save_per_sample", action="store_false")
    parser.add_argument(
        "--save-logits-npz",
        action="store_true",
        help="Save process/direct/full logits and primary route codes as a compressed NPZ.",
    )
    parser.add_argument(
        "--logits-npz-name",
        default=None,
        help="Optional NPZ filename. Defaults to routing_logits_<seed>.npz inferred from paths.",
    )
    parser.add_argument("--debug-lenient", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    loader = select_loader(cfg, args.split)
    model, meta = load_model(cfg, args.checkpoint, device=args.device, debug_lenient=args.debug_lenient)
    data = collect(model, loader)
    certificate = json.loads(Path(args.certificate).read_text(encoding="utf-8"))
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    non_sufficient_policies = [
        x.strip() for x in args.non_sufficient_policies.split(",") if x.strip()
    ]
    invalid = [p for p in non_sufficient_policies if p not in ("full", "direct", "abstain")]
    if invalid:
        raise ValueError(f"Unknown non-sufficient policies: {invalid}")
    baseline_family_policies = [
        x.strip() for x in args.baseline_family_policies.split(",") if x.strip()
    ]
    invalid_bf = [p for p in baseline_family_policies if p not in ("random", "majority")]
    if invalid_bf:
        raise ValueError(f"Unknown baseline family policies: {invalid_bf}")
    result = diagnose_routing(
        data,
        certificate=certificate,
        thresholds=thresholds,
        modality_leaky_policy=args.modality_leaky_policy,
        non_sufficient_policies=non_sufficient_policies,
        baseline_family_policies=baseline_family_policies,
        baseline_random_seed=args.baseline_random_seed,
        save_per_sample=bool(args.save_per_sample),
    )
    result["checkpoint"] = meta
    result["split"] = args.split
    result["certificate_path"] = str(args.certificate)
    out_dir = Path(args.output_dir)
    if args.save_logits_npz and result.get("available", True):
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / (
            args.logits_npz_name
            if args.logits_npz_name
            else f"routing_logits_{_infer_seed_label(args.certificate, args.checkpoint)}.npz"
        )
        _write_logits_npz(npz_path, data, result)
        result["logits_npz"] = npz_path.name
    result.pop("_primary_route_code", None)
    result.pop("_predicted_family", None)
    result.pop("_family_confidence", None)
    write_json(out_dir / "selective_deployment_routing.json", result)
    write_markdown(out_dir / "selective_deployment_routing.md", render_markdown(result))
    print(f"Wrote {out_dir / 'selective_deployment_routing.json'}")
    print(f"Wrote {out_dir / 'selective_deployment_routing.md'}")


def collect(model: torch.nn.Module, loader: Any) -> dict[str, Any]:
    device = next(model.parameters()).device
    store: dict[str, list[torch.Tensor]] = {
        "process": [],
        "direct": [],
        "full": [],
        "deployed": [],
        "Y": [],
        "Y_family": [],
    }
    sample_ids: list[str] = []
    pathway_available = True
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch["Y"].shape[0]) if isinstance(batch.get("Y"), torch.Tensor) else 0
            sample_ids.extend(_extract_sample_ids(batch, batch_size, start_index=len(sample_ids)))
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            logits = outputs["logits"]
            batch_pathway_available = all(
                key in outputs for key in ("logits_mediated", "logits_direct", "logits_pathway")
            )
            pathway_available = pathway_available and batch_pathway_available
            store["deployed"].append(logits.detach().cpu())
            store["process"].append(
                outputs.get("logits_mediated", torch.full_like(logits, float("nan"))).detach().cpu()
            )
            store["direct"].append(
                outputs.get("logits_direct", torch.full_like(logits, float("nan"))).detach().cpu()
            )
            store["full"].append(outputs.get("logits_pathway", logits).detach().cpu())
            store["Y"].append(batch["Y"].detach().cpu())
            store["Y_family"].append(batch["Y_family"].detach().cpu())
    out = {key: torch.cat(parts, dim=0) for key, parts in store.items()}
    out["pathway_available"] = bool(pathway_available)
    out["sample_id"] = sample_ids
    return out


def diagnose_routing(
    data: dict[str, torch.Tensor],
    *,
    certificate: dict[str, Any],
    thresholds: list[float],
    modality_leaky_policy: str,
    non_sufficient_policies: list[str],
    baseline_family_policies: list[str] | None = None,
    baseline_random_seed: int = 0,
    save_per_sample: bool = True,
) -> dict[str, Any]:
    y = data["Y"].to(torch.long)
    y_family = data["Y_family"].to(torch.long)
    process = data["process"]
    direct = data["direct"]
    full = data["full"]
    num_classes = int(full.shape[-1])
    if not bool(data.get("pathway_available", True)):
        return {
            **FAMILY_ROUTER_METADATA,
            "available": False,
            "reason": "dual-pathway outputs unavailable; selective process-sufficiency routing is not applicable",
            "num_samples": int(y.numel()),
            "num_classes": int(num_classes),
            "baseline": {
                "full_additive_macro_f1": macro_f1_from_logits(full, y, num_classes),
                "full_additive_accuracy": accuracy_from_logits(full, y),
            },
            "policy_variants": [],
        }
    baseline = {
        "process_macro_f1": macro_f1_from_logits(process, y, num_classes),
        "full_additive_macro_f1": macro_f1_from_logits(full, y, num_classes),
        "direct_macro_f1": macro_f1_from_logits(direct, y, num_classes),
        "process_accuracy": accuracy_from_logits(process, y),
        "full_additive_accuracy": accuracy_from_logits(full, y),
        "direct_accuracy": accuracy_from_logits(direct, y),
    }
    predicted_family, predicted_confidence = _predict_family_from_process_logits(process)
    family_pred_acc = float((predicted_family == y_family).float().mean().item()) if y.numel() else float("nan")
    num_families = int(max(REAL_LABEL_TO_FAMILY) + 1)
    family_confusion = _family_confusion(predicted_family, y_family, num_families)
    sample_ids = _normalized_sample_ids(data.get("sample_id"), int(y.numel()))
    primary_ns_policy = non_sufficient_policies[0] if non_sufficient_policies else "full"
    primary_threshold = float(thresholds[0]) if thresholds else 0.0
    primary_family_policy = _family_policy(
        certificate,
        modality_leaky_policy=modality_leaky_policy,
        non_sufficient_policy=primary_ns_policy,
    )
    primary_per_sample, primary_route_code = _per_sample_records(
        true_labels=y,
        true_families=y_family,
        sample_ids=sample_ids,
        process_logits=process,
        direct_logits=direct,
        full_logits=full,
        route_family=predicted_family,
        route_confidence=predicted_confidence,
        family_policy=primary_family_policy,
        threshold=primary_threshold,
    )

    # ---- Baseline family sources for Table 2 (random / majority) -----------
    # These ignore the process logits entirely and demonstrate how much of the
    # deployable routing performance comes from the process-only predictor vs.
    # the per-family certificate. They go through the same _route() pipeline
    # and the same family_policy variants as the predicted-family routing.
    baseline_family_policies = list(baseline_family_policies or [])
    baseline_family_sources: dict[str, dict[str, Any]] = {}
    if "random" in baseline_family_policies:
        rand_gen = torch.Generator().manual_seed(int(baseline_random_seed))
        random_family = torch.randint(
            low=0, high=num_families, size=(int(y.numel()),),
            generator=rand_gen, dtype=torch.long,
        )
        baseline_family_sources["random"] = {
            "family": random_family,
            "confidence": torch.ones_like(random_family, dtype=torch.float32),
            "constant_family": None,
            "seed": int(baseline_random_seed),
        }
    if "majority" in baseline_family_policies:
        # "Majority" = pick the family with the largest test-set support and
        # apply it uniformly to every sample. This is the most charitable
        # constant-prediction baseline. Family ids with zero support are
        # ignored.
        if y_family.numel() > 0:
            counts = torch.bincount(
                y_family.to(torch.long).view(-1), minlength=num_families
            )
            majority_fid = int(torch.argmax(counts).item())
        else:
            majority_fid = 0
        majority_family = torch.full((int(y.numel()),), majority_fid, dtype=torch.long)
        baseline_family_sources["majority"] = {
            "family": majority_family,
            "confidence": torch.ones_like(majority_family, dtype=torch.float32),
            "constant_family": majority_fid,
            "family_counts": counts.tolist() if y_family.numel() else [],
        }

    variants = []
    for ns_policy in non_sufficient_policies:
        family_policy = _family_policy(
            certificate,
            modality_leaky_policy=modality_leaky_policy,
            non_sufficient_policy=ns_policy,
        )
        oracle = _route(
            true_labels=y,
            process_logits=process,
            direct_logits=direct,
            full_logits=full,
            route_family=y_family,
            route_confidence=torch.ones_like(y_family, dtype=torch.float32),
            family_policy=family_policy,
            threshold=0.0,
            num_classes=num_classes,
        )
        predicted_rows = [
            _route(
                true_labels=y,
                process_logits=process,
                direct_logits=direct,
                full_logits=full,
                route_family=predicted_family,
                route_confidence=predicted_confidence,
                family_policy=family_policy,
                threshold=threshold,
                num_classes=num_classes,
            )
            for threshold in thresholds
        ]
        # Process-logit-free baselines: random / majority family selection.
        baseline_routes = {}
        for src_name, src in baseline_family_sources.items():
            row = _route(
                true_labels=y,
                process_logits=process,
                direct_logits=direct,
                full_logits=full,
                route_family=src["family"],
                route_confidence=src["confidence"],
                family_policy=family_policy,
                threshold=0.0,
                num_classes=num_classes,
            )
            row["family_source"] = src_name
            if src.get("constant_family") is not None:
                row["constant_family"] = int(src["constant_family"])
                row["constant_family_name"] = (
                    REAL_FAMILY_NAMES[int(src["constant_family"])]
                    if 0 <= int(src["constant_family"]) < len(REAL_FAMILY_NAMES)
                    else str(src["constant_family"])
                )
            if "seed" in src:
                row["seed"] = int(src["seed"])
            baseline_routes[src_name] = row
        variants.append(
            {
                "non_sufficient_policy": ns_policy,
                "modality_leaky_policy": modality_leaky_policy,
                "family_policy": family_policy,
                "oracle_family_routing": oracle,
                "predicted_family_routing": predicted_rows,
                "baseline_family_routing": baseline_routes,
            }
        )

    return {
        **FAMILY_ROUTER_METADATA,
        "available": True,
        "num_samples": int(y.numel()),
        "num_classes": num_classes,
        "num_families": num_families,
        "per_sample_saved": bool(save_per_sample),
        "per_sample_route_variant": {
            "family_source": "predicted",
            "non_sufficient_policy": primary_ns_policy,
            "modality_leaky_policy": modality_leaky_policy,
            "threshold": primary_threshold,
        },
        "modality_leaky_policy": modality_leaky_policy,
        "non_sufficient_policies_evaluated": list(non_sufficient_policies),
        "baseline_family_policies_evaluated": list(baseline_family_policies),
        "family_prediction_accuracy": family_pred_acc,
        "family_prediction_accuracy_note": (
            "Computed from process-head class-softmax aggregation into family "
            "probabilities; the auxiliary family classifier is not used for routing."
        ),
        "family_prediction_confusion": family_confusion,
        "per_sample": primary_per_sample if save_per_sample else [],
        "_primary_route_code": primary_route_code,
        "_predicted_family": predicted_family.to(torch.long).tolist(),
        "_family_confidence": predicted_confidence.to(torch.float32).tolist(),
        "baseline": baseline,
        "policy_variants": variants,
        "note": (
            "Each entry in `policy_variants` is one routing policy. Within a variant, "
            "oracle-family routing is an analysis upper bound (uses true Y_family); "
            "predicted-family routing uses process-only class logits to infer the "
            "family and is the deployable protocol. baseline_family_routing reports "
            "the same pipeline applied to random and constant-majority family targets "
            "(process logits ignored), which establishes the process-free lower bound "
            "of the routing protocol. The variant with the higher predicted-family "
            "macro-F1 at the same coverage is the recommended deployment policy."
        ),
    }


def _family_policy(
    certificate: dict[str, Any],
    *,
    modality_leaky_policy: str,
    non_sufficient_policy: str = "full",
) -> dict[str, dict[str, Any]]:
    policy: dict[str, dict[str, Any]] = {}
    for row in certificate.get("families", []):
        fam = int(row.get("family", -1))
        cert = str(row.get("sufficiency_certificate", "undetermined"))
        if cert == "process_sufficient":
            route = "process"
        elif cert == "classification_sufficient_modality_leaky":
            route = modality_leaky_policy
        elif cert == "non_sufficient":
            route = non_sufficient_policy
        else:
            route = "abstain"
        policy[str(fam)] = {
            "family_name": row.get("family_name", REAL_FAMILY_NAMES[fam] if 0 <= fam < len(REAL_FAMILY_NAMES) else str(fam)),
            "certificate": cert,
            "route": route,
            "n": row.get("n"),
        }
    return policy


def _predict_family_from_process_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = logits.softmax(dim=-1)
    num_families = int(max(REAL_LABEL_TO_FAMILY) + 1)
    family_probs = torch.zeros(logits.shape[0], num_families, dtype=probs.dtype)
    for cls_idx, fam_idx in enumerate(REAL_LABEL_TO_FAMILY):
        if cls_idx < probs.shape[-1]:
            family_probs[:, int(fam_idx)] += probs[:, cls_idx]
    confidence, family = family_probs.max(dim=-1)
    return family.to(torch.long), confidence.to(torch.float32)


def _family_confusion(
    predicted: torch.Tensor, true: torch.Tensor, num_families: int
) -> dict[str, Any]:
    """Confusion matrix for predicted family vs. true Y_family.

    Rows = true family, columns = predicted family (sklearn convention).
    Includes per-row counts/recalls so reviewers can see which families are
    confused for which without re-deriving from the matrix.
    """
    pred = predicted.to(torch.long).view(-1)
    tru = true.to(torch.long).view(-1)
    matrix = torch.zeros(num_families, num_families, dtype=torch.long)
    for t, p in zip(tru.tolist(), pred.tolist()):
        if 0 <= t < num_families and 0 <= p < num_families:
            matrix[t, p] += 1
    family_names = [
        REAL_FAMILY_NAMES[i] if 0 <= i < len(REAL_FAMILY_NAMES) else str(i)
        for i in range(num_families)
    ]
    row_totals = matrix.sum(dim=1)
    row_recall = []
    for i in range(num_families):
        total = int(row_totals[i].item())
        if total == 0:
            row_recall.append(float("nan"))
        else:
            row_recall.append(float(matrix[i, i].item()) / float(total))
    return {
        "family_names": family_names,
        "matrix": matrix.tolist(),
        "row_totals": row_totals.tolist(),
        "row_recall": row_recall,
        "axis_note": "rows = true Y_family, columns = predicted family",
    }


def _per_sample_records(
    *,
    true_labels: torch.Tensor,
    true_families: torch.Tensor,
    sample_ids: list[str],
    process_logits: torch.Tensor,
    direct_logits: torch.Tensor,
    full_logits: torch.Tensor,
    route_family: torch.Tensor,
    route_confidence: torch.Tensor,
    family_policy: dict[str, dict[str, Any]],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[int]]:
    process_pred = process_logits.argmax(dim=-1).to(torch.long)
    direct_pred = direct_logits.argmax(dim=-1).to(torch.long)
    full_pred = full_logits.argmax(dim=-1).to(torch.long)
    pathway_pred = {
        "process": process_pred,
        "direct": direct_pred,
        "full": full_pred,
    }
    records: list[dict[str, Any]] = []
    route_codes: list[int] = []
    for idx in range(int(true_labels.numel())):
        fam = int(route_family[idx].item())
        conf = float(route_confidence[idx].item())
        if conf < float(threshold):
            route = "abstain"
        else:
            route = str(family_policy.get(str(fam), {"route": "abstain"})["route"])
        if route not in ROUTE_CODE:
            raise ValueError(f"Unknown route '{route}' produced by family_policy at idx={idx}")
        route_codes.append(int(ROUTE_CODE[route]))
        covered = route != "abstain"
        final_pred = int(pathway_pred[route][idx].item()) if covered else None
        true_label = int(true_labels[idx].item())
        records.append(
            {
                "sample_index": idx,
                "sample_id": sample_ids[idx] if idx < len(sample_ids) else f"idx_{idx}",
                "true_label": true_label,
                "true_family": _family_name(int(true_families[idx].item())),
                "predicted_family": _family_name(fam),
                "family_confidence": conf,
                "route": route,
                "covered": bool(covered),
                "process_pred": int(process_pred[idx].item()),
                "direct_pred": int(direct_pred[idx].item()),
                "full_pred": int(full_pred[idx].item()),
                "final_pred": final_pred,
                "correct": bool(final_pred == true_label) if covered else None,
            }
        )
    return records, route_codes


def _extract_sample_ids(batch: dict[str, Any], batch_size: int, *, start_index: int) -> list[str]:
    for key in ("sample_id", "sample_ids", "id", "ids", "path", "paths"):
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return [str(value.item())]
            return [str(v) for v in value.detach().cpu().view(-1).tolist()]
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [value]
    return [f"idx_{start_index + i}" for i in range(batch_size)]


def _normalized_sample_ids(value: Any, n: int) -> list[str]:
    if isinstance(value, list):
        out = [str(v) for v in value[:n]]
    else:
        out = []
    if len(out) < n:
        out.extend(f"idx_{i}" for i in range(len(out), n))
    return out


def _family_name(family_idx: int) -> str:
    if 0 <= int(family_idx) < len(REAL_FAMILY_NAMES):
        return REAL_FAMILY_NAMES[int(family_idx)]
    return str(family_idx)


def _write_logits_npz(path: Path, data: dict[str, Any], result: dict[str, Any]) -> None:
    import numpy as np

    route_code = result.get("_primary_route_code")
    if route_code is None:
        raise ValueError("Primary route codes are unavailable; cannot write logits NPZ")
    np.savez_compressed(
        path,
        process_logits=data["process"].detach().cpu().numpy(),
        direct_logits=data["direct"].detach().cpu().numpy(),
        full_logits=data["full"].detach().cpu().numpy(),
        true_label=data["Y"].to(torch.long).detach().cpu().numpy(),
        true_family=data["Y_family"].to(torch.long).detach().cpu().numpy(),
        predicted_family=np.asarray(result.get("_predicted_family", []), dtype=np.int64),
        family_confidence=np.asarray(result.get("_family_confidence", []), dtype=np.float32),
        route_code=np.asarray(route_code, dtype=np.int64),
    )


def _infer_seed_label(*paths: str) -> str:
    for raw in paths:
        for part in Path(raw).parts:
            if part.startswith("seed"):
                return part
    return "seed_unknown"


def _route(
    *,
    true_labels: torch.Tensor,
    process_logits: torch.Tensor,
    direct_logits: torch.Tensor,
    full_logits: torch.Tensor,
    route_family: torch.Tensor,
    route_confidence: torch.Tensor,
    family_policy: dict[str, dict[str, Any]],
    threshold: float,
    num_classes: int,
) -> dict[str, Any]:
    keep_indices: list[int] = []
    routed_logits: list[torch.Tensor] = []
    routed_predictions: list[int] = []
    routed_paths: list[str] = []
    route_counts: Counter[str] = Counter()
    pathway_lookup = {
        "process": process_logits,
        "direct": direct_logits,
        "full": full_logits,
    }
    for idx in range(int(true_labels.numel())):
        fam = int(route_family[idx].item())
        conf = float(route_confidence[idx].item())
        if conf < float(threshold):
            route = "abstain"
        else:
            route = str(family_policy.get(str(fam), {"route": "abstain"})["route"])
        route_counts[route] += 1
        if route == "abstain":
            continue
        if route not in pathway_lookup:
            # Defensive: an unknown route name should not silently route to process.
            raise ValueError(f"Unknown route '{route}' produced by family_policy at idx={idx}")
        keep_indices.append(idx)
        selected_logits = pathway_lookup[route][idx]
        routed_logits.append(selected_logits)
        routed_predictions.append(int(selected_logits.argmax(dim=-1).item()))
        routed_paths.append(route)
    if keep_indices:
        idx_tensor = torch.tensor(keep_indices, dtype=torch.long)
        logits = torch.stack(routed_logits, dim=0)
        labels = true_labels[idx_tensor]
        macro_f1 = macro_f1_from_logits(logits, labels, num_classes)
        accuracy = accuracy_from_logits(logits, labels)
    else:
        macro_f1 = float("nan")
        accuracy = float("nan")
    n_total = int(true_labels.numel())
    n_covered = len(keep_indices)
    return {
        "threshold": float(threshold),
        "num_total": n_total,
        "num_covered": n_covered,
        "coverage": float(n_covered / n_total) if n_total else float("nan"),
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "route_counts": dict(route_counts),
        "covered_indices": keep_indices,
        "pred_labels": routed_predictions,
        "route_paths": routed_paths,
    }


def render_markdown(result: dict[str, Any]) -> str:
    base = result["baseline"]
    if not result.get("available", True):
        lines = [
            "# Selective Process-Sufficiency Deployment Routing",
            "",
            f"- `split`: {result.get('split', 'NA')}",
            f"- `num_samples`: {result.get('num_samples', 'NA')}",
            f"- `available`: false",
            f"- `reason`: {result.get('reason', 'routing is not applicable')}",
            "",
            "## Baseline",
            "",
            "| path | macro-F1 | accuracy |",
            "|---|---:|---:|",
            f"| full-additive | {_fmt(base.get('full_additive_macro_f1'))} | {_fmt(base.get('full_additive_accuracy'))} |",
            "",
            "## Note",
            "",
            "Dual-pathway outputs were unavailable, so process/direct routing is intentionally not reported.",
        ]
        return "\n".join(lines) + "\n"

    lines = [
        "# Selective Process-Sufficiency Deployment Routing",
        "",
        f"- `split`: {result['split']}",
        f"- `num_samples`: {result['num_samples']}",
        f"- `family_prediction_accuracy`: {result['family_prediction_accuracy']:.4g}",
        f"- `family_router_source`: `{result.get('family_router_source', 'NA')}`",
        f"- `uses_auxiliary_family_head_for_routing`: `{result.get('uses_auxiliary_family_head_for_routing', False)}`",
        f"- `modality_leaky_policy`: `{result['modality_leaky_policy']}`",
        f"- `non_sufficient_policies_evaluated`: `{result.get('non_sufficient_policies_evaluated', [])}`",
        "",
        "## Baselines",
        "",
        "| path | macro-F1 | accuracy |",
        "|---|---:|---:|",
        f"| process-only | {base['process_macro_f1']:.4g} | {base['process_accuracy']:.4g} |",
        f"| direct-only | {base['direct_macro_f1']:.4g} | {base.get('direct_accuracy', float('nan')):.4g} |",
        f"| full-additive | {base['full_additive_macro_f1']:.4g} | {base['full_additive_accuracy']:.4g} |",
    ]
    for variant in result.get("policy_variants", []):
        ns_policy = variant["non_sufficient_policy"]
        oracle = variant["oracle_family_routing"]
        lines.extend(
            [
                "",
                f"## Variant: non_sufficient -> `{ns_policy}`",
                "",
                "### Family Policy",
                "",
                "| family | certificate | route | n |",
                "|---|---|---|---:|",
            ]
        )
        for fam, row in sorted(variant["family_policy"].items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {row['family_name']} | {row['certificate']} | {row['route']} | {row.get('n', 'NA')} |"
            )
        lines.extend(
            [
                "",
                "### Oracle-Family Routing",
                "",
                f"- `coverage`: {oracle['coverage']:.4g}",
                f"- `macro_f1`: {oracle['macro_f1']:.4g}",
                f"- `accuracy`: {oracle['accuracy']:.4g}",
                f"- `route_counts`: `{oracle['route_counts']}`",
                "",
                "### Predicted-Family Routing",
                "",
                "| confidence threshold | coverage | macro-F1 | accuracy | route counts |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for row in variant["predicted_family_routing"]:
            lines.append(
                f"| {row['threshold']:.3g} | {row['coverage']:.4g} | {_fmt(row['macro_f1'])} | {_fmt(row['accuracy'])} | `{row['route_counts']}` |"
            )

        baseline_routes = variant.get("baseline_family_routing", {}) or {}
        if baseline_routes:
            lines.extend(
                [
                    "",
                    "### Process-Free Baseline-Family Routing",
                    "",
                    "These rows ignore the process-only logits when picking a family. "
                    "They establish the lower bound of the routing protocol — anything "
                    "above this line is genuinely paying for the process-only family "
                    "predictor, not just for the per-family certificate.",
                    "",
                    "| baseline | coverage | macro-F1 | accuracy | route counts | details |",
                    "|---|---:|---:|---:|---|---|",
                ]
            )
            for src_name in sorted(baseline_routes):
                row = baseline_routes[src_name]
                detail = ""
                if "constant_family_name" in row:
                    detail = f"constant family = `{row['constant_family_name']}`"
                elif "seed" in row:
                    detail = f"seed = {row['seed']}"
                lines.append(
                    f"| {src_name} | {row['coverage']:.4g} | {_fmt(row['macro_f1'])} | "
                    f"{_fmt(row['accuracy'])} | `{row['route_counts']}` | {detail} |"
                )

    # Family-prediction confusion (between process-logit-derived family and
    # the true Y_family). Reviewers want to know which families the routing
    # head misroutes into which others, not just the scalar accuracy.
    confusion = result.get("family_prediction_confusion")
    if confusion:
        lines.extend(["", "## Family Prediction Confusion", "", confusion["axis_note"], ""])
        names = confusion["family_names"]
        header = "| true \\\\ pred | " + " | ".join(names) + " | n | recall |"
        sep = "|---|" + "|".join(["---:"] * len(names)) + "|---:|---:|"
        lines.append(header)
        lines.append(sep)
        for i, row in enumerate(confusion["matrix"]):
            cells = " | ".join(str(int(v)) for v in row)
            n_i = int(confusion["row_totals"][i])
            recall_i = confusion["row_recall"][i]
            recall_str = "NA" if recall_i != recall_i else f"{recall_i:.4g}"
            lines.append(f"| {names[i]} | {cells} | {n_i} | {recall_str} |")

    # Cross-variant comparison summary
    lines.extend(["", "## Cross-Variant Summary (Oracle)", "", "| non_sufficient policy | coverage | macro-F1 | accuracy |", "|---|---:|---:|---:|"])
    for variant in result.get("policy_variants", []):
        oracle = variant["oracle_family_routing"]
        lines.append(
            f"| {variant['non_sufficient_policy']} | {oracle['coverage']:.4g} | {_fmt(oracle['macro_f1'])} | {_fmt(oracle['accuracy'])} |"
        )

    lines.extend(["", "## Note", "", result["note"]])
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4g}"
    except Exception:
        return "NA"


if __name__ == "__main__":
    main()
