"""Train and evaluate an independent calibrated family router.

This is an evaluation-time router: it consumes CMR-Net features from a frozen
checkpoint and predicts the coarse defect family used by the process-sufficiency
certificate. It does not update CMR-Net.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmr_net.data.dataset import REAL_FAMILY_NAMES

from _causal_audit_common import load_config, load_model, macro_f1_from_logits, select_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--certificate", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="results/round21_family_router")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--thresholds", default="0,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--modality-leaky-policy", choices=("process", "direct", "full", "abstain"), default="process")
    parser.add_argument("--non-sufficient-policy", choices=("direct", "full", "abstain"), default="direct")
    parser.add_argument("--debug-lenient", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = _device(args.device)
    model, meta = load_model(cfg, args.checkpoint, device=str(device), debug_lenient=args.debug_lenient)
    splits = {
        "train": _collect(model, select_loader(cfg, "train"), device),
        "val": _collect(model, select_loader(cfg, "val"), device),
        "test": _collect(model, select_loader(cfg, "test"), device),
    }
    num_families = int(max(int(splits["train"]["family"].max().item()), int(splits["test"]["family"].max().item())) + 1)
    router = _FamilyRouter(splits["train"]["features"].shape[1], int(args.hidden_dim), num_families).to(device)
    _train_router(
        router,
        splits["train"]["features"].to(device),
        splits["train"]["family"].to(device),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    temperature = _fit_temperature(
        router,
        splits["val"]["features"].to(device),
        splits["val"]["family"].to(device),
    )
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    certificate = {}
    if args.certificate:
        certificate = json.loads(Path(args.certificate).read_text(encoding="utf-8"))
    result = _evaluate_router(
        router,
        splits["test"],
        temperature=temperature,
        thresholds=thresholds,
        certificate=certificate,
        modality_leaky_policy=args.modality_leaky_policy,
        non_sufficient_policy=args.non_sufficient_policy,
        num_families=num_families,
    )
    result["checkpoint"] = meta
    result["config"] = str(args.config)
    result["certificate"] = str(args.certificate) if args.certificate else None
    result["feature_dim"] = int(splits["train"]["features"].shape[1])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibrated_family_router.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "calibrated_family_router.md").write_text(_render_markdown(result), encoding="utf-8")
    print(f"Wrote {out_dir / 'calibrated_family_router.json'}")
    print(f"Wrote {out_dir / 'calibrated_family_router.md'}")


class _FamilyRouter(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, num_families: int) -> None:
        super().__init__()
        if d_hidden <= 0:
            self.net = nn.Linear(d_in, num_families)
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, d_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_hidden, num_families),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _collect(model: nn.Module, loader: Any, device: torch.device) -> dict[str, torch.Tensor]:
    parts: dict[str, list[torch.Tensor]] = {
        "features": [],
        "family": [],
        "label": [],
        "process": [],
        "direct": [],
        "full": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            process = outputs.get("logits_mediated", outputs["logits"])
            direct = outputs.get("logits_direct", torch.zeros_like(process))
            full = outputs.get("logits_pathway", outputs["logits"])
            z_p = outputs.get("Z_P")
            if not isinstance(z_p, torch.Tensor):
                z_p = process.new_zeros(process.shape[0], 0)
            prob = process.softmax(dim=-1)
            top2 = torch.topk(prob, k=min(2, prob.shape[-1]), dim=-1).values
            max_prob = top2[:, 0:1]
            margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1) if top2.shape[1] > 1 else max_prob
            entropy = -(prob * prob.clamp_min(1.0e-12).log()).sum(dim=-1, keepdim=True)
            features = torch.cat([z_p.float(), process.float(), max_prob, margin, entropy], dim=-1)
            parts["features"].append(features.detach().cpu())
            parts["family"].append(batch["Y_family"].detach().cpu())
            parts["label"].append(batch["Y"].detach().cpu())
            parts["process"].append(process.detach().cpu())
            parts["direct"].append(direct.detach().cpu())
            parts["full"].append(full.detach().cpu())
    return {key: torch.cat(value, dim=0) for key, value in parts.items()}


def _train_router(
    router: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    router.train()
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(max(1, epochs)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(router(x), y.long())
        loss.backward()
        opt.step()


def _fit_temperature(router: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    router.eval()
    with torch.no_grad():
        logits = router(x_val)
    best_temp = 1.0
    best_loss = math.inf
    for step in range(91):
        temp = 0.5 + step * 0.05
        loss = F.cross_entropy(logits / temp, y_val.long()).item()
        if loss < best_loss:
            best_loss = loss
            best_temp = temp
    return float(best_temp)


def _evaluate_router(
    router: nn.Module,
    data: dict[str, torch.Tensor],
    *,
    temperature: float,
    thresholds: list[float],
    certificate: dict[str, Any],
    modality_leaky_policy: str,
    non_sufficient_policy: str,
    num_families: int,
) -> dict[str, Any]:
    router.eval()
    device = next(router.parameters()).device
    with torch.no_grad():
        logits = router(data["features"].to(device)).cpu()
    probs = (logits / max(float(temperature), 1.0e-6)).softmax(dim=-1)
    confidence, pred_family = probs.max(dim=-1)
    family = data["family"].long()
    labels = data["label"].long()
    accuracy = float((pred_family == family).float().mean().item()) if family.numel() else math.nan
    ece = _ece(confidence, pred_family, family)
    policy = _family_policy(
        certificate,
        modality_leaky_policy=modality_leaky_policy,
        non_sufficient_policy=non_sufficient_policy,
    )
    curves = [
        _route(
            labels,
            pred_family,
            confidence,
            data["process"],
            data["direct"],
            data["full"],
            policy,
            threshold,
        )
        for threshold in thresholds
    ]
    return {
        "family_router_accuracy": accuracy,
        "family_router_ece": ece,
        "temperature": float(temperature),
        "num_samples": int(labels.numel()),
        "num_families": int(num_families),
        "family_confusion_matrix": _confusion(pred_family, family, num_families),
        "coverage_macro_f1_curve": [
            {"threshold": row["threshold"], "coverage": row["coverage"], "macro_f1": row["macro_f1"]}
            for row in curves
        ],
        "coverage_accuracy_curve": [
            {"threshold": row["threshold"], "coverage": row["coverage"], "accuracy": row["accuracy"]}
            for row in curves
        ],
        "routing_rows": curves,
        "policy": policy,
    }


def _family_policy(
    certificate: dict[str, Any],
    *,
    modality_leaky_policy: str,
    non_sufficient_policy: str,
) -> dict[str, str]:
    policy: dict[str, str] = {}
    for row in certificate.get("families", []):
        family = str(int(row.get("family", -1)))
        cert = str(row.get("sufficiency_certificate", "undetermined"))
        if cert == "process_sufficient":
            route = "process"
        elif cert == "classification_sufficient_modality_leaky":
            route = modality_leaky_policy
        elif cert == "non_sufficient":
            route = non_sufficient_policy
        else:
            route = "abstain"
        policy[family] = route
    return policy


def _route(
    labels: torch.Tensor,
    route_family: torch.Tensor,
    confidence: torch.Tensor,
    process: torch.Tensor,
    direct: torch.Tensor,
    full: torch.Tensor,
    policy: dict[str, str],
    threshold: float,
) -> dict[str, Any]:
    logits_by_route = {"process": process, "direct": direct, "full": full}
    keep: list[int] = []
    preds: list[int] = []
    route_counts: Counter[str] = Counter()
    for idx in range(int(labels.numel())):
        if float(confidence[idx].item()) < float(threshold):
            route = "abstain"
        else:
            route = policy.get(str(int(route_family[idx].item())), "abstain")
        route_counts[route] += 1
        if route == "abstain":
            continue
        keep.append(idx)
        preds.append(int(logits_by_route[route][idx].argmax(dim=-1).item()))
    if not keep:
        return {
            "threshold": float(threshold),
            "coverage": 0.0,
            "macro_f1": math.nan,
            "accuracy": math.nan,
            "num_covered": 0,
            "route_counts": dict(route_counts),
        }
    covered_labels = labels[torch.tensor(keep, dtype=torch.long)].tolist()
    num_classes = int(max(labels.max().item(), full.shape[-1] - 1) + 1)
    return {
        "threshold": float(threshold),
        "coverage": len(keep) / int(labels.numel()),
        "macro_f1": _macro_f1(covered_labels, preds, num_classes),
        "accuracy": float(sum(int(y == p) for y, p in zip(covered_labels, preds, strict=False)) / len(keep)),
        "num_covered": len(keep),
        "route_counts": dict(route_counts),
    }


def _ece(confidence: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, bins: int = 10) -> float:
    total = max(1, int(target.numel()))
    ece = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        mask = (confidence >= lo) & (confidence < hi if b < bins - 1 else confidence <= hi)
        count = int(mask.sum().item())
        if count == 0:
            continue
        acc = (pred[mask] == target[mask]).float().mean().item()
        conf = confidence[mask].mean().item()
        ece += (count / total) * abs(acc - conf)
    return float(ece)


def _confusion(pred: torch.Tensor, true: torch.Tensor, num_families: int) -> dict[str, Any]:
    matrix = torch.zeros(num_families, num_families, dtype=torch.long)
    for t, p in zip(true.long().tolist(), pred.long().tolist(), strict=False):
        if 0 <= t < num_families and 0 <= p < num_families:
            matrix[t, p] += 1
    return {
        "family_names": [
            REAL_FAMILY_NAMES[i] if i < len(REAL_FAMILY_NAMES) else str(i)
            for i in range(num_families)
        ],
        "matrix": matrix.tolist(),
        "axis_note": "rows=true family, columns=predicted family",
    }


def _macro_f1(labels: list[int], pred: list[int], num_classes: int) -> float:
    scores = []
    for cls in range(num_classes):
        tp = sum(1 for y, p in zip(labels, pred, strict=False) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(labels, pred, strict=False) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(labels, pred, strict=False) if y == cls and p != cls)
        if tp + fn == 0:
            continue
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(sum(scores) / len(scores)) if scores else math.nan


def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(device)


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Calibrated Family Router",
        "",
        f"- `family_router_accuracy`: {_fmt(result.get('family_router_accuracy'))}",
        f"- `family_router_ece`: {_fmt(result.get('family_router_ece'))}",
        f"- `temperature`: {_fmt(result.get('temperature'))}",
        "",
        "| threshold | coverage | macro-F1 | accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for row in result.get("routing_rows", []):
        lines.append(
            f"| {_fmt(row.get('threshold'))} | {_fmt(row.get('coverage'))} | "
            f"{_fmt(row.get('macro_f1'))} | {_fmt(row.get('accuracy'))} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.4g}"


if __name__ == "__main__":
    main()
