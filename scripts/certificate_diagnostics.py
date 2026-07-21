from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmr_net import CMRNet
from cmr_net.data import build_loaders
from cmr_net.data.dataset import REAL_FAMILY_NAMES
from cmr_net.evaluation.diagnostics import _analytical_chsic_permutation

AUDIO_RESIDUAL_KEYS = (
    "Z_A_perp",
    "Z_A_residual",
    "audio_residual",
    "audio_residualized",
)
VIDEO_RESIDUAL_KEYS = (
    "Z_V_perp",
    "Z_V_residual",
    "video_residual",
    "video_residualized",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, nargs="+", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "certificate")
    parser.add_argument("--n-permutation", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--min-family-n", type=int, default=40)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--equivalence-delta", type=float, default=5.0e-5)
    parser.add_argument("--f1-equivalence-margin", type=float, default=0.03)
    parser.add_argument("--direct-gap-margin", type=float, default=0.03)
    parser.add_argument("--delta-grid", default="1e-5,3e-5,5e-5,7e-5,1e-4,5e-4")
    parser.add_argument("--debug-lenient", action="store_true")
    return parser.parse_args()


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device(value)


def load_config(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path)


def select_loader(cfg: DictConfig, split: str):
    train_loader, val_loader, test_loader = build_loaders(cfg)
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    return test_loader


def select_state_dict(payload: Any) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(payload, dict):
        for key in (
            "best_stage4_ema_state_dict",
            "best_ema_state_dict",
            "ema_state_dict",
            "model_state_dict",
            "state_dict",
            "model",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return normalize_state(candidate), key
        if payload and all(isinstance(key, str) for key in payload):
            return normalize_state(payload), "checkpoint-root"
    raise TypeError("State dictionary not found in checkpoint")


def normalize_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean = key[7:] if key.startswith("module.") else key
        if isinstance(value, torch.Tensor):
            normalized[clean] = value
    return normalized


def load_model(
    cfg: DictConfig,
    checkpoint: Path,
    device: torch.device,
    debug_lenient: bool,
) -> tuple[CMRNet, dict[str, Any]]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu")
    state, source = select_state_dict(payload)
    model = CMRNet(cfg)
    result = model.load_state_dict(state, strict=not debug_lenient)
    model.to(device)
    model.eval()
    metadata = {
        "checkpoint": str(checkpoint),
        "state_source": source,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
    return model, metadata


def require_tensor(outputs: dict[str, Any], keys: tuple[str, ...], name: str) -> torch.Tensor:
    for key in keys:
        value = outputs.get(key)
        if isinstance(value, torch.Tensor):
            return value
    raise KeyError(f"Required output not found: {name}")


def collect(model: torch.nn.Module, loader: Any) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    store: dict[str, list[torch.Tensor]] = {
        "logits_process": [],
        "logits_direct": [],
        "logits_full": [],
        "Z_P": [],
        "Z_A_perp": [],
        "Z_V_perp": [],
        "T": [],
        "C": [],
        "Y": [],
        "Y_family": [],
    }
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            outputs = model(batch)
            process_logits = outputs.get("logits_mediated")
            direct_logits = outputs.get("logits_direct")
            full_logits = outputs.get("logits_pathway")
            if not all(isinstance(value, torch.Tensor) for value in (process_logits, direct_logits, full_logits)):
                raise KeyError("Process, Direct, and Full pathway logits are required")
            audio_residual = require_tensor(outputs, AUDIO_RESIDUAL_KEYS, "Z_A_perp")
            video_residual = require_tensor(outputs, VIDEO_RESIDUAL_KEYS, "Z_V_perp")
            store["logits_process"].append(process_logits.detach().cpu())
            store["logits_direct"].append(direct_logits.detach().cpu())
            store["logits_full"].append(full_logits.detach().cpu())
            store["Z_P"].append(outputs["Z_P"].detach().cpu())
            store["Z_A_perp"].append(audio_residual.detach().cpu())
            store["Z_V_perp"].append(video_residual.detach().cpu())
            for key in ("T", "C", "Y", "Y_family"):
                store[key].append(batch[key].detach().cpu())
    return {key: torch.cat(parts, dim=0) for key, parts in store.items()}


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_classes: int | None = None) -> float:
    predictions = logits.argmax(dim=-1).to(torch.long)
    labels = labels.to(torch.long)
    if num_classes is None:
        num_classes = int(logits.shape[-1])
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    for label, prediction in zip(labels.view(-1), predictions.view(-1), strict=False):
        label_value = int(label)
        prediction_value = int(prediction)
        if 0 <= label_value < num_classes and 0 <= prediction_value < num_classes:
            confusion[label_value, prediction_value] += 1
    true_positive = confusion.diag()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1.0)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1.0)
    score = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
    support = confusion.sum(dim=1) > 0
    return float(score[support].mean().item()) if support.any() else float("nan")


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def bootstrap_gap(
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
    labels: torch.Tensor,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    n = int(labels.numel())
    num_classes = int(max(left_logits.shape[-1], right_logits.shape[-1]))
    values: list[float] = []
    for _ in range(n_bootstrap):
        indices = torch.randint(0, n, (n,), generator=generator)
        left_score = macro_f1(left_logits[indices], labels[indices], num_classes)
        right_score = macro_f1(right_logits[indices], labels[indices], num_classes)
        values.append(right_score - left_score)
    return {
        "mean": float(sum(values) / len(values)),
        "low": percentile(values, 0.025),
        "high": percentile(values, 0.975),
    }


def candidate_diagnostics(
    data: dict[str, torch.Tensor],
    family: int,
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, Any]:
    mask = data["Y_family"] == family
    labels = data["Y"][mask]
    n = int(mask.sum().item())
    result: dict[str, Any] = {
        "family": family,
        "family_name": REAL_FAMILY_NAMES[family] if family < len(REAL_FAMILY_NAMES) else str(family),
        "n": n,
        "process_macro_f1": macro_f1(data["logits_process"][mask], labels),
        "direct_macro_f1": macro_f1(data["logits_direct"][mask], labels),
        "full_macro_f1": macro_f1(data["logits_full"][mask], labels),
    }
    if n < args.min_family_n:
        result["candidate_state"] = "IND"
        return result
    full_gap = bootstrap_gap(
        data["logits_process"][mask],
        data["logits_full"][mask],
        labels,
        args.n_bootstrap,
        args.bootstrap_seed + seed_offset * 10000 + family,
    )
    direct_gap = bootstrap_gap(
        data["logits_process"][mask],
        data["logits_direct"][mask],
        labels,
        args.n_bootstrap,
        args.bootstrap_seed + seed_offset * 10000 + 1000 + family,
    )
    condition = torch.cat((data["Z_P"][mask], data["C"][mask]), dim=-1)
    tests: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
        "audio": (data["Z_A_perp"][mask], data["T"][mask], condition),
        "video": (data["Z_V_perp"][mask], data["T"][mask], condition),
    }
    if int(labels.unique().numel()) >= 2:
        y_onehot = torch.nn.functional.one_hot(
            labels.to(torch.long), num_classes=int(data["logits_full"].shape[-1])
        ).to(torch.float32)
        tests["y"] = (y_onehot, data["T"][mask], condition)
    observed: dict[str, float | None] = {"audio": None, "video": None, "y": None}
    for offset, (name, values) in enumerate(tests.items()):
        stats = _analytical_chsic_permutation(
            X=values[0],
            Y=values[1],
            Z=values[2],
            ridge=args.ridge,
            n_permutation=args.n_permutation,
            seed=args.bootstrap_seed + seed_offset * 1000 + family * 10 + offset,
        )
        observed[name] = float(stats["observed"])
        result[f"chsic_{name}"] = float(stats["observed"])
        result[f"p_{name}"] = float(stats["p_value"])
        result[f"ok_{name}"] = bool(float(stats["observed"]) <= args.equivalence_delta)
    y_ok = True if observed["y"] is None else bool(observed["y"] <= args.equivalence_delta)
    audio_ok = bool(observed["audio"] is not None and observed["audio"] <= args.equivalence_delta)
    video_ok = bool(observed["video"] is not None and observed["video"] <= args.equivalence_delta)
    ni = bool(full_gap["high"] <= args.f1_equivalence_margin)
    superiority = bool(direct_gap["low"] > args.direct_gap_margin)
    if ni and y_ok and audio_ok and video_ok and not superiority:
        state = "PS"
    elif ni and y_ok and ((not audio_ok) or (not video_ok) or superiority):
        state = "PS-RMD"
    else:
        state = "PI"
    result.update(
        {
            "full_minus_process": result["full_macro_f1"] - result["process_macro_f1"],
            "direct_minus_process": result["direct_macro_f1"] - result["process_macro_f1"],
            "full_minus_process_ci_low": full_gap["low"],
            "full_minus_process_ci_high": full_gap["high"],
            "direct_minus_process_ci_low": direct_gap["low"],
            "direct_minus_process_ci_high": direct_gap["high"],
            "predictive_noninferiority": ni,
            "residual_pathway_superiority": superiority,
            "candidate_state": state,
        }
    )
    return result


def pooled_state(
    data_by_seed: list[dict[str, torch.Tensor]],
    candidates: list[dict[str, Any]],
    family: int,
    args: argparse.Namespace,
    active_seed_indices: list[int],
    equivalence_delta: float,
    seed_value: int,
) -> dict[str, Any]:
    first_data = data_by_seed[active_seed_indices[0]]
    mask = first_data["Y_family"] == family
    labels = first_data["Y"][mask]
    n = int(mask.sum().item())
    if n < args.min_family_n:
        return {"state": "IND", "n": n}
    generator = torch.Generator().manual_seed(seed_value)
    full_values: list[float] = []
    direct_values: list[float] = []
    num_classes = int(first_data["logits_full"].shape[-1])
    for _ in range(args.n_bootstrap):
        sampled_positions = torch.randint(
            0,
            len(active_seed_indices),
            (len(active_seed_indices),),
            generator=generator,
        )
        sampled_seeds = [active_seed_indices[int(position)] for position in sampled_positions]
        sampled_instances = torch.randint(0, n, (n,), generator=generator)
        full_gaps: list[float] = []
        direct_gaps: list[float] = []
        for seed_index in sampled_seeds:
            data = data_by_seed[seed_index]
            process_score = macro_f1(
                data["logits_process"][mask][sampled_instances],
                labels[sampled_instances],
                num_classes,
            )
            full_score = macro_f1(
                data["logits_full"][mask][sampled_instances],
                labels[sampled_instances],
                num_classes,
            )
            direct_score = macro_f1(
                data["logits_direct"][mask][sampled_instances],
                labels[sampled_instances],
                num_classes,
            )
            full_gaps.append(full_score - process_score)
            direct_gaps.append(direct_score - process_score)
        full_values.append(float(sum(full_gaps) / len(full_gaps)))
        direct_values.append(float(sum(direct_gaps) / len(direct_gaps)))
    upper = percentile(full_values, 0.975)
    lower = percentile(direct_values, 0.025)
    threshold = len(active_seed_indices) // 2 + 1
    selected_candidates = [candidates[index] for index in active_seed_indices]
    y_count = sum(
        1
        for row in selected_candidates
        if row.get("chsic_y") is None or float(row["chsic_y"]) <= equivalence_delta
    )
    audio_count = sum(
        1
        for row in selected_candidates
        if row.get("chsic_audio") is not None and float(row["chsic_audio"]) <= equivalence_delta
    )
    video_count = sum(
        1
        for row in selected_candidates
        if row.get("chsic_video") is not None and float(row["chsic_video"]) <= equivalence_delta
    )
    ni = bool(upper <= args.f1_equivalence_margin)
    superiority = bool(lower > args.direct_gap_margin)
    y_ok = y_count >= threshold
    audio_ok = audio_count >= threshold
    video_ok = video_count >= threshold
    if ni and y_ok and audio_ok and video_ok and not superiority:
        state = "PS"
    elif ni and y_ok and ((not audio_ok) or (not video_ok) or superiority):
        state = "PS-RMD"
    else:
        state = "PI"
    return {
        "state": state,
        "n": n,
        "upper_full_minus_process": upper,
        "lower_direct_minus_process": lower,
        "predictive_noninferiority": ni,
        "residual_pathway_superiority": superiority,
        "y_pass_count": y_count,
        "audio_pass_count": audio_count,
        "video_pass_count": video_count,
        "majority_threshold": threshold,
    }


def validate_alignment(reference: dict[str, torch.Tensor], current: dict[str, torch.Tensor]) -> None:
    for key in ("Y", "Y_family"):
        if not torch.equal(reference[key], current[key]):
            raise ValueError(f"Checkpoint predictions are not aligned for {key}")


def analyze_family(
    data_by_seed: list[dict[str, torch.Tensor]],
    candidates_by_seed: list[list[dict[str, Any]]],
    family: int,
    args: argparse.Namespace,
    delta_grid: list[float],
) -> dict[str, Any]:
    candidates = [rows[family] for rows in candidates_by_seed]
    active = list(range(len(data_by_seed)))
    pooled = pooled_state(
        data_by_seed,
        candidates,
        family,
        args,
        active,
        args.equivalence_delta,
        args.bootstrap_seed + family * 100000,
    )
    loso_states: list[str] = []
    for omitted in active:
        retained = [index for index in active if index != omitted]
        loso = pooled_state(
            data_by_seed,
            candidates,
            family,
            args,
            retained,
            args.equivalence_delta,
            args.bootstrap_seed + 500000 + family * 10000 + omitted,
        )
        loso_states.append(str(loso["state"]))
    released = "IND" if any(state != pooled["state"] for state in loso_states) else str(pooled["state"])
    scan_states: dict[str, str] = {}
    for index, delta in enumerate(delta_grid):
        scan = pooled_state(
            data_by_seed,
            candidates,
            family,
            args,
            active,
            delta,
            args.bootstrap_seed + 900000 + family * 10000 + index,
        )
        scan_states[f"{delta:.8g}"] = str(scan["state"])
    boundary_sensitive = any(state != pooled["state"] for state in scan_states.values())
    family_name = REAL_FAMILY_NAMES[family] if family < len(REAL_FAMILY_NAMES) else str(family)
    return {
        "family": family,
        "family_name": family_name,
        "n": int(pooled["n"]),
        "candidate_diagnostics": candidates,
        "pooled": pooled,
        "loso_states": loso_states,
        "released_certificate": released,
        "boundary_sensitive": boundary_sensitive,
        "delta_scan": scan_states,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Family-Level Process-Sufficiency Certificates",
        "",
        "| Family | n | Pooled state | Released certificate | Boundary-sensitive | LOSO states |",
        "|---|---:|---|---|---|---|",
    ]
    for row in result["families"]:
        lines.append(
            f"| {row['family_name']} | {row['n']} | {row['pooled']['state']} | "
            f"{row['released_certificate']} | {str(row['boundary_sensitive']).lower()} | "
            f"{', '.join(row['loso_states'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    loader = select_loader(cfg, args.split)
    device = device_from_arg(args.device)
    data_by_seed: list[dict[str, torch.Tensor]] = []
    checkpoint_metadata: list[dict[str, Any]] = []
    for checkpoint in args.checkpoint:
        model, metadata = load_model(cfg, checkpoint, device, args.debug_lenient)
        data = collect(model, loader)
        if data_by_seed:
            validate_alignment(data_by_seed[0], data)
        data_by_seed.append(data)
        checkpoint_metadata.append(metadata)
    families = sorted(int(value) for value in data_by_seed[0]["Y_family"].unique().tolist())
    candidates_by_seed: list[list[dict[str, Any]]] = []
    for seed_index, data in enumerate(data_by_seed):
        rows = [candidate_diagnostics(data, family, args, seed_index) for family in families]
        candidates_by_seed.append(rows)
    delta_grid = [float(value.strip()) for value in args.delta_grid.split(",") if value.strip()]
    result = {
        "split": args.split,
        "checkpoints": checkpoint_metadata,
        "n_seeds": len(data_by_seed),
        "n_permutation": args.n_permutation,
        "n_bootstrap": args.n_bootstrap,
        "min_family_n": args.min_family_n,
        "ridge": args.ridge,
        "equivalence_delta": args.equivalence_delta,
        "f1_equivalence_margin": args.f1_equivalence_margin,
        "direct_gap_margin": args.direct_gap_margin,
        "families": [
            analyze_family(data_by_seed, candidates_by_seed, family, args, delta_grid)
            for family in families
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "family_certificates.json"
    markdown_path = args.output_dir / "family_certificates.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
