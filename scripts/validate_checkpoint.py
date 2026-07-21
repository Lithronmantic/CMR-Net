from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

EXPECTED_CONFIG = {
    "model.d_C": 7,
    "model.d_P_obs": 6,
    "model.d_z_T": 64,
    "model.d_Z_P": 32,
    "model.d_Z_P_anchor": 16,
    "model.d_Z_A": 32,
    "model.d_Z_V": 32,
    "model.num_classes": 12,
    "model.num_families": 6,
    "model.pathway_decomposition.enabled": True,
    "model.pathway_decomposition.inference_mediated_only": False,
    "model.pathway_decomposition.direct_condition_on_mediated": False,
    "model.pathway_decomposition.direct_condition_on_context": False,
    "model.pathway_decomposition.direct_bottleneck_dim": 32,
    "model.pathway_decomposition.direct_gate_enabled": True,
    "model.pathway_decomposition.direct_gate_alpha_max": 0.3,
    "model.av_t_adversary.enabled": True,
    "model.av_t_adversary.use_residual_for_direct": True,
}

DIRECT_INPUT_WEIGHT_KEYS = (
    "direct_head.modality_proj.1.weight",
    "direct_head.modality_proj.0.weight",
    "direct_head.head.1.weight",
    "direct_head.head.0.weight",
)

DISALLOWED_PREFIXES = (
    "process_teacher.",
    "teacher.",
    "process_teacher_model.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    return parser.parse_args()


def load_config(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path)


def cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    cursor = cfg
    for token in dotted.split("."):
        if isinstance(cursor, dict):
            if token not in cursor:
                return default
            cursor = cursor[token]
        else:
            try:
                cursor = cursor[token]
            except Exception:
                return default
    return cursor


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
                return normalize_keys(candidate), key
        if payload and all(isinstance(key, str) for key in payload):
            return normalize_keys(payload), "checkpoint-root"
    raise TypeError("State dictionary not found in checkpoint")


def normalize_keys(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean = key[7:] if key.startswith("module.") else key
        if isinstance(value, torch.Tensor):
            normalized[clean] = value
    return normalized


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    return repr(value)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu")
    state, state_source = select_state_dict(payload)
    errors: list[str] = []
    passes: list[str] = []

    for dotted, expected in EXPECTED_CONFIG.items():
        actual = cfg_get(cfg, dotted, None)
        if actual == expected:
            passes.append(f"{dotted} = {format_value(actual)}")
        else:
            errors.append(
                f"{dotted}: expected {format_value(expected)}, found {format_value(actual)}"
            )

    direct_key = None
    direct_shape = None
    for key in DIRECT_INPUT_WEIGHT_KEYS:
        tensor = state.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            direct_key = key
            direct_shape = tuple(int(value) for value in tensor.shape)
            break

    if direct_shape is None:
        errors.append("Direct-head input weight not found")
    else:
        direct_input_dim = direct_shape[1]
        if direct_input_dim == 64:
            passes.append(f"{direct_key} input dimension = 64")
        else:
            errors.append(f"{direct_key} input dimension = {direct_input_dim}; expected 64")

    disallowed = sorted(key for key in state if key.startswith(DISALLOWED_PREFIXES))
    if disallowed:
        errors.append(f"Checkpoint contains {len(disallowed)} unsupported teacher parameters")

    print("CMR-Net checkpoint validation")
    print(f"config: {args.config}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"state source: {state_source}")

    if passes:
        print("PASS")
        for item in passes:
            print(item)

    if errors:
        print("FAIL")
        for item in errors:
            print(item)
        return 1

    print("Validation completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
