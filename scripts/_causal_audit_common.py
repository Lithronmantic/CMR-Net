"""Shared helpers for process-sufficiency audit scripts.

These helpers are deliberately lightweight and independent from Hydra entry
points so the audit scripts can be run on the server with an explicit
``--config`` / ``--checkpoint`` pair.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
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


def cfg_get(cfg: object, path: str, default: object = None) -> object:
    cursor = cfg
    for part in path.split("."):
        if isinstance(cursor, dict):
            if part not in cursor:
                return default
            cursor = cursor[part]
            continue
        if not hasattr(cursor, part):
            return default
        cursor = getattr(cursor, part)
    return cursor


def load_config(path: str | Path) -> DictConfig:
    path = Path(path)
    cfg = OmegaConf.load(path)
    defaults = cfg.get("defaults") if isinstance(cfg, DictConfig) else None
    if defaults is None:
        return cfg

    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    if not default_path.exists() or path.resolve() == default_path.resolve():
        cfg.pop("defaults", None)
        return cfg

    base = OmegaConf.load(default_path)
    cfg.pop("defaults", None)
    return OmegaConf.merge(base, cfg)


def select_loader(cfg: DictConfig, split: str):
    train_loader, val_loader, test_loader = build_loaders(cfg)
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    if split == "test":
        return test_loader
    raise ValueError(f"Unknown split '{split}'")


def load_model(cfg: DictConfig, checkpoint: str | Path, *, device: str = "auto", debug_lenient: bool = False) -> tuple[CMRNet, dict[str, Any]]:
    torch_device = _device(device)
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    raw = torch.load(path, map_location="cpu")
    state, source_key, uses_ema = select_state_dict(raw)
    cfg_for_model, inferred_overrides = reconcile_model_config_with_checkpoint(cfg, state)
    model = CMRNet(cfg_for_model)
    state, ignored = normalize_checkpoint_state(model, state)
    try:
        result = model.load_state_dict(state, strict=not debug_lenient)
    except RuntimeError:
        if not debug_lenient:
            raise
        result = model.load_state_dict(state, strict=False)
    model.to(torch_device)
    model.eval()
    meta = {
        "checkpoint": str(path),
        "source_key": source_key,
        "uses_ema": bool(uses_ema),
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "ignored_legacy_keys": ignored,
        "inferred_config_overrides": inferred_overrides,
        "device": str(torch_device),
    }
    return model, meta


def select_state_dict(checkpoint: object) -> tuple[dict, str, bool]:
    if isinstance(checkpoint, dict):
        for key, uses_ema in (
            ("best_stage4_ema_state_dict", True),
            ("best_ema_state_dict", True),
            ("ema_state_dict", True),
            ("model_state_dict", False),
            ("state_dict", False),
            ("model", False),
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, key, uses_ema
        return checkpoint, "checkpoint", False
    return checkpoint, "raw", False


def normalize_checkpoint_state(model: CMRNet, state: dict) -> tuple[dict, list[str]]:
    current_keys = set(model.state_dict().keys())
    has_current_dih_state = any(key.startswith("dih.") for key in current_keys)
    normalized: dict = {}
    ignored: list[str] = []
    for key, value in state.items():
        clean_key = key[7:] if isinstance(key, str) and key.startswith("module.") else key
        if clean_key in current_keys:
            normalized[clean_key] = value
            continue
        if isinstance(clean_key, str) and _is_known_legacy_key(clean_key, has_current_dih_state):
            ignored.append(clean_key)
            continue
        normalized[clean_key] = value
    return normalized, ignored


def reconcile_model_config_with_checkpoint(cfg: DictConfig, state: dict) -> tuple[DictConfig, dict[str, Any]]:
    """Patch architecture-only config fields that are unambiguous in a checkpoint.

    Audit scripts are often launched with hand-written override configs rather
    than Hydra's fully composed runtime config. If the override file omits the
    canonical defaults, CMRNet falls back to constructor defaults such as
    ``hidden_dim=128`` even though the checkpoint was trained with
    ``hidden_dim=256``. The state dict is authoritative for these MLP head
    widths, so infer them before instantiating the model.
    """

    cfg_out = _clone_cfg(cfg)
    overrides: dict[str, Any] = {}

    _infer_linear_out(
        cfg_out,
        state,
        "classifier.net.0.weight",
        "model.hidden_dim",
        overrides,
    )
    _infer_linear_out(
        cfg_out,
        state,
        "zp_classifier.net.0.weight",
        "model.zp_classifier_hidden_dim",
        overrides,
    )
    aux_values = [
        _linear_out_dim(state, key)
        for key in (
            "audio_aux_classifier.net.0.weight",
            "video_aux_classifier.net.0.weight",
            "family_classifier.net.0.weight",
            "risk_classifier.net.0.weight",
        )
    ]
    aux_values = [value for value in aux_values if value is not None]
    if aux_values and len(set(aux_values)) == 1:
        _set_cfg_value(cfg_out, "model.modality_aux_hidden_dim", aux_values[0], overrides)

    _infer_linear_out(
        cfg_out,
        state,
        "mediated_head.net.1.weight",
        "model.pathway_decomposition.mediated_hidden_dim",
        overrides,
    )
    _infer_linear_out(
        cfg_out,
        state,
        "direct_head.modality_proj.1.weight",
        "model.pathway_decomposition.direct_bottleneck_dim",
        overrides,
    )
    _infer_linear_out(
        cfg_out,
        state,
        "direct_head.head.1.weight",
        "model.pathway_decomposition.direct_hidden_dim",
        overrides,
    )
    _infer_linear_out(
        cfg_out,
        state,
        "direct_head.gate_net.1.weight",
        "model.pathway_decomposition.direct_gate_hidden_dim",
        overrides,
    )

    av_subtractor_values = [
        _linear_out_dim(state, "audio_t_adversary.subtractor.net.1.weight"),
        _linear_out_dim(state, "video_t_adversary.subtractor.net.1.weight"),
    ]
    av_subtractor_values = [value for value in av_subtractor_values if value is not None]
    if av_subtractor_values and len(set(av_subtractor_values)) == 1:
        _set_cfg_value(cfg_out, "model.av_t_adversary.subtractor_hidden_dim", av_subtractor_values[0], overrides)
    av_adversary_values = [
        _linear_out_dim(state, "audio_t_adversary.adversary.1.weight"),
        _linear_out_dim(state, "video_t_adversary.adversary.1.weight"),
    ]
    av_adversary_values = [value for value in av_adversary_values if value is not None]
    if av_adversary_values and len(set(av_adversary_values)) == 1:
        _set_cfg_value(cfg_out, "model.av_t_adversary.adversary_hidden_dim", av_adversary_values[0], overrides)

    return cfg_out, overrides


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    if isinstance(cfg, DictConfig):
        return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    return OmegaConf.create(deepcopy(cfg))


def _infer_linear_out(
    cfg: DictConfig,
    state: dict,
    weight_key: str,
    cfg_path: str,
    overrides: dict[str, Any],
) -> None:
    value = _linear_out_dim(state, weight_key)
    if value is not None:
        _set_cfg_value(cfg, cfg_path, value, overrides)


def _linear_out_dim(state: dict, weight_key: str) -> int | None:
    tensor = state.get(weight_key)
    if tensor is None:
        tensor = state.get(f"module.{weight_key}")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
        return None
    return int(tensor.shape[0])


def _set_cfg_value(cfg: DictConfig, path: str, value: int, overrides: dict[str, Any]) -> None:
    current = cfg_get(cfg, path, None)
    if current == value:
        return
    OmegaConf.update(cfg, path, int(value), merge=True, force_add=True)
    overrides[path] = {"from_config": current, "from_checkpoint": int(value)}


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def macro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int | None = None) -> float:
    pred = logits.argmax(dim=-1).to(torch.long)
    labels = labels.to(torch.long)
    if num_classes is None:
        num_classes = int(max(int(logits.shape[-1]), int(labels.max().item()) + 1 if labels.numel() else 0))
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    for y, p in zip(labels.view(-1), pred.view(-1), strict=False):
        if 0 <= int(y) < num_classes and 0 <= int(p) < num_classes:
            confusion[int(y), int(p)] += 1
    tp = confusion.diag()
    precision = tp / confusion.sum(dim=0).clamp_min(1.0)
    recall = tp / confusion.sum(dim=1).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
    support = confusion.sum(dim=1) > 0
    return float(f1[support].mean().item()) if support.any() else float("nan")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return float("nan")
    return float((logits.argmax(dim=-1).cpu() == labels.cpu()).float().mean().item())


def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(device)


def _is_known_legacy_key(key: str, has_current_dih_state: bool) -> bool:
    if key in {"phi_T.pair_indices", "phi_T.legendre_p1_scale"}:
        return True
    return key.startswith("dih.") and not has_current_dih_state
