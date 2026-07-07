"""Evaluation entry point for CMR-Net."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmr_net import CMRNet
from cmr_net.data import build_loaders
from cmr_net.evaluation import (
    evaluate_classification,
    evaluate_dih_counterfactual,
    evaluate_dih_pathway_counterfactual,
    evaluate_fusion_pathways,
    evaluate_hsic_analytical_falsifiability,
    evaluate_hsic_falsifiability,
    evaluate_identifiability_mcc,
    evaluate_modality_auxiliary,
    evaluate_negative_pair_degradation,
    evaluate_pathway_decomposition,
    evaluate_pathway_intervention,
)
from cmr_net.losses import ConditionalHSICLoss


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run all CMR-Net diagnostics and write Markdown results."""
    project_root = Path(get_original_cwd())
    # Round-17: auto-merge the training-time ``model`` topology so the
    # eval-time module tree matches the checkpoint. Without this, a
    # user must re-pass every ``model.pathway_decomposition.*``,
    # ``model.av_t_adversary.*`` (etc.) flag on the eval CLI or strict
    # checkpoint loading raises ``Unexpected key(s)`` for the trained
    # modules. ``resolved_config.yaml`` is written by ``train`` /
    # ``train_ddp`` next to the per-run results directory and is the
    # ground truth for how the checkpoint was constructed.
    _merge_training_model_config(cfg, project_root)
    _require_test_evaluation_approval(cfg)
    _, _, test_loader = build_loaders(cfg)
    model = CMRNet(cfg)
    # P0.3 fix (round-2 audit): default to strict checkpoint loading so a
    # typo in run_name / seed / checkpoint_path can no longer silently
    # evaluate random or partially-loaded weights.
    debug_lenient = bool(_cfg_get(cfg, "evaluation.debug_lenient_checkpoint", False))
    checkpoint_meta = _load_checkpoint_if_available(
        model,
        _checkpoint_path(cfg, project_root),
        debug_lenient=debug_lenient,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.training.device != "cpu" else "cpu")
    model.to(device)
    print(
        "[evaluate] "
        f"device={device} "
        f"hsic_bootstrap={int(cfg.evaluation.hsic_bootstrap)} "
        f"checkpoint_loaded={checkpoint_meta.get('loaded')}",
        flush=True,
    )
    # State the inference path explicitly so the diagnostics file is
    # unambiguous about whether outputs.logits is the process-only head or the
    # additive dual-pathway head.
    # The model.py forward() override only fires in eval mode when this flag
    # is True; logits_mediated / logits_direct / logits_pathway are
    # preserved on `outputs` either way so the pathway diagnostics keep
    # working regardless of which mode this report is run under.
    _pathway_enabled = bool(
        _cfg_get(cfg, "model.pathway_decomposition.enabled", False)
    )
    _inference_mediated_only = bool(
        _cfg_get(cfg, "model.pathway_decomposition.inference_mediated_only", False)
    )
    if _pathway_enabled and _inference_mediated_only:
        _inference_path = (
            "PROCESS-ONLY (direct head excluded from outputs.logits)"
        )
    elif _pathway_enabled:
        _inference_path = (
            "FULL DUAL (additive logits = logits_mediated + logits_direct)"
        )
    else:
        _inference_path = "PATHWAY-DECOMPOSITION DISABLED (single-head outputs.logits)"
    print(f"[evaluate] inference path: {_inference_path}", flush=True)

    physical_proxies = list(
        _cfg_get(
            cfg,
            "evaluation.physical_proxies",
            [
                "current_mean",
                "voltage_mean",
                "pressure_mean",
                "feed_mean",
                "heat_input",
            ],
        )
    )
    # Construct the eval-time HSIC module with the SAME settings the trainer
    # would have used so the buffered RFF basis is bit-identical between
    # training and evaluation.
    hsic_rff_features = int(
        cfg.loss.get("hsic_rff_features", 512) if hasattr(cfg, "loss") else 512
    )
    hsic_seed = int(cfg.project.seed) if hasattr(cfg, "project") else 42
    eval_hsic = ConditionalHSICLoss(num_features=hsic_rff_features, seed=hsic_seed)
    diagnostics = {
        "metadata": {
            "checkpoint": checkpoint_meta,
            "results_dir": str(_results_dir(cfg, project_root)),
            "seed": int(_cfg_get(cfg, "project.seed", 42)),
            # Round-18 Option A: persist the inference path used to produce
            # this diagnostics bundle so downstream artefacts (paper tables,
            # ablation rows) cannot mix up process-only and full-dual runs.
            "inference_path": _inference_path,
            "pathway_decomposition_enabled": _pathway_enabled,
            "pathway_inference_mediated_only": _inference_mediated_only,
        },
    }
    diagnostics["classification"] = _run_diagnostic(
        "classification",
        lambda: evaluate_classification(model, test_loader),
    )
    diagnostics["identifiability_mcc"] = _run_diagnostic(
        "identifiability_mcc",
        lambda: evaluate_identifiability_mcc(model, test_loader, physical_proxies),
    )
    diagnostics["hsic_falsifiability"] = _run_diagnostic(
        "hsic_falsifiability",
        lambda: evaluate_hsic_falsifiability(
            model,
            test_loader,
            n_bootstrap=int(cfg.evaluation.hsic_bootstrap),
            hsic_loss=eval_hsic,
        ),
    )
    # Round-14 M2: analytical (full-Gram) conditional HSIC with a permutation
    # null. This is the proper hypothesis test for X ⫫ Y | Z; the RFF version
    # above only estimates the value and a bootstrap CI of the value, which
    # cannot reject H0 when the value is at the 1e-7 noise floor of the RFF
    # estimator. Off by default if either flag is missing in the config.
    hsic_analytical_enabled = bool(
        _cfg_get(cfg, "evaluation.hsic_analytical_enabled", True)
    )
    if hsic_analytical_enabled:
        n_permutation = int(_cfg_get(cfg, "evaluation.hsic_n_permutation", 500))
        hsic_analytical_max_samples = _cfg_get(
            cfg, "evaluation.hsic_analytical_max_samples", None
        )
        max_samples = (
            int(hsic_analytical_max_samples)
            if hsic_analytical_max_samples not in (None, "", "null")
            else None
        )
        hsic_analytical_ridge = float(
            _cfg_get(cfg, "evaluation.hsic_analytical_ridge", 1.0e-3)
        )
        diagnostics["hsic_analytical_falsifiability"] = _run_diagnostic(
            "hsic_analytical_falsifiability",
            lambda: evaluate_hsic_analytical_falsifiability(
                model,
                test_loader,
                n_permutation=n_permutation,
                ridge=hsic_analytical_ridge,
                max_samples=max_samples,
                seed=int(_cfg_get(cfg, "project.seed", 42)),
            ),
        )
    diagnostics["dih_counterfactual"] = _run_diagnostic(
        "dih_counterfactual",
        lambda: evaluate_dih_counterfactual(model, test_loader),
    )
    diagnostics["negative_pair_degradation"] = _run_diagnostic(
        "negative_pair_degradation",
        lambda: evaluate_negative_pair_degradation(model, test_loader),
    )
    diagnostics["modality_auxiliary"] = _run_diagnostic(
        "modality_auxiliary",
        lambda: evaluate_modality_auxiliary(model, test_loader),
    )
    diagnostics["fusion_pathways"] = _run_diagnostic(
        "fusion_pathways",
        lambda: evaluate_fusion_pathways(model, test_loader),
    )
    # Dual-pathway process/modality decomposition + intervention-style stress
    # tests.
    # The decomposition evaluator reports per-pathway and joint
    # classification metrics; the intervention evaluator perturbs one
    # pathway while holding the other fixed to demonstrate that the two
    # pathways are non-degenerate. Both are gated by independent config
    # flags so each can be turned off without disabling the other.
    pathway_decomp_enabled = bool(
        _cfg_get(cfg, "evaluation.pathway_decomposition_enabled", True)
    )
    if pathway_decomp_enabled:
        diagnostics["pathway_decomposition"] = _run_diagnostic(
            "pathway_decomposition",
            lambda: evaluate_pathway_decomposition(model, test_loader),
        )
    pathway_intv_enabled = bool(
        _cfg_get(cfg, "evaluation.pathway_intervention_enabled", True)
    )
    if pathway_intv_enabled:
        n_perturb = int(_cfg_get(cfg, "evaluation.pathway_intervention_n_perturb", 8))
        perturb_scale = float(
            _cfg_get(cfg, "evaluation.pathway_intervention_perturb_scale", 1.0)
        )
        diagnostics["pathway_intervention"] = _run_diagnostic(
            "pathway_intervention",
            lambda: evaluate_pathway_intervention(
                model,
                test_loader,
                n_perturb=n_perturb,
                perturb_scale=perturb_scale,
                seed=int(_cfg_get(cfg, "project.seed", 42)),
            ),
        )
    # Legacy DIH-style process-summary replacement evaluation. The historical
    # key name is retained for compatibility, but the report should interpret
    # it as a stress test, not as causal identification.
    # Gated by the same ``pathway_decomposition_enabled`` flag because
    # both flavors require the dual-pathway heads to be present.
    if pathway_decomp_enabled:
        diagnostics["dih_pathway_counterfactual"] = _run_diagnostic(
            "dih_pathway_counterfactual",
            lambda: evaluate_dih_pathway_counterfactual(model, test_loader),
        )

    results_dir = _results_dir(cfg, project_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = results_dir / "diagnostics.md"
    json_path = results_dir / "diagnostics.json"
    markdown_path.write_text(_to_markdown(diagnostics), encoding="utf-8")
    json_path.write_text(
        json.dumps(_json_safe(diagnostics), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Wrote diagnostics to {markdown_path}")


def _run_diagnostic(name: str, fn: Any) -> Any:
    start = time.perf_counter()
    print(f"[evaluate] start {name}", flush=True)
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"[evaluate] done {name} elapsed_sec={elapsed:.1f}", flush=True)
    return result


def _require_test_evaluation_approval(cfg: DictConfig) -> None:
    allowed = bool(_cfg_get(cfg, "evaluation.allow_test_evaluation", False))
    allowed = allowed or os.environ.get("CMR_NET_ALLOW_TEST_EVALUATION", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }
    if allowed:
        return
    raise RuntimeError(
        "scripts/evaluate.py evaluates the held-out test split. Development "
        "iterations must use analysis/repair_harness/dev_val_contract.py on val. "
        "For the one-time Phase 5 test run, pass "
        "evaluation.allow_test_evaluation=true or set "
        "CMR_NET_ALLOW_TEST_EVALUATION=1."
    )

def _checkpoint_path(cfg: DictConfig, project_root: Path) -> Path | None:
    explicit = _cfg_get(cfg, "evaluation.checkpoint_path", None)
    if explicit not in (None, ""):
        return Path(str(explicit))

    checkpoint_name = str(_cfg_get(cfg, "output.checkpoint_name", "cmr_net_final.pt"))
    candidates = []
    run_name = _run_name(cfg)
    checkpoints_root = Path(str(_cfg_get(cfg, "output.checkpoint_dir", "checkpoints")))
    checkpoints_root = checkpoints_root if checkpoints_root.is_absolute() else project_root / checkpoints_root
    if run_name:
        candidates.append(checkpoints_root / run_name / checkpoint_name)
    candidates.append(checkpoints_root / checkpoint_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    warnings.warn(
        "No evaluation checkpoint_path provided and no inferred checkpoint found. "
        f"Tried: {[str(p) for p in candidates]}. Evaluating current model weights.",
        stacklevel=2,
    )
    return None


def _load_checkpoint_if_available(
    model: CMRNet,
    checkpoint_path: Path | None,
    debug_lenient: bool = False,
) -> dict[str, object]:
    """Load a checkpoint into ``model`` with strict-by-default semantics.

    P0.2: Prefer full-objective stage-4 EMA over generic EMA so the
    evaluated weights match the EMA weights that were used during validation
    (training-time macro-F1 ~ eval macro-F1 by construction).
    P0.3: Strict loading is the default. Set
    ``evaluation.debug_lenient_checkpoint=true`` to fall back to
    ``strict=False`` and to tolerate a missing checkpoint file. Otherwise
    we raise ``FileNotFoundError`` / ``RuntimeError`` so a typo cannot
    silently evaluate random or partially-loaded weights.
    """

    if not checkpoint_path:
        message = "No checkpoint path provided / inferred."
        if debug_lenient:
            warnings.warn(message + " Evaluating current (random) weights.", stacklevel=2)
            return {
                "path": None,
                "loaded": False,
                "missing": [],
                "unexpected": [],
                "source_key": None,
                "uses_ema": False,
                "debug_lenient": True,
            }
        raise FileNotFoundError(
            message
            + " Set evaluation.checkpoint_path=<path> or run training first; "
            + "or pass evaluation.debug_lenient_checkpoint=true to override."
        )

    path = Path(checkpoint_path)
    if not path.exists():
        message = f"Checkpoint not found: {path}"
        if debug_lenient:
            warnings.warn(message + "; evaluating current weights.", stacklevel=2)
            return {
                "path": str(path),
                "loaded": False,
                "missing": [],
                "unexpected": [],
                "source_key": None,
                "uses_ema": False,
                "debug_lenient": True,
            }
        raise FileNotFoundError(
            message + " (set evaluation.debug_lenient_checkpoint=true to override)"
        )

    checkpoint = torch.load(path, map_location="cpu")
    state, source_key, uses_ema = _select_state_dict(checkpoint)
    state, ignored_keys = _normalize_checkpoint_state(model, state)

    strict = not debug_lenient
    try:
        result = model.load_state_dict(state, strict=strict)
        missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    except RuntimeError as exc:
        if debug_lenient:
            warnings.warn(f"Strict load failed; falling back to strict=False: {exc}", stacklevel=2)
            result = model.load_state_dict(state, strict=False)
            missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
        else:
            raise

    if (missing or unexpected) and not debug_lenient:
        raise RuntimeError(
            f"Strict checkpoint load reported missing={missing} unexpected={unexpected}; "
            "set evaluation.debug_lenient_checkpoint=true to bypass."
        )
    if ignored_keys:
        print(f"Ignored legacy checkpoint keys: {len(ignored_keys)}")
    if missing or unexpected:
        warnings.warn(
            f"Checkpoint loaded with missing={missing} unexpected={unexpected}",
            stacklevel=2,
        )
    print(f"Loaded checkpoint from {path} (source_key={source_key}, uses_ema={uses_ema})")
    return {
        "path": str(path),
        "loaded": True,
        "missing": missing,
        "unexpected": unexpected,
        "ignored_legacy_keys": ignored_keys,
        "source_key": source_key,
        "uses_ema": uses_ema,
        "debug_lenient": debug_lenient,
    }


def _select_state_dict(checkpoint: object) -> tuple[dict, str, bool]:
    """Choose the best state dict from a (possibly nested) checkpoint object.

    Preference order: ``best_stage4_ema_state_dict`` > ``best_ema_state_dict`` > ``ema_state_dict`` >
    ``model_state_dict`` > ``state_dict`` > ``model`` > the object itself
    if it already looks like a state dict.

    The canonical eval target is the best stage-4 EMA snapshot, because
    stage 4 is the only phase where prediction, mechanism, DIH, negative
    pairs, and HSIC have all been active. Older checkpoints fall back to
    the global best EMA or the last EMA.
    """

    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("best_stage4_ema_state_dict"), dict):
            return checkpoint["best_stage4_ema_state_dict"], "best_stage4_ema_state_dict", True
        if isinstance(checkpoint.get("best_ema_state_dict"), dict):
            return checkpoint["best_ema_state_dict"], "best_ema_state_dict", True
        if isinstance(checkpoint.get("ema_state_dict"), dict):
            return checkpoint["ema_state_dict"], "ema_state_dict", True
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, key, False
        # bare state dict
        return checkpoint, "checkpoint", False
    return checkpoint, "raw", False


def _normalize_checkpoint_state(model: CMRNet, state: dict) -> tuple[dict, list[str]]:
    """Drop only known-safe legacy keys before strict loading.

    DIH used to register references to shared CMR-Net submodules, which
    produced duplicate ``dih.*`` state_dict entries. The fixed DIH keeps plain
    references, so those duplicates are ignored. We also ignore deterministic
    non-persistent treatment buffers from older checkpoints.
    """

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


def _is_known_legacy_key(key: str, has_current_dih_state: bool) -> bool:
    if key in {"phi_T.pair_indices", "phi_T.legendre_p1_scale"}:
        return True
    return key.startswith("dih.") and not has_current_dih_state


def _results_dir(cfg: DictConfig, project_root: Path) -> Path:
    raw_results_dir = Path(str(_cfg_get(cfg, "evaluation.results_dir", "results")))
    results_dir = raw_results_dir if raw_results_dir.is_absolute() else project_root / raw_results_dir
    run_name = _run_name(cfg)
    if run_name and _is_default_relative(raw_results_dir, "results"):
        results_dir = results_dir / run_name
    return results_dir


def _merge_training_model_config(cfg: DictConfig, project_root: Path) -> None:
    """Overlay the training-time ``model`` config into the eval-time cfg.

    Reads ``<results_dir>/resolved_config.yaml`` (written by ``train`` /
    ``train_ddp`` after Hydra resolution) and merges only its ``model``
    section into the current cfg. This guarantees that ``CMRNet(cfg)``
    rebuilds the exact module tree the checkpoint was trained with —
    enable flags (``pathway_decomposition.enabled``,
    ``av_t_adversary.enabled`` …), dimension knobs, gate parameters,
    and so on — so strict checkpoint loading succeeds without the user
    having to repeat every ``model.*`` flag on the eval CLI.

    Set ``evaluation.use_resolved_model_config=false`` to disable. Honors
    only the ``model`` subtree; loss / training / data / evaluation
    sections remain under the user's eval-time control.
    """

    if not bool(_cfg_get(cfg, "evaluation.use_resolved_model_config", True)):
        return
    results_dir = _results_dir(cfg, project_root)
    candidate = results_dir / "resolved_config.yaml"
    if not candidate.exists():
        # Quiet — the user may legitimately be evaluating an external
        # checkpoint (no co-located resolved_config); strict load will
        # then surface the real mismatch.
        return
    try:
        resolved = OmegaConf.load(candidate)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Could not parse resolved_config.yaml at {candidate}: {exc}. "
            "Falling back to default eval-time model config.",
            stacklevel=2,
        )
        return
    resolved_model = OmegaConf.select(resolved, "model")
    if resolved_model is None:
        return
    # ``inference_mediated_only`` is an eval/deployment policy, not a
    # checkpoint topology key.  Preserve the eval-time value across the
    # resolved training-model merge so users can switch between full-dual and
    # process-only inference without rebuilding the module tree or
    # disabling strict checkpoint loading.
    eval_inference_mediated_only = bool(
        _cfg_get(cfg, "model.pathway_decomposition.inference_mediated_only", False)
    )
    # Merge: resolved-training values win on the model subtree. The user
    # who genuinely wants a model.* override at eval-time can set
    # ``evaluation.use_resolved_model_config=false`` and pass all the
    # flags explicitly.
    #
    # Hydra cfg is struct by default, which would reject keys present in
    # resolved_config but absent from the current ``configs/default.yaml``
    # (e.g. ``model.T_len`` — read by ``model.py`` with a
    # ``data.process_seq_len`` fallback, never declared in default.yaml,
    # but historically baked into resolved configs). Relax struct around
    # the merge so any such drifted-but-real key flows through; restore
    # struct afterwards so downstream typos still fail loudly.
    was_struct = OmegaConf.is_struct(cfg.model)
    OmegaConf.set_struct(cfg.model, False)
    try:
        cfg.model = OmegaConf.merge(cfg.model, resolved_model)
        if bool(_cfg_get(cfg, "model.pathway_decomposition.enabled", False)):
            OmegaConf.update(
                cfg,
                "model.pathway_decomposition.inference_mediated_only",
                eval_inference_mediated_only,
                merge=False,
            )
    finally:
        OmegaConf.set_struct(cfg.model, was_struct if was_struct is not None else True)
    print(
        "[evaluate] merged training-time model config from "
        f"{candidate.relative_to(project_root) if candidate.is_relative_to(project_root) else candidate} "
        "(disable with evaluation.use_resolved_model_config=false)",
        flush=True,
    )


def _run_name(cfg: DictConfig) -> str | None:
    explicit = _cfg_get(cfg, "output.run_name", None)
    if explicit not in (None, ""):
        return _safe_path_part(str(explicit))
    if not bool(_cfg_get(cfg, "output.isolate_by_seed", True)):
        return None
    return _safe_path_part(f"seed{int(_cfg_get(cfg, 'project.seed', 42))}")


def _is_default_relative(path: Path, default_name: str) -> bool:
    return not path.is_absolute() and path.as_posix().rstrip("/") == default_name


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_")


def _cfg_get(cfg: object, path: str, default: object = None) -> object:
    current = cfg
    for part in path.split("."):
        if current is None:
            return default
        try:
            if OmegaConf.is_config(current):
                if part not in current:
                    return default
                current = current[part]
                continue
            if isinstance(current, dict):
                if part not in current:
                    return default
                current = current[part]
                continue
            try:
                current = getattr(current, part)
            except (AttributeError, TypeError):
                try:
                    current = current[part]
                except (KeyError, TypeError, IndexError):
                    return default
        except Exception:
            return default
    return default if current is None else current


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


_PATHWAY_DECOMP_SECTION = "pathway_decomposition"
_PATHWAY_INTV_SECTION = "pathway_intervention"
_DIH_PATHWAY_CF_SECTION = "dih_pathway_counterfactual"


def _to_markdown(diagnostics: dict) -> str:
    lines: list[str] = ["# CMR-Net Diagnostics", ""]
    for section, value in diagnostics.items():
        if section == _PATHWAY_DECOMP_SECTION and isinstance(value, dict):
            lines.extend(_render_pathway_decomposition(value))
            continue
        if section == _PATHWAY_INTV_SECTION and isinstance(value, dict):
            lines.extend(_render_pathway_intervention(value))
            continue
        if section == _DIH_PATHWAY_CF_SECTION and isinstance(value, dict):
            lines.extend(_render_dih_pathway_counterfactual(value))
            continue
        lines.append(f"## {section}")
        lines.append("")
        if isinstance(value, dict):
            for k, v in value.items():
                lines.append(f"- `{k}`: {_format_scalar(v)}")
        else:
            lines.append(str(value))
        lines.append("")
    return "\n".join(lines)


def _render_pathway_decomposition(value: dict) -> list[str]:
    """Render the dual-pathway decomposition diagnostic.

    Lays out process / direct / full macro-F1, the two incremental contributions,
    the per-pathway logit norms, and argmax agreement. Falls back to a
    short "not available" block when the model did not emit pathway
    logits (i.e. ``model.pathway_decomposition.enabled`` was false).
    """

    lines: list[str] = ["## Pathway Decomposition", ""]
    if not value.get("available", False):
        reason = value.get("reason", "unavailable")
        lines.append(f"_Not available: {reason}_")
        lines.append("")
        return lines

    rows = [
        ("num_samples", value.get("num_samples")),
        ("process_macro_f1", value.get("mediated_macro_f1")),
        ("direct_macro_f1", value.get("direct_macro_f1")),
        ("full_macro_f1", value.get("full_macro_f1")),
        ("full_minus_process_macro_f1", value.get("full_minus_mediated_macro_f1")),
        ("full_minus_direct_macro_f1", value.get("full_minus_direct_macro_f1")),
        ("process_accuracy", value.get("mediated_accuracy")),
        ("direct_accuracy", value.get("direct_accuracy")),
        ("full_accuracy", value.get("full_accuracy")),
        ("process_logit_norm", value.get("mediated_logit_norm")),
        ("direct_logit_norm", value.get("direct_logit_norm")),
        ("process_logit_share", value.get("mediated_logit_share")),
        ("logit_cosine_alignment", value.get("logit_cosine_alignment")),
        ("argmax_agreement_frac", value.get("argmax_agreement_frac")),
    ]
    for k, v in rows:
        lines.append(f"- `{k}`: {_format_scalar(v)}")
    for recall_key in (
        "mediated_per_class_recall",
        "direct_per_class_recall",
        "full_per_class_recall",
    ):
        if recall_key in value:
            lines.append(f"- `{recall_key}`: {_format_scalar(value[recall_key])}")
    interp = value.get("interpretation")
    if interp:
        lines.append("")
        lines.append(f"_Interpretation:_ {interp}")
    lines.append("")
    return lines


def _render_pathway_intervention(value: dict) -> list[str]:
    """Render the pathway-intervention diagnostic.

    Reports the baseline and per-perturbation macro-F1, the two
    ``delta_macro_f1`` figures (which carry the causal-asymmetry signal),
    the noise-variant logit deltas, and the pathway-strength asymmetry
    ratio. Falls back to a short "not available" block when the loader
    was empty or the model lacked the dual-pathway heads.
    """

    lines: list[str] = ["## Pathway Intervention", ""]
    if not value.get("available", False):
        reason = value.get("reason", "unavailable")
        lines.append(f"_Not available: {reason}_")
        lines.append("")
        return lines

    rows = [
        ("n_perturb", value.get("n_perturb")),
        ("perturb_scale_noise", value.get("perturb_scale_noise")),
        ("num_samples", value.get("num_samples")),
        ("num_classes_present_in_test", value.get("num_classes_present_in_test")),
        ("baseline_macro_f1", value.get("baseline_macro_f1")),
        ("macro_f1_after_perturb_Z_D", value.get("macro_f1_after_perturb_Z_D")),
        ("macro_f1_after_perturb_Z_D_std", value.get("macro_f1_after_perturb_Z_D_std")),
        ("macro_f1_after_perturb_Z_P", value.get("macro_f1_after_perturb_Z_P")),
        ("macro_f1_after_perturb_Z_P_std", value.get("macro_f1_after_perturb_Z_P_std")),
        ("delta_macro_f1_perturb_Z_D", value.get("delta_macro_f1_perturb_Z_D")),
        ("delta_macro_f1_perturb_Z_P", value.get("delta_macro_f1_perturb_Z_P")),
        ("perturb_Z_D_logit_l2", value.get("perturb_Z_D_logit_l2")),
        ("perturb_Z_P_logit_l2", value.get("perturb_Z_P_logit_l2")),
        ("perturb_Z_D_prob_l1", value.get("perturb_Z_D_prob_l1")),
        ("perturb_Z_P_prob_l1", value.get("perturb_Z_P_prob_l1")),
        ("perturb_Z_D_pred_change_frac", value.get("perturb_Z_D_pred_change_frac")),
        ("perturb_Z_P_pred_change_frac", value.get("perturb_Z_P_pred_change_frac")),
        ("perturb_Z_D_temperature_jsd_T2", value.get("perturb_Z_D_temperature_jsd_T2")),
        ("perturb_Z_P_temperature_jsd_T2", value.get("perturb_Z_P_temperature_jsd_T2")),
        ("perturb_Z_D_temperature_jsd_T5", value.get("perturb_Z_D_temperature_jsd_T5")),
        ("perturb_Z_P_temperature_jsd_T5", value.get("perturb_Z_P_temperature_jsd_T5")),
        ("perturb_Z_D_top1_margin_delta", value.get("perturb_Z_D_top1_margin_delta")),
        ("perturb_Z_P_top1_margin_delta", value.get("perturb_Z_P_top1_margin_delta")),
        (
            "perturb_Z_D_centered_logit_cosine_change",
            value.get("perturb_Z_D_centered_logit_cosine_change"),
        ),
        (
            "perturb_Z_P_centered_logit_cosine_change",
            value.get("perturb_Z_P_centered_logit_cosine_change"),
        ),
        ("noise_Z_D_logit_l2", value.get("noise_Z_D_logit_l2")),
        ("noise_Z_P_logit_l2", value.get("noise_Z_P_logit_l2")),
        ("pathway_strength_asymmetry", value.get("pathway_strength_asymmetry")),
    ]
    for k, v in rows:
        lines.append(f"- `{k}`: {_format_scalar(v)}")
    interp = value.get("interpretation")
    if interp:
        lines.append("")
        lines.append(f"_Interpretation:_ {interp}")
    lines.append("")
    return lines


def _render_dih_pathway_counterfactual(value: dict) -> list[str]:
    """Render the round-17 dual-flavor DIH counterfactual diagnostic.

    Surfaces both flavors side-by-side so the analyst can read the
    contrast in a single block:

    - ``process_summary_replacement_process_only``: the process-summary
      replacement stress test propagated through the process-only head.
    - ``dih_full_dual_pathway``: same Z_P_do, but Z_A/Z_V are also
      regenerated so the direct head sees a consistent intervention,
      and the final prediction is ``logits_process_do + logits_direct_do``.

    ``counterfactual_accuracy_gap`` is retained as a legacy key; it should be
    interpreted as the difference between process-only and full stress-test
    accuracy, not as a causal effect.
    """

    lines: list[str] = ["## DIH Pathway Counterfactual", ""]
    if not value.get("available", False):
        reason = value.get("reason", "unavailable")
        lines.append(f"_Not available: {reason}_")
        lines.append("")
        return lines

    lines.append(f"- `num_pairs`: {_format_scalar(value.get('num_pairs'))}")
    lines.append(
        f"- `counterfactual_accuracy_gap`: "
        f"{_format_scalar(value.get('counterfactual_accuracy_gap'))}"
    )
    lines.append("")
    for flavor_key, flavor_title in (
        ("dih_mediated_only", "### process_summary_replacement_process_only"),
        ("dih_full_dual_pathway", "### process_summary_replacement_full_additive"),
    ):
        flavor = value.get(flavor_key)
        if not isinstance(flavor, dict):
            continue
        lines.append(flavor_title)
        lines.append("")
        for k in (
            "ate_pred",
            "ate_observed",
            "ate_error",
            "cate_mae",
            "counterfactual_accuracy",
        ):
            if k in flavor:
                lines.append(f"- `{k}`: {_format_scalar(flavor[k])}")
        lines.append("")
    interp = value.get("interpretation")
    if interp:
        lines.append(f"_Interpretation:_ {interp}")
        lines.append("")
    return lines


def _format_scalar(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.6g}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_format_scalar(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_scalar(v) for v in value) + "]"
    return str(value)


if __name__ == "__main__":
    main()
