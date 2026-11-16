"""Resilient overnight Task 1 orchestration.

Phases are isolated: a failed phase falls back to defaults / prior checkpoints
and later phases still run. Re-running resumes completed sweep configs.

Run:
  python -m pipelines.task1_overnight_flow

Outputs (never overwrites task_1_predictions.csv):
  assignment1_2026/task_1_predictions_overnight.csv
  artifacts/task1/overnight_summary.json
  artifacts/task1/overnight_resolved.yaml
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from prefect import flow, task
from prefect.logging import get_run_logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task1.data import project_root
from src.task1.mf_sweep import (
    DEFAULT_GRID,
    DEFAULT_SVDPP_GRID,
    _member_fields,
    run_mf_sweep,
    run_svdpp_sweep,
)
from src.task1.pipeline import load_config, run_task1_pipeline

OVERNIGHT_CONFIG = ROOT / "configs" / "task1_overnight.yaml"
FALLBACK_CONFIG = ROOT / "configs" / "task1.yaml"
ARTIFACTS = ROOT / "artifacts" / "task1"
SUMMARY_PATH = ARTIFACTS / "overnight_summary.json"
RESOLVED_PATH = ARTIFACTS / "overnight_resolved.yaml"
PHASE_STATUS_PATH = ARTIFACTS / "overnight_phase_status.json"
WINNER_CSV_NAME = "task_1_predictions.csv"
OVERNIGHT_CSV_NAME = "task_1_predictions_overnight.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _update_phase(name: str, **fields: Any) -> Dict[str, Any]:
    status = _load_json(PHASE_STATUS_PATH)
    phases = status.setdefault("phases", {})
    row = phases.get(name, {})
    row.update(fields)
    row["updated_at"] = _now()
    phases[name] = row
    status["updated_at"] = _now()
    _write_json(PHASE_STATUS_PATH, status)
    return status


def _assert_safe_output(cfg: Dict[str, Any]) -> None:
    out = str(cfg.get("output_file", ""))
    if out == WINNER_CSV_NAME or out.endswith("/" + WINNER_CSV_NAME):
        raise RuntimeError(
            f"Refusing to overwrite Gradescope winner CSV ({WINNER_CSV_NAME}). "
            f"Use {OVERNIGHT_CSV_NAME}."
        )


def _fallback_mf_members(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    members = ((base_cfg.get("mf_ensemble") or {}).get("members")) or []
    if members:
        return [dict(m) for m in members]
    mf = dict(base_cfg.get("mf") or {})
    return [
        {
            "n_factors": int(mf.get("n_factors", 32)),
            "n_epochs": int(mf.get("n_epochs", 50)),
            "lr_all": float(mf.get("lr_all", 0.005)),
            "reg_all": float(mf.get("reg_all", 0.1)),
        }
    ]


def _fallback_svdpp(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    sp = dict(base_cfg.get("svdpp") or {})
    sp.setdefault("enabled", True)
    sp.setdefault("n_factors", 20)
    sp.setdefault("n_epochs", 30)
    sp.setdefault("lr_all", 0.005)
    sp.setdefault("reg_all", 0.1)
    return sp


def _build_resolved_config(
    base_cfg: Dict[str, Any],
    mf_members: List[Dict[str, Any]],
    svdpp_params: Optional[Dict[str, Any]],
    *,
    neural_enabled: bool,
    svdpp_enabled: bool,
) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg["output_file"] = OVERNIGHT_CSV_NAME
    cfg["mlflow_experiment"] = cfg.get("mlflow_experiment") or "task1_overnight"
    cfg["mf_ensemble"] = {"enabled": True, "members": mf_members}
    if svdpp_enabled and svdpp_params is not None:
        sp = dict(svdpp_params)
        sp["enabled"] = True
        sp.setdefault("rating_scale", [0.0, 5.0])
        sp.setdefault("verbose", False)
        cfg["svdpp"] = sp
    else:
        cfg["svdpp"] = {**(cfg.get("svdpp") or {}), "enabled": False}
    cfg["knn"] = {**(cfg.get("knn") or {}), "enabled": False}
    cfg.setdefault("lgbm", {})
    cfg["lgbm"]["enabled"] = True
    cfg["lgbm"]["mode"] = "residual"
    cfg.setdefault("neural", {})
    cfg["neural"]["enabled"] = bool(neural_enabled)
    _assert_safe_output(cfg)
    return cfg


def _write_resolved_yaml(cfg: Dict[str, Any]) -> Path:
    dump = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    RESOLVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESOLVED_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(dump, f, sort_keys=False)
    return RESOLVED_PATH


@task(name="overnight_mf_sweep", retries=1, retry_delay_seconds=30)
def phase_mf_sweep(config_path: str) -> Dict[str, Any]:
    logger = get_run_logger()
    _update_phase("mf_sweep", status="running", started_at=_now())
    try:
        summary = run_mf_sweep(
            config_path=config_path,
            grid=DEFAULT_GRID,
            experiment="task1_overnight_mf_sweep",
            resume=True,
        )
        top3 = [_member_fields(r) for r in summary["top3"]]
        payload = {
            "ok": True,
            "top3": top3,
            "best": summary.get("best"),
            "n_ok": summary.get("n_ok"),
            "n_failed": summary.get("n_failed"),
            "source": "sweep",
        }
        _update_phase("mf_sweep", status="ok", result=payload)
        logger.info("MF sweep ok: n_ok=%s best_mse=%s", payload["n_ok"], summary["best"]["val_mse"])
        return payload
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        logger.exception("MF sweep failed; will use fallback ensemble")
        traceback.print_exc()
        # Prefer prior successful results file if present
        prior = _load_json(ARTIFACTS / "mf_sweep_results.json")
        if prior.get("top3"):
            top3 = [_member_fields(r) for r in prior["top3"]]
            source = "prior_results"
        else:
            base = load_config(config_path)
            top3 = _fallback_mf_members(base)
            source = "config_fallback"
        payload = {"ok": False, "error": err, "top3": top3, "source": source}
        _update_phase("mf_sweep", status="failed_fallback", result=payload)
        return payload


@task(name="overnight_svdpp_sweep", retries=1, retry_delay_seconds=60)
def phase_svdpp_sweep(config_path: str) -> Dict[str, Any]:
    logger = get_run_logger()
    _update_phase("svdpp_sweep", status="running", started_at=_now())
    try:
        summary = run_svdpp_sweep(
            config_path=config_path,
            grid=DEFAULT_SVDPP_GRID,
            experiment="task1_overnight_svdpp_sweep",
            resume=True,
        )
        best = summary["best"]
        params = _member_fields(best)
        params["use_svdpp"] = True
        payload = {
            "ok": True,
            "enabled": True,
            "params": params,
            "best_mse": best.get("val_mse"),
            "n_ok": summary.get("n_ok"),
            "n_failed": summary.get("n_failed"),
            "source": "sweep",
        }
        _update_phase("svdpp_sweep", status="ok", result=payload)
        logger.info("SVD++ sweep ok: best_mse=%s", payload["best_mse"])
        return payload
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        logger.exception("SVD++ sweep failed; falling back")
        traceback.print_exc()
        prior = _load_json(ARTIFACTS / "svdpp_sweep_results.json")
        if prior.get("best"):
            params = _member_fields(prior["best"])
            params["use_svdpp"] = True
            payload = {
                "ok": False,
                "enabled": True,
                "params": params,
                "error": err,
                "source": "prior_results",
            }
        else:
            base = load_config(config_path)
            params = _fallback_svdpp(base)
            payload = {
                "ok": False,
                "enabled": True,
                "params": {
                    "n_factors": int(params.get("n_factors", 20)),
                    "n_epochs": int(params.get("n_epochs", 30)),
                    "lr_all": float(params.get("lr_all", 0.005)),
                    "reg_all": float(params.get("reg_all", 0.1)),
                    "use_svdpp": True,
                },
                "error": err,
                "source": "config_fallback",
            }
        _update_phase("svdpp_sweep", status="failed_fallback", result=payload)
        return payload


@task(name="overnight_final_pipeline", retries=0)
def phase_final_pipeline(
    base_config_path: str,
    mf_payload: Dict[str, Any],
    svdpp_payload: Dict[str, Any],
) -> Dict[str, Any]:
    logger = get_run_logger()
    _update_phase("final_pipeline", status="running", started_at=_now())
    base = load_config(base_config_path)
    mf_members = mf_payload["top3"]
    svdpp_params = svdpp_payload.get("params")
    svdpp_enabled = bool(svdpp_payload.get("enabled", True))

    attempts = [
        {"neural": True, "svdpp": svdpp_enabled, "lgbm_mode": "residual", "label": "full"},
        {"neural": False, "svdpp": svdpp_enabled, "lgbm_mode": "residual", "label": "no_neural"},
        {"neural": False, "svdpp": False, "lgbm_mode": "residual", "label": "mf_residual_only"},
        {"neural": False, "svdpp": False, "lgbm_mode": "rating", "label": "mf_rating_only"},
    ]

    last_error: Optional[str] = None
    for attempt in attempts:
        label = attempt["label"]
        logger.info("Final pipeline attempt: %s", label)
        try:
            cfg = _build_resolved_config(
                base,
                mf_members,
                svdpp_params,
                neural_enabled=bool(attempt["neural"]),
                svdpp_enabled=bool(attempt["svdpp"]),
            )
            cfg["lgbm"]["mode"] = attempt["lgbm_mode"]
            resolved = _write_resolved_yaml(cfg)
            # Safety: never point at winner CSV
            assert cfg["output_file"] == OVERNIGHT_CSV_NAME

            result = run_task1_pipeline(
                config_path=resolved,
                run_name=f"task1_overnight_{label}",
                enable_mlflow=True,
            )
            sub = result.get("submission_path")
            if sub and Path(sub).name == WINNER_CSV_NAME:
                raise RuntimeError(f"Pipeline wrote winner CSV unexpectedly: {sub}")

            payload = {
                "ok": True,
                "attempt": label,
                "submission_path": sub,
                "blend_metrics": result.get("blend_metrics"),
                "candidate_metrics": result.get("candidate_metrics"),
                "blend_weights": result.get("blend_weights"),
                "resolved_config": str(resolved),
            }
            _update_phase("final_pipeline", status="ok", result=payload)
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Final pipeline attempt %s failed: %s", label, last_error)
            traceback.print_exc()
            _update_phase(
                "final_pipeline",
                status="attempt_failed",
                last_attempt=label,
                last_error=last_error,
            )

    payload = {"ok": False, "error": last_error or "all attempts failed"}
    _update_phase("final_pipeline", status="failed", result=payload)
    return payload


@flow(name="task1_overnight", log_prints=True)
def task1_overnight_flow(
    config_path: Optional[str] = None,
    skip_mf_sweep: bool = False,
    skip_svdpp_sweep: bool = False,
) -> Dict[str, Any]:
    """Run overnight sweeps then final blend with per-phase fallbacks."""
    logger = get_run_logger()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(config_path) if config_path else OVERNIGHT_CONFIG
    if not cfg_path.exists():
        logger.warning("Missing %s; falling back to %s", cfg_path, FALLBACK_CONFIG)
        cfg_path = FALLBACK_CONFIG
    cfg_path_s = str(cfg_path)

    # Preserve winner CSV if somehow present next to overnight output
    winner = project_root() / "assignment1_2026" / WINNER_CSV_NAME
    if winner.exists():
        bak = winner.with_suffix(".csv.pre_overnight_bak")
        if not bak.exists():
            shutil.copy2(winner, bak)
            logger.info("Backed up Gradescope CSV to %s", bak)

    _update_phase("overnight_flow", status="running", started_at=_now(), config=cfg_path_s)
    print(f"Overnight config: {cfg_path_s}", flush=True)
    print("Safe output target:", OVERNIGHT_CSV_NAME, flush=True)

    if skip_mf_sweep:
        base = load_config(cfg_path_s)
        mf_payload = {
            "ok": True,
            "top3": _fallback_mf_members(base),
            "source": "skip_use_config",
        }
        _update_phase("mf_sweep", status="skipped", result=mf_payload)
    else:
        mf_payload = phase_mf_sweep(cfg_path_s)

    if skip_svdpp_sweep:
        base = load_config(cfg_path_s)
        sp = _fallback_svdpp(base)
        svdpp_payload = {
            "ok": True,
            "enabled": True,
            "params": {
                "n_factors": int(sp.get("n_factors", 20)),
                "n_epochs": int(sp.get("n_epochs", 30)),
                "lr_all": float(sp.get("lr_all", 0.005)),
                "reg_all": float(sp.get("reg_all", 0.1)),
                "use_svdpp": True,
            },
            "source": "skip_use_config",
        }
        _update_phase("svdpp_sweep", status="skipped", result=svdpp_payload)
    else:
        svdpp_payload = phase_svdpp_sweep(cfg_path_s)

    final_payload = phase_final_pipeline(cfg_path_s, mf_payload, svdpp_payload)

    summary = {
        "finished_at": _now(),
        "config": cfg_path_s,
        "mf_sweep": mf_payload,
        "svdpp_sweep": svdpp_payload,
        "final": final_payload,
        "output_file": OVERNIGHT_CSV_NAME,
        "winner_csv_untouched": WINNER_CSV_NAME,
        "phase_status_path": str(PHASE_STATUS_PATH),
    }
    _write_json(SUMMARY_PATH, summary)
    _update_phase(
        "overnight_flow",
        status="ok" if final_payload.get("ok") else "failed",
        summary_path=str(SUMMARY_PATH),
    )

    print("\n=== Overnight summary ===", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    if final_payload.get("ok"):
        print("Wrote:", final_payload.get("submission_path"), flush=True)
        print("Metrics:", final_payload.get("blend_metrics"), flush=True)
    else:
        print("Overnight final pipeline FAILED — see", SUMMARY_PATH, flush=True)
    return summary


if __name__ == "__main__":
    task1_overnight_flow()
