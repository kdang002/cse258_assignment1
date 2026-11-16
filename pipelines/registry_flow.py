"""Prefect + MLflow registry: seed champions, record public scores, restore, retrain.

  python -m pipelines.registry_flow seed
  python -m pipelines.registry_flow record --task 4 --public 0.862
  python -m pipelines.registry_flow restore --task 2
  python -m pipelines.registry_flow retrain --task 4
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from prefect import flow, task
from prefect.logging import get_run_logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.registry import (
    PIPELINE_BY_TASK,
    champion_config_path,
    record_public_score,
    register_offline_bundle,
    restore_champion_csv,
)

# Public scores as of 2026-11-05 leaderboard row 195, plus prior winners.
SEED_BUNDLES = [
    dict(
        task=1,
        config="configs/task1_candidate.yaml",
        public=0.6141114686206307,
        stage="Production",
        submission="assignment1_2026/task_1_predictions.csv",
        artifacts_dir="artifacts/task1_candidate",
        note="OOF LGBM + 3-seed SVD bag",
    ),
    dict(
        task=1,
        config="configs/task1.yaml",
        public=0.6192479467681701,
        stage="Staging",
        submission="artifacts/task1/task_1_predictions.csv",
        artifacts_dir="artifacts/task1",
        note="prior diverse SVD+LGBM 0.619",
    ),
    dict(
        task=2,
        config="configs/task2_transfer.yaml",
        public=0.9169,
        stage="Production",
        submission="artifacts/task2_transfer/task_2_predictions_candidate.csv",
        artifacts_dir="artifacts/task2_transfer",
        note="BPR-only multi-seed transfer",
    ),
    dict(
        task=2,
        config="configs/task2_gpu.yaml",
        public=0.9023,
        stage="Staging",
        submission="artifacts/task2_gpu/task_2_predictions_candidate.csv",
        artifacts_dir="artifacts/task2_gpu",
        note="GPU BPR lost to 0.9169",
    ),
    dict(
        task=3,
        config="configs/task3_candidate.yaml",
        public=0.7829052412731236,
        stage="Production",
        submission="assignment1_2026/task_3_predictions.csv",
        artifacts_dir="artifacts/task3_candidate",
        note="bidirectional LOO time features",
    ),
    dict(
        task=4,
        config="configs/task4_candidate.yaml",
        public=0.862,
        stage="Production",
        submission="assignment1_2026/task_4_predictions.csv",
        artifacts_dir="artifacts/task4_candidate",
        note="random-user + interaction negatives",
    ),
    dict(
        task=5,
        config="configs/task5.yaml",
        public=0.86862,
        stage="Production",
        submission="artifacts/task5/task_5_predictions_candidate.csv",
        artifacts_dir="artifacts/task5",
        note="interaction-frequency negatives",
    ),
    dict(
        task=5,
        config="configs/task5_gpu.yaml",
        public=0.8676514645219275,
        stage="Staging",
        submission="artifacts/task5_gpu/task_5_predictions_candidate.csv",
        artifacts_dir="artifacts/task5_gpu",
        note="GPU BPR lost to 0.8686",
    ),
]


@task(name="seed_one_champion")
def seed_one_champion(spec: Dict[str, Any]) -> str:
    logger = get_run_logger()
    version = register_offline_bundle(
        task=int(spec["task"]),
        config_path=spec["config"],
        public=spec.get("public"),
        stage=spec["stage"],
        submission=spec.get("submission"),
        artifacts_dir=spec.get("artifacts_dir"),
        run_name=spec.get("note") or f"task{spec['task']}_{spec['stage']}",
        note=spec.get("note") or "",
    )
    logger.info(
        "task=%s stage=%s public=%s version=%s",
        spec["task"],
        spec["stage"],
        spec.get("public"),
        version,
    )
    return version


@flow(name="seed_mlflow_registry", log_prints=True)
def seed_registry_flow() -> Dict[str, str]:
    """Register current winners/losers from configs + CSVs. No retrain."""
    out: Dict[str, str] = {}
    for spec in SEED_BUNDLES:
        key = f"task{spec['task']}_{spec['stage']}"
        out[key] = seed_one_champion(spec)
    return out


@flow(name="record_public_score", log_prints=True)
def record_public_flow(task: int, public: float, note: str = "") -> Dict[str, Any]:
    """Promote Staging → Production if public beats the locked score; else restore Production CSV."""
    result = record_public_score(int(task), float(public), note=note)
    print(result["reason"])
    return result


@flow(name="restore_champion_csv", log_prints=True)
def restore_champion_flow(task: int, stage: str = "Production") -> str:
    path = restore_champion_csv(int(task), stage=stage)
    print(f"Restored {path}")
    return str(path)


@task(name="retrain_from_registered_params", retries=2, retry_delay_seconds=60)
def retrain_from_registered_params(task: int, stage: str = "Production") -> Dict[str, Any]:
    logger = get_run_logger()
    cfg_path = champion_config_path(int(task), stage=stage)
    logger.info("Retraining task %s from %s %s", task, stage, cfg_path)
    mod_name, fn_name = PIPELINE_BY_TASK[int(task)].split(":")
    fn = getattr(importlib.import_module(mod_name), fn_name)
    return fn(config_path=cfg_path, run_name=f"task{task}_retrain_{stage}", enable_mlflow=True)


@flow(name="retrain_champion", log_prints=True)
def retrain_champion_flow(task: int, stage: str = "Production") -> Dict[str, Any]:
    result = retrain_from_registered_params(int(task), stage=stage)
    print("blend:", result.get("blend_metrics"))
    print("submission:", result.get("submission_path"))
    return result


def _cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed")
    rec = sub.add_parser("record")
    rec.add_argument("--task", type=int, required=True)
    rec.add_argument("--public", type=float, required=True)
    rec.add_argument("--note", default="")
    rst = sub.add_parser("restore")
    rst.add_argument("--task", type=int, required=True)
    rst.add_argument("--stage", default="Production")
    tr = sub.add_parser("retrain")
    tr.add_argument("--task", type=int, required=True)
    tr.add_argument("--stage", default="Production")
    args = parser.parse_args(argv)

    if args.cmd == "seed":
        seed_registry_flow()
    elif args.cmd == "record":
        record_public_flow(args.task, args.public, note=args.note)
    elif args.cmd == "restore":
        restore_champion_flow(args.task, stage=args.stage)
    elif args.cmd == "retrain":
        retrain_champion_flow(args.task, stage=args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
