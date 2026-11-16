"""Prefect orchestration for Task 1 rating prediction.

Run:
  python -m pipelines.task1_flow
  # or
  python pipelines/task1_flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from prefect import flow, task
from prefect.logging import get_run_logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task1.pipeline import load_config, run_task1_pipeline


@task(name="load_task1_config", retries=0)
def load_task1_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    return load_config(config_path)


@task(name="train_and_evaluate_task1", retries=2, retry_delay_seconds=60)
def train_and_evaluate_task1(
    config_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Starting Task 1 hybrid pipeline (MF -> LightGBM -> blend)")
    result = run_task1_pipeline(
        config_path=config_path,
        run_name=run_name,
        enable_mlflow=True,
    )
    logger.info(
        "blend_val_rmse=%.6f mf_val_rmse=%.6f submission=%s",
        result["blend_metrics"]["blend_val_rmse"],
        result["mf_metrics"]["mf_val_rmse"],
        result.get("submission_path"),
    )
    return result


@flow(name="task1_rating_pipeline", log_prints=True)
def task1_rating_flow(
    config_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Orchestrate Task 1 training, MLflow logging, and submission generation.

    Future tasks (2–6) should follow the same pattern: one Prefect flow per task,
    shared MLflow tracking URI under ./mlruns, artifacts under ./artifacts/taskN.
    """
    cfg = load_task1_config(config_path)
    print(f"Loaded config from {cfg.get('_config_path')}")
    result = train_and_evaluate_task1(
        config_path=config_path,
        run_name=run_name or "task1_prefect_run",
    )
    print("Candidate metrics:", result["candidate_metrics"])
    print("Blend weights:", result["blend_weights"])
    print("Blend metrics:", result["blend_metrics"])
    if result.get("submission_path"):
        print("Wrote submission:", result["submission_path"])
    return result


if __name__ == "__main__":
    task1_rating_flow()
