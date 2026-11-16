"""Prefect orchestration for Task 3 time-aware rating prediction.

Run:
  python -m pipelines.task3_flow
  # or Gradescope-facing (no MLflow):
  python -c "from src.task3.pipeline import run_task3_pipeline; run_task3_pipeline(enable_mlflow=False)"
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

from src.task3.pipeline import load_config, run_task3_pipeline


@task(name="load_task3_config", retries=0)
def load_task3_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    return load_config(config_path)


@task(name="train_and_evaluate_task3", retries=2, retry_delay_seconds=60)
def train_and_evaluate_task3(
    config_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Starting Task 3 time-aware pipeline (MF -> time-LGBM -> blend)")
    result = run_task3_pipeline(
        config_path=config_path,
        run_name=run_name,
        enable_mlflow=True,
    )
    logger.info(
        "blend_val_mse=%.6f mf_val_mse=%.6f submission=%s",
        result["blend_metrics"]["blend_val_mse"],
        result["mf_metrics"]["mf_val_mse"],
        result.get("submission_path"),
    )
    return result


@flow(name="task3_rating_time_pipeline", log_prints=True)
def task3_rating_flow(
    config_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_task3_config(config_path)
    print(f"Loaded config from {cfg.get('_config_path')}")
    result = train_and_evaluate_task3(
        config_path=config_path,
        run_name=run_name or "task3_prefect_run",
    )
    print("Candidate metrics:", result["candidate_metrics"])
    print("Blend weights:", result["blend_weights"])
    print("Blend metrics:", result["blend_metrics"])
    if result.get("submission_path"):
        print("Wrote submission:", result["submission_path"])
    return result


if __name__ == "__main__":
    task3_rating_flow()
