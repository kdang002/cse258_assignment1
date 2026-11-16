"""Prefect orchestration for Task 2 pairwise interaction prediction.

Run:
  python -m pipelines.task2_flow
  # or
  python pipelines/task2_flow.py
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

from src.task2.pipeline import load_config, run_task2_pipeline


@task(name="load_task2_config", retries=0)
def load_task2_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    return load_config(config_path)


@task(name="train_and_evaluate_task2", retries=2, retry_delay_seconds=60)
def train_and_evaluate_task2(
    config_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Starting Task 2 pipeline (BPR -> ALS -> LightGBM -> blend)")
    result = run_task2_pipeline(
        config_path=config_path,
        run_name=run_name,
        enable_mlflow=True,
    )
    logger.info(
        "blend_val_accuracy=%.6f bpr_val_accuracy=%.6f submission=%s",
        result["blend_metrics"]["blend_val_accuracy"],
        result["bpr_metrics"]["bpr_val_accuracy"],
        result.get("submission_path"),
    )
    return result


@flow(name="task2_interaction_pipeline", log_prints=True)
def task2_interaction_flow(
    config_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Orchestrate Task 2 training, MLflow logging, and submission generation."""
    cfg = load_task2_config(config_path)
    print(f"Loaded config from {cfg.get('_config_path')}")
    result = train_and_evaluate_task2(
        config_path=config_path,
        run_name=run_name or "task2_prefect_run",
    )
    print("Candidate metrics:", result["candidate_metrics"])
    print("Blend weights:", result["blend_weights"])
    print("Blend metrics:", result["blend_metrics"])
    if result.get("submission_path"):
        print("Wrote submission:", result["submission_path"])
    return result


if __name__ == "__main__":
    task2_interaction_flow()
