"""Prefect run for Task 6 next-gap prediction.

Writes assignment1_2026/task_6_predictions.csv. Does not touch tasks 1–5.

  python -m pipelines.task6_flow
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

from src.task6.pipeline import run_task6_pipeline


@task(name="train_task6", retries=2, retry_delay_seconds=60)
def train_task6(config_path: Optional[str] = None, run_name: Optional[str] = None) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Task 6 next-gap MAE model")
    result = run_task6_pipeline(config_path=config_path, run_name=run_name, enable_mlflow=True)
    logger.info("val_mae=%.0f submission=%s", result["blend_metrics"]["val_mae"], result.get("submission_path"))
    return result


@flow(name="task6_next_gap", log_prints=True)
def task6_flow(config_path: Optional[str] = None, run_name: Optional[str] = None) -> Dict[str, Any]:
    return train_task6(config_path=config_path, run_name=run_name or "task6_full")


if __name__ == "__main__":
    task6_flow()
