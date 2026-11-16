"""Prefect run for the Task 1 arena winner, full-data scale-up.

Writes task_1_predictions_arena_knn.csv. Does not replace task_1_predictions.csv.

  python -m pipelines.task1_arena_flow
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

from src.task1.arena_scale import run_task1_arena_scale


@task(name="train_task1_arena_knn", retries=2, retry_delay_seconds=60)
def train_task1_arena_knn(config_path: Optional[str] = None, run_name: Optional[str] = None) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Task 1 residual item-kNN scale-up (staging CSV only)")
    result = run_task1_arena_scale(config_path=config_path, run_name=run_name, enable_mlflow=True)
    logger.info("val_mse=%.6f submission=%s", result["blend_metrics"]["val_mse"], result.get("submission_path"))
    return result


@flow(name="task1_arena_knn", log_prints=True)
def task1_arena_flow(config_path: Optional[str] = None, run_name: Optional[str] = None) -> Dict[str, Any]:
    return train_task1_arena_knn(config_path=config_path, run_name=run_name or "task1_arena_knn_full")


if __name__ == "__main__":
    task1_arena_flow()
