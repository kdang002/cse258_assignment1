"""Prefect flow: Task 2 BPR hyperparameter sweep, then optional full pipeline.

Run overnight:
  python -m pipelines.task2_bpr_sweep
  # after sweep auto-promotes top-3 into configs/task2.yaml:
  python -m pipelines.task2_flow

Or one-shot (sweep then full pipeline):
  python -m pipelines.task2_bpr_sweep --then-full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from prefect import flow, task
from prefect.logging import get_run_logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task2.bpr_sweep import run_bpr_sweep
from src.task2.pipeline import run_task2_pipeline


@task(name="run_bpr_hyperparam_sweep")
def run_bpr_hyperparam_sweep() -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Starting Task 2 BPR sweep (MLflow experiment task2_bpr_sweep)")
    summary = run_bpr_sweep()
    best = summary["best"]
    logger.info(
        "Best BPR: factors=%s iterations=%s lr=%s reg=%s val_acc=%.6f",
        best["factors"],
        best["iterations"],
        best["learning_rate"],
        best["regularization"],
        best["val_accuracy"],
    )
    return summary


@task(name="run_full_task2_after_sweep")
def run_full_task2_after_sweep() -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Running full Task 2 pipeline with promoted BPR ensemble")
    return run_task2_pipeline(
        run_name="task2_after_sweep",
        enable_mlflow=True,
    )


@flow(name="task2_bpr_sweep", log_prints=True)
def task2_bpr_sweep_flow(then_full: bool = False) -> Dict[str, Any]:
    summary = run_bpr_hyperparam_sweep()
    print("Best:", summary["best"])
    print("Top3:", summary["top3"])
    print("Top3 equal blend accuracy:", summary["top3_equal_blend_accuracy"])
    print("Wrote artifacts/task2/bpr_sweep_results.json")
    if then_full:
        result = run_full_task2_after_sweep()
        print("Full pipeline blend:", result["blend_metrics"])
        print("Submission:", result.get("submission_path"))
        summary["full_pipeline"] = {
            "blend_metrics": result["blend_metrics"],
            "submission_path": result.get("submission_path"),
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2 BPR sweep")
    parser.add_argument(
        "--then-full",
        action="store_true",
        help="After sweep, run full Task 2 pipeline with promoted top-3",
    )
    args = parser.parse_args()
    task2_bpr_sweep_flow(then_full=args.then_full)
