"""Prefect flow: Task 1 MF hyperparameter sweep.

Run:
  python -m pipelines.task1_mf_sweep
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from prefect import flow, task
from prefect.logging import get_run_logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task1.mf_sweep import run_mf_sweep


@task(name="run_mf_hyperparam_sweep")
def run_mf_hyperparam_sweep() -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Starting Task 1 MF sweep (logged to MLflow experiment task1_mf_sweep)")
    summary = run_mf_sweep()
    best = summary["best"]
    logger.info(
        "Best MF: factors=%s epochs=%s lr=%s reg=%s val_mse=%.6f",
        best["n_factors"],
        best["n_epochs"],
        best["lr_all"],
        best["reg_all"],
        best["val_mse"],
    )
    return summary


@flow(name="task1_mf_sweep", log_prints=True)
def task1_mf_sweep_flow() -> Dict[str, Any]:
    summary = run_mf_hyperparam_sweep()
    print("Best:", summary["best"])
    print("Top3 blend:", summary["top3_equal_blend"])
    print("Wrote artifacts/task1/mf_sweep_results.json")
    return summary


if __name__ == "__main__":
    task1_mf_sweep_flow()
