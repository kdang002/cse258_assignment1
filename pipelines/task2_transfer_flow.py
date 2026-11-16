"""Task 2 Gradescope-transfer run: BPR-only multi-seed ensemble.

Writes assignment1_2026/task_2_predictions_candidate.csv
(does NOT overwrite the locked task_2_predictions.csv winner).

Hardware notes (i7-9700k 8c / 32GB):
  - parallel_workers=2 × num_threads=3 ≈ 6 cores; leaves headroom for co-agents
  - GPU BPR requires RMM (not installed); CPU implicit BPR used instead

Run:
  python -m pipelines.task2_transfer_flow
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

DEFAULT_CONFIG = ROOT / "configs" / "task2_transfer.yaml"


@task(name="load_task2_transfer_config")
def load_task2_transfer_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    return load_config(config_path or str(DEFAULT_CONFIG))


@task(name="train_task2_transfer")
def train_task2_transfer(config_path: Optional[str] = None) -> Dict[str, Any]:
    logger = get_run_logger()
    path = config_path or str(DEFAULT_CONFIG)
    logger.info("Task 2 transfer pipeline (BPR-only multi-seed) config=%s", path)
    result = run_task2_pipeline(
        config_path=path,
        run_name="task2_transfer_bpr_only",
        enable_mlflow=True,
    )
    logger.info(
        "blend_val_accuracy=%.6f bpr_val_accuracy=%.6f submission=%s",
        result["blend_metrics"]["blend_val_accuracy"],
        result["bpr_metrics"]["bpr_val_accuracy"],
        result.get("submission_path"),
    )
    return result


@flow(name="task2_transfer_pipeline", log_prints=True)
def task2_transfer_flow(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_task2_transfer_config(config_path)
    print(f"Loaded config from {cfg.get('_config_path')}")
    print(f"output_file={cfg.get('output_file')} (candidate — locked winner untouched)")
    result = train_task2_transfer(config_path=config_path or str(DEFAULT_CONFIG))
    print("Candidate metrics:", result["candidate_metrics"])
    print("Blend weights:", result["blend_weights"])
    print("Blend metrics:", result["blend_metrics"])
    if result.get("submission_path"):
        print("Wrote candidate submission:", result["submission_path"])
    return result


if __name__ == "__main__":
    task2_transfer_flow()
