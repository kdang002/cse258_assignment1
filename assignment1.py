"""
CSE258 Assignment 1 (2026)

Working solutions for each task. Run:
    python assignment1.py

Prediction CSVs match the baseline format for Gradescope upload.
Local MLOps (Prefect + MLflow) lives under pipelines/ and is optional.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task1.pipeline import run_task1_pipeline
from src.task2.pipeline import run_task2_pipeline
from src.task3.pipeline import run_task3_pipeline
from src.task4.pipeline import run_task4_pipeline
from src.task5.pipeline import run_task5_pipeline
from src.task6.pipeline import run_task6_pipeline

# Same layout convention as baselines.ipynb
dataDir = str(ROOT / "assignment1_2026")


# =============================================================================
# Task 1: rating prediction (user, item) -> rating
# =============================================================================
# Approach: equal-weight ensemble of tuned Surprise SVD models
# (compact latent factors + stronger regularization), then LightGBM
# residual correction on user/item stats, blended with user mean.
# =============================================================================

def run_task1() -> dict:
    """Fit Task 1 models and write task_1_predictions.csv (no Prefect)."""
    print("Task 1: training tuned SVD ensemble + LightGBM residual ...")
    result = run_task1_pipeline(
        config_path=ROOT / "configs" / "task1.yaml",
        run_name="assignment1_task1",
        enable_mlflow=True,
    )
    out = result.get("submission_path")
    print(f"Task 1: wrote {out}")
    print(f"Task 1: val metrics {result.get('blend_metrics')}")
    return result


# =============================================================================
# Task 2: interaction prediction (pairwise which item)
# =============================================================================
# Approach: tuned implicit BPR ensemble (+ ALS) scored as user·item dots,
# popularity/rating-affinity features, LightGBM pairwise classifier on
# score margins, blended on local pairwise validation accuracy.
# =============================================================================

def run_task2() -> dict:
    """Fit Task 2 models and write task_2_predictions.csv (Prefect/MLflow tracked)."""
    print("Task 2: training BPR-only multi-seed transfer stack ...")
    result = run_task2_pipeline(
        config_path=ROOT / "configs" / "task2_transfer.yaml",
        run_name="assignment1_task2",
        enable_mlflow=True,
    )
    out = result.get("submission_path")
    print(f"Task 2: wrote {out}")
    print(f"Task 2: val metrics {result.get('blend_metrics')}")
    return result


# =============================================================================
# Task 3: rating prediction with timestamp
# =============================================================================
# Approach: random-user val + OOF MF→LGBM residual on bidirectional time features;
# blend on val; hardened full-train single-MF refit. MLflow registers params.
# =============================================================================

def run_task3() -> dict:
    """Fit Task 3 models and write task_3_predictions.csv (Prefect/MLflow tracked)."""
    print("Task 3: training time-aware SVD + LightGBM residual ...")
    result = run_task3_pipeline(
        config_path=ROOT / "configs" / "task3.yaml",
        run_name="assignment1_task3",
        enable_mlflow=True,
    )
    out = result.get("submission_path")
    print(f"Task 3: wrote {out}")
    print(f"Task 3: val metrics {result.get('blend_metrics')}")
    return result


# =============================================================================
# Task 4: interaction prediction with timestamp
# =============================================================================
# Approach: time-split val; BPR/ALS + timestamp/recency features into
# LightGBM pairwise; blend on val. CSV includes timestamp for Gradescope keys.
# =============================================================================

def run_task4() -> dict:
    """Fit Task 4 models and write task_4_predictions.csv (no Prefect)."""
    print("Task 4: training time-aware BPR/ALS + LightGBM pairwise ...")
    result = run_task4_pipeline(
        config_path=ROOT / "configs" / "task4.yaml",
        run_name="assignment1_task4",
        enable_mlflow=True,
    )
    out = result.get("submission_path")
    print(f"Task 4: wrote {out}")
    print(f"Task 4: val metrics {result.get('blend_metrics')}")
    return result


# =============================================================================
# Task 5: sequential next-item prediction
# =============================================================================
# Approach: leave-last-out sequential pairs with interaction-frequency
# negatives; Markov/window + BPR/ALS → LightGBM pairwise; val blend.
# MLflow registers params; Prefect restores Production on a worse public score.
# =============================================================================

def run_task5() -> dict:
    """Fit Task 5 models and write task_5_predictions.csv (Prefect/MLflow tracked)."""
    print("Task 5: training sequential + BPR/ALS + LightGBM pairwise ...")
    result = run_task5_pipeline(
        config_path=ROOT / "configs" / "task5.yaml",
        run_name="assignment1_task5",
        enable_mlflow=True,
    )
    out = result.get("submission_path")
    print(f"Task 5: wrote {out}")
    print(f"Task 5: val metrics {result.get('blend_metrics')}")
    return result


# =============================================================================
# Task 6: time-delta prediction  [TODO]
# =============================================================================

def run_task6() -> dict:
    """Fit Task 6 and write task_6_predictions.csv. Does not touch tasks 1–5."""
    print("Task 6: training next-gap model ...")
    result = run_task6_pipeline(
        config_path=ROOT / "configs" / "task6.yaml",
        run_name="assignment1_task6",
        enable_mlflow=True,
    )
    print(f"Task 6: wrote {result.get('submission_path')}")
    print(f"Task 6: val metrics {result.get('blend_metrics')}")
    return result


if __name__ == "__main__":
    # Baseline-style entry: generate submitted prediction files.
    # Task 2 overnight MLOps (preferred; does not block Task 1 agent):
    #   python -m pipelines.task2_bpr_sweep --then-full
    #   python -m pipelines.task2_flow
    run_task1()
    # run_task2()
    # run_task3()
    # run_task4()
    # run_task5()
    # run_task6()
