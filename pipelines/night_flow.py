"""8-hour overnight: train the weaker tasks with Prefect retries + MLflow.

GPU BPR jobs go through WSL (one 3080). CPU rating jobs run natively.
T2/T5 write staging CSVs so Production 0.9169 / 0.86862 stay restorable.

  python -m pipelines.night_flow
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from prefect import flow, task
from prefect.logging import get_run_logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATUS_PATH = ROOT / "artifacts" / "night_status.json"
WSL_SCRIPT_WIN = ROOT / "scripts" / "wsl_run_task.sh"
WSL_SCRIPT_LINUX = (
    "/mnt/c/Users/kietd/OneDrive/UCSD - Projects/CSE258/assignment1_2026/"
    "scripts/wsl_run_task.sh"
)


def _unix_script() -> None:
    data = WSL_SCRIPT_WIN.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    WSL_SCRIPT_WIN.write_bytes(data)


def _write_status(update: Dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    if STATUS_PATH.is_file():
        try:
            payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.update(update)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@task(name="wsl_train", retries=2, retry_delay_seconds=120)
def wsl_train(label: str, python_c: str) -> str:
    logger = get_run_logger()
    logger.info("WSL GPU job %s", label)
    _unix_script()
    cmd = [
        "wsl",
        "-d",
        "Ubuntu",
        "--",
        "bash",
        WSL_SCRIPT_LINUX,
        "-c",
        python_c,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit {proc.returncode}")
    logger.info("%s finished in %.0fs", label, time.time() - t0)
    _write_status({label: {"ok": True, "seconds": time.time() - t0}})
    return label


@task(name="cpu_task3", retries=2, retry_delay_seconds=60)
def cpu_task3() -> Dict[str, Any]:
    from src.task3.pipeline import run_task3_pipeline

    result = run_task3_pipeline(
        config_path=str(ROOT / "configs" / "task3_night.yaml"),
        run_name="task3_night",
        enable_mlflow=True,
    )
    _write_status(
        {
            "task3_night": {
                "ok": True,
                "blend": result.get("blend_metrics"),
                "submission": result.get("submission_path"),
            }
        }
    )
    return result


@task(name="cpu_task1", retries=1, retry_delay_seconds=60)
def cpu_task1() -> Dict[str, Any]:
    from src.task1.pipeline import run_task1_pipeline

    result = run_task1_pipeline(
        config_path=str(ROOT / "configs" / "task1_candidate.yaml"),
        run_name="task1_night_confirm",
        enable_mlflow=True,
    )
    _write_status(
        {
            "task1_night": {
                "ok": True,
                "blend": result.get("blend_metrics"),
                "submission": result.get("submission_path"),
            }
        }
    )
    return result


@flow(name="assignment1_night_push", log_prints=True)
def night_push_flow(include_task1: bool = True, skip_task4: bool = False) -> Dict[str, Any]:
    """T4 GPU → T5 item2vec GPU → T2 interaction GPU → T3 CPU → optional T1."""
    _write_status({"started": datetime.now().isoformat(timespec="seconds")})
    out: Dict[str, Any] = {}

    if not skip_task4:
        wsl_train(
            "task4_night",
            "from src.task4.pipeline import run_task4_pipeline; "
            "run_task4_pipeline('configs/task4_night.yaml', "
            "run_name='task4_night_gpu', enable_mlflow=True)",
        )
        out["task4"] = "ok"

    wsl_train(
        "task5_night",
        "from src.task5.pipeline import run_task5_pipeline; "
        "run_task5_pipeline('configs/task5_night.yaml', "
        "run_name='task5_night_item2vec', enable_mlflow=True)",
    )
    out["task5"] = "ok"

    wsl_train(
        "task2_night",
        "from src.task2.pipeline import run_task2_pipeline; "
        "run_task2_pipeline('configs/task2_night.yaml', "
        "run_name='task2_night_interaction', enable_mlflow=True)",
    )
    out["task2"] = "ok"

    t3 = cpu_task3()
    out["task3"] = t3.get("blend_metrics")

    if include_task1:
        t1 = cpu_task1()
        out["task1"] = t1.get("blend_metrics")

    _write_status({"finished": datetime.now().isoformat(timespec="seconds"), "ok": True})
    return out


if __name__ == "__main__":
    skip_t4 = "--skip-task4" in sys.argv
    night_push_flow(include_task1=True, skip_task4=skip_t4)
