"""Overnight counting-set train: staging CSVs only.

Order: Task 4 (stronger BPR) → Task 2 (interaction-neg) → Task 5 (item2vec staging).
Never overwrites locked live Gradescope winners (T1–T6 live CSVs stay put).

  python scripts/overnight_counting.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assignment1_2026"
STATUS = ROOT / "artifacts" / "overnight_counting_status.json"
LOG = ROOT / "artifacts" / "overnight_counting.log"
WSL_SCRIPT = ROOT / "scripts" / "wsl_run_task.sh"
WSL_SCRIPT_LINUX = (
    "/mnt/c/Users/kietd/OneDrive/UCSD - Projects/CSE258/assignment1_2026/"
    "scripts/wsl_run_task.sh"
)

LOCKED = [
    "task_1_predictions.csv",
    "task_2_predictions.csv",
    "task_3_predictions.csv",
    "task_4_predictions.csv",
    "task_5_predictions.csv",
    "task_6_predictions.csv",
]

JOBS = [
    (
        "task4_overnight",
        "from src.task4.pipeline import run_task4_pipeline; "
        "run_task4_pipeline('configs/task4_overnight.yaml', "
        "run_name='task4_overnight_gpu', enable_mlflow=True)",
    ),
    (
        "task2_night",
        "from src.task2.pipeline import run_task2_pipeline; "
        "run_task2_pipeline('configs/task2_night.yaml', "
        "run_name='task2_overnight_interaction', enable_mlflow=True)",
    ),
    (
        "task5_night",
        "from src.task5.pipeline import run_task5_pipeline; "
        "run_task5_pipeline('configs/task5_night.yaml', "
        "run_name='task5_overnight_item2vec', enable_mlflow=True)",
    ),
]


def _log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_locked() -> dict:
    snaps = {}
    for name in LOCKED:
        p = DATA / name
        if not p.is_file():
            snaps[name] = None
            continue
        snaps[name] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
    return snaps


def assert_locked_untouched(before: dict) -> None:
    after = snapshot_locked()
    for name, prev in before.items():
        cur = after.get(name)
        if prev != cur:
            raise RuntimeError(
                f"LOCKED FILE CHANGED during overnight: {name}\n"
                f"  before={prev}\n  after={cur}"
            )


def write_status(update: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if STATUS.is_file():
        try:
            payload = json.loads(STATUS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.update(update)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def unixify_wsl_script() -> None:
    data = WSL_SCRIPT.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    WSL_SCRIPT.write_bytes(data)


def run_wsl(label: str, python_c: str) -> None:
    _log(f"START {label}")
    write_status({label: {"status": "running", "started": datetime.now().isoformat(timespec="seconds")}})
    unixify_wsl_script()
    t0 = time.time()
    proc = subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "bash", WSL_SCRIPT_LINUX, "-c", python_c],
        cwd=str(ROOT),
    )
    secs = time.time() - t0
    ok = proc.returncode == 0
    write_status(
        {
            label: {
                "status": "ok" if ok else "failed",
                "seconds": secs,
                "exit": proc.returncode,
                "finished": datetime.now().isoformat(timespec="seconds"),
            }
        }
    )
    _log(f"END {label} exit={proc.returncode} seconds={secs:.0f}")
    if not ok:
        raise RuntimeError(f"{label} failed with exit {proc.returncode}")


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    _log("overnight_counting start (staging only)")
    before = snapshot_locked()
    write_status({"started": datetime.now().isoformat(timespec="seconds"), "locked_before": before})

    try:
        for label, code in JOBS:
            run_wsl(label, code)
            assert_locked_untouched(before)
        write_status({"finished": datetime.now().isoformat(timespec="seconds"), "ok": True})
        _log("overnight_counting finished OK")
        return 0
    except Exception as exc:
        write_status(
            {
                "finished": datetime.now().isoformat(timespec="seconds"),
                "ok": False,
                "error": str(exc),
            }
        )
        _log(f"overnight_counting FAILED: {exc}")
        try:
            assert_locked_untouched(before)
            _log("locked live CSVs still intact after failure")
        except Exception as lock_exc:
            _log(f"CRITICAL: {lock_exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
