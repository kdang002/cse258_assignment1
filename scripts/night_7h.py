"""Archive locked prediction CSVs, then keep the 3080 busy for 7 hours.

WSL runs implicit CUDA BPR/ALS for tasks 5, 4, and 2 and writes
task_N_predictions.csv. The rest of the window is CUDA biased MF for tasks
3 and 1 on this Windows interpreter.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assignment1_2026"
HOURS = 7.0

ARCHIVES = {
    "task_1_predictions.csv": "task_1_predictions_public_0.61925.csv",
    "task_2_predictions.csv": "task_2_predictions_public_0.9169.csv",
    "task_3_predictions.csv": "task_3_predictions_public_0.79080.csv",
    "task_4_predictions.csv": "task_4_predictions_public_0.8495.csv",
    "task_5_predictions.csv": "task_5_predictions_public_0.86862.csv",
}

MF_PLAN = [
    (3, 128, 50, 42),
    (3, 256, 45, 42),
    (3, 192, 50, 43),
    (3, 384, 35, 44),
    (1, 128, 40, 42),
    (1, 256, 35, 42),
    (3, 128, 60, 45),
    (1, 192, 40, 43),
    (3, 256, 50, 46),
    (1, 384, 30, 44),
]


def archive() -> None:
    for src_name, dst_name in ARCHIVES.items():
        src = DATA / src_name
        dst = DATA / dst_name
        if not src.exists():
            print(f"archive skip, missing {src_name}", flush=True)
            continue
        if dst.exists():
            dst = DATA / f"{dst.stem}_{int(time.time())}{dst.suffix}"
        src.rename(dst)
        print(f"archived {src_name} -> {dst.name}", flush=True)


def _wsl_path(path: Path) -> str:
    resolved = str(path.resolve())
    drive, rest = resolved[0].lower(), resolved[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def run_wsl() -> None:
    script = _wsl_path(ROOT / "scripts" / "wsl_gpu_night.sh")
    print(f"starting WSL implicit CUDA pipelines via {script}", flush=True)
    proc = subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "bash", script],
        cwd=str(ROOT),
    )
    print(f"WSL pipelines exit={proc.returncode}", flush=True)


def run_torch_until(deadline: float) -> None:
    cycle = 0
    while time.time() < deadline:
        task, factors, epochs, seed = MF_PLAN[cycle % len(MF_PLAN)]
        seed = seed + (cycle // len(MF_PLAN)) * 10
        cycle += 1
        print(
            f"CUDA MF cycle {cycle} task={task} factors={factors} "
            f"epochs={epochs} seed={seed}",
            flush=True,
        )
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "gpu_mf_night.py"),
                "--task",
                str(task),
                "--factors",
                str(factors),
                "--epochs",
                str(epochs),
                "--seed",
                str(seed),
            ],
            cwd=str(ROOT),
        )
        print(f"CUDA MF exit={proc.returncode}", flush=True)
        if proc.returncode != 0 and time.time() < deadline:
            time.sleep(5)


def main() -> None:
    deadline = time.time() + HOURS * 3600
    print(f"night window {HOURS:.0f}h, deadline epoch {deadline:.0f}", flush=True)
    archive()
    run_wsl()
    if time.time() < deadline:
        run_torch_until(deadline)
    print("7h window finished", flush=True)


if __name__ == "__main__":
    main()
