"""Promote a candidate prediction CSV over the locked winner only if public score improves."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "artifacts" / "leaderboard.json"

TASK_KEYS: Dict[int, list[str]] = {
    1: ["user", "item", "rating"],
    2: ["user", "item1", "item2", "which"],
    3: ["user", "item", "timestamp", "rating"],
    4: ["user", "timestamp", "item1", "item2", "which"],
    5: ["user", "item1", "item2", "which"],
}

TEST_FILES: Dict[int, str] = {
    1: "task_1_test.csv",
    2: "task_2_test.csv",
    3: "task_3_test.csv",
    4: "task_4_test.csv",
    5: "task_5_test.csv",
}


@dataclass
class PromoteResult:
    promoted: bool
    reason: str
    locked_public: float
    public: float
    locked_file: Path
    candidate: Path
    archive: Optional[Path] = None


def _load_ledger() -> Dict[str, Any]:
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_ledger(ledger: Dict[str, Any]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
        f.write("\n")


def _is_better(direction: str, public: float, locked: float) -> bool:
    if direction == "lower":
        return public < locked
    if direction == "higher":
        return public > locked
    raise ValueError(f"Unknown direction: {direction}")


def _validate_schema(task: int, candidate: Path) -> None:
    keys = TASK_KEYS[task]
    pred = pd.read_csv(candidate)
    missing = [c for c in keys if c not in pred.columns]
    if missing:
        raise ValueError(f"Candidate missing columns {missing}; need {keys}")

    test_path = ROOT / "assignment1_2026" / TEST_FILES[task]
    if not test_path.exists():
        return
    test = pd.read_csv(test_path)
    if len(pred) != len(test):
        raise ValueError(f"Row count mismatch: candidate={len(pred)} test={len(test)}")

    join_cols = [c for c in keys if c != "rating" and c != "which"]
    # Soft check: key columns present on both sides
    for c in join_cols:
        if c not in test.columns:
            continue
        if pred[c].astype(str).tolist() != test[c].astype(str).tolist():
            # Allow reorder only if multisets match on keys
            if (
                pred[join_cols].astype(str).apply(tuple, axis=1).sort_values().tolist()
                != test[join_cols].astype(str).apply(tuple, axis=1).sort_values().tolist()
            ):
                raise ValueError("Candidate keys do not match test file keys")


def promote(
    task: int,
    public: float,
    candidate: Path | None = None,
    note: str = "",
) -> PromoteResult:
    """Copy candidate over locked CSV only if public score is strictly better."""
    if task not in TASK_KEYS:
        raise ValueError(f"task must be 1..5, got {task}")

    ledger = _load_ledger()
    entry = ledger[str(task)]
    direction = entry["direction"]
    locked_public = float(entry["locked_public"])
    locked_file = ROOT / entry["locked_file"]
    if candidate is None:
        candidate = locked_file.with_name(
            locked_file.name.replace(".csv", "_candidate.csv")
        )
    candidate = Path(candidate)
    if not candidate.is_absolute():
        candidate = ROOT / candidate

    if not candidate.exists():
        raise FileNotFoundError(f"Candidate not found: {candidate}")

    _validate_schema(task, candidate)

    better = _is_better(direction, public, locked_public)
    archive: Optional[Path] = None
    if better:
        archive_dir = ROOT / "artifacts" / f"task{task}" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f"{locked_file.stem}_{locked_public:.6f}.csv"
        if locked_file.exists():
            shutil.copy2(locked_file, archive)
        shutil.copy2(candidate, locked_file)
        entry["locked_public"] = float(public)
        reason = (
            f"promoted: {public} beats locked {locked_public} ({direction})"
        )
    else:
        reason = (
            f"rejected: {public} does not beat locked {locked_public} ({direction})"
        )

    entry.setdefault("history", []).append(
        {
            "date": date.today().isoformat(),
            "public": float(public),
            "note": note or reason,
            "promoted": better,
            "candidate": str(candidate.relative_to(ROOT))
            if candidate.is_relative_to(ROOT)
            else str(candidate),
        }
    )
    _save_ledger(ledger)

    return PromoteResult(
        promoted=better,
        reason=reason,
        locked_public=locked_public,
        public=float(public),
        locked_file=locked_file,
        candidate=candidate,
        archive=archive,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("public", type=float, help="Gradescope public score")
    parser.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help="Path to candidate CSV (default: task_N_predictions_candidate.csv)",
    )
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()
    from src.registry import record_public_score

    payload = record_public_score(args.task, args.public, note=args.note)
    print(payload["reason"])
    print(f"version={payload.get('version')} locked_public={payload.get('locked_public')}")
    raise SystemExit(0 if payload["promoted"] else 1)


if __name__ == "__main__":
    main()
