"""15% user screen: keep rating-0 vs drop rating-0 for Tasks 1 and 3.

Also score on the 1-5 subset only, which matches the claim that the test
set has no zeros.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "drop0_screen"


def mse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def run(path: Path, label: str, frac_users: float = 0.15, seed: int = 42) -> dict:
    df = pd.read_csv(path, dtype={"user": str, "item": str, "rating": float})
    users = df["user"].unique()
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(frac_users * len(users))))
    keep = set(rng.choice(users, size=n_keep, replace=False))
    df = df[df["user"].isin(keep)].reset_index(drop=True)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = int(round(0.1 * len(df)))
    val = df.iloc[idx[:n_val]].reset_index(drop=True)
    tr = df.iloc[idx[n_val:]].reset_index(drop=True)
    print(
        f"=== {label} sample rows={len(df)} train={len(tr)} val={len(val)} "
        f"frac0_train={(tr.rating == 0).mean():.4f} frac0_val={(val.rating == 0).mean():.4f}",
        flush=True,
    )

    def fit_predict(train: pd.DataFrame, scale: tuple[float, float], drop0: bool, clip_lo: float, clip_hi: float) -> np.ndarray:
        t = train[train["rating"] > 0].copy() if drop0 else train
        lo, hi = scale
        data = Dataset.load_from_df(t[["user", "item", "rating"]], Reader(rating_scale=(lo, hi)))
        algo = SVD(n_factors=32, n_epochs=20, lr_all=0.005, reg_all=0.1, random_state=seed, verbose=False)
        algo.fit(data.build_full_trainset())
        pred = np.array([algo.predict(u, i, clip=False).est for u, i in zip(val["user"], val["item"])])
        return np.clip(pred, clip_lo, clip_hi)

    variants = [
        ("keep0_scale05_clip05", dict(scale=(0.0, 5.0), drop0=False, clip_lo=0.0, clip_hi=5.0)),
        ("drop0_scale15_clip15", dict(scale=(1.0, 5.0), drop0=True, clip_lo=1.0, clip_hi=5.0)),
        ("drop0_scale15_clip05", dict(scale=(1.0, 5.0), drop0=True, clip_lo=0.0, clip_hi=5.0)),
    ]
    y = val["rating"].to_numpy()
    z = y == 0
    nz = y > 0
    scores = {}
    for name, kw in variants:
        p = fit_predict(tr, **kw)
        row = {
            "mse_all": mse(y, p),
            "mse_on0": mse(y[z], p[z]) if z.any() else None,
            "mse_on1to5": mse(y[nz], p[nz]),
            "mean_pred_on0": float(p[z].mean()) if z.any() else None,
            "n_val0": int(z.sum()),
            "n_val1to5": int(nz.sum()),
        }
        scores[name] = row
        print(
            f"  {name:22} all={row['mse_all']:.5f}  on0={row['mse_on0']:.5f}  "
            f"on1to5={row['mse_on1to5']:.5f}  mean_pred_on0={row['mean_pred_on0']:.3f}",
            flush=True,
        )
    # Metric that matches the friend's claim: only score 1-5 holdout rows.
    best = min(scores, key=lambda n: scores[n]["mse_on1to5"])
    keep = scores["keep0_scale05_clip05"]["mse_on1to5"]
    drop = scores["drop0_scale15_clip15"]["mse_on1to5"]
    return {
        "label": label,
        "rows": int(len(df)),
        "val_rows": int(len(val)),
        "frac0_train": float((tr.rating == 0).mean()),
        "best_on1to5": best,
        "keep0_mse_on1to5": keep,
        "drop0_mse_on1to5": drop,
        "relative_gain_on1to5": (keep - drop) / keep,
        "scores": scores,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {
        "1": run(ROOT / "assignment1_2026" / "tasks_1_2_train.csv", "task1"),
        "3": run(ROOT / "assignment1_2026" / "tasks_3_4_train.csv", "task3"),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nWrote", OUT / "results.json", flush=True)
    for task, row in results.items():
        print(
            f"task {task}: drop0 vs keep0 on1to5 gain {row['relative_gain_on1to5']:.2%} "
            f"({row['keep0_mse_on1to5']:.5f} -> {row['drop0_mse_on1to5']:.5f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
