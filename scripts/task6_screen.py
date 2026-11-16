"""20% user screen for Task 6 long-gap models vs locked L1 family.

Writes artifacts/task6_screen/results.json. Does not touch Gradescope CSVs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hardware import training_threads
from src.task6.pipeline import _bounds, _load_train, _mae

DAY = 86400.0
OUT = ROOT / "artifacts" / "task6_screen"
FEATS = [
    "log1p_median",
    "log1p_mean",
    "log1p_last",
    "log1p_geom",
    "log1p_count",
    "log1p_med3",
    "log1p_p75",
    "log1p_p90",
    "log1p_iqr",
    "frac_gt_2med",
    "last_over_median",
    "rating",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "log1p_span_days",
    "log1p_item_count",
    "log1p_item_med",
    "item_known",
]


def user_sample(df: pd.DataFrame, frac: float = 0.20, seed: int = 42) -> pd.DataFrame:
    sizes = df.groupby("user", sort=False).size()
    users = sizes.index.to_numpy()
    order = np.random.default_rng(seed).permutation(len(users))
    target = int(round(frac * len(df)))
    taken, rows = [], 0
    for i in order:
        taken.append(users[i])
        rows += int(sizes.iloc[i])
        if rows >= target and len(taken) > 1:
            break
    keep = set(taken)
    out = df[df["user"].isin(keep)].copy()
    print(f"sample users {len(keep)} rows {len(out)} ({len(out)/len(df):.1%})", flush=True)
    return out


def history_table(train: pd.DataFrame) -> pd.DataFrame:
    users = train["user"].to_numpy()
    items = train["item"].to_numpy()
    ts = train["timestamp"].to_numpy(np.int64)
    rating = train["rating"].to_numpy(np.float64)
    starts, ends = _bounds(users)
    parts = []
    for s, e in zip(starts, ends):
        gaps = np.diff(ts[s:e].astype(np.float64))
        n_gaps = len(gaps)
        if n_gaps < 2:
            continue
        n_rows = n_gaps - 1
        logs = np.log1p(gaps)
        med = np.empty(n_rows)
        geom = np.empty(n_rows)
        med3 = np.empty(n_rows)
        p75 = np.empty(n_rows)
        p90 = np.empty(n_rows)
        iqr = np.empty(n_rows)
        frac = np.empty(n_rows)
        for i, k in enumerate(range(1, n_gaps)):
            g = gaps[:k]
            med[i] = np.median(g)
            geom[i] = np.median(logs[:k])
            med3[i] = np.median(g[max(0, k - 3) : k])
            p75[i] = np.quantile(g, 0.75)
            p90[i] = np.quantile(g, 0.90)
            q25, q75 = np.quantile(g, [0.25, 0.75])
            iqr[i] = max(q75 - q25, 0.0)
            frac[i] = float(np.mean(g > 2.0 * med[i])) if med[i] > 0 else 0.0
        ks = np.arange(1, n_gaps, dtype=np.float64)
        csum = np.cumsum(gaps)
        last = gaps[:-1]
        mean = csum[:-1] / ks
        start_ts = ts[s + 1 : e - 1].astype(np.float64)
        hour = (start_ts % DAY) / 3600.0
        dow = np.floor(start_ts / DAY) % 7.0
        ang_h = 2.0 * np.pi * hour / 24.0
        ang_d = 2.0 * np.pi * dow / 7.0
        parts.append(
            pd.DataFrame(
                {
                    "user": users[s + 2 : e],
                    "item": items[s + 2 : e],
                    "y": gaps[1:],
                    "is_last": np.arange(n_rows) == (n_rows - 1),
                    "log1p_median": np.log1p(med),
                    "log1p_mean": np.log1p(mean),
                    "log1p_last": np.log1p(last),
                    "log1p_geom": geom,
                    "log1p_count": np.log1p(ks),
                    "log1p_med3": np.log1p(med3),
                    "log1p_p75": np.log1p(p75),
                    "log1p_p90": np.log1p(p90),
                    "log1p_iqr": np.log1p(iqr),
                    "frac_gt_2med": frac,
                    "last_over_median": last / (med + 1.0),
                    "rating": rating[s + 1 : e - 1],
                    "hour_sin": np.sin(ang_h),
                    "hour_cos": np.cos(ang_h),
                    "dow_sin": np.sin(ang_d),
                    "dow_cos": np.cos(ang_d),
                    "log1p_span_days": np.log1p(np.maximum((start_ts - float(ts[s])) / DAY, 0.0)),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def attach_items(frame: pd.DataFrame, fit_mask: np.ndarray) -> pd.DataFrame:
    fit = frame.loc[fit_mask, ["item", "y"]]
    count = fit.groupby("item")["y"].size()
    med = fit.groupby("item")["y"].median()
    global_log = float(np.median(np.log1p(fit["y"].to_numpy(np.float64))))
    out = frame.copy()
    c = out["item"].map(count).fillna(0.0).to_numpy(np.float64)
    m = out["item"].map(med)
    out["log1p_item_count"] = np.log1p(c)
    out["log1p_item_med"] = np.log1p(m.fillna(np.expm1(global_log)).to_numpy(np.float64))
    out["item_known"] = m.notna().astype(np.float64)
    return out


def fit_model(kind: str, x_tr, y_tr, x_es, y_es, seed: int, weight=None):
    common = dict(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=47,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=training_threads(),
        verbosity=-1,
    )
    if kind == "l1":
        model = lgb.LGBMRegressor(objective="regression_l1", **common)
        ytr, yes = y_tr, y_es
        inv = lambda p: np.clip(p, 0, None)
    elif kind == "log_l1":
        model = lgb.LGBMRegressor(objective="regression_l1", **common)
        ytr, yes = np.log1p(y_tr), np.log1p(y_es)
        inv = lambda p: np.clip(np.expm1(p), 0, None)
    elif kind.startswith("q"):
        alpha = float(kind.split("_")[1])
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **common)
        ytr, yes = y_tr, y_es
        inv = lambda p: np.clip(p, 0, None)
    else:
        raise ValueError(kind)
    model.fit(
        x_tr,
        ytr,
        sample_weight=weight,
        eval_set=[(x_es, yes)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model, inv


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = _load_train(ROOT / "assignment1_2026" / "task_6_train.csv")
    sample = user_sample(train, 0.20, 42)
    hist = history_table(sample)
    y = hist["y"].to_numpy(np.float64)
    is_last = hist["is_last"].to_numpy(bool)
    frame = attach_items(hist, ~is_last)
    x = frame[FEATS].to_numpy(np.float64)
    y_val = y[is_last]
    x_val = x[is_last]
    fit_idx = np.flatnonzero(~is_last)
    rng = np.random.default_rng(42)
    rng.shuffle(fit_idx)
    n_es = max(1, int(round(0.1 * len(fit_idx))))
    es_idx, tr_idx = fit_idx[:n_es], fit_idx[n_es:]
    tail = y_val >= np.quantile(y_val, 0.8)
    scores = {}

    plans = [
        ("lgbm_l1", "l1", None),
        ("lgbm_log_l1", "log_l1", None),
        ("quantile_0.60", "q_0.60", None),
        ("quantile_0.65", "q_0.65", None),
        ("quantile_0.70", "q_0.70", None),
        ("l1_w_sqrt", "l1", np.sqrt(y[tr_idx] + 1.0)),
        ("l1_w_cbrt", "l1", np.cbrt(y[tr_idx] + 1.0)),
    ]
    preds = {}
    for name, kind, w in plans:
        print(f"fitting {name} ...", flush=True)
        model, inv = fit_model(kind, x[tr_idx], y[tr_idx], x[es_idx], y[es_idx], 42, w)
        pred = inv(model.predict(x_val))
        preds[name] = pred
        scores[name] = {
            "mae": _mae(y_val, pred),
            "tail_mae": _mae(y_val[tail], pred[tail]),
            "role": "incumbent" if name == "lgbm_l1" else "challenger",
        }
        print(f"  {name:16} MAE {scores[name]['mae']:.0f}  tail {scores[name]['tail_mae']:.0f}", flush=True)

    # Blends of L1 and a higher quantile.
    for qname, w in (("quantile_0.65", 0.35), ("quantile_0.65", 0.50), ("quantile_0.70", 0.35)):
        name = f"blend_l1_{qname}_{w:.2f}"
        pred = (1 - w) * preds["lgbm_l1"] + w * preds[qname]
        scores[name] = {
            "mae": _mae(y_val, pred),
            "tail_mae": _mae(y_val[tail], pred[tail]),
            "role": "blend",
        }
        print(f"  {name:16} MAE {scores[name]['mae']:.0f}  tail {scores[name]['tail_mae']:.0f}", flush=True)

    # Soft floor: max(L1, geom_median * scale)
    geom = np.expm1(x_val[:, FEATS.index("log1p_geom")])
    for scale in (1.0, 1.25, 1.5):
        name = f"l1_floor_geom_{scale:.2f}"
        pred = np.maximum(preds["lgbm_l1"], scale * geom)
        scores[name] = {
            "mae": _mae(y_val, pred),
            "tail_mae": _mae(y_val[tail], pred[tail]),
            "role": "floor",
        }
        print(f"  {name:16} MAE {scores[name]['mae']:.0f}  tail {scores[name]['tail_mae']:.0f}", flush=True)

    incumbent = scores["lgbm_l1"]["mae"]
    winner = min(scores, key=lambda n: scores[n]["mae"])
    gain = (incumbent - scores[winner]["mae"]) / incumbent
    out = {
        "rows": int(len(sample)),
        "val_rows": int(is_last.sum()),
        "incumbent": "lgbm_l1",
        "incumbent_mae": incumbent,
        "winner": winner,
        "winner_mae": scores[winner]["mae"],
        "relative_gain": gain,
        "beats_incumbent": bool(gain >= 0.015),
        "scores": scores,
        "features": FEATS,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwinner {winner} gain {gain:.2%} scale? {out['beats_incumbent']}", flush=True)


if __name__ == "__main__":
    main()
