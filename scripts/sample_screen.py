"""15–20% architecture screen for Tasks 3, 4, and 6.

Users are kept whole. The slice is a prefix of a shuffled user list until
about 20% of rows are included. Every architecture is scored on the same
holdout. The current model family is one of the entrants.

Nothing here writes a Gradescope CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task6.pipeline import (  # noqa: E402
    FEATURE_COLS,
    _attach_items,
    _history_table,
    _item_stats,
    _mae,
)

DAY = 86400.0
OUT = ROOT / "artifacts" / "sample_screen"
DATA = ROOT / "assignment1_2026"


def user_sample(df: pd.DataFrame, frac: float = 0.20, seed: int = 42) -> pd.DataFrame:
    """Keep entire users until the row count reaches `frac` of the file."""
    sizes = df.groupby("user", sort=False).size()
    users = sizes.index.to_numpy()
    order = np.random.default_rng(seed).permutation(len(users))
    target = int(round(frac * len(df)))
    taken = []
    rows = 0
    for i in order:
        taken.append(users[i])
        rows += int(sizes.iloc[i])
        if rows >= target and len(taken) > 1:
            break
    keep = set(taken)
    out = df[df["user"].isin(keep)].copy()
    print(f"sample users {len(keep)} rows {len(out)} ({len(out) / len(df):.1%} of {len(df)})", flush=True)
    return out


def _lgbm(objective: str, seed: int, alpha: float | None = None) -> lgb.LGBMRegressor:
    kwargs = dict(
        objective=objective,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=6,
        verbosity=-1,
    )
    if alpha is not None:
        kwargs["alpha"] = alpha
    return lgb.LGBMRegressor(**kwargs)


def _fit_predict(model, x_tr, y_tr, x_es, y_es, x_va):
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_es, y_es)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    return model.predict(x_va)


def screen_task6(seed: int = 42) -> dict:
    print("\n=== Task 6 ===", flush=True)
    train = pd.read_csv(
        DATA / "task_6_train.csv",
        dtype={"user": str, "item": str, "timestamp": np.int64, "rating": np.float64},
    )
    sample = user_sample(train, 0.20, seed)
    hist = _history_table(sample)
    y = hist["y"].to_numpy(np.float64)
    is_last = hist["is_last"].to_numpy(bool)
    count, med, global_log = _item_stats(hist, ~is_last)
    frame = _attach_items(hist, count, med, global_log)
    x = frame[FEATURE_COLS].to_numpy(np.float64)
    y_val = y[is_last]
    x_val = x[is_last]
    fit_idx = np.flatnonzero(~is_last)
    rng = np.random.default_rng(seed)
    rng.shuffle(fit_idx)
    n_es = max(1, int(round(0.1 * len(fit_idx))))
    es_idx, tr_idx = fit_idx[:n_es], fit_idx[n_es:]
    tail = y_val >= np.quantile(y_val, 0.8)

    scores = {}
    # Incumbent family: MAE regression.
    pred_l1 = _fit_predict(_lgbm("regression_l1", seed), x[tr_idx], y[tr_idx], x[es_idx], y[es_idx], x_val)
    pred_l1 = np.clip(pred_l1, 0, None)
    scores["lgbm_l1"] = {"mae": _mae(y_val, pred_l1), "tail_mae": _mae(y_val[tail], pred_l1[tail]), "role": "incumbent"}

    for alpha in (0.65, 0.80):
        pred = _fit_predict(
            _lgbm("quantile", seed, alpha),
            x[tr_idx],
            y[tr_idx],
            x[es_idx],
            y[es_idx],
            x_val,
        )
        pred = np.clip(pred, 0, None)
        name = f"quantile_{alpha:.2f}"
        scores[name] = {"mae": _mae(y_val, pred), "tail_mae": _mae(y_val[tail], pred[tail]), "role": "long_gap"}

    # Pointwise max of the MAE model and the user's own 75th percentile gap.
    p75 = np.expm1(x_val[:, FEATURE_COLS.index("log1p_med3")])
    # med3 is the median of the last 3, not p75. Use a real p75 from log features is unavailable.
    # Blend: longer of the L1 prediction and 1.5x the history median, clipped by the quantile model.
    hist_med = np.expm1(x_val[:, FEATURE_COLS.index("log1p_median")])
    pred_long = np.maximum(pred_l1, 1.5 * hist_med)
    scores["l1_or_1.5_median"] = {
        "mae": _mae(y_val, pred_long),
        "tail_mae": _mae(y_val[tail], pred_long[tail]),
        "role": "long_gap",
    }
    for name, row in scores.items():
        print(f"  {name:20} MAE {row['mae']:.0f}  tail {row['tail_mae']:.0f}", flush=True)
    incumbent = scores["lgbm_l1"]["mae"]
    winner = min(scores, key=lambda n: scores[n]["mae"])
    return {
        "rows": int(len(sample)),
        "val_rows": int(is_last.sum()),
        "incumbent": "lgbm_l1",
        "incumbent_mae": incumbent,
        "winner": winner,
        "beats_incumbent": bool(scores[winner]["mae"] < incumbent * 0.99),
        "scores": scores,
    }


def _holdout_rows(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out val_fraction of each user's rows, at least one when the user has 5+ events."""
    rng = np.random.default_rng(seed)
    parts_tr, parts_va = [], []
    for _, g in df.groupby("user", sort=False):
        n = len(g)
        if n < 5:
            parts_tr.append(g)
            continue
        n_val = max(1, int(round(n * val_fraction)))
        pick = rng.choice(n, size=n_val, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[pick] = True
        parts_va.append(g.iloc[mask])
        parts_tr.append(g.iloc[~mask])
    train = pd.concat(parts_tr, ignore_index=True)
    val = pd.concat(parts_va, ignore_index=True)
    return train, val


def _svd_predict(fit: pd.DataFrame, query: pd.DataFrame, seed: int) -> np.ndarray:
    from surprise import SVD, Dataset, Reader

    data = Dataset.load_from_df(fit[["user", "item", "rating"]], Reader(rating_scale=(0, 5)))
    algo = SVD(n_factors=16, n_epochs=12, lr_all=0.005, reg_all=0.1, random_state=seed, verbose=False)
    algo.fit(data.build_full_trainset())
    return np.array([algo.predict(u, i, clip=False).est for u, i in zip(query["user"], query["item"])])


def _temporal_predict(fit: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    """Koren-style biases scored at the query row's own timestamp."""
    users = pd.unique(fit["user"])
    items = pd.unique(fit["item"])
    uidx = {u: i for i, u in enumerate(users)}
    iidx = {i: j for j, i in enumerate(items)}
    u = fit["user"].map(uidx).to_numpy(np.int64)
    i = fit["item"].map(iidx).to_numpy(np.int64)
    r = fit["rating"].to_numpy(np.float64)
    t = fit["timestamp"].to_numpy(np.float64)
    n_bins = 16
    edges = np.quantile(t, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.searchsorted(edges, t, side="right")
    nu = np.bincount(u, minlength=len(uidx)).astype(np.float64)
    ni = np.bincount(i, minlength=len(iidx)).astype(np.float64)
    tmean = np.bincount(u, weights=t, minlength=len(uidx)) / np.maximum(nu, 1)
    alpha = float(r.mean())
    bu = np.zeros(len(uidx))
    bi = np.zeros(len(iidx))
    bib = np.zeros(len(iidx) * n_bins)
    au = np.zeros(len(uidx))
    ib = i * n_bins + bins
    nib = np.bincount(ib, minlength=len(bib)).astype(np.float64)
    dev = np.sign(t - tmean[u]) * np.abs((t - tmean[u]) / DAY) ** 0.4
    dv2 = np.bincount(u, weights=dev * dev, minlength=len(uidx))
    lam_u, lam_i, lam_bin, lam_drift = 2.0, 5.0, 20.0, 5.0
    for _ in range(12):
        rest = bib[ib] + au[u] * dev
        alpha = float(np.mean(r - bu[u] - bi[i] - rest))
        bu = np.bincount(u, weights=r - alpha - bi[i] - rest, minlength=len(uidx)) / (nu + lam_u)
        bi = np.bincount(i, weights=r - alpha - bu[u] - rest, minlength=len(iidx)) / (ni + lam_i)
        r0 = r - alpha - bu[u] - bi[i]
        bib = np.bincount(ib, weights=r0 - au[u] * dev, minlength=len(bib)) / (nib + lam_bin)
        au = np.bincount(u, weights=dev * (r0 - bib[ib]), minlength=len(uidx)) / (dv2 + lam_drift)

    qu = query["user"].map(uidx).fillna(-1).to_numpy(np.int64)
    qi = query["item"].map(iidx).fillna(-1).to_numpy(np.int64)
    qt = query["timestamp"].to_numpy(np.float64)
    qbin = np.searchsorted(edges, qt, side="right")
    pred = np.full(len(query), alpha)
    seen_u = qu >= 0
    seen_i = qi >= 0
    pred[seen_u] += bu[qu[seen_u]]
    pred[seen_i] += bi[qi[seen_i]]
    both = seen_u & seen_i
    pred[both] += bib[qi[both] * n_bins + qbin[both]]
    qdev = np.sign(qt - tmean[np.maximum(qu, 0)]) * np.abs((qt - tmean[np.maximum(qu, 0)]) / DAY) ** 0.4
    pred[seen_u] += au[qu[seen_u]] * qdev[seen_u]
    return np.clip(pred, 0, 5)


def _mse(y, p) -> float:
    return float(np.mean((y - np.clip(p, 0, 5)) ** 2))


def screen_task3(seed: int = 42) -> dict:
    print("\n=== Task 3 ===", flush=True)
    train = pd.read_csv(DATA / "tasks_3_4_train.csv")
    train["user"] = train["user"].astype(str)
    train["item"] = train["item"].astype(str)
    train["timestamp"] = train["timestamp"].astype(np.int64)
    train["rating"] = train["rating"].astype(np.float64)
    sample = user_sample(train, 0.20, seed)
    fit, val = _holdout_rows(sample, 0.10, seed)
    y = val["rating"].to_numpy(np.float64)
    print(f"  fit {len(fit)} val {len(val)}", flush=True)
    print("  SVD ...", flush=True)
    svd = _svd_predict(fit, val, seed)
    print("  temporal bias at the query timestamp ...", flush=True)
    temporal = _temporal_predict(fit, val)
    scores = {"svd": {"mse": _mse(y, svd), "role": "incumbent_core"}, "temporal_bias": {"mse": _mse(y, temporal), "role": "timestamp_bias"}}

    def tree(name, pred_fit_query, pred_val):
        # Fit the tree on a 80k subsample of fit rows. The SVD/temporal columns are the architecture.
        rng = np.random.default_rng(seed)
        take = rng.choice(len(fit), size=min(80000, len(fit)), replace=False)
        sub = fit.iloc[take]
        if name == "lgbm_svd":
            base = _svd_predict(fit, sub, seed)
        else:
            base = _temporal_predict(fit, sub)
        hour = (sub["timestamp"].to_numpy() % DAY) / 3600.0
        x_tr = np.column_stack([base, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)])
        hour_v = (val["timestamp"].to_numpy() % DAY) / 3600.0
        x_va = np.column_stack([pred_val, np.sin(2 * np.pi * hour_v / 24), np.cos(2 * np.pi * hour_v / 24)])
        n_es = max(1, int(0.1 * len(x_tr)))
        pred = _fit_predict(_lgbm("regression", seed), x_tr[n_es:], sub["rating"].to_numpy()[n_es:], x_tr[:n_es], sub["rating"].to_numpy()[:n_es], x_va)
        scores[name] = {"mse": _mse(y, pred), "role": "incumbent" if name == "lgbm_svd" else "timestamp_bias"}

    print("  LGBM on SVD ...", flush=True)
    tree("lgbm_svd", svd, svd)
    print("  LGBM on temporal bias ...", flush=True)
    tree("lgbm_temporal", temporal, temporal)
    for name, row in scores.items():
        print(f"  {name:20} MSE {row['mse']:.5f}", flush=True)
    incumbent = scores["lgbm_svd"]["mse"]
    winner = min(scores, key=lambda n: scores[n]["mse"])
    return {
        "rows": int(len(sample)),
        "val_rows": int(len(val)),
        "incumbent": "lgbm_svd",
        "incumbent_mse": incumbent,
        "winner": winner,
        "beats_incumbent": bool(scores[winner]["mse"] < incumbent - 0.005),
        "scores": scores,
    }


def _index_users(fit: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for user, g in fit.groupby("user", sort=False):
        g = g.sort_values("timestamp")
        out[user] = (
            g["timestamp"].to_numpy(np.int64),
            g["item"].to_numpy(),
            g["rating"].to_numpy(np.float64),
        )
    return out


def _local_feats(index, user, ts, item, radius):
    ts_a, it_a, r_a = index.get(user, (np.array([], np.int64), np.array([]), np.array([])))
    if len(ts_a) == 0:
        return 0.0, np.nan, 0.0
    lo = np.searchsorted(ts_a, ts - radius)
    hi = np.searchsorted(ts_a, ts + radius, side="right")
    sl_it = it_a[lo:hi]
    sl_r = r_a[lo:hi]
    sl_ts = ts_a[lo:hi]
    keep = ~((sl_it == item) & (np.abs(sl_ts.astype(np.int64) - int(ts)) <= 1))
    sl_it, sl_r = sl_it[keep], sl_r[keep]
    if len(sl_r) == 0:
        return 0.0, np.nan, 0.0
    return float(len(sl_r)), float(sl_r.mean()), float(np.any(sl_it == item))


def _pair_frame(positives: pd.DataFrame, fit: pd.DataFrame, index, rng: np.random.Generator) -> pd.DataFrame:
    count_s = fit.groupby("item").size()
    items = count_s.index.to_numpy()
    weights = count_s.to_numpy(np.float64)
    weights = weights / weights.sum()
    counts = count_s.to_dict()
    means = fit.groupby("item")["rating"].mean().to_dict()
    first = fit.groupby("item")["timestamp"].min().to_dict()
    user_items = fit.groupby("user")["item"].agg(set).to_dict()
    rows = []
    pos_items = positives["item"].to_numpy()
    pos_users = positives["user"].to_numpy()
    pos_ts = positives["timestamp"].to_numpy(np.int64)
    n = len(positives)
    neg_idx = rng.choice(len(items), size=n, p=weights)
    for k in range(n):
        user = pos_users[k]
        ts = int(pos_ts[k])
        pos = pos_items[k]
        seen = user_items.get(user, set())
        neg = items[neg_idx[k]]
        tries = 0
        while neg == pos or neg in seen:
            neg = items[rng.integers(0, len(items))]
            tries += 1
            if tries > 20:
                break
        if rng.random() < 0.5:
            item1, item2, label = pos, neg, 1
        else:
            item1, item2, label = neg, pos, 0
        c1, m1, h1 = _local_feats(index, user, ts, item1, DAY)
        c7, _, h7 = _local_feats(index, user, ts, item1, 7 * DAY)
        d1, n1, g1 = _local_feats(index, user, ts, item2, DAY)
        d7, _, g7 = _local_feats(index, user, ts, item2, 7 * DAY)

        def _nz(value: float) -> float:
            return 0.0 if np.isnan(value) else value

        rows.append(
            {
                "label": label,
                "pop_diff": np.log1p(counts.get(item1, 0)) - np.log1p(counts.get(item2, 0)),
                "mean_diff": float(means.get(item1, 4.0) - means.get(item2, 4.0)),
                "age_diff": (ts - float(first.get(item1, ts))) / DAY - (ts - float(first.get(item2, ts))) / DAY,
                "near1_count_diff": c1 - d1,
                "near1_mean_diff": _nz(m1) - _nz(n1),
                "near1_hit_diff": h1 - g1,
                "near7_count_diff": c7 - d7,
                "near7_hit_diff": h7 - g7,
            }
        )
    return pd.DataFrame(rows)


def screen_task4(seed: int = 42) -> dict:
    print("\n=== Task 4 ===", flush=True)
    train = pd.read_csv(DATA / "tasks_3_4_train.csv")
    train["user"] = train["user"].astype(str)
    train["item"] = train["item"].astype(str)
    train["timestamp"] = train["timestamp"].astype(np.int64)
    train["rating"] = train["rating"].astype(np.float64)
    sample = user_sample(train, 0.20, seed)
    fit, val = _holdout_rows(sample, 0.10, seed)
    rng = np.random.default_rng(seed)
    if len(val) > 20000:
        val = val.iloc[rng.choice(len(val), size=20000, replace=False)]
    if len(fit) > 40000:
        fit_pos = fit.iloc[rng.choice(len(fit), size=40000, replace=False)]
    else:
        fit_pos = fit
    print(f"  indexing {fit.user.nunique()} users, train pairs {len(fit_pos)} val pairs {len(val)}", flush=True)
    index = _index_users(fit)
    print("  building pair tables ...", flush=True)
    tr = _pair_frame(fit_pos, fit, index, rng)
    va = _pair_frame(val, fit, index, np.random.default_rng(seed + 1))
    y_tr = tr["label"].to_numpy(np.int64)
    y_va = va["label"].to_numpy(np.int64)
    base_cols = ["pop_diff", "mean_diff", "age_diff"]
    local_cols = ["near1_count_diff", "near1_mean_diff", "near1_hit_diff", "near7_count_diff", "near7_hit_diff"]

    def acc(cols, name, role):
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=6,
            verbosity=-1,
        )
        model.fit(tr[cols], y_tr)
        pred = model.predict(va[cols])
        score = float(np.mean(pred == y_va))
        print(f"  {name:20} acc {score:.4f}", flush=True)
        return {"accuracy": score, "role": role}

    scores = {
        "base_features": acc(base_cols, "base_features", "incumbent"),
        "base_plus_timelocal": acc(base_cols + local_cols, "base_plus_timelocal", "time_local"),
        "timelocal_only": acc(local_cols, "timelocal_only", "time_local"),
    }
    incumbent = scores["base_features"]["accuracy"]
    winner = max(scores, key=lambda n: scores[n]["accuracy"])
    return {
        "rows": int(len(sample)),
        "val_pairs": int(len(va)),
        "incumbent": "base_features",
        "incumbent_accuracy": incumbent,
        "winner": winner,
        "beats_incumbent": bool(scores[winner]["accuracy"] > incumbent + 0.005),
        "scores": scores,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {
        "6": screen_task6(),
        "3": screen_task3(),
        "4": screen_task4(),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nWrote", OUT / "results.json", flush=True)
    for task, row in results.items():
        print(f"task {task}: winner {row['winner']} beats_incumbent {row['beats_incumbent']}", flush=True)


if __name__ == "__main__":
    main()
