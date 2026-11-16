"""One CUDA biased-MF run for Task 1 or Task 3. Writes task_N_predictions.csv.

Keeps every run whose validation MSE is close to the best and averages those
test predictions, so later seeds can only replace the file with a better blend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task1.metrics import clip_ratings  # noqa: E402
from src.task1.submit import write_task1_predictions  # noqa: E402
from src.task3.data import train_val_split_random_user  # noqa: E402
from src.task3.submit import write_task3_predictions  # noqa: E402


def _ids(values: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    return np.fromiter((vocab.get(str(v), 0) for v in values), dtype=np.int64, count=len(values))


def _vocab(series: pd.Series) -> dict[str, int]:
    uniq = pd.unique(series.astype(str))
    return {u: i + 1 for i, u in enumerate(uniq)}


class BiasedMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, k: int, n_extra: int) -> None:
        super().__init__()
        self.user = nn.Embedding(n_users, k)
        self.item = nn.Embedding(n_items, k)
        self.user_b = nn.Embedding(n_users, 1)
        self.item_b = nn.Embedding(n_items, 1)
        self.global_b = nn.Parameter(torch.tensor(3.7))
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)
        nn.init.zeros_(self.user_b.weight)
        nn.init.zeros_(self.item_b.weight)
        self.mlp = None
        if n_extra:
            self.mlp = nn.Sequential(
                nn.Linear(k + n_extra, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, users, items, extra):
        user_f = self.user(users)
        item_f = self.item(items)
        dot = (user_f * item_f).sum(-1)
        pred = self.global_b + self.user_b(users).squeeze(-1) + self.item_b(items).squeeze(-1) + dot
        if extra is not None and self.mlp is not None:
            pred = pred + self.mlp(torch.cat([user_f * item_f, extra], dim=-1)).squeeze(-1)
        return pred.clamp(0.0, 5.0)


def _time_extra(df: pd.DataFrame, ts_mean: float, ts_std: float) -> np.ndarray:
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    z = ((ts - ts_mean) / ts_std).astype(np.float32)
    return z.reshape(-1, 1)


def _load_bank(path: Path) -> tuple[list[np.ndarray], list[float]]:
    if not path.exists():
        return [], []
    blob = np.load(path, allow_pickle=False)
    preds = [blob[k] for k in blob.files if k.startswith("p")]
    mses = blob["mses"].tolist()
    return preds, [float(m) for m in mses]


def _save_blend(path: Path, preds: list[np.ndarray], mses: list[float], test_df, task: int) -> None:
    best = min(mses)
    keep = [p for p, m in zip(preds, mses) if m <= best * 1.01]
    blended = clip_ratings(np.mean(np.vstack(keep), axis=0))
    out = ROOT / "assignment1_2026" / f"task_{task}_predictions.csv"
    if task == 3:
        write_task3_predictions(test_df, blended, out)
    else:
        write_task1_predictions(test_df, blended, out)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"p{i}": p for i, p in enumerate(preds)}
    payload["mses"] = np.asarray(mses, dtype=np.float64)
    np.savez(path, **payload)
    print(
        f"task {task}: wrote {out.name} from {len(keep)}/{len(preds)} runs "
        f"best_val_mse={best:.5f}",
        flush=True,
    )


def run(task: int, factors: int, epochs: int, seed: int, batch_size: int) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this process")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    data = ROOT / "assignment1_2026"
    if task == 1:
        train = pd.read_csv(data / "tasks_1_2_train.csv")
        test = pd.read_csv(data / "task_1_test.csv")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(train))
        n_val = int(round(len(train) * 0.1))
        val_df = train.iloc[order[:n_val]].reset_index(drop=True)
        fit_df = train.iloc[order[n_val:]].reset_index(drop=True)
    else:
        train = pd.read_csv(data / "tasks_3_4_train.csv")
        test = pd.read_csv(data / "task_3_test.csv")
        fit_df, val_df, _ = train_val_split_random_user(train, 0.1, seed)

    for frame in (fit_df, val_df, test):
        frame["user"] = frame["user"].astype(str)
        frame["item"] = frame["item"].astype(str)
    user_vocab = _vocab(fit_df["user"])
    item_vocab = _vocab(fit_df["item"])
    n_extra = 1 if task == 3 else 0
    ts_mean = float(fit_df["timestamp"].mean()) if task == 3 else 0.0
    ts_std = float(fit_df["timestamp"].std()) or 1.0 if task == 3 else 1.0

    def pack(frame: pd.DataFrame):
        u = torch.from_numpy(_ids(frame["user"], user_vocab)).to(device)
        i = torch.from_numpy(_ids(frame["item"], item_vocab)).to(device)
        extra = None
        if task == 3:
            extra = torch.from_numpy(_time_extra(frame, ts_mean, ts_std)).to(device)
        return u, i, extra

    u_fit, i_fit, x_fit = pack(fit_df)
    y_fit = torch.from_numpy(fit_df["rating"].to_numpy(dtype=np.float32)).to(device)
    u_val, i_val, x_val = pack(val_df)
    y_val = val_df["rating"].to_numpy(dtype=np.float64)
    u_test, i_test, x_test = pack(test)

    model = BiasedMF(len(user_vocab) + 1, len(item_vocab) + 1, factors, n_extra).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01 if factors <= 128 else 0.006, weight_decay=1e-6)
    n = len(fit_df)
    best_val = float("inf")
    best_state = None
    stall = 0
    print(
        f"CUDA MF task={task} factors={factors} epochs={epochs} seed={seed} "
        f"device={torch.cuda.get_device_name(0)} fit={n}",
        flush=True,
    )
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            extra = None if x_fit is None else x_fit[idx]
            pred = model(u_fit[idx], i_fit[idx], extra)
            loss = torch.mean((pred - y_fit[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)
        model.eval()
        with torch.no_grad():
            chunks = []
            for start in range(0, len(val_df), batch_size):
                sl = slice(start, start + batch_size)
                extra = None if x_val is None else x_val[sl]
                chunks.append(model(u_val[sl], i_val[sl], extra).detach().float().cpu().numpy())
            val_mse = float(np.mean((np.concatenate(chunks) - y_val) ** 2))
        print(
            f"  epoch {epoch + 1}/{epochs} train_mse={total / max(seen, 1):.4f} val_mse={val_mse:.4f}",
            flush=True,
        )
        if val_mse + 1e-5 < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= 8 and epoch >= 15:
                print(f"  early stop at epoch {epoch + 1}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        chunks = []
        for start in range(0, len(test), batch_size):
            sl = slice(start, start + batch_size)
            extra = None if x_test is None else x_test[sl]
            chunks.append(model(u_test[sl], i_test[sl], extra).detach().float().cpu().numpy())
        test_pred = clip_ratings(np.concatenate(chunks))
    bank = ROOT / "artifacts" / "gpu_night" / f"task{task}_bank.npz"
    preds, mses = _load_bank(bank)
    preds.append(test_pred.astype(np.float32))
    mses.append(best_val)
    _save_blend(bank, preds, mses, test, task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=[1, 3], required=True)
    parser.add_argument("--factors", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()
    run(args.task, args.factors, args.epochs, args.seed, args.batch_size)


if __name__ == "__main__":
    main()
