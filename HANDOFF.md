# CSE258 Assignment 1 — Live handoff

Durable status for orchestrator chats and subagents. Prefer this file + [`masterplan.md`](masterplan.md) over chat history.

## Goal

Beat Gradescope baselines on Goodreads-style recsys tasks. Submit ≥3 of 6 tasks; only the **best 3** count. Full brief: [`masterplan.md`](masterplan.md).

## Leaderboard (2026-11-15)

Gradescope filename is always `task_X_predictions.csv`.

| Task | Public | vs last | Locked file |
|---|---|---|---|
| 1 — rating (↓) | **0.596313** | beat 0.614111 | drop-0 / clip 1–5 |
| 2 — pairwise (↑) | **0.9169** | held | transfer BPR-only |
| 3 — rating+time (↓) | **0.757718** | beat 0.782905 | drop-0 / clip 1–5 |
| 4 — pairwise+time (↑) | **0.8657** | held | GPU BPR 128/256/192 on protocol stack |
| 5 — sequential (↑) | **0.868619** | held | interaction-neg Production |
| 6 — time delta (↓) | **12880501.21** | held | L1 + q0.70 blend, long-tail features |

Counting set is still **Task 2, Task 5, Task 4**. Drop-0 cut ~2.9% MSE on T1 and ~3.2% on T3. Task 6 remains ~5.0% under the published baseline (13,563,799 → 12,880,501).

## MLOps (do not skip)

Train and recover through Prefect + MLflow. Do **not** train with `enable_mlflow=False`. A worse public score is recovered by restoring the Production alias, not by hunting renamed CSVs.

```bash
# One-time: register current winners/losers from configs + existing CSVs
python -m pipelines.registry_flow seed

# Train (logs params/metrics, registers Staging)
python -m pipelines.task4_flow
python -m pipelines.task1_flow
# … task2_flow, task3_flow, task5_flow

# After a Gradescope upload
python -m pipelines.registry_flow record --task 4 --public 0.862
# or: python scripts/promote.py 4 0.862

# If the upload lost: copy Production submission back to task_N_predictions.csv
python -m pipelines.registry_flow restore --task 2

# Retrain from registered Production params (Prefect retries twice)
python -m pipelines.registry_flow retrain --task 4
```

Tracking URI: `sqlite:///mlflow.db` (see each `configs/taskN.yaml`). UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

New pipeline runs call [`src/registry.py`](src/registry.py) `log_and_register_run` and land as **Staging**. `record` promotes to **Production** only when public beats [`artifacts/leaderboard.json`](artifacts/leaderboard.json). Writing `assignment1_2026/task_N_predictions.csv` archives the previous live file in that folder.

## Code map

| Area | Paths |
|---|---|
| Gradescope entry | [`assignment1.py`](assignment1.py) — `run_task1()` … `run_task5()` (MLflow **on**) |
| Registry | [`src/registry.py`](src/registry.py), [`pipelines/registry_flow.py`](pipelines/registry_flow.py) |
| Prefect train | [`pipelines/task1_flow.py`](pipelines/task1_flow.py) … `task5_flow.py` (retries=2) |
| Ledger | [`artifacts/leaderboard.json`](artifacts/leaderboard.json) |
| Task 1 | [`src/task1/`](src/task1/), Production [`configs/task1.yaml`](configs/task1.yaml) |
| Task 2 | [`src/task2/`](src/task2/), Production [`configs/task2_transfer.yaml`](configs/task2_transfer.yaml) |
| Task 3 | [`src/task3/`](src/task3/), Production [`configs/task3.yaml`](configs/task3.yaml) |
| Task 4 | [`src/task4/`](src/task4/), Production [`configs/task4_night.yaml`](configs/task4_night.yaml) |
| Task 5 | [`src/task5/`](src/task5/), [`configs/task5.yaml`](configs/task5.yaml) |
| Task 6 | [`src/task6/`](src/task6/), [`configs/task6.yaml`](configs/task6.yaml) |
| Data | [`assignment1_2026/`](assignment1_2026/) |
| Writeup | [`writeup.txt`](writeup.txt) |

### Stacks

- **Task 1:** OOF LightGBM + 3-seed bag of diverse SVD members; **drop rating-0; clip [1,5]**. Public **0.596313**. Prior 0.614111 archived as `task_1_predictions_public_0.614111.csv`.
- **Task 2:** BPR-only multi-seed transfer. Public **0.9169**. GPU capacity run is Staging only.
- **Task 3:** random-user split, OOF residual, bidirectional LOO (k=5/20, ±1d/±30d); **drop rating-0; clip [1,5]**. Public **0.757718**. Prior 0.782905 archived as `task_3_predictions_public_0.782905.csv`.
- **Task 4:** random-user + interaction negatives + GPU BPR 128/256/192. Public **0.8657**. Config: [`configs/task4_night.yaml`](configs/task4_night.yaml).
- **Task 5:** interaction-frequency negatives. Public **0.868619**. GPU / item2vec are Staging only.
- **Task 6:** L1 + quantile-0.70 blend (weight 0.35) with long-tail gap features. Public **12,880,501** versus baseline **13,563,799**. Config: [`configs/task6.yaml`](configs/task6.yaml). Previous 13,162,685 archived as `task_6_predictions_public_13162684.75.csv`.

**Task 3/4 CSV:** timestamp must be an integer string. Task 4 header is `user,timestamp,item1,item2,which`.

## Orchestrator protocol

1. Source of truth = this handoff + registered MLflow models + leaderboard.json.
2. Train via Prefect so retries and MLflow registration happen.
3. After public scores, `record` (promote or restore Production).
4. One GPU BPR job at a time (one 3080).
5. Do not rerun `scripts/gpu_mf_night.py` for Task 1/3.
6. Never overwrite locked winners until a new public beats `leaderboard.json`.

## Open / next

- [x] Task 1 public **0.596313** locked (drop-0).
- [x] Task 3 public **0.757718** locked (drop-0).
- [x] Task 6 public **12880501.21** locked (beat 13162684.75).
- [x] Task 4 public **0.8657** locked.
- [x] Task 2 **0.9169** and Task 5 **0.868619** held.
- [x] Fill [`writeup.txt`](writeup.txt) (markdown, all six tasks).
- [x] Overnight 2026-11-16 finished (T4/T2/T5 staging). T4/T5 local vals flat/worse; T2 interaction-neg public **0.916** lost vs **0.9169** — live CSV restored.
- [x] Prediction CSV cleanup: live `task_1..6_predictions.csv` + T1/T3 prior-winner archives only.
- Task 4 is still the largest gap vs rank 3 (0.9538).

## Do not

- Train with `enable_mlflow=False`
- Skip Prefect when a run might fail (GPU / long SVD++)
- Overwrite Production params because local val looked better (T1 overnight, T2/T5 GPU)
- Skip Task 3/4 `timestamp` in submission headers
- Overwrite `task_1_predictions.csv` unless a new public MSE beats 0.596313
- Overwrite `task_3_predictions.csv` unless a new public MSE beats 0.757718
- Overwrite `task_6_predictions.csv` unless a new public MAE beats 12880501.21
