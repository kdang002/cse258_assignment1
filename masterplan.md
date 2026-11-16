# CSE258 Assignment 1 (2026) — Masterplan

Live status / Gradescope lock: see `HANDOFF.md`.

Source: `Assignment 1 2026.pdf`. Prefer this file over re-reading the PDF.

## Goal

Build recommender-system predictors for Goodreads book-review / interaction data. Beat provided baselines on Gradescope (public + private test splits). Complete at least 3 of 6 tasks; if more are submitted, only the best 3 count.

## Deliverables

| File | Purpose |
|---|---|
| `assignment1.py` | Working solution code; clearly document which sections map to each task. Autograder does **not** execute this for scoring, but staff may run it to verify outputs / check plagiarism / assign partial credit. |
| `writeup.txt` | Brief plain-text description of the approach for each submitted task (a few sentences each). Prepare before the deadline. |
| `task_1_predictions.csv` … `task_6_predictions.csv` | Prediction files for each attempted task. Match the format produced by the baseline notebook. |

**Required coverage:** submit something for **at least 3 of 6** tasks.

**Deadline:** beginning of week 8 (exact time on Gradescope). Upload to the leaderboard regularly.

**Individual work only.** Do not scrape Goodreads or reverse-engineer anonymization hashes.

## Data layout (`assignment1_2026/`)

| File | Role |
|---|---|
| `tasks_1_2_train.csv` | Train for tasks 1–2. Columns: `user,item,timestamp,rating` |
| `tasks_3_4_train.csv` | Train for tasks 3–4 (same schema; different sample) |
| `task_5_train.csv` | Train for task 5 |
| `task_6_train.csv` | Train for task 6 |
| `task_X_test.csv` | Test inputs for task X (missing the target field to predict) |
| `baselines.ipynb` | Reference baselines + valid output-file format |

Test inputs may include: `user`, `item`, `item1`/`item2`, `timestamp`.  
Predicted fields always go in the **final column**:

| Field | Meaning |
|---|---|
| `rating` | Predicted rating |
| `which` | Which of `item1`/`item2` is the positive interaction (write one of the two item IDs) |
| `delta` | Predicted time delta until next interaction |

## Tasks (complete ≥ 3)

### Task 1 — Rating prediction
- Predict `rating` for a `(user, item)` pair.
- Train: `tasks_1_2_train.csv`; Test: `task_1_test.csv` (`user,item` → append `rating`).
- Baseline: Surprise SVD latent-factor model (timestamps ignored).

### Task 2 — Interaction (pairwise) prediction
- Given `item1` and `item2`, predict which is the positive interaction (`which`).
- Train: `tasks_1_2_train.csv`; Test: `task_2_test.csv`.
- Baseline: Bayesian Personalized Ranking (BPR); timestamps ignored.

### Task 3 — Rating prediction with timestamp
- Same as Task 1, but **timestamp is available** and may be used.
- Train: `tasks_3_4_train.csv`; Test: `task_3_test.csv`.
- Baseline: same LFM as Task 1 (**baseline ignores timestamp** — opportunity to improve).

### Task 4 — Interaction prediction with timestamp
- Same as Task 2, but **timestamp is available** and may be used.
- Train: `tasks_3_4_train.csv`; Test: `task_4_test.csv`.
- Baseline: BPR (**baseline ignores timestamp**).

### Task 5 — Sequential next-item interaction prediction
- Pairwise positive-item prediction concerned with the user’s **most recent** interaction (what comes next).
- Items are ordered sequentially; **timestamp is not in the test set** — use ordering only.
- Train: `task_5_train.csv`; Test: `task_5_test.csv`.
- Baseline: BPR (**baseline ignores ordering/timestamp**).

### Task 6 — Time-delta prediction
- Predict how long until the user’s next interaction (`delta`).
- Train: `task_6_train.csv`; Test: `task_6_test.csv` (`user,item` → append `delta`).
- Baseline: always predict the user’s **median** historical delta.

## Grading (33 marks total)

Best **3** task scores + report:

| Component | Per task | Notes |
|---|---|---|
| Beat baseline on **unseen** (private) test | 4 | Full marks need a solution **substantially** better than baseline |
| Beat baseline on **seen** (public) test | 2 | Consolation if you overfit the public half |
| Class ranking / leaderboard | 4 | Heavier ranking emphasis this year |
| Written report (`writeup.txt`) | 3 total | Few sentences per task; full marks unless issues suspected |

Leaderboard shows performance on **half** the test data; final score uses the **whole** dataset.

Make-up assignment exists (last year’s A1) for catastrophic stuck cases — usually only enough to pass.

## Published baseline scores (from `baselines.ipynb`)

```
public  = {task1: 0.7429, task2: 0.8716, task3: 0.9820, task4: 0.8036, task5: 0.8189, task6: 13563799}
private = {task1: 0.7738, task2: 0.8736, task3: 0.9582, task4: 0.8007, task5: 0.8195, task6: 13481806}
```

Interpretation notes:
- Tasks 1 & 3: lower error is better (rating MSE-style metrics on the board).
- Tasks 2, 4, 5: higher accuracy-style scores are better (pairwise correctness).
- Task 6: lower absolute error on deltas is better (large numeric scale).

Confirm metric direction on Gradescope when submitting; use these numbers as “must beat” targets.

## Success criteria (definition of done)

1. [ ] Choose ≥ 3 tasks (recommend prioritizing ones with clearest gains over baselines).
2. [ ] Implement models that **beat public baselines**, then verify private-side improvement on Gradescope.
3. [ ] Generate valid `task_X_predictions.csv` files in the same format as baselines.
4. [ ] Produce documented `assignment1.py` with clear task sections.
5. [ ] Write concise `writeup.txt` covering each submitted task.
6. [ ] Upload prediction files to Gradescope leaderboard; iterate until substantially above baseline on private scores where possible.
7. [ ] Do not violate competition rules (no scraping / de-anonymizing).

## Suggested execution plan

### Phase 0 — Setup
- Reproduce baseline outputs from `baselines.ipynb` / equivalent Python.
- Upload baseline predictions once to confirm file format + metric polarity.

### Phase 1 — Pick 3 strong tasks
Good starting bets (based on where baselines leave headroom):
- **Task 1**: stronger MF / biases / regularization / more factors than SVD(n_factors=1).
- **Task 2** or **4**: better ranking features / popularity / user–item affinity beyond tiny BPR.
- **Task 3** or **4**: use timestamps (baselines ignore them).
- **Task 5**: use sequence/order (baseline ignores it) — e.g. recency, Markov / sequential signals.
- **Task 6**: better than global/user median (e.g. recent-window stats, item effects, time-of-day).

### Phase 2 — Iterate per task
For each chosen task:
1. Hold out a local validation split from train.
2. Improve model / features until local val beats baseline by a clear margin.
3. Write predictions → Gradescope → compare public score.
4. Guard against overfitting the public half; prefer methods that also help private.

### Phase 3 — Package
- Clean `assignment1.py` (one runnable path that regenerates submitted CSVs).
- Finalize `writeup.txt`.
- Confirm all submitted prediction files match code outputs.

## MLOps (Task 1+)

| Concern | Choice |
|---|---|
| Orchestration | Prefect 3 flows under [`pipelines/`](pipelines/) |
| Experiment tracking | MLflow SQLite `sqlite:///mlflow.db` (+ `./mlartifacts`), experiment `task1_rating` |
| Config | [`configs/task1.yaml`](configs/task1.yaml) |
| Code | [`src/task1/`](src/task1/) + Gradescope entry [`assignment1.py`](assignment1.py) |
| Artifacts | `./artifacts/task1/` (models, metrics, predictions) |

**Run Task 1:**

```bash
pip install -r requirements.txt
# Gradescope-facing (no Prefect/MLflow):
python assignment1.py
# Local MLOps experiments:
python -m pipelines.task1_flow
# MF hyperparameter sweep (logs to MLflow experiment task1_mf_sweep):
python -m pipelines.task1_mf_sweep
```

Task 1 MF tuning notes (local 10% val MSE; Gradescope test is usually lower):
- Weak large models overfit; best region is compact SVD (~16–40 factors) with `reg_all≈0.08–0.12`.
- Current config: 3-model MF ensemble + LightGBM residual + learned blend with user mean.
- Sweep artifacts: `artifacts/task1/mf_sweep_results.json`

**Inspect runs:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Future tasks should add `pipelines/taskN_flow.py` following the same Prefect + MLflow pattern.

## MLOps (Task 2)

| Concern | Choice |
|---|---|
| Orchestration | Prefect flows [`pipelines/task2_flow.py`](pipelines/task2_flow.py), [`pipelines/task2_bpr_sweep.py`](pipelines/task2_bpr_sweep.py) |
| Experiment tracking | Same `sqlite:///mlflow.db`; experiments `task2_interaction`, `task2_bpr_sweep` |
| Config | [`configs/task2.yaml`](configs/task2.yaml) |
| Code | [`src/task2/`](src/task2/) + [`assignment1.py`](assignment1.py) `run_task2()` |
| Artifacts | `./artifacts/task2/` |

**Run Task 2:**

```bash
pip install -r requirements.txt
# Gradescope-facing (no Prefect/MLflow):
python -c "from assignment1 import run_task2; run_task2()"
# BPR hyperparameter sweep (auto-promotes top-3 into configs/task2.yaml):
python -m pipelines.task2_bpr_sweep
# Full stack after sweep (or alone with current yaml ensemble):
python -m pipelines.task2_flow
# Overnight one-shot (sweep then full pipeline):
python -m pipelines.task2_bpr_sweep --then-full
```

Task 2 notes:
- Metric: pairwise accuracy (higher better). Beat public/private ~0.8716 / 0.8736.
- Stack: tuned `implicit` BPR ensemble + ALS + popularity/rating features + LightGBM pairwise + val blend.
- Shared train file with Task 1 (`tasks_1_2_train.csv`); do not overwrite Task 1 artifacts.

## Constraints & reminders

- Reasonable existing approaches are enough; inventing novel methods is optional.
- Code must correspond to submitted predictions.
- Staff may zero grades if code does not appear to generate the submitted solution.
- Prefer this `masterplan.md` as the working brief; only open the PDF if something here is ambiguous or Gradescope differs.
