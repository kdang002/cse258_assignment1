#!/usr/bin/env bash
set -eu
SITE=/home/kietd/cse258-gpu/lib/python3.12/site-packages
export LD_LIBRARY_PATH="${SITE}/nvidia/cu13/lib:${SITE}/librmm/lib64:${SITE}/rapids_logger/lib64"
export PYTHONPATH="/mnt/c/Users/kietd/OneDrive/UCSD - Projects/CSE258/assignment1_2026"
export MLFLOW_DISABLE_AGENT_HINT=1
cd "$PYTHONPATH"
PY=/home/kietd/cse258-gpu/bin/python
nvidia-smi -L
"$PY" -c "import implicit.gpu as g; print('implicit_cuda', bool(g.HAS_CUDA), 'gpus', g.get_device_count() if g.HAS_CUDA else 0)"
echo "=== Task 5 GPU BPR/ALS ==="
"$PY" -c "from src.task5.pipeline import run_task5_pipeline; run_task5_pipeline('configs/task5_gpu.yaml', enable_mlflow=False)"
echo "=== Task 4 GPU BPR/ALS ==="
"$PY" -c "from src.task4.pipeline import run_task4_pipeline; run_task4_pipeline('configs/task4_gpu.yaml', enable_mlflow=False)"
echo "=== Task 2 GPU BPR/ALS ==="
"$PY" -c "from src.task2.pipeline import run_task2_pipeline; run_task2_pipeline('configs/task2_gpu.yaml', enable_mlflow=False)"
echo "=== WSL GPU pipelines done ==="
