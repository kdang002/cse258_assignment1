#!/usr/bin/env bash
set -eu
SITE=/home/kietd/cse258-gpu/lib/python3.12/site-packages
export LD_LIBRARY_PATH="${SITE}/nvidia/cu13/lib:${SITE}/librmm/lib64:${SITE}/rapids_logger/lib64"
export PYTHONPATH="/mnt/c/Users/kietd/OneDrive/UCSD - Projects/CSE258/assignment1_2026"
export MLFLOW_DISABLE_AGENT_HINT=1
cd "$PYTHONPATH"
nvidia-smi -L
/home/kietd/cse258-gpu/bin/python -c "import implicit.gpu as g; print('implicit_cuda', bool(g.HAS_CUDA))"
exec /home/kietd/cse258-gpu/bin/python "$@"
