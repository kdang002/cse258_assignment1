#!/usr/bin/env bash
set -eu
ROOT=/home/kietd/cse258-gpu/lib/python3.12/site-packages
export LD_LIBRARY_PATH="${ROOT}/nvidia/cu13/lib:${ROOT}/librmm/lib64:${ROOT}/rapids_logger/lib64"
export PYTHONPATH="/mnt/c/Users/kietd/OneDrive/UCSD - Projects/CSE258/assignment1_2026"
export MLFLOW_DISABLE_AGENT_HINT=1
exec /home/kietd/cse258-gpu/bin/python "$@"
