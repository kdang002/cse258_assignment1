#!/usr/bin/env bash
set -euo pipefail
python3 -m venv "$HOME/cse258-gpu"
"$HOME/cse258-gpu/bin/pip" install -U pip
"$HOME/cse258-gpu/bin/pip" install implicit "rmm-cu13"
"$HOME/cse258-gpu/bin/python" - <<'PY'
import implicit.gpu as g
print("implicit_cuda", g.HAS_CUDA)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_missing", type(exc).__name__, exc)
PY
