#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem"
PY="$ROOT/.venv/bin/python"

echo "Python: $PY"
"$PY" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sem cuda")
PY

echo
echo "Rodando teste fuzzy/Fourier..."
"$PY" "$ROOT/lidc_fourier_segmentacao.py"
