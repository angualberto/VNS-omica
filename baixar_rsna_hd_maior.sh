#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25"
PROJECT="$ROOT/iaimgem"
PY="$PROJECT/.venv/bin/python"
KAGGLEHUB_CACHE_DIR="$ROOT/kagglehub_cache"
RSNA_OUT_DIR="$ROOT/rsna_breast_cancer_detection"

mkdir -p "$KAGGLEHUB_CACHE_DIR" "$RSNA_OUT_DIR"

echo "Cache KaggleHub: $KAGGLEHUB_CACHE_DIR"
echo "Destino RSNA: $RSNA_OUT_DIR"
df -h "$ROOT"

KAGGLEHUB_CACHE="$KAGGLEHUB_CACHE_DIR" "$PY" - <<'PY'
import kagglehub

path = kagglehub.competition_download(
    "rsna-breast-cancer-detection",
    output_dir="/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/rsna_breast_cancer_detection",
)
print("Path to competition files:", path)
PY
