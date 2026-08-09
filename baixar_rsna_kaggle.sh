#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem"
PY="$ROOT/.venv/bin/python"
KAGGLE="$ROOT/.venv/bin/kaggle"
OUT_DIR="$ROOT/dados/rsna-breast-cancer-detection"

if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
	echo "KAGGLE_API_TOKEN nao definido. Rode primeiro:" >&2
	echo "  export KAGGLE_API_TOKEN='seu_token'" >&2
	exit 1
fi

if [ ! -x "$KAGGLE" ]; then
	"$PY" -m pip install kaggle
fi

mkdir -p "$OUT_DIR"
"$KAGGLE" competitions download -c rsna-breast-cancer-detection -p "$OUT_DIR"

echo "Download concluido em: $OUT_DIR"
