#!/usr/bin/env bash
set -euo pipefail

PY="/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem/.venv/bin/python"
ROOT="/media/angualberto/HD500_TRABALHO/cbis_ddsm"
CSV="$ROOT/calc_case_description_train_set_pixelwise.csv"
OUT="$ROOT/saida_analise_cores_alizams_fortran_full"
CHUNKS_DIR="$OUT/chunks"
CHUNK_LINES=1000
WORKERS=8

mkdir -p "$OUT" "$CHUNKS_DIR"

# Split CSV (keep header separately)
HEADER_FILE="$CHUNKS_DIR/header.csv"
head -n1 "$CSV" > "$HEADER_FILE"

tail -n +2 "$CSV" | split -l $CHUNK_LINES - "$CHUNKS_DIR/chunk_"

i=0
for f in "$CHUNKS_DIR"/chunk_*; do
  i=$((i+1))
  chunk_base="$CHUNKS_DIR/chunk_${i}"
  chunk_csv="${chunk_base}.csv"
  cat "$HEADER_FILE" > "$chunk_csv"
  cat "$f" >> "$chunk_csv"
  chunk_out="$OUT/chunk_${i}"
  mkdir -p "$chunk_out"
  if [ -f "$chunk_out/cores_alizams_por_imagem.csv" ]; then
    echo "skip chunk $i already done" >> "$OUT/run.log"
    continue
  fi
  echo "processing chunk $i -> $chunk_csv" >> "$OUT/run.log"
  "$PY" $(pwd)/analisar_alizams_cores_dicom_csv.py --csv "$chunk_csv" --images-dir "$ROOT" --out-dir "$chunk_out" --backend fortran_openmp --workers $WORKERS
  echo "done chunk $i" >> "$OUT/run.log"
done

# Merge per-chunk CSVs
merge_out="$OUT/cores_alizams_por_imagem_all.csv"
first_chunk="$OUT/chunk_1/cores_alizams_por_imagem.csv"
if [ -f "$first_chunk" ]; then
  head -n1 "$first_chunk" > "$merge_out"
  for d in "$OUT"/chunk_*; do
    if [ -f "$d/cores_alizams_por_imagem.csv" ]; then
      tail -n +2 "$d/cores_alizams_por_imagem.csv" >> "$merge_out"
    fi
  done
  echo "merged to $merge_out" >> "$OUT/run.log"
else
  echo "no chunk outputs found, aborting merge" >> "$OUT/run.log"
  exit 1
fi

# Deduplicate
python - <<PY
import pandas as pd
p='$merge_out'
df=pd.read_csv(p,dtype=str)
df=df.drop_duplicates(['patient_id','image_id'])
out='$OUT/cores_alizams_por_imagem_all_dedup.csv'
df.to_csv(out,index=False)
print('dedup wrote',out,len(df))
PY

# Build fuzzy features and run inference
"$PY" $(pwd)/fuzzy_dataset_features.py --predictions-csv "$OUT/cores_alizams_por_imagem_all_dedup.csv" --colors-csv "$OUT/cores_alizams_por_imagem_all_dedup.csv" --output-csv "$OUT/fuzzy_features_predictions_all.csv"
"$PY" $(pwd)/run_fuzzy_inference.py --features_csv "$OUT/fuzzy_features_predictions_all.csv" --output_dir "$OUT/fuzzy_results_all"

echo "pipeline complete" >> "$OUT/run.log"
