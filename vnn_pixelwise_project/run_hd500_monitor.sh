#!/usr/bin/env bash
set -euo pipefail
OUT="/media/angualberto/HD500_TRABALHO/cbis_ddsm/saida_analise_cores_alizams_fortran_full"
LOG="$OUT/run_hd500_chunks.log"
INTERVAL=${1:-300}
ONCE=false
if [ "$1" == "--once" ]; then
  ONCE=true
  INTERVAL=0
fi
status() {
  now=$(date +"%Y-%m-%d %H:%M:%S")
  chunks_dir="$OUT/chunks"
  total_requests=0
  if [ -d "$chunks_dir" ]; then
    total_requests=$(ls "$chunks_dir"/chunk_*.csv 2>/dev/null | wc -l || true)
  fi
  processed=0
  for d in "$OUT"/chunk_*; do
    if [ -f "$d/cores_alizams_por_imagem.csv" ]; then
      processed=$((processed+1))
    fi
  done
  merged="no"
  if [ -f "$OUT/cores_alizams_por_imagem_all.csv" ] || [ -f "$OUT/cores_alizams_por_imagem_all_dedup.csv" ]; then
    merged="yes"
  fi
  fuzzy_done="no"
  if [ -d "$OUT/fuzzy_results_all" ]; then
    fuzzy_done="yes"
  fi
  last_lines=""
  if [ -f "$LOG" ]; then
    last_lines=$(tail -n 8 "$LOG" | sed 's/^/    /')
  fi
  cat <<EOF
PROGRESS SNAPSHOT: $now
  chunks_csv_created: $total_requests
  chunks_processed_dirs_with_output: $processed
  merged_csv_present: $merged
  fuzzy_inference_done: $fuzzy_done
  run_log: $LOG
  recent_log:
$last_lines
EOF
}

if [ "$ONCE" = true ]; then
  status
  exit 0
fi
# Daemon loop
while true; do
  status >> "$OUT/monitor_status.log"
  sleep "$INTERVAL"
done
