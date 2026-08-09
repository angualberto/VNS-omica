#!/usr/bin/env bash
set -euo pipefail
OUT="/media/angualberto/HD500_TRABALHO/cbis_ddsm/saida_analise_cores_alizams_fortran_full"
LOG="$OUT/run_hd500_chunks.log"
FLAG="$OUT/PROCESSING_DONE.flag"
while true; do
  if [ -f "$FLAG" ]; then
    exit 0
  fi
  merged="$OUT/cores_alizams_por_imagem_all_dedup.csv"
  fuzzy_dir="$OUT/fuzzy_results_all"
  if [ -f "$merged" ] && [ -d "$fuzzy_dir" ]; then
    # create a small summary
    metrics="$fuzzy_dir/fuzzy_metrics.json"
    summary_file="$OUT/final_summary.txt"
    if [ -f "$metrics" ]; then
      echo "Processing finished: $(date)" > "$summary_file"
      jq '.selected_thresholds, .metrics' "$metrics" >> "$summary_file" 2>/dev/null || cat "$metrics" >> "$summary_file"
    else
      echo "Processing finished (no metrics found) at $(date)" > "$summary_file"
    fi
    # desktop notification if available
    if command -v notify-send >/dev/null 2>&1; then
      notify-send "HD500 processing complete" "Resultados em: $OUT" -u normal
    fi
    echo "COMPLETED $(date)" >> "$OUT/monitor_status.log"
    touch "$FLAG"
    exit 0
  fi
  sleep 30
done
