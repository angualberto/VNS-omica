#!/usr/bin/env python3
import sys
import ast
import json
import csv
from pathlib import Path

if len(sys.argv) < 2:
    print('Usage: export_metrics_from_log.py /path/to/logfile')
    sys.exit(2)

log_path = Path(sys.argv[1])
if not log_path.exists():
    print('Log file not found:', log_path)
    sys.exit(1)

out_dir = log_path.parent
csv_path = out_dir / 'metrics_by_epoch.csv'
json_path = out_dir / 'metrics_by_epoch.json'

entries = []
with log_path.open('r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
        if s.startswith('{') and 'epoch' in s:
            try:
                # Use ast.literal_eval to parse single-quoted dicts
                d = ast.literal_eval(s)
                entries.append(d)
            except Exception as e:
                # try to fix common trailing commas
                try:
                    s2 = s.rstrip(',')
                    d = ast.literal_eval(s2)
                    entries.append(d)
                except Exception:
                    print('Failed to parse line:', s[:200])

if not entries:
    print('No epoch entries found in log')
    sys.exit(0)

# Normalize keys and write CSV
all_keys = set()
for e in entries:
    all_keys.update(e.keys())
all_keys = sorted(all_keys)

with csv_path.open('w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=all_keys)
    writer.writeheader()
    for e in entries:
        # convert non-serializable fields
        row = {k: e.get(k) for k in all_keys}
        # stringify confusion_matrix
        if 'confusion_matrix' in row and row['confusion_matrix'] is not None:
            row['confusion_matrix'] = json.dumps(row['confusion_matrix'])
        writer.writerow(row)

with json_path.open('w') as jf:
    json.dump(entries, jf, indent=2)

print(f'Wrote {len(entries)} entries to {csv_path} and {json_path}')
