#!/usr/bin/env python3
"""Monitora um diretório de run e regenera o relatório quando metrics_by_epoch.csv aparecer.
Uso:
  python watch_and_regen_report.py --watch-dir /path/to/run --root /media/angualberto/HD500_TRABALHO/cbis_ddsm --out /path/to/report.pdf
"""
import argparse
import os
import time
import subprocess
import sys
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--watch-dir', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--interval', type=int, default=20)
    args = p.parse_args()

    target = os.path.join(args.watch_dir, 'metrics_by_epoch.csv')
    log_path = os.path.join(os.path.dirname(__file__), 'watch_and_regen_report.log')

    with open(log_path, 'a') as log:
        log.write(f"Watcher started: watch_dir={args.watch_dir}, root={args.root}, out={args.out}\n")
        while True:
            if os.path.exists(target):
                log.write(f"Found metrics file: {target}\n")
                cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'generate_training_report.py'), '--root', args.root, '--out', args.out]
                log.write('Running: ' + ' '.join(cmd) + '\n')
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    log.write('Returncode: ' + str(r.returncode) + '\n')
                    log.write('stdout:\n' + (r.stdout or '') + '\n')
                    log.write('stderr:\n' + (r.stderr or '') + '\n')
                except Exception as e:
                    log.write('Exception when running report: ' + str(e) + '\n')
                log.write('Watcher exiting after regeneration.\n')
                # Try to open the generated PDF if possible (non-blocking)
                try:
                    if os.path.exists(args.out):
                        if sys.platform == 'darwin':
                            subprocess.Popen(['open', args.out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        elif os.name == 'nt':
                            os.startfile(args.out)
                        else:
                            # Linux/others: use xdg-open when available
                            if shutil.which('xdg-open'):
                                subprocess.Popen(['xdg-open', args.out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    log.write('Failed to open PDF automatically: ' + str(e) + '\n')
                break
            log.write(f"Not found yet: {target}. Sleeping {args.interval}s\n")
            log.flush()
            time.sleep(args.interval)

if __name__ == '__main__':
    main()
