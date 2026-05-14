#!/usr/bin/env python3
"""Parse plaintext training stdout logs and aggregate best validation AUC/LogLoss.

Usage:
  python tools/aggregate_from_stdout.py run_outputs/*.log

The script looks for lines containing `Args:` (a Python-style dict),
`Total parameters:`, and validation lines like
`Epoch 1 Validation | AUC: 0.732, LogLoss: 0.336`.
It writes `results.csv` and appends a short summary to `特征工程.md`.
"""
import sys
import re
import csv
import ast
from pathlib import Path
from datetime import datetime

RE_VALID = re.compile(r"Epoch\s*(\d+)\s*Validation\s*\|\s*AUC:\s*([0-9.]+)\s*,\s*LogLoss:\s*([0-9.]+)")
RE_TOTAL = re.compile(r"Total parameters:\s*([0-9,]+)")


def parse_file(p: Path):
    text = p.read_text(encoding='utf-8', errors='ignore')
    # Args dict
    args = {}
    for line in text.splitlines():
        if 'Args:' in line:
            try:
                payload = line.split('Args:', 1)[1].strip()
                args = ast.literal_eval(payload)
            except Exception:
                args = {}
            break

    seed = args.get('seed') if isinstance(args, dict) else None
    sidecar = args.get('sidecar_config') if isinstance(args, dict) else None

    # total params
    total = None
    m = RE_TOTAL.search(text)
    if m:
        total = int(m.group(1).replace(',', ''))

    # find all validation lines and pick best AUC (max)
    best = None
    for m in RE_VALID.finditer(text):
        epoch = int(m.group(1))
        auc = float(m.group(2))
        logloss = float(m.group(3))
        if best is None or auc > best['auc']:
            best = dict(epoch=epoch, auc=auc, logloss=logloss)

    return {
        'file': str(p),
        'seed': seed,
        'sidecar': sidecar,
        'total_params': total,
        'best_epoch': best['epoch'] if best else None,
        'best_auc': best['auc'] if best else None,
        'best_logloss': best['logloss'] if best else None,
    }


def write_csv(rows, out='results.csv'):
    keys = ['file', 'seed', 'sidecar', 'total_params', 'best_epoch', 'best_auc', 'best_logloss']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def append_md(rows, md_path='特征工程.md'):
    p = Path(md_path)
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    lines = [f"\n**自动汇总 {now}**\n"]
    lines.append('| file | seed | sidecar | total_params | best_epoch | best_auc | best_logloss |')
    lines.append('|---|---:|---|---:|---:|---:|---:|')
    for r in rows:
        lines.append(f"| {Path(r['file']).name} | {r['seed']} | {r['sidecar']} | {r['total_params']} | {r['best_epoch']} | {r['best_auc']} | {r['best_logloss']} |")
    # if md doesn't exist, create it
    if not p.exists():
        p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    else:
        p.write_text(p.read_text(encoding='utf-8') + '\n'.join(lines) + '\n', encoding='utf-8')
    return md_path


def main(argv):
    if len(argv) < 2:
        print('Usage: python tools/aggregate_from_stdout.py <logfile> [<logfile> ...]')
        return 2
    files = []
    for g in argv[1:]:
        if ('*' in g or '?' in g):
            files.extend(sorted(Path('.').glob(g)))
        else:
            files.append(Path(g))
    if not files:
        print('No files matched')
        return 2
    rows = [parse_file(p) for p in files]
    csv_path = write_csv(rows)
    try:
        append_md(rows)
        print(f'Appended summary to 特征工程.md and wrote {csv_path}')
    except Exception as e:
        print(f'Wrote {csv_path} but failed to append MD: {e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
