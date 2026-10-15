"""Summarize evaluation records without loading images or pretrained weights."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent/'reported_results')
    args = parser.parse_args()
    for ds in ['HUS', 'PU']:
        for tag in ['ldman_main', 'ours/main']:
            files = [args.root/tag/f'{ds}_s{s}.json' for s in range(10, 20)]
            runs = [json.loads(f.read_text()) for f in files if f.exists()]
            if not runs:
                print(ds, tag, 'no records'); continue
            metrics = []
            for k in ['OA', 'AA', 'K']:
                values = [r[k] for r in runs]
                sd = statistics.stdev(values) if len(values)>1 else 0.0
                metrics.append(f'{k}={statistics.mean(values):.2f} +/- {sd:.2f}')
            print(ds, tag, f'n={len(runs)}', ', '.join(metrics))


if __name__ == '__main__':
    main()
