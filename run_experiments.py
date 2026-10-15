"""Portable launcher for the recorded ATOF configurations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['HUS', 'PU'], required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true', help='Build main satellite features (requires CUDA).')
    mode.add_argument('--experiment', help='Experiment tag in configs/experiments.json.')
    mode.add_argument('--suite', choices=['ablation', 'views'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    commands = []
    ds = args.dataset
    if args.prepare:
        env.setdefault('DEV', 'cuda')
        commands = [
            ['build_feats.py', ds],
            ['feat_v2.py', ds, 'vit_large_patch16_dinov3.sat493m', '4', '5', '8,16,24', '64'],
            ['build_guided.py', ds, 'tok:vit_large_sat:u4:c0-2:l24', 'gms'],
        ]
    else:
        env.setdefault('DEV', 'cpu')
        config = json.loads((ROOT/'configs/experiments.json').read_text())
        suites = {'ablation': ['abl_spec', 'abl_ms', 'abl_ms_gms', 'abl_tok',
                  'abl_tok_ms', 'abl_tok_gms', 'fus_g0', 'main', 'abl_spec_full'],
                  'views': ['abl_c0', 'abl_c02', 'main']}
        tags = suites[args.suite] if args.suite else [args.experiment]
        for tag in tags:
            if tag not in config or ds not in config[tag]:
                parser.error(f'No recorded configuration for {ds}/{tag}.')
            cfg = config[tag][ds]
            cmd = ['run_ours.py', ds, tag, cfg['spec']]
            for name in ['seeds', 'split', 'gammas', 'rule', 'C', 'head', 'post', 'balanced']:
                cmd += ['--'+name, str(cfg[name])]
            commands.append(cmd)
    for command in commands:
        argv = [sys.executable, str(ROOT/'code'/command[0]), *command[1:]]
        print(subprocess.list2cmdline(argv), flush=True)
        if not args.dry_run:
            subprocess.run(argv, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
