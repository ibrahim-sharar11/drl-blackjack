import os
import sys
import argparse
import subprocess
from pathlib import Path


def run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apps', nargs='+', default=['blackjack','formflow'])
    ap.add_argument('--algos', nargs='+', default=['ppo','a2c'])
    ap.add_argument('--personas', nargs='+', default=['survivor','explorer'])
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--timesteps', type=int, default=200000)
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--runs_dir', default='runs')
    ap.add_argument('--record_gif', action='store_true')
    args = ap.parse_args()

    py = sys.executable or 'python'
    for app in args.apps:
        for algo in args.algos:
            for persona in args.personas:
                # Train
                run([py, '-m', 'src.train', '--app', app, '--algo', algo, '--persona', persona, '--seed', str(args.seed), '--timesteps', str(args.timesteps)])
                # Eval latest
                ev = [py, '-m', 'src.eval', '--app', app, '--algo', algo, '--persona', persona, '--seed', str(args.seed), '--episodes', str(args.episodes)]
                if args.record_gif:
                    ev.append('--record_gif')
                run(ev)
    print('All experiments completed. Check the runs/ directory for artifacts.')


if __name__ == '__main__':
    main()
