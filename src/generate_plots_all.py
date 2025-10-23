import os
import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', default='runs')
    args = ap.parse_args()

    py = 'python'
    for root, dirs, files in os.walk(args.runs_dir):
        if 'episodes.csv' in files:
            csv_path = os.path.join(root, 'episodes.csv')
            out_png = os.path.join(root, 'return_curve.png')
            try:
                subprocess.run([py, 'notebooks/plots.py', '--csv', csv_path, '--out', out_png], check=True)
            except Exception:
                pass
            # Heuristic metric plots
            # Blackjack
            if os.path.basename(root).startswith('blackjack'):
                for metric in ['win','lose','draw']:
                    out = os.path.join(root, f'{metric}_hist.png')
                    subprocess.run([py, 'notebooks/plots.py', '--csv', csv_path, '--out', out, '--metric', metric])
            # FormFlow
            if os.path.basename(root).startswith('formflow'):
                for metric in ['distinct_pages','distinct_selectors','validation_errors']:
                    out = os.path.join(root, f'{metric}_hist.png')
                    subprocess.run([py, 'notebooks/plots.py', '--csv', csv_path, '--out', out, '--metric', metric])

    print('Plots generated where CSVs were found.')


if __name__ == '__main__':
    main()

