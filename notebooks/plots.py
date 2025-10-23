import argparse, os, pandas as pd
import matplotlib.pyplot as plt

def plot_learning(csv_path, out_png):
    df = pd.read_csv(csv_path)
    ax = df['return'].plot(title='Episode Returns')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

def plot_metric_hist(csv_path, metric, out_png):
    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        raise ValueError(f"{metric} not in CSV columns: {list(df.columns)}")
    df[metric].plot(kind='hist', bins=30, title=f'{metric} distribution')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default=None)
    args = ap.parse_args()
    if args.metric:
        plot_metric_hist(args.csv, args.metric, args.out)
    else:
        plot_learning(args.csv, args.out)
