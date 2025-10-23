import os
import json
import argparse
import subprocess
from pathlib import Path


APP_PREFIX_METRICS = {
    'blackjack': ['win', 'lose', 'draw'],
    'formflow': ['distinct_pages', 'distinct_selectors', 'validation_errors'],
    'minigrid': ['visited_cells', 'success'],
    'tetris': ['lines_cleared_total', 'holes_count', 'max_height'],
}


def safe_run(cmd):
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False


def generate_plots(run_dir: Path):
    csv = run_dir / 'episodes.csv'
    if not csv.exists():
        return
    # Return curve
    out = run_dir / 'return_curve.png'
    if not out.exists():
        safe_run(['python', 'notebooks/plots.py', '--csv', str(csv), '--out', str(out)])
    # Metric hists based on app prefix
    prefix = run_dir.name.split('-')[0]
    metrics = APP_PREFIX_METRICS.get(prefix, [])
    for m in metrics:
        out_m = run_dir / f'{m}_hist.png'
        if not out_m.exists():
            safe_run(['python', 'notebooks/plots.py', '--csv', str(csv), '--out', str(out_m), '--metric', m])


def collect_runs(runs_dir: Path):
    items = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        cfg = d / 'config.json'
        agg = d / 'aggregate.json'
        if cfg.exists() and agg.exists():
            try:
                with open(cfg, 'r', encoding='utf-8') as f:
                    cfg_obj = json.load(f)
                with open(agg, 'r', encoding='utf-8') as f:
                    agg_obj = json.load(f)
                items.append({'dir': d, 'config': cfg_obj, 'agg': agg_obj})
            except Exception:
                pass
    return items


def find_preview(eval_dir: Path):
    if not eval_dir.exists():
        return None
    gifs = sorted(eval_dir.glob('episode_*.gif'))
    if gifs:
        return gifs[0]
    pngs = sorted(eval_dir.glob('episode_*_frame_*.png'))
    if pngs:
        return pngs[0]
    return None


def write_html(out_html: Path, runs):
    parts = []
    parts.append('<!doctype html><meta charset="utf-8"><title>DRL for Automated Testing — Report</title>')
    parts.append('<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222} h1,h2{margin:0.4em 0} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px} .card{border:1px solid #ddd;border-radius:8px;padding:12px} img{max-width:100%;height:auto;border:1px solid #eee;border-radius:6px} code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style>')
    parts.append('<h1>DRL for Automated Testing — Results</h1>')
    parts.append('<p>This report is auto-generated from artifacts under <code>runs/</code>. It includes per-run aggregates, plots, and preview frames/GIFs when available.</p>')
    # Group by app
    by_app = {}
    for item in runs:
        name = item['dir'].name
        app = name.split('-')[0]
        by_app.setdefault(app, []).append(item)
    for app, items in by_app.items():
        parts.append(f'<h2>{app.title()}</h2>')
        parts.append('<div class="grid">')
        for it in items:
            rd = it['dir']
            agg = it['agg']
            parts.append('<div class="card">')
            parts.append(f'<h3 style="margin-top:0">{rd.name}</h3>')
            # Plots
            rc = rd / 'return_curve.png'
            if rc.exists():
                parts.append(f'<div><img src="{rc.as_posix()}" alt="return curve"></div>')
            # Metrics bullets (top few)
            parts.append('<ul>')
            for k in ['episodes','return_mean','return_std','length_mean']:
                if k in agg:
                    parts.append(f'<li><b>{k}</b>: {agg[k]}</li>')
            # Pick a few app-specific metrics
            prefix = rd.name.split('-')[0]
            for k in APP_PREFIX_METRICS.get(prefix, [])[:3]:
                km = f'{k}_mean'
                if km in agg:
                    parts.append(f'<li><b>{km}</b>: {agg[km]}</li>')
            parts.append('</ul>')
            # Preview
            pv = find_preview(rd / 'eval')
            if pv is not None:
                parts.append(f'<div><img src="{pv.as_posix()}" alt="preview"></div>')
            parts.append('</div>')
        parts.append('</div>')
    out_html.write_text('\n'.join(parts), encoding='utf-8')


def append_report_md(report_md: Path, runs):
    lines = []
    lines.append('\n')
    lines.append('Auto-Generated Results Summary')
    lines.append('===============================')
    lines.append('')
    lines.append('Below are aggregated snapshots from existing runs under `runs/`, including return statistics and key domain metrics.')
    lines.append('')
    for it in runs:
        rd = it['dir'].name
        agg = it['agg']
        lines.append(f'- {rd}: episodes={agg.get("episodes")}, return_mean={agg.get("return_mean"):.3f}, return_std={agg.get("return_std"):.3f}, length_mean={agg.get("length_mean"):.2f}')
    with open(report_md, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description='Build plots and a consolidated report from runs/')
    ap.add_argument('--runs_dir', default='runs')
    ap.add_argument('--html_out', default='AMAZING_REPORT.html')
    ap.add_argument('--append_report_md', action='store_true')
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs = collect_runs(runs_dir)
    # Generate plots where missing
    for it in runs:
        generate_plots(it['dir'])
    # Write HTML
    write_html(Path(args.html_out), runs)
    # Optionally append summary to REPORT.md if present
    if args.append_report_md:
        report_md = Path('REPORT.md')
        if report_md.exists():
            append_report_md(report_md, runs)
    print(f'Wrote HTML report to {args.html_out}.')


if __name__ == '__main__':
    main()

