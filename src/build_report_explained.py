import os
import json
import argparse
from pathlib import Path
import subprocess


APP_PREFIX_METRICS = {
    'blackjack': ['win', 'lose', 'draw'],
    'formflow': ['distinct_pages', 'distinct_selectors', 'validation_errors'],
}

APP_EXPLANATIONS = {
    'blackjack': {
        'intro': (
            "This card summarizes a Blackjack agent run. The return curve shows training stability and reward progression. "
            "Outcome histograms (win/lose/draw) reflect frequencies across episodes. The preview (GIF/frames) visualizes hands "
            "dealt and simple state bars (sums, bet/bankroll if enabled)."
        ),
        'metrics': {
            'win': 'Episodes the agent won (higher is better).',
            'lose': 'Episodes the agent lost (lower is better).',
            'draw': 'Episodes that ended in a push.'
        }
    },
    'formflow': {
        'intro': (
            "This card summarizes a FormFlow (web flow) agent run. The return curve shows learning over episodes. "
            "Histograms capture domain metrics: page/DOM coverage and validation errors discovered. Previews encode current page, "
            "field validity, latency spikes, and error counters."
        ),
        'metrics': {
            'distinct_pages': 'Unique pages visited (coverage).',
            'distinct_selectors': 'Unique selectors clicked (DOM coverage proxy).',
            'validation_errors': 'Validation issues encountered (issue detection).'
        }
    },
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


def latest_by_key(items):
    latest = {}
    for it in items:
        dname = it['dir'].name
        key = (
            dname.split('-')[0],  # app
            dname.split('-')[1],  # algo
            dname.split('-')[2],  # persona
        )
        if key not in latest or dname > latest[key]['dir'].name:
            latest[key] = it
    return latest


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
    parts.append('<!doctype html><meta charset="utf-8"><title>DRL for Automated Testing — Results</title>')
    parts.append('<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222} h1,h2{margin:0.4em 0} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px} .card{border:1px solid #ddd;border-radius:8px;padding:12px} img{max-width:100%;height:auto;border:1px solid #eee;border-radius:6px} code{background:#f5f5f5;padding:2px 4px;border-radius:4px} .muted{color:#666} ul{margin:0.4em 0 0.8em 1.2em}</style>')
    parts.append('<h1>DRL for Automated Testing — Results</h1>')
    parts.append('<p>This report is auto-generated from artifacts under <code>runs/</code>. It includes per-run aggregates, plots, and preview frames/GIFs when available.</p>')
    parts.append('<div class="muted"><b>How to read:</b> The return curve summarizes episode returns across training (higher and more stable is better). Histograms show distributions of app-specific metrics. The preview (GIF or first PNG frame) provides a quick visual of agent behaviour during evaluation.</div>')
    # Group by app
    by_app = {}
    for item in runs:
        name = item['dir'].name
        app = name.split('-')[0]
        by_app.setdefault(app, []).append(item)
    latest = latest_by_key(runs)
    for app, items in by_app.items():
        parts.append(f'<h2>{app.title()}</h2>')
        # Comparison summary table (latest per algo/persona)
        parts.append('<div class="card">')
        parts.append('<h3 style="margin-top:0">Comparison Snapshot</h3>')
        parts.append('<div class="muted">Side-by-side metrics for the latest runs per algorithm and persona.</div>')
        # Build table header based on app
        if app == 'blackjack':
            cols = ['algo','persona','return_mean','win_mean','lose_mean','draw_mean']
        elif app == 'formflow':
            cols = ['algo','persona','return_mean','success_mean','distinct_pages_mean','distinct_selectors_mean','validation_errors_mean']
        else:
            cols = ['algo','persona','return_mean']
        parts.append('<table style="width:100%;border-collapse:collapse">')
        parts.append('<tr>' + ''.join([f'<th style="text-align:left;border-bottom:1px solid #ddd;padding:4px 6px">{c}</th>' for c in cols]) + '</tr>')
        # Personas to show first if present
        pref_personas = ['survivor','explorer','speedrunner']
        for algo in ['ppo','a2c']:
            for persona in pref_personas:
                k = (app, algo, persona)
                it = latest.get(k)
                if not it: continue
                agg = it['agg']
                row = []
                for c in cols:
                    if c in ['algo','persona']:
                        row.append({'algo': algo, 'persona': persona}[c])
                    else:
                        val = agg.get(c)
                        row.append(f"{val:.3f}" if isinstance(val,(int,float)) else (str(val) if val is not None else ''))
                parts.append('<tr>' + ''.join([f'<td style="padding:4px 6px;border-bottom:1px solid #f0f0f0">{cell}</td>' for cell in row]) + '</tr>')
        parts.append('</table>')
        parts.append('</div>')
        parts.append('<div class="grid">')
        for it in items:
            rd = it['dir']
            agg = it['agg']
            parts.append('<div class="card">')
            parts.append(f'<h3 style="margin-top:0">{rd.name}</h3>')
            app_prefix = rd.name.split('-')[0]
            expl = APP_EXPLANATIONS.get(app_prefix, {}).get('intro')
            if expl:
                parts.append(f'<p class="muted">{expl}</p>')
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
            metrics = APP_PREFIX_METRICS.get(app_prefix, [])
            for k in metrics[:3]:
                km = f'{k}_mean'
                if km in agg:
                    desc = APP_EXPLANATIONS.get(app_prefix, {}).get('metrics', {}).get(k, None)
                    if desc:
                        parts.append(f'<li><b>{km}</b>: {agg[km]} — <span class="muted">{desc}</span></li>')
                    else:
                        parts.append(f'<li><b>{km}</b>: {agg[km]}</li>')
            parts.append('</ul>')
            # Preview
            pv = find_preview(rd / 'eval')
            if pv is not None:
                parts.append(f'<div><img src="{pv.as_posix()}" alt="preview"></div>')
            parts.append('</div>')
        parts.append('</div>')
    out_html.write_text('\n'.join(parts), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='Build plots and a consolidated report from runs/ with explanations')
    ap.add_argument('--runs_dir', default='runs')
    ap.add_argument('--html_out', default='AMAZING_REPORT.html')
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs = collect_runs(runs_dir)
    for it in runs:
        generate_plots(it['dir'])
    write_html(Path(args.html_out), runs)
    print(f'Wrote HTML report to {args.html_out}.')


if __name__ == '__main__':
    main()
