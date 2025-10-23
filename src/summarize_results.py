import argparse
import json
from pathlib import Path


KEYS_BY_APP = {
    'blackjack': ['return_mean', 'win_mean', 'lose_mean', 'draw_mean'],
    'formflow': ['return_mean', 'success_mean', 'distinct_pages_mean', 'validation_errors_mean'],
}


def load_runs(runs_dir: Path):
    items = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        parts = name.split('-')
        if len(parts) < 4:
            continue
        app, algo, persona = parts[0], parts[1], parts[2]
        agg = d / 'aggregate.json'
        if agg.exists():
            try:
                with open(agg, 'r', encoding='utf-8') as f:
                    items.append((app, algo, persona, name, json.load(f)))
            except Exception:
                pass
    return items


def choose_latest(items):
    # items already sorted by dir name; last timestamp suffix is latest
    by_key = {}
    for app, algo, persona, name, agg in items:
        by_key[(app, algo, persona)] = (name, agg)
    return by_key


def format_block(app, data):
    lines = []
    lines.append(f"### {app.title()}")
    keys = KEYS_BY_APP.get(app, ['return_mean', 'success_mean'])
    # Compare algos for each persona
    personas = sorted({p for (_, _, p) in data.keys()})
    algos = sorted({a for (_, a, _) in data.keys()})
    for persona in personas:
        lines.append(f"- Persona: {persona}")
        for algo in algos:
            name, agg = data.get((app, algo, persona), (None, None))
            if agg is None:
                continue
            metrics = ', '.join([f"{k}={agg.get(k):.3f}" for k in keys if k in agg])
            lines.append(f"  - {algo.upper()}: {metrics} ({name})")
    # Compare personas for each algo
    for algo in algos:
        lines.append(f"- Algo: {algo.upper()}")
        for persona in personas:
            name, agg = data.get((app, algo, persona), (None, None))
            if agg is None:
                continue
            metrics = ', '.join([f"{k}={agg.get(k):.3f}" for k in keys if k in agg])
            lines.append(f"  - {persona}: {metrics} ({name})")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='Append a Results & Discussion summary to REPORT.md')
    ap.add_argument('--runs_dir', default='runs')
    ap.add_argument('--report_md', default='REPORT.md')
    args = ap.parse_args()

    runs = load_runs(Path(args.runs_dir))
    latest = choose_latest(runs)
    # Group per app
    by_app = {}
    for (app, algo, persona), (name, agg) in latest.items():
        by_app.setdefault(app, {})[(app, algo, persona)] = (name, agg)

    lines = []
    lines.append('\n')
    lines.append('Results & Discussion')
    lines.append('=====================')
    lines.append('')
    lines.append('Comparisons below use the latest run per (app, algo, persona). Metrics are summarized to support persona and algorithm trade-off discussion.')
    lines.append('')
    for app, data in by_app.items():
        lines.append(format_block(app, data))
        lines.append('')

    with open(args.report_md, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Appended Results & Discussion to {args.report_md}")


if __name__ == '__main__':
    main()

