DRL for Automated Testing

Automate testing of applications using trained deep RL agents (not scripted bots). This repo contains:
- MiniGrid (2D navigation)
- FormFlow (simulated multi-page web flow)
- Tetris-like (grid puzzle with line clears)

Overview
- Algorithms: PPO and A2C, with configurable personas via reward weights (survivor, explorer, speedrunner).
- Metrics: Per-episode and aggregate CSV/JSON exports; plotting helpers for learning curves and distributions.
- Reproducible: Fixed seeds, pinned dependencies, saved models and logs.

Repo Structure
- `envs/` — environment wrappers
  - `envs/minigrid_env.py` — MiniGrid wrapper with shaped rewards and coverage metric.
  - `envs/formflow_env.py` — Simulated web flow with validation/latency/coverage and issue-detection rewards.
  - `envs/tetris_env.py` — Tetris-like grid puzzle with line clears and structural metrics.
- `src/` — training & eval
  - `src/train.py` — Train SB3 agents (PPO/A2C) with personas.
  - `src/eval.py` — Evaluate trained agents; export metrics; optional GIFs for MiniGrid.
  - `src/make_env.py` — Factory to build env from configs.
  - `src/metrics.py` — Episode logger and aggregation to JSON.
- `configs/` — YAML configs for apps, algos, personas.
- `notebooks/plots.py` — Quick plotting for returns or metric histograms.
- `runs/` — Saved artifacts (models, episodes.csv, aggregate.json, plots).

Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Quick Start
Train at least two models on each app with distinct personas to compare:
```
# MiniGrid
python -m src.train --algo ppo --app minigrid --persona survivor --seed 7
python -m src.train --algo a2c --app minigrid --persona explorer --seed 7

# FormFlow
python -m src.train --algo a2c --app formflow --persona survivor --seed 7
python -m src.train --algo ppo --app formflow --persona speedrunner --seed 7

# Tetris-like
python -m src.train --algo ppo --app tetris --persona survivor --seed 7
python -m src.train --algo a2c --app tetris --persona speedrunner --seed 7
```

Evaluate and export aggregate metrics:
```
python -m src.eval --algo ppo --app minigrid --persona survivor --seed 7 --episodes 50
python -m src.eval --algo a2c --app formflow --persona survivor --seed 7 --episodes 50
python -m src.eval --algo ppo --app tetris --persona speedrunner --seed 7 --episodes 50
```

Plot learning curve or a metric distribution:
```
# Episode returns
python notebooks/plots.py --csv runs/<run-dir>/episodes.csv --out runs/<run-dir>/return_curve.png

# Metric histogram (e.g., distinct_selectors for FormFlow)
python notebooks/plots.py --csv runs/<run-dir>/eval/episodes.csv --metric distinct_selectors --out runs/<run-dir>/eval/distinct_selectors_hist.png
```

Environments
MiniGrid
- Observation: `obs["image"]` normalized to [0,1].
- Actions: as per selected MiniGrid ID (default `MiniGrid-Empty-5x5-v0`).
- Shaping keys: `step_cost`, `explore_bonus`, `success`, `speed_bonus`.
- Metrics: `steps`, `visited_cells`, `success`.

FormFlow (simulated web flow)
- Pages: landing -> signup -> profile -> review -> submit.
- Actions: `next_page`, `prev_page`, `type_input`, `clear_input`, `toggle_checkbox`, `click_random_selector`, `submit_page`.
- Observation: one-hot page + form/latency/timer scalars.
- Rewards/checks: `validation_error_bonus`, `latency_penalty`, `page_progress`, `page_coverage_bonus`, `dom_coverage_bonus`, `softlock_penalty`, `success`, `speed_bonus`, `step_cost`.
- Metrics: `steps`, `distinct_pages`, `distinct_selectors`, `validation_errors`, `latency_spike`, `softlock`, `success`.

Tetris-like
- Actions: `noop`, `left`, `right`, `rotate`, `hard_drop`.
- Observation: flattened 20x10 board + piece one-hot + normalized (x,y).
- Rewards/checks: `line_clear_bonus`, `hole_penalty`, `height_penalty`, `topout_penalty`, plus generic `step_cost`, `success`, `speed_bonus`.
- Metrics: `lines_cleared_total`, `rows_cleared_last`, `holes_count`, `max_height`, `topout`, `success`.

Personas (Rewards)
Personas are defined in `configs/persona/*.yaml` and map to reward weights used by each env. For example:
- Survivor: modest exploration, prioritizes finishing and avoiding risky states.
- Explorer: strong coverage incentives (pages/selectors/cells), neutral on speed.
- Speedrunner: prioritizes reaching success quickly, little coverage shaping.

Reproducibility
- Seeds fixed across numpy/torch and envs; specify with `--seed`.
- Dependencies pinned in `requirements.txt`.
- Each run saves:
  - `runs/<run-dir>/model.zip` — trained model
  - `runs/<run-dir>/episodes.csv` — per-episode metrics
  - `runs/<run-dir>/aggregate.json` — summary stats
  - `runs/<run-dir>/eval/...` — eval metrics and plots

Results
- Example artifacts are saved under `runs/`, including:
  - `episodes.csv` per-episode metrics and `aggregate.json` summary
  - Optional GIFs when recorded during eval: `runs/<run-dir>/eval/episode_*.gif`
  - PNG frames fallback: `runs/<run-dir>/eval/episode_*_frame_*.png`

Results (Current Repo)
- Open the consolidated `AMAZING_REPORT.html` for a grid of runs with plots and previews.
- Included example runs cover Blackjack and FormFlow with PPO/A2C and Survivor/Explorer personas (see `runs/`).

Record GIFs (All apps)
```
python -m src.eval --algo ppo --app minigrid --persona survivor --seed 7 --episodes 3 --record_gif
python -m src.eval --algo a2c --app formflow --persona explorer --seed 7 --episodes 3 --record_gif
python -m src.eval --algo ppo --app tetris --persona speedrunner --seed 7 --episodes 3 --record_gif
# GIFs are saved under the run's eval directory (or PNG frames if GIF not available)
```

Fault Injection (FormFlow)
- Configure in `configs/app/formflow.yaml`:
  - `invalid_prob`: chance of invalid input when typing
  - `latency_spike_prob`: chance of latency spikes per step

Notebook Report
- See `notebooks/report.ipynb` for a full interactive report with:
  - Code summaries and config snapshots
  - Auto-discovery of runs and aggregate stats
  - Return curves and metric histograms
  - Inline previews of recorded frames/GIFs
- If you prefer a root-level report, open `REPORT.ipynb` (same content, easier to find).
 - If your IDE cannot open notebooks, view the static `REPORT.md` in the repo root (embeds plots/frames).

Full Report (Single Doc)
- For the complete rubric-aligned write-up with architecture, personas, commands, results, curated visuals, and links to artifacts, open `REPORT.md` (canonical combined report).

GIF Batch Conversion
- Convert PNG frames under `runs/*/eval` into GIFs:
```
python -m src.make_gifs --runs_dir runs --fps 10
```
Notes: Requires `imageio`. If not installed, the script will warn and skip GIF creation.

HTML Report
- Open `AMAZING_REPORT.html` for a polished, static report with embedded visuals.

Build/Refresh the HTML report and plots from existing runs:
```
python -m src.build_report --append_report_md
```
This generates per-run plots (learning curves and metric histograms), writes `AMAZING_REPORT.html`, and appends a concise summary to `REPORT.md`.

Blackjack & Viewer
- New app id: `blackjack`. Train/eval like other apps:
```
python -m src.train --app blackjack --algo ppo --persona survivor --seed 7 --timesteps 200000
python -m src.eval --app blackjack --algo ppo --persona survivor --seed 7 --episodes 50 --record_gif
```
- Animated human viewer (optional):
```
python -m apps.blackjack_pygame           # no betting
python -m apps.blackjack_pygame --betting # enable betting+double
```
- Persona weights include Blackjack keys: `bust_penalty`, `win_reward`, `lose_penalty`, `draw_bonus`, `blackjack_bonus`.

Experiment Matrix Runner
- Run a sweep over apps/algos/personas with a single command:
```
python -m src.exp_matrix --apps blackjack formflow --algos ppo a2c --personas survivor explorer --seed 7 --timesteps 200000 --episodes 50 --record_gif
```
- Artifacts are saved under `runs/` for each configuration.
