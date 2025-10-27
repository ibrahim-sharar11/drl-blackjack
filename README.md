# DRL for Automated Testing — Blackjack + FormFlow

Automate testing of applications using trained deep RL agents on two apps:
- Blackjack (game)
- FormFlow (simulated multi-page web flow)

## Executive Summary
The project uses the PPO and A2C algorithms from Stable Baselines3, with all settings configured through YAML files.
Three personas were trained: Survivor as the main baseline, and Explorer and Speedrunner as variations created by adjusting reward weights.
Each run produces both per-episode CSV files and an overall JSON summary that includes key metrics like win and loss rates, validation errors, coverage, softlocks, and overall success.
All experiments are fully reproducible with fixed random seeds, pinned dependencies, and clearly listed commands. Results and outputs are saved in the runs folder.

## Submission Focus
- This report and all commands/results are scoped to Blackjack and FormFlow only.
- Other example envs (MiniGrid/Tetris) may exist in the codebase but are not part of this submission.

## Repository Structure
- `README.md` — main report and usage instructions (this file)
- `requirements.txt` — pinned dependencies
- `envs/` — application environments
  - `envs/blackjack_env.py` — Blackjack environment with optional betting and reward shaping
  - `envs/formflow_env.py` — Web flow simulator with validation/latency/coverage signals
- `src/` — training, evaluation, metrics, utilities
  - `src/train.py` — Train PPO/A2C with personas; saves artifacts to `runs/`
  - `src/eval.py` — Evaluate trained agents; export eval metrics; optional GIFs
  - `src/make_env.py` — Factory wiring app config + persona weights into an environment
  - `src/metrics.py` — Episode logger (CSV) and aggregate stats (JSON)
  - `src/build_report.py` — Build plots and `AMAZING_REPORT.html` from `runs/`
  - `src/exp_matrix.py` — sweep helper to train/eval multiple combos
  - `src/generate_plots_all.py` — generate return curves and metric histograms across runs
  - `src/make_gifs.py` — convert PNG frames under `runs/*/eval` into GIFs
  - `src/make_legends.py` — render legend image used in report cards
  - `src/summarize_results.py` — aggregate results and print/append summaries
  - `src/utils.py` — config loader (`load_configs`) and global seeding helpers
- `configs/` — YAML configs
  - `configs/app/` — per-app settings (e.g., `blackjack.yaml`, `formflow.yaml`)
  - `configs/algo/` — PPO/A2C hyperparameters
  - `configs/persona/` — reward weights (survivor/explorer/speedrunner)
- `apps/`
  - `apps/blackjack_pygame.py` — optional human viewer for Blackjack
- `notebooks/`
  - `notebooks/plots.py` — plotting script used by build_report
- `assets/`
  - `assets/README.md` — optional artwork/sounds guidance for the viewer
- `runs/` — per-run artifacts (created by training/eval)
  - `<app>-<algo>-<persona>-seed<seed>-<ts>/` — `model.zip`, `episodes.csv`, `aggregate.json`, `eval/`

## Setup (Python)
1) Create venv and install pinned dependencies
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
2) Verify install (optional)
```
python -c "import gymnasium,stable_baselines3,torch; print('OK')"
```

## Setup (Docker, optional CPU/headless)
Create a `Dockerfile`:
```
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libxrender1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV SDL_VIDEODRIVER=dummy MPLBACKEND=Agg
CMD ["python", "-m", "src.train", "--app", "blackjack", "--algo", "ppo", "--persona", "survivor", "--seed", "7", "--timesteps", "10000"]
```
Build and run:
```
docker build -t drl-testing .
# Windows PowerShell (mount runs/ to host):
docker run --rm -v ${PWD}/runs:/app/runs drl-testing
# Linux/macOS:
docker run --rm -v "$PWD/runs:/app/runs" drl-testing
```

## Dependency Pinning
- All versions are pinned in `requirements.txt` (Gymnasium, SB3, Torch, Numpy, Pandas, Matplotlib, Pygame, ImageIO, PyYAML, etc.).

## Quick Start (5–10 minute smoke test)
```
# Train short runs (10k timesteps) to verify setup
python -m src.train --app blackjack --algo ppo --persona survivor --seed 7 --timesteps 10000
python -m src.train --app formflow  --algo a2c --persona survivor --seed 7 --timesteps 10000

# Evaluate 5 episodes and record GIFs
python -m src.eval  --app blackjack --algo ppo --persona survivor --seed 7 --episodes 5 --record_gif
python -m src.eval  --app formflow  --algo a2c --persona survivor --seed 7 --episodes 5 --record_gif
```

## How to Train/Eval (One-Liners)
- Train competence baselines (200k steps):
```
python -m src.train --app blackjack --algo ppo --persona survivor --seed 7 --timesteps 200000
python -m src.train --app blackjack --algo a2c --persona survivor --seed 7 --timesteps 200000
python -m src.train --app formflow  --algo ppo --persona survivor --seed 7 --timesteps 200000
python -m src.train --app formflow  --algo a2c --persona survivor --seed 7 --timesteps 200000
```
- Evaluate (50 episodes; add `--record_gif` for clips):
```
python -m src.eval --app blackjack --algo ppo --persona survivor --seed 7 --episodes 50
python -m src.eval --app formflow  --algo a2c --persona survivor --seed 7 --episodes 50
```

## Environments: Actions, Observations, Rewards, Metrics

### Blackjack (`envs/blackjack_env.py`)
- Actions: `hit`, `stand`; optional `double` (first decision) and betting bins.
- Observation (8D normalized): `[player_sum, dealer_upcard, usable_ace, steps_left, bankroll, bet, rounds_left, doubled]`.
- Rewards/Checks (persona-weighted): `step_cost`, `bust_penalty`, `win_reward`, `lose_penalty`, `draw_bonus`, `blackjack_bonus`, `success`, `speed_bonus`, `approach_21_bonus`, `early_stand_penalty`, `safe_hit_bonus`.
- Metrics (info): `win`, `lose`, `draw`, `player_bust`, `dealer_bust`, `bankroll`, `bet`, `action_hit/stand/double`.

Actions
| Action | Meaning |
|-------|---------|
| hit   | Draw a card |
| stand | End player turn, resolve dealer |
| double (opt) | Double bet, take one card, stand |

Observation (8D)
| Feature | Description |
|---------|-------------|
| player_sum | Player hand total (normalized) |
| dealer_upcard | Dealer visible card (normalized) |
| usable_ace | 1 if Ace counted as 11 without bust |
| steps_left | Fraction of steps remaining |
| bankroll | Bankroll progress (betting mode) |
| bet | Current bet fraction (betting mode) |
| rounds_left | Fraction of rounds remaining |
| doubled | 1 if doubled this round |

Reward Keys (examples)
| Key | Effect |
|-----|--------|
| win_reward / lose_penalty / draw_bonus | Outcome shaping |
| bust_penalty | Penalize player busts |
| blackjack_bonus | Bonus for natural blackjack |
| approach_21_bonus | Reward moving closer to 21 safely |
| early_stand_penalty | Discourage very early stand |
| safe_hit_bonus | Encourage safe early hits (<=11) |

### FormFlow (`envs/formflow_env.py`)
- Pages: landing → signup → profile → review → submit.
- Actions: `next`, `prev`, `type_input`, `clear_input`, `toggle_checkbox`, `click_random_selector`, `submit`.
- Observation (11D): one‑hot(page) + scalars for `field_filled`, `field_valid`, `checkbox`, `errors_on_page`, `latency_bucket`, `steps_left_norm`.
- Rewards/Checks (persona‑weighted): `validation_error_bonus`, `latency_penalty`, `page_progress`, `page_coverage_bonus`, `dom_coverage_bonus`, `softlock_penalty`, `success`, `speed_bonus`, `step_cost`.
- Metrics (info): `distinct_pages`, `distinct_selectors`, `validation_errors`, `latency_spike`, `softlock`, `success`, action flags.

Actions
| Action | Meaning |
|--------|---------|
| next | Attempt advance to next page |
| prev | Go back a page |
| type_input | Fill field (may trigger validation error) |
| clear_input | Clear field |
| toggle_checkbox | Toggle checkbox |
| click_random_selector | Click random selector (coverage proxy) |
| submit | Submit on final page |

Observation (11D)
| Feature | Description |
|---------|-------------|
| page one‑hot (5) | Current page indicator |
| field_filled | 0/1, whether field has content |
| field_valid | 0/1, whether field passes validation |
| checkbox | 0/1, opt‑in state |
| errors_on_page | 0..3 normalized error count |
| latency_bucket | 0..3 normalized latency level |
| steps_left_norm | Fraction of steps remaining |

Reward Keys (examples)
| Key | Effect |
|-----|--------|
| validation_error_bonus | Incentivize finding validation issues |
| latency_penalty | Penalize latency spikes |
| page_progress | Reward steady forward progress |
| page_coverage_bonus / dom_coverage_bonus | Encourage coverage |
| softlock_penalty | Penalize loops/softlocks |
| success / speed_bonus | Reward timely completion |

## Experiments
- Featured competence runs (Survivor persona, 200k steps, 50‑episode eval):
  - Blackjack PPO: `runs/blackjack-ppo-survivor-seed7-1761503633`
  - Blackjack A2C: `runs/blackjack-a2c-survivor-seed7-1761503840`
  - FormFlow PPO: `runs/formflow-ppo-survivor-seed7-1761504146`
  - FormFlow A2C: `runs/formflow-a2c-survivor-seed7-1761504286`

## Reproduction Steps (Exact Commands Used)
```
python -m src.train --app blackjack --algo ppo --persona survivor --seed 7 --timesteps 200000
python -m src.train --app blackjack --algo a2c --persona survivor --seed 7 --timesteps 200000
python -m src.train --app formflow  --algo ppo --persona survivor --seed 7 --timesteps 200000
python -m src.train --app formflow  --algo a2c --persona survivor --seed 7 --timesteps 200000
python -m src.eval  --app blackjack --algo ppo --persona survivor --seed 7 --episodes 50
python -m src.eval  --app blackjack --algo a2c --persona survivor --seed 7 --episodes 50
python -m src.eval  --app formflow  --algo ppo --persona survivor --seed 7 --episodes 50
python -m src.eval  --app formflow  --algo a2c --persona survivor --seed 7 --episodes 50
```

## Short Clips/GIFs per App
- Add `--record_gif` to any eval command to save previews in `runs/<run>/eval/episode_*.gif` (PNG fallback supported).
- Examples from featured runs:
  - Blackjack PPO Survivor: `runs/blackjack-ppo-survivor-seed7-1761503633/eval/episode_1.gif`
  - FormFlow  PPO Survivor: `runs/formflow-ppo-survivor-seed7-1761504146/eval/episode_1.gif`

## Results Pointers
- Each run contains: `model.zip`, `episodes.csv`, `aggregate.json`, `return_curve.png`, and `eval/` artifacts.
- A consolidated, auto‑generated HTML view is available at `AMAZING_REPORT.html`.

## Architecture & Decoupling
- Clean separation of environment code (apps) from training/eval and metrics.
- Config‑driven env construction via `src/make_env.py` makes the framework reusable across apps/personas.
- Metrics: `src/metrics.py` logs per‑episode CSV and aggregates to JSON.

## Code Snippets

### Factory: build env from configs + persona
```
# src/make_env.py (excerpt)
from envs.formflow_env import FormFlowEnv
from envs.blackjack_env import BlackjackEnv

def make_env(app_cfg, persona_cfg):
    weights = persona_cfg["weights"]
    app_id = app_cfg["id"]
    if app_id == "formflow":
        return FormFlowEnv(
            max_steps=app_cfg.get("max_steps", 150),
            seed=app_cfg.get("seed", 7),
            reward_weights=weights,
            reward_scale=app_cfg.get("reward_scale", 1.0),
            invalid_prob=app_cfg.get("invalid_prob", 0.2),
            latency_spike_prob=app_cfg.get("latency_spike_prob", 0.05),
        )
    elif app_id == "blackjack":
        return BlackjackEnv(
            max_steps=app_cfg.get("max_steps", 100),
            seed=app_cfg.get("seed", 7),
            reward_weights=weights,
            reward_scale=app_cfg.get("reward_scale", 1.0),
            num_decks=app_cfg.get("num_decks", 1),
            penetration=app_cfg.get("penetration", 0.75),
            rounds_per_episode=app_cfg.get("rounds_per_episode", 1),
            bankroll_start=app_cfg.get("bankroll_start", 0),
            bankroll_target=app_cfg.get("bankroll_target", 0),
            bet_bins=app_cfg.get("bet_bins", 0),
            min_bet=app_cfg.get("min_bet", 1),
            max_bet=app_cfg.get("max_bet", 10),
            payout_blackjack=app_cfg.get("payout_blackjack", 1.5),
            dealer_hits_soft17=app_cfg.get("dealer_hits_soft17", False),
            allow_double=app_cfg.get("allow_double", True),
            bet_scaled_reward=app_cfg.get("bet_scaled_reward", False),
        )
    else:
        raise ValueError(f"Unknown app id: {app_id}")
```

### Training loop with metrics logging
```
# src/train.py (excerpt)
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from src.metrics import EpisodeLogger

ALGOS = {"ppo": PPO, "a2c": A2C}

env = make_env(cfg["app"], cfg["persona"])  # built from YAML + persona
env = Monitor(env)
env = DummyVecEnv([lambda: env])

Algo = ALGOS[args.algo]
model = Algo(policy, env, seed=args.seed, verbose=1, **kwargs)
cb = EpisodeLogger(out_dir)
model.learn(total_timesteps=total_ts, callback=cb, progress_bar=True)
model.save(os.path.join(out_dir, "model"))
```

### Evaluation + per-episode metrics
```
# src/eval.py (excerpt)
model = ALGOS[args.algo].load(model_path, env=venv)
logger = EpisodeLogger(os.path.join(run_dir, "eval"))
for ep in range(args.episodes):
    obs = venv.reset(); done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = venv.step(action)
        logger.locals = {"infos": infos, "rewards": reward, "dones": dones}
        logger._on_step()
        done = bool(dones[0])
```

