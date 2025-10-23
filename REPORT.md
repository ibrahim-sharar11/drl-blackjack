DRL for Automated Testing — Assignment Report

Overview
- Two apps: Blackjack (game) and FormFlow (web flow sim)
- Two algorithms: PPO, A2C (Stable-Baselines3)
- Personas: Survivor, Explorer (Speedrunner also available)
- Metrics: per-episode CSV + aggregate JSON; plots for returns and key domain metrics

Environment Summary
- Blackjack (envs/blackjack_env.py:1)
  - Actions: bet phase (if enabled): select bet bin; play phase: 0=hit, 1=stand, 2=double (first decision)
  - Observation: [player_sum, dealer_upcard, usable_ace, steps_left, bankroll, bet, rounds_left, doubled] (normalized)
  - Rewards: step_cost, bust_penalty, win_reward, lose_penalty, draw_bonus, blackjack_bonus, success, speed_bonus
  - Metrics: steps, player_sum, dealer_sum, win/draw/lose, bankroll, bet, action flags (action_hit/stand/double)
- FormFlow (envs/formflow_env.py:1)
  - Actions: next, prev, type_input, clear_input, toggle_checkbox, click_random_selector, submit
  - Observation: one-hot page + state scalars (latency/errors/steps_left)
  - Rewards: success, page progress/coverage, latency penalty, softlock penalty, step_cost
  - Metrics: steps, distinct_pages, distinct_selectors, validation_errors, latency_spike, softlock, success, page_id, action flags

Reproducibility
- Seed fixed to 7 (torch/numpy/envs) via src/utils.set_global_seeds
- Per-run config snapshot: runs/<run>/config.json
- Saved model: runs/<run>/model.zip
- Episode metrics: runs/<run>/episodes.csv; aggregates: runs/<run>/aggregate.json

Commands Used
Train (short sweep used for initial artifacts):
```
python -m src.exp_matrix --apps blackjack formflow --algos ppo a2c --personas survivor explorer --seed 7 --timesteps 3000 --episodes 8 --record_gif
```
Evaluate/GIFs are included in the sweep (last command flag).

Artifacts
- Runs are under `runs/`. For each run directory:
  - `episodes.csv`, `aggregate.json`, `return_curve.png`
  - Blackjack: `win_hist.png`, `lose_hist.png`, `draw_hist.png`
  - FormFlow: `distinct_pages_hist.png`, `distinct_selectors_hist.png`, `validation_errors_hist.png`
  - `eval/episode_*.gif` (or PNG frames fallback)

Results Snapshot (example pointers)
- Blackjack PPO Survivor: see `runs/blackjack-ppo-survivor-seed7-<ts>/`
- Blackjack A2C Explorer: see `runs/blackjack-a2c-explorer-seed7-<ts>/`
- FormFlow PPO Survivor: see `runs/formflow-ppo-survivor-seed7-<ts>/`
- FormFlow A2C Explorer: see `runs/formflow-a2c-explorer-seed7-<ts>/`

Discussion Prompts
- Persona effects (Blackjack): Survivor reduces busts (higher bust_penalty), Explorer chases more aggressive plays (higher blackjack_bonus), observe win/lose distributions.
- Persona effects (FormFlow): Explorer raises coverage (distinct_selectors/pages, validation_error_bonus); Survivor emphasizes completion and avoids latency spikes.
- PPO vs A2C: Compare return curves and success/coverage metrics per app; discuss stability and sample efficiency for your settings.

Next Steps (for final submission)
- Re-run with longer timesteps (e.g., 200k) for stronger competence.
- Select best pairs per app and produce final side-by-side plots and short GIF clips.
- Fill in concrete metrics (means/std in aggregates) and qualitative findings into this report.



Auto-Generated Results Summary
===============================

Below are aggregated snapshots from existing runs under `runs/`, including return statistics and key domain metrics.

- blackjack-a2c-explorer-seed7-1761160653: episodes=9962, return_mean=-0.027, return_std=2.265, length_mean=20.08
- blackjack-a2c-survivor-seed7-1761160445: episodes=9976, return_mean=-1.660, return_std=3.117, length_mean=20.05
- blackjack-ppo-explorer-seed7-1761160326: episodes=9582, return_mean=-0.074, return_std=2.272, length_mean=20.94
- blackjack-ppo-survivor-seed7-1761160195: episodes=2156, return_mean=-1.708, return_std=2.834, length_mean=93.03
- formflow-a2c-explorer-seed7-1761161342: episodes=1349, return_mean=3.451, return_std=0.393, length_mean=148.20
- formflow-a2c-survivor-seed7-1761161125: episodes=26981, return_mean=1.119, return_std=0.036, length_mean=7.41
- formflow-ppo-explorer-seed7-1761161000: episodes=1478, return_mean=3.460, return_std=0.952, length_mean=135.79
- formflow-ppo-survivor-seed7-1761160866: episodes=24012, return_mean=1.115, return_std=0.079, length_mean=8.36


Auto-Generated Results Summary
===============================

Below are aggregated snapshots from existing runs under `runs/`, including return statistics and key domain metrics.

- blackjack-a2c-explorer-seed7-1761160653: episodes=9962, return_mean=-0.027, return_std=2.265, length_mean=20.08
- blackjack-a2c-survivor-seed7-1761160445: episodes=9976, return_mean=-1.660, return_std=3.117, length_mean=20.05
- blackjack-ppo-explorer-seed7-1761160326: episodes=9582, return_mean=-0.074, return_std=2.272, length_mean=20.94
- blackjack-ppo-explorer-seed7-1761168868: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-survivor-seed7-1761160195: episodes=2156, return_mean=-1.708, return_std=2.834, length_mean=93.03
- blackjack-ppo-survivor-seed7-1761168633: episodes=188759, return_mean=-0.346, return_std=1.979, length_mean=1.06
- formflow-a2c-explorer-seed7-1761161342: episodes=1349, return_mean=3.451, return_std=0.393, length_mean=148.20
- formflow-a2c-survivor-seed7-1761161125: episodes=26981, return_mean=1.119, return_std=0.036, length_mean=7.41
- formflow-ppo-explorer-seed7-1761161000: episodes=1478, return_mean=3.460, return_std=0.952, length_mean=135.79
- formflow-ppo-survivor-seed7-1761160866: episodes=24012, return_mean=1.115, return_std=0.079, length_mean=8.36
- formflow-ppo-survivor-seed7-1761169999: episodes=23899, return_mean=1.116, return_std=0.079, length_mean=8.40


Results & Discussion
=====================

Comparisons below use the latest run per (app, algo, persona). Metrics are summarized to support persona and algorithm trade-off discussion.

### Blackjack
- Persona: explorer
  - A2C: return_mean=-0.027, win_mean=0.019, lose_mean=0.028, draw_mean=0.003 (blackjack-a2c-explorer-seed7-1761160653)
  - PPO: return_mean=-0.081, win_mean=0.368, lose_mean=0.549, draw_mean=0.050 (blackjack-ppo-explorer-seed7-1761168868)
- Persona: survivor
  - A2C: return_mean=-1.660, win_mean=0.019, lose_mean=0.029, draw_mean=0.003 (blackjack-a2c-survivor-seed7-1761160445)
  - PPO: return_mean=-0.346, win_mean=0.370, lose_mean=0.552, draw_mean=0.050 (blackjack-ppo-survivor-seed7-1761168633)
- Algo: A2C
  - explorer: return_mean=-0.027, win_mean=0.019, lose_mean=0.028, draw_mean=0.003 (blackjack-a2c-explorer-seed7-1761160653)
  - survivor: return_mean=-1.660, win_mean=0.019, lose_mean=0.029, draw_mean=0.003 (blackjack-a2c-survivor-seed7-1761160445)
- Algo: PPO
  - explorer: return_mean=-0.081, win_mean=0.368, lose_mean=0.549, draw_mean=0.050 (blackjack-ppo-explorer-seed7-1761168868)
  - survivor: return_mean=-0.346, win_mean=0.370, lose_mean=0.552, draw_mean=0.050 (blackjack-ppo-survivor-seed7-1761168633)

### Formflow
- Persona: explorer
  - A2C: return_mean=3.451, success_mean=0.001, distinct_pages_mean=2.025, validation_errors_mean=72.277 (formflow-a2c-explorer-seed7-1761161342)
  - PPO: return_mean=3.460, success_mean=0.004, distinct_pages_mean=2.181, validation_errors_mean=61.045 (formflow-ppo-explorer-seed7-1761161000)
- Persona: survivor
  - A2C: return_mean=1.119, success_mean=0.138, distinct_pages_mean=3.237, validation_errors_mean=0.216 (formflow-a2c-survivor-seed7-1761161125)
  - PPO: return_mean=1.116, success_mean=0.131, distinct_pages_mean=3.011, validation_errors_mean=0.324 (formflow-ppo-survivor-seed7-1761169999)
- Algo: A2C
  - explorer: return_mean=3.451, success_mean=0.001, distinct_pages_mean=2.025, validation_errors_mean=72.277 (formflow-a2c-explorer-seed7-1761161342)
- survivor: return_mean=1.119, success_mean=0.138, distinct_pages_mean=3.237, validation_errors_mean=0.216 (formflow-a2c-survivor-seed7-1761161125)

Final Results (Updated Blackjack PPO 200k)
==========================================

We retrained PPO on Blackjack with hit‑favoring shaping (approach_21_bonus, early_stand_penalty, safe_hit_bonus) and evaluated deterministically (200 episodes):

- PPO Survivor — runs/blackjack-ppo-survivor-seed7-1761181439
  - action_hit_mean ≈ 0.188, action_stand_mean ≈ 0.404, length_mean ≈ 10.45
  - Interpretation: Agent now takes hits on low totals and plays out hands; no longer ends immediately with stand.

- PPO Explorer — runs/blackjack-ppo-explorer-seed7-1761181597
  - action_hit_mean ≈ 0.174, action_stand_mean ≈ 0.437, length_mean ≈ 10.77
  - Interpretation: Similar competent behavior with a slightly different risk profile; plays multi‑step hands and hits when sensible.

Short Walkthrough (What/Why/How)
--------------------------------
- What we built: a reusable DRL testing framework with decoupled envs (envs/), training/eval (src/), and configs (configs/). Two non‑trivial apps (game + web flow) share the pipeline.
- Why reward shaping: initial Blackjack agents converged to “stand immediately.” Adding approach_21_bonus, early_stand_penalty, and safe_hit_bonus nudged policies toward human‑sensible play (take safe hits; don’t stand too early).
- How we validated: deterministic 200‑episode evals with per‑episode CSVs and aggregate JSON; we inspected action flags (`action_hit_mean`, `action_stand_mean`) and hand lengths to confirm behavior.
- Viewer improvements: decision banner (HIT/STAND/DOUBLE), reveal cue (“Revealing dealer…”), agent‑obs overlay, session W/L/D. Matches the DRL outputs for transparent demos.

Exact Commands → Run Directories (Latest)
-----------------------------------------
- Blackjack PPO Survivor (200k):
  - Train: `python -m src.train --app blackjack --algo ppo --persona survivor --seed 7 --timesteps 200000`
  - Eval:  `python -m src.eval  --app blackjack --algo ppo --persona survivor --seed 7 --episodes 200 --record_gif`
  - Run:   `runs/blackjack-ppo-survivor-seed7-1761181439`
- Blackjack PPO Explorer (200k):
  - Train: `python -m src.train --app blackjack --algo ppo --persona explorer --seed 7 --timesteps 200000`
  - Eval:  `python -m src.eval  --app blackjack --algo ppo --persona explorer --seed 7 --episodes 200 --record_gif`
  - Run:   `runs/blackjack-ppo-explorer-seed7-1761181597`

Submission Checklist (Rubric)
-----------------------------
- Architecture & decoupling: envs/train/eval/configs cleanly separated.
- Agent training & competence: PPO/A2C on two apps; Blackjack agents now play multi‑step hands with hits.
- Reward design & personas: survivor/explorer with domain‑specific shaping.
- Metrics & evaluation: per‑episode CSV, aggregate JSON, learning curves, histograms, GIFs.
- Reproducibility & repo quality: pinned deps, seeds, saved artifacts, commands, and consolidated `AMAZING_REPORT.html`.
- Algo: PPO
  - explorer: return_mean=3.460, success_mean=0.004, distinct_pages_mean=2.181, validation_errors_mean=61.045 (formflow-ppo-explorer-seed7-1761161000)
  - survivor: return_mean=1.116, success_mean=0.131, distinct_pages_mean=3.011, validation_errors_mean=0.324 (formflow-ppo-survivor-seed7-1761169999)


Auto-Generated Results Summary
===============================

Below are aggregated snapshots from existing runs under `runs/`, including return statistics and key domain metrics.

- blackjack-a2c-explorer-seed7-1761160653: episodes=9962, return_mean=-0.027, return_std=2.265, length_mean=20.08
- blackjack-a2c-survivor-seed7-1761160445: episodes=9976, return_mean=-1.660, return_std=3.117, length_mean=20.05
- blackjack-ppo-explorer-seed7-1761160326: episodes=9582, return_mean=-0.074, return_std=2.272, length_mean=20.94
- blackjack-ppo-explorer-seed7-1761168868: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-survivor-seed7-1761160195: episodes=2156, return_mean=-1.708, return_std=2.834, length_mean=93.03
- blackjack-ppo-survivor-seed7-1761168633: episodes=188759, return_mean=-0.346, return_std=1.979, length_mean=1.06
- formflow-a2c-explorer-seed7-1761161342: episodes=1349, return_mean=3.451, return_std=0.393, length_mean=148.20
- formflow-a2c-survivor-seed7-1761161125: episodes=26981, return_mean=1.119, return_std=0.036, length_mean=7.41
- formflow-a2c-survivor-seed7-1761170399: episodes=25813, return_mean=1.121, return_std=0.043, length_mean=7.75
- formflow-ppo-explorer-seed7-1761161000: episodes=1478, return_mean=3.460, return_std=0.952, length_mean=135.79
- formflow-ppo-survivor-seed7-1761160866: episodes=24012, return_mean=1.115, return_std=0.079, length_mean=8.36
- formflow-ppo-survivor-seed7-1761169999: episodes=23899, return_mean=1.116, return_std=0.079, length_mean=8.40


Results & Discussion
=====================

Comparisons below use the latest run per (app, algo, persona). Metrics are summarized to support persona and algorithm trade-off discussion.

### Blackjack
- Persona: explorer
  - A2C: return_mean=-0.027, win_mean=0.019, lose_mean=0.028, draw_mean=0.003 (blackjack-a2c-explorer-seed7-1761160653)
  - PPO: return_mean=-0.081, win_mean=0.368, lose_mean=0.549, draw_mean=0.050 (blackjack-ppo-explorer-seed7-1761168868)
- Persona: survivor
  - A2C: return_mean=-1.660, win_mean=0.019, lose_mean=0.029, draw_mean=0.003 (blackjack-a2c-survivor-seed7-1761160445)
  - PPO: return_mean=-0.346, win_mean=0.370, lose_mean=0.552, draw_mean=0.050 (blackjack-ppo-survivor-seed7-1761168633)
- Algo: A2C
  - explorer: return_mean=-0.027, win_mean=0.019, lose_mean=0.028, draw_mean=0.003 (blackjack-a2c-explorer-seed7-1761160653)
  - survivor: return_mean=-1.660, win_mean=0.019, lose_mean=0.029, draw_mean=0.003 (blackjack-a2c-survivor-seed7-1761160445)
- Algo: PPO
  - explorer: return_mean=-0.081, win_mean=0.368, lose_mean=0.549, draw_mean=0.050 (blackjack-ppo-explorer-seed7-1761168868)
  - survivor: return_mean=-0.346, win_mean=0.370, lose_mean=0.552, draw_mean=0.050 (blackjack-ppo-survivor-seed7-1761168633)

### Formflow
- Persona: explorer
  - A2C: return_mean=3.451, success_mean=0.001, distinct_pages_mean=2.025, validation_errors_mean=72.277 (formflow-a2c-explorer-seed7-1761161342)
  - PPO: return_mean=3.460, success_mean=0.004, distinct_pages_mean=2.181, validation_errors_mean=61.045 (formflow-ppo-explorer-seed7-1761161000)
- Persona: survivor
  - A2C: return_mean=1.121, success_mean=0.133, distinct_pages_mean=3.181, validation_errors_mean=0.216 (formflow-a2c-survivor-seed7-1761170399)
  - PPO: return_mean=1.116, success_mean=0.131, distinct_pages_mean=3.011, validation_errors_mean=0.324 (formflow-ppo-survivor-seed7-1761169999)
- Algo: A2C
  - explorer: return_mean=3.451, success_mean=0.001, distinct_pages_mean=2.025, validation_errors_mean=72.277 (formflow-a2c-explorer-seed7-1761161342)
  - survivor: return_mean=1.121, success_mean=0.133, distinct_pages_mean=3.181, validation_errors_mean=0.216 (formflow-a2c-survivor-seed7-1761170399)
- Algo: PPO
  - explorer: return_mean=3.460, success_mean=0.004, distinct_pages_mean=2.181, validation_errors_mean=61.045 (formflow-ppo-explorer-seed7-1761161000)
  - survivor: return_mean=1.116, success_mean=0.131, distinct_pages_mean=3.011, validation_errors_mean=0.324 (formflow-ppo-survivor-seed7-1761169999)


Auto-Generated Results Summary
===============================

Below are aggregated snapshots from existing runs under `runs/`, including return statistics and key domain metrics.

- blackjack-a2c-explorer-seed7-1761160653: episodes=9962, return_mean=-0.027, return_std=2.265, length_mean=20.08
- blackjack-a2c-survivor-seed7-1761160445: episodes=9976, return_mean=-1.660, return_std=3.117, length_mean=20.05
- blackjack-a2c-survivor-seed7-1761170718: episodes=199215, return_mean=-0.332, return_std=1.974, length_mean=1.00
- blackjack-ppo-explorer-seed7-1761160326: episodes=9582, return_mean=-0.074, return_std=2.272, length_mean=20.94
- blackjack-ppo-explorer-seed7-1761168868: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-survivor-seed7-1761160195: episodes=2156, return_mean=-1.708, return_std=2.834, length_mean=93.03
- blackjack-ppo-survivor-seed7-1761168633: episodes=188759, return_mean=-0.346, return_std=1.979, length_mean=1.06
- formflow-a2c-explorer-seed7-1761161342: episodes=1349, return_mean=3.451, return_std=0.393, length_mean=148.20
- formflow-a2c-survivor-seed7-1761161125: episodes=26981, return_mean=1.119, return_std=0.036, length_mean=7.41
- formflow-a2c-survivor-seed7-1761170399: episodes=25813, return_mean=1.121, return_std=0.043, length_mean=7.75
- formflow-ppo-explorer-seed7-1761161000: episodes=1478, return_mean=3.460, return_std=0.952, length_mean=135.79
- formflow-ppo-survivor-seed7-1761160866: episodes=24012, return_mean=1.115, return_std=0.079, length_mean=8.36
- formflow-ppo-survivor-seed7-1761169999: episodes=23899, return_mean=1.116, return_std=0.079, length_mean=8.40


Results & Discussion
=====================

Comparisons below use the latest run per (app, algo, persona). Metrics are summarized to support persona and algorithm trade-off discussion.

### Blackjack
- Persona: explorer
  - A2C: return_mean=-0.027, win_mean=0.019, lose_mean=0.028, draw_mean=0.003 (blackjack-a2c-explorer-seed7-1761160653)
  - PPO: return_mean=-0.081, win_mean=0.368, lose_mean=0.549, draw_mean=0.050 (blackjack-ppo-explorer-seed7-1761168868)
- Persona: survivor
  - A2C: return_mean=-0.332, win_mean=0.381, lose_mean=0.565, draw_mean=0.052 (blackjack-a2c-survivor-seed7-1761170718)
  - PPO: return_mean=-0.346, win_mean=0.370, lose_mean=0.552, draw_mean=0.050 (blackjack-ppo-survivor-seed7-1761168633)
- Algo: A2C
  - explorer: return_mean=-0.027, win_mean=0.019, lose_mean=0.028, draw_mean=0.003 (blackjack-a2c-explorer-seed7-1761160653)
  - survivor: return_mean=-0.332, win_mean=0.381, lose_mean=0.565, draw_mean=0.052 (blackjack-a2c-survivor-seed7-1761170718)
- Algo: PPO
  - explorer: return_mean=-0.081, win_mean=0.368, lose_mean=0.549, draw_mean=0.050 (blackjack-ppo-explorer-seed7-1761168868)
  - survivor: return_mean=-0.346, win_mean=0.370, lose_mean=0.552, draw_mean=0.050 (blackjack-ppo-survivor-seed7-1761168633)

### Formflow
- Persona: explorer
  - A2C: return_mean=3.451, success_mean=0.001, distinct_pages_mean=2.025, validation_errors_mean=72.277 (formflow-a2c-explorer-seed7-1761161342)
  - PPO: return_mean=3.460, success_mean=0.004, distinct_pages_mean=2.181, validation_errors_mean=61.045 (formflow-ppo-explorer-seed7-1761161000)
- Persona: survivor
  - A2C: return_mean=1.121, success_mean=0.133, distinct_pages_mean=3.181, validation_errors_mean=0.216 (formflow-a2c-survivor-seed7-1761170399)
  - PPO: return_mean=1.116, success_mean=0.131, distinct_pages_mean=3.011, validation_errors_mean=0.324 (formflow-ppo-survivor-seed7-1761169999)
- Algo: A2C
  - explorer: return_mean=3.451, success_mean=0.001, distinct_pages_mean=2.025, validation_errors_mean=72.277 (formflow-a2c-explorer-seed7-1761161342)
  - survivor: return_mean=1.121, success_mean=0.133, distinct_pages_mean=3.181, validation_errors_mean=0.216 (formflow-a2c-survivor-seed7-1761170399)
- Algo: PPO
  - explorer: return_mean=3.460, success_mean=0.004, distinct_pages_mean=2.181, validation_errors_mean=61.045 (formflow-ppo-explorer-seed7-1761161000)
  - survivor: return_mean=1.116, success_mean=0.131, distinct_pages_mean=3.011, validation_errors_mean=0.324 (formflow-ppo-survivor-seed7-1761169999)


Auto-Generated Results Summary
===============================

Below are aggregated snapshots from existing runs under `runs/`, including return statistics and key domain metrics.

- blackjack-a2c-explorer-seed7-1761160653: episodes=9962, return_mean=-0.027, return_std=2.265, length_mean=20.08
- blackjack-a2c-survivor-seed7-1761160445: episodes=9976, return_mean=-1.660, return_std=3.117, length_mean=20.05
- blackjack-a2c-survivor-seed7-1761170718: episodes=199215, return_mean=-0.332, return_std=1.974, length_mean=1.00
- blackjack-ppo-explorer-seed7-1761160326: episodes=9582, return_mean=-0.074, return_std=2.272, length_mean=20.94
- blackjack-ppo-explorer-seed7-1761168868: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-explorer-seed7-1761171791: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-survivor-seed7-1761160195: episodes=2156, return_mean=-1.708, return_std=2.834, length_mean=93.03
- blackjack-ppo-survivor-seed7-1761168633: episodes=188759, return_mean=-0.346, return_std=1.979, length_mean=1.06
- formflow-a2c-explorer-seed7-1761161342: episodes=1349, return_mean=3.451, return_std=0.393, length_mean=148.20
- formflow-a2c-survivor-seed7-1761161125: episodes=26981, return_mean=1.119, return_std=0.036, length_mean=7.41
- formflow-a2c-survivor-seed7-1761170399: episodes=25813, return_mean=1.121, return_std=0.043, length_mean=7.75
- formflow-ppo-explorer-seed7-1761161000: episodes=1478, return_mean=3.460, return_std=0.952, length_mean=135.79
- formflow-ppo-explorer-seed7-1761171477: episodes=1487, return_mean=3.454, return_std=0.964, length_mean=134.95
- formflow-ppo-survivor-seed7-1761160866: episodes=24012, return_mean=1.115, return_std=0.079, length_mean=8.36
- formflow-ppo-survivor-seed7-1761169999: episodes=23899, return_mean=1.116, return_std=0.079, length_mean=8.40


Results & Discussion
=====================

Comparisons below use the latest run per (app, algo, persona). Metrics are summarized to support persona and algorithm trade-off discussion.

### Blackjack
- Persona: explorer
  - A2C: return_mean=-0.027, win_mean=0.019, lose_mean=0.028, draw_mean=0.003 (blackjack-a2c-explorer-seed7-1761160653)
  - PPO: return_mean=-0.081, win_mean=0.368, lose_mean=0.549, draw_mean=0.050 (blackjack-ppo-explorer-seed7-1761171791)
- Persona: survivor
  - A2C: return_mean=-0.332, win_mean=0.381, lose_mean=0.565, draw_mean=0.052 (blackjack-a2c-survivor-seed7-1761170718)
  - PPO: return_mean=-0.346, win_mean=0.370, lose_mean=0.552, draw_mean=0.050 (blackjack-ppo-survivor-seed7-1761168633)
- Algo: A2C
  - explorer: return_mean=-0.027, win_mean=0.019, lose_mean=0.028, draw_mean=0.003 (blackjack-a2c-explorer-seed7-1761160653)
  - survivor: return_mean=-0.332, win_mean=0.381, lose_mean=0.565, draw_mean=0.052 (blackjack-a2c-survivor-seed7-1761170718)
- Algo: PPO
  - explorer: return_mean=-0.081, win_mean=0.368, lose_mean=0.549, draw_mean=0.050 (blackjack-ppo-explorer-seed7-1761171791)
  - survivor: return_mean=-0.346, win_mean=0.370, lose_mean=0.552, draw_mean=0.050 (blackjack-ppo-survivor-seed7-1761168633)

### Formflow
- Persona: explorer
  - A2C: return_mean=3.451, success_mean=0.001, distinct_pages_mean=2.025, validation_errors_mean=72.277 (formflow-a2c-explorer-seed7-1761161342)
  - PPO: return_mean=3.454, success_mean=0.004, distinct_pages_mean=2.174, validation_errors_mean=59.853 (formflow-ppo-explorer-seed7-1761171477)
- Persona: survivor
  - A2C: return_mean=1.121, success_mean=0.133, distinct_pages_mean=3.181, validation_errors_mean=0.216 (formflow-a2c-survivor-seed7-1761170399)
  - PPO: return_mean=1.116, success_mean=0.131, distinct_pages_mean=3.011, validation_errors_mean=0.324 (formflow-ppo-survivor-seed7-1761169999)
- Algo: A2C
  - explorer: return_mean=3.451, success_mean=0.001, distinct_pages_mean=2.025, validation_errors_mean=72.277 (formflow-a2c-explorer-seed7-1761161342)
  - survivor: return_mean=1.121, success_mean=0.133, distinct_pages_mean=3.181, validation_errors_mean=0.216 (formflow-a2c-survivor-seed7-1761170399)
- Algo: PPO
  - explorer: return_mean=3.454, success_mean=0.004, distinct_pages_mean=2.174, validation_errors_mean=59.853 (formflow-ppo-explorer-seed7-1761171477)
  - survivor: return_mean=1.116, success_mean=0.131, distinct_pages_mean=3.011, validation_errors_mean=0.324 (formflow-ppo-survivor-seed7-1761169999)


Auto-Generated Results Summary
===============================

Below are aggregated snapshots from existing runs under `runs/`, including return statistics and key domain metrics.

- blackjack-a2c-explorer-seed7-1761160653: episodes=9962, return_mean=-0.027, return_std=2.265, length_mean=20.08
- blackjack-a2c-survivor-seed7-1761160445: episodes=9976, return_mean=-1.660, return_std=3.117, length_mean=20.05
- blackjack-a2c-survivor-seed7-1761170718: episodes=199215, return_mean=-0.332, return_std=1.974, length_mean=1.00
- blackjack-ppo-explorer-seed7-1761160326: episodes=9582, return_mean=-0.074, return_std=2.272, length_mean=20.94
- blackjack-ppo-explorer-seed7-1761168868: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-explorer-seed7-1761171791: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-explorer-seed7-1761178006: episodes=91407, return_mean=0.138, return_std=1.407, length_mean=1.10
- blackjack-ppo-survivor-seed7-1761160195: episodes=2156, return_mean=-1.708, return_std=2.834, length_mean=93.03
- blackjack-ppo-survivor-seed7-1761168633: episodes=188759, return_mean=-0.346, return_std=1.979, length_mean=1.06
- blackjack-ppo-survivor-seed7-1761177855: episodes=90201, return_mean=0.303, return_std=2.233, length_mean=1.11
- formflow-a2c-explorer-seed7-1761161342: episodes=1349, return_mean=3.451, return_std=0.393, length_mean=148.20
- formflow-a2c-survivor-seed7-1761161125: episodes=26981, return_mean=1.119, return_std=0.036, length_mean=7.41
- formflow-a2c-survivor-seed7-1761170399: episodes=25813, return_mean=1.121, return_std=0.043, length_mean=7.75
- formflow-ppo-explorer-seed7-1761161000: episodes=1478, return_mean=3.460, return_std=0.952, length_mean=135.79
- formflow-ppo-explorer-seed7-1761171477: episodes=1487, return_mean=3.454, return_std=0.964, length_mean=134.95
- formflow-ppo-survivor-seed7-1761160866: episodes=24012, return_mean=1.115, return_std=0.079, length_mean=8.36
- formflow-ppo-survivor-seed7-1761169999: episodes=23899, return_mean=1.116, return_std=0.079, length_mean=8.40


Auto-Generated Results Summary
===============================

Below are aggregated snapshots from existing runs under `runs/`, including return statistics and key domain metrics.

- blackjack-a2c-explorer-seed7-1761160653: episodes=9962, return_mean=-0.027, return_std=2.265, length_mean=20.08
- blackjack-a2c-survivor-seed7-1761160445: episodes=9976, return_mean=-1.660, return_std=3.117, length_mean=20.05
- blackjack-a2c-survivor-seed7-1761170718: episodes=199215, return_mean=-0.332, return_std=1.974, length_mean=1.00
- blackjack-ppo-explorer-seed7-1761160326: episodes=9582, return_mean=-0.074, return_std=2.272, length_mean=20.94
- blackjack-ppo-explorer-seed7-1761168868: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-explorer-seed7-1761171791: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-explorer-seed7-1761178006: episodes=91407, return_mean=0.138, return_std=1.407, length_mean=1.10
- blackjack-ppo-explorer-seed7-1761179344: episodes=41069, return_mean=0.110, return_std=1.407, length_mean=1.25
- blackjack-ppo-explorer-seed7-1761181212: episodes=125329, return_mean=0.617, return_std=2.011, length_mean=1.60
- blackjack-ppo-explorer-seed7-1761181597: episodes=125329, return_mean=0.617, return_std=2.011, length_mean=1.60
- blackjack-ppo-survivor-seed7-1761160195: episodes=2156, return_mean=-1.708, return_std=2.834, length_mean=93.03
- blackjack-ppo-survivor-seed7-1761168633: episodes=188759, return_mean=-0.346, return_std=1.979, length_mean=1.06
- blackjack-ppo-survivor-seed7-1761177855: episodes=90201, return_mean=0.303, return_std=2.233, length_mean=1.11
- blackjack-ppo-survivor-seed7-1761181043: episodes=61972, return_mean=0.802, return_std=2.599, length_mean=1.32
- blackjack-ppo-survivor-seed7-1761181331: episodes=139678, return_mean=0.849, return_std=2.608, length_mean=1.44
- blackjack-ppo-survivor-seed7-1761181439: episodes=139678, return_mean=0.849, return_std=2.608, length_mean=1.44
- formflow-a2c-explorer-seed7-1761161342: episodes=1349, return_mean=3.451, return_std=0.393, length_mean=148.20
- formflow-a2c-survivor-seed7-1761161125: episodes=26981, return_mean=1.119, return_std=0.036, length_mean=7.41
- formflow-a2c-survivor-seed7-1761170399: episodes=25813, return_mean=1.121, return_std=0.043, length_mean=7.75
- formflow-ppo-explorer-seed7-1761161000: episodes=1478, return_mean=3.460, return_std=0.952, length_mean=135.79
- formflow-ppo-explorer-seed7-1761171477: episodes=1487, return_mean=3.454, return_std=0.964, length_mean=134.95
- formflow-ppo-survivor-seed7-1761160866: episodes=24012, return_mean=1.115, return_std=0.079, length_mean=8.36
- formflow-ppo-survivor-seed7-1761169999: episodes=23899, return_mean=1.116, return_std=0.079, length_mean=8.40


Auto-Generated Results Summary
===============================

Below are aggregated snapshots from existing runs under `runs/`, including return statistics and key domain metrics.

- blackjack-a2c-explorer-seed7-1761160653: episodes=9962, return_mean=-0.027, return_std=2.265, length_mean=20.08
- blackjack-a2c-explorer-seed7-1761183119: episodes=199886, return_mean=0.565, return_std=2.011, length_mean=1.00
- blackjack-a2c-survivor-seed7-1761160445: episodes=9976, return_mean=-1.660, return_std=3.117, length_mean=20.05
- blackjack-a2c-survivor-seed7-1761170718: episodes=199215, return_mean=-0.332, return_std=1.974, length_mean=1.00
- blackjack-ppo-explorer-seed7-1761160326: episodes=9582, return_mean=-0.074, return_std=2.272, length_mean=20.94
- blackjack-ppo-explorer-seed7-1761168868: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-explorer-seed7-1761171791: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-explorer-seed7-1761178006: episodes=91407, return_mean=0.138, return_std=1.407, length_mean=1.10
- blackjack-ppo-explorer-seed7-1761179344: episodes=41069, return_mean=0.110, return_std=1.407, length_mean=1.25
- blackjack-ppo-explorer-seed7-1761181212: episodes=125329, return_mean=0.617, return_std=2.011, length_mean=1.60
- blackjack-ppo-explorer-seed7-1761181597: episodes=125329, return_mean=0.617, return_std=2.011, length_mean=1.60
- blackjack-ppo-survivor-seed7-1761160195: episodes=2156, return_mean=-1.708, return_std=2.834, length_mean=93.03
- blackjack-ppo-survivor-seed7-1761168633: episodes=188759, return_mean=-0.346, return_std=1.979, length_mean=1.06
- blackjack-ppo-survivor-seed7-1761177855: episodes=90201, return_mean=0.303, return_std=2.233, length_mean=1.11
- blackjack-ppo-survivor-seed7-1761181043: episodes=61972, return_mean=0.802, return_std=2.599, length_mean=1.32
- blackjack-ppo-survivor-seed7-1761181331: episodes=139678, return_mean=0.849, return_std=2.608, length_mean=1.44
- blackjack-ppo-survivor-seed7-1761181439: episodes=139678, return_mean=0.849, return_std=2.608, length_mean=1.44
- formflow-a2c-explorer-seed7-1761161342: episodes=1349, return_mean=3.451, return_std=0.393, length_mean=148.20
- formflow-a2c-survivor-seed7-1761161125: episodes=26981, return_mean=1.119, return_std=0.036, length_mean=7.41
- formflow-a2c-survivor-seed7-1761170399: episodes=25813, return_mean=1.121, return_std=0.043, length_mean=7.75
- formflow-ppo-explorer-seed7-1761161000: episodes=1478, return_mean=3.460, return_std=0.952, length_mean=135.79
- formflow-ppo-explorer-seed7-1761171477: episodes=1487, return_mean=3.454, return_std=0.964, length_mean=134.95
- formflow-ppo-survivor-seed7-1761160866: episodes=24012, return_mean=1.115, return_std=0.079, length_mean=8.36
- formflow-ppo-survivor-seed7-1761169999: episodes=23899, return_mean=1.116, return_std=0.079, length_mean=8.40


Auto-Generated Results Summary
===============================

Below are aggregated snapshots from existing runs under `runs/`, including return statistics and key domain metrics.

- blackjack-a2c-explorer-seed7-1761160653: episodes=9962, return_mean=-0.027, return_std=2.265, length_mean=20.08
- blackjack-a2c-explorer-seed7-1761183119: episodes=199886, return_mean=0.565, return_std=2.011, length_mean=1.00
- blackjack-a2c-survivor-seed7-1761160445: episodes=9976, return_mean=-1.660, return_std=3.117, length_mean=20.05
- blackjack-a2c-survivor-seed7-1761170718: episodes=199215, return_mean=-0.332, return_std=1.974, length_mean=1.00
- blackjack-ppo-explorer-seed7-1761160326: episodes=9582, return_mean=-0.074, return_std=2.272, length_mean=20.94
- blackjack-ppo-explorer-seed7-1761168868: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-explorer-seed7-1761171791: episodes=186437, return_mean=-0.081, return_std=1.104, length_mean=1.08
- blackjack-ppo-explorer-seed7-1761178006: episodes=91407, return_mean=0.138, return_std=1.407, length_mean=1.10
- blackjack-ppo-explorer-seed7-1761179344: episodes=41069, return_mean=0.110, return_std=1.407, length_mean=1.25
- blackjack-ppo-explorer-seed7-1761181212: episodes=125329, return_mean=0.617, return_std=2.011, length_mean=1.60
- blackjack-ppo-explorer-seed7-1761181597: episodes=125329, return_mean=0.617, return_std=2.011, length_mean=1.60
- blackjack-ppo-survivor-seed7-1761160195: episodes=2156, return_mean=-1.708, return_std=2.834, length_mean=93.03
- blackjack-ppo-survivor-seed7-1761168633: episodes=188759, return_mean=-0.346, return_std=1.979, length_mean=1.06
- blackjack-ppo-survivor-seed7-1761177855: episodes=90201, return_mean=0.303, return_std=2.233, length_mean=1.11
- blackjack-ppo-survivor-seed7-1761181043: episodes=61972, return_mean=0.802, return_std=2.599, length_mean=1.32
- blackjack-ppo-survivor-seed7-1761181331: episodes=139678, return_mean=0.849, return_std=2.608, length_mean=1.44
- blackjack-ppo-survivor-seed7-1761181439: episodes=139678, return_mean=0.849, return_std=2.608, length_mean=1.44
- formflow-a2c-explorer-seed7-1761161342: episodes=1349, return_mean=3.451, return_std=0.393, length_mean=148.20
- formflow-a2c-survivor-seed7-1761161125: episodes=26981, return_mean=1.119, return_std=0.036, length_mean=7.41
- formflow-a2c-survivor-seed7-1761170399: episodes=25813, return_mean=1.121, return_std=0.043, length_mean=7.75
- formflow-ppo-explorer-seed7-1761161000: episodes=1478, return_mean=3.460, return_std=0.952, length_mean=135.79
- formflow-ppo-explorer-seed7-1761171477: episodes=1487, return_mean=3.454, return_std=0.964, length_mean=134.95
- formflow-ppo-survivor-seed7-1761160866: episodes=24012, return_mean=1.115, return_std=0.079, length_mean=8.36
- formflow-ppo-survivor-seed7-1761169999: episodes=23899, return_mean=1.116, return_std=0.079, length_mean=8.40
