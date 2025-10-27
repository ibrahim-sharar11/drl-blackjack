"""
Environment factory with lazy imports so missing optional envs
don’t break unrelated apps. Only import the specific env when used.
"""

from envs.formflow_env import FormFlowEnv
from envs.blackjack_env import BlackjackEnv

def make_env(app_cfg, persona_cfg):
    weights = persona_cfg["weights"]
    app_id = app_cfg["id"]
    if app_id == "minigrid":
        # Lazy import to avoid hard dependency when not using MiniGrid
        from envs.minigrid_env import MiniGridWrapper
        return MiniGridWrapper(
            env_id=app_cfg.get("env_id", "MiniGrid-Empty-8x8-v0"),
            max_steps=app_cfg.get("max_steps", 200),
            reward_weights=weights,
            seed=app_cfg.get("seed", 7),
            obs_mode=app_cfg.get("obs_mode", "partial"),
            reward_scale=app_cfg.get("reward_scale", 1.0),
            render_mode=app_cfg.get("render_mode", None),
        )
    elif app_id == "formflow":
        return FormFlowEnv(
            max_steps=app_cfg.get("max_steps", 150),
            seed=app_cfg.get("seed", 7),
            reward_weights=weights,
            reward_scale=app_cfg.get("reward_scale", 1.0),
            invalid_prob=app_cfg.get("invalid_prob", 0.2),
            latency_spike_prob=app_cfg.get("latency_spike_prob", 0.05),
        )
    elif app_id == "tetris":
        # Lazy import to avoid hard dependency when not using Tetris
        from envs.tetris_env import TetrisEnv
        return TetrisEnv(
            max_steps=app_cfg.get("max_steps", 2000),
            seed=app_cfg.get("seed", 7),
            reward_weights=weights,
            reward_scale=app_cfg.get("reward_scale", 1.0),
            target_lines=app_cfg.get("target_lines", 10),
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
