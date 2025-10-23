import os, argparse, json, time
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from src.utils import load_configs, set_global_seeds
from src.make_env import make_env
from src.metrics import EpisodeLogger, aggregate_csv

ALGOS = {"ppo": PPO, "a2c": A2C}

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ppo", choices=["ppo","a2c"])
    p.add_argument("--app", default="minigrid", choices=["minigrid","formflow","tetris","blackjack"])
    p.add_argument("--persona", default="survivor", choices=["survivor","explorer","speedrunner"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="runs")
    p.add_argument("--timesteps", type=int, default=None, help="Override timesteps from config for quick runs")
    return p.parse_args()

def main():
    args = parse()
    cfg = load_configs(app=args.app, algo=args.algo, persona=args.persona)
    set_global_seeds(args.seed)
    run_id = f"{args.app}-{args.algo}-{args.persona}-seed{args.seed}-{int(time.time())}"
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(out_dir, exist_ok=True)
    env = make_env(cfg["app"], cfg["persona"])
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    Algo = ALGOS[args.algo]
    policy = cfg["algo"].get("policy","MlpPolicy")
    kwargs = {k:v for k,v in cfg["algo"].items() if k not in ["name","timesteps","policy"]}
    model = Algo(policy, env, seed=args.seed, verbose=1, **kwargs)
    cb = EpisodeLogger(out_dir)
    total_ts = int(cfg["algo"]["timesteps"]) if args.timesteps is None else int(args.timesteps)
    model.learn(total_timesteps=total_ts, callback=cb, progress_bar=True)
    model.save(os.path.join(out_dir, "model"))
    # Write a JSON snapshot of merged configs; coerce non-serializable values to strings
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
    ep_csv = os.path.join(out_dir, "episodes.csv")
    if os.path.exists(ep_csv):
        aggregate_csv(ep_csv, os.path.join(out_dir, "aggregate.json"))
    print("Saved to", out_dir)

if __name__ == "__main__":
    main()
