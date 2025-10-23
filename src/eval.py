import os, argparse
try:
    import imageio  # optional; if missing we fall back to PNG frames
    HAS_IMAGEIO = True
except Exception:
    imageio = None
    HAS_IMAGEIO = False
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from src.utils import load_configs, set_global_seeds
from src.make_env import make_env
from src.metrics import EpisodeLogger, aggregate_csv

ALGOS = {"ppo": PPO, "a2c": A2C}

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["ppo","a2c"])
    p.add_argument("--app", required=True, choices=["minigrid","formflow","tetris","blackjack"])
    p.add_argument("--persona", required=True, choices=["survivor","explorer","speedrunner"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--run_subdir", default=None)
    p.add_argument("--record_gif", action="store_true", help="Record per-episode GIFs (MiniGrid only)")
    return p.parse_args()

def main():
    args = parse()
    cfg = load_configs(app=args.app, algo=args.algo, persona=args.persona)
    set_global_seeds(args.seed)
    if args.run_subdir is None:
        prefix = f"{args.app}-{args.algo}-{args.persona}-seed{args.seed}"
        candidates = [d for d in os.listdir(args.runs_dir) if d.startswith(prefix)]
        if not candidates:
            raise FileNotFoundError("No matching runs found.")
        candidates.sort()
        run_dir = os.path.join(args.runs_dir, candidates[-1])
    else:
        run_dir = os.path.join(args.runs_dir, args.run_subdir)
    model_path = os.path.join(run_dir, "model.zip")
    # Enable rgb_array rendering for MiniGrid when recording GIFs
    if args.record_gif and args.app == "minigrid":
        cfg["app"]["render_mode"] = "rgb_array"
    env = make_env(cfg["app"], cfg["persona"])
    env = Monitor(env)
    venv = DummyVecEnv([lambda: env])
    Algo = ALGOS[args.algo]
    model = Algo.load(model_path, env=venv)
    logger = EpisodeLogger(os.path.join(run_dir, "eval"))
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        frames = []
        if args.record_gif:
            if args.app == "minigrid":
                frame = venv.envs[0].base_env.render()
            else:
                # try to get render from wrapped env
                inner = getattr(venv.envs[0], "env", venv.envs[0])
                frame = getattr(inner, "render", lambda: None)()
            if frame is not None:
                frames.append(frame)
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            logger.locals = {"infos": infos, "rewards": reward, "dones": dones}
            logger._on_step()
            if args.record_gif:
                if args.app == "minigrid":
                    frame = venv.envs[0].base_env.render()
                else:
                    inner = getattr(venv.envs[0], "env", venv.envs[0])
                    frame = getattr(inner, "render", lambda: None)()
                if frame is not None:
                    frames.append(frame)
            done = bool(dones[0])
        if args.record_gif and frames:
            if HAS_IMAGEIO:
                out_gif = os.path.join(eval_dir, f"episode_{ep+1}.gif")
                imageio.mimsave(out_gif, frames, fps=10)
            else:
                # Fallback: save PNG frames
                for i, fr in enumerate(frames, start=1):
                    out_png = os.path.join(eval_dir, f"episode_{ep+1}_frame_{i:03d}.png")
                    try:
                        from matplotlib import pyplot as plt
                        plt.imsave(out_png, fr)
                    except Exception:
                        pass
    ep_csv = os.path.join(run_dir, "eval", "episodes.csv")
    if os.path.exists(ep_csv):
        aggregate_csv(ep_csv, os.path.join(run_dir, "eval", "aggregate.json"))
    print("Evaluated:", run_dir)

if __name__ == "__main__":
    main()
