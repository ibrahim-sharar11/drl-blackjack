import os, csv, json
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict

class EpisodeLogger(BaseCallback):
    def __init__(self, out_dir, verbose=0):
        super().__init__(verbose)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.episode = 0
        self.ep_rewards = 0.0
        self.ep_len = 0
        self.csv_path = os.path.join(out_dir, "episodes.csv")
        self.fieldnames = None
        self.buffer_infos = defaultdict(list)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        if rewards is not None:
            self.ep_rewards += float(rewards)
            self.ep_len += 1
        if infos:
            last_info = infos[0]
            for k,v in last_info.items():
                self.buffer_infos[k].append(v)
        if dones is not None and bool(dones):
            agg = {}
            for k,v in self.buffer_infos.items():
                try:
                    if v and isinstance(v[0], (int,float)):
                        agg[k] = sum(v)/len(v)
                    else:
                        agg[k] = v[-1] if v else None
                except Exception:
                    agg[k] = None
            row = {"episode": self.episode, "return": self.ep_rewards, "length": self.ep_len}
            row.update(agg)
            if self.fieldnames is None:
                self.fieldnames = list(row.keys())
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=self.fieldnames)
                    w.writeheader()
                    w.writerow(row)
            else:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=self.fieldnames)
                    w.writerow(row)
            self.episode += 1
            self.ep_rewards = 0.0
            self.ep_len = 0
            self.buffer_infos.clear()
        return True

def aggregate_csv(csv_path, out_json):
    import pandas as pd
    df = pd.read_csv(csv_path)
    agg = {
        "episodes": int(len(df)),
        "return_mean": float(df["return"].mean()),
        "return_std": float(df["return"].std()),
        "length_mean": float(df["length"].mean()),
    }
    for col in df.columns:
        if col not in ["episode", "return", "length"] and pd.api.types.is_numeric_dtype(df[col]):
            agg[col+"_mean"] = float(df[col].mean())
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    return agg
