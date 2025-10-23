import os, json, random, yaml, numpy as np

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_configs(app, algo, persona):
    cfg = {}
    cfg['app'] = load_yaml(f"configs/app/{app}.yaml")
    cfg['algo'] = load_yaml(f"configs/algo/{algo}.yaml")
    cfg['persona'] = load_yaml(f"configs/persona/{persona}.yaml")
    return cfg

def set_global_seeds(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
