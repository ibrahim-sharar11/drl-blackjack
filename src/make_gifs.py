import argparse
import os
from pathlib import Path

def find_eval_dirs(runs_dir: Path):
    for run in runs_dir.glob('*'):
        if (run / 'eval').is_dir():
            yield run / 'eval'

def group_frames(eval_dir: Path, pattern: str):
    frames = sorted(eval_dir.glob(pattern))
    buckets = {}
    for p in frames:
        # episode_1_frame_001.png -> key by episode_1
        name = p.stem
        if '_frame_' in name:
            key = name.split('_frame_')[0]
        else:
            key = name
        buckets.setdefault(key, []).append(p)
    return buckets

def save_gif(frames, out_path: Path, fps: int):
    try:
        import imageio
    except Exception as e:
        print(f"[WARN] imageio not available; cannot write GIF: {out_path}")
        print("       Install via: pip install imageio")
        return False
    images = []
    for f in frames:
        try:
            images.append(imageio.v2.imread(f))
        except Exception as e:
            print(f"[WARN] cannot read frame {f}: {e}")
    if not images:
        print(f"[WARN] no readable frames for {out_path}")
        return False
    imageio.mimsave(out_path, images, fps=fps)
    return True

def main():
    ap = argparse.ArgumentParser(description='Batch convert eval PNG frames to GIFs under runs/*/eval')
    ap.add_argument('--runs_dir', default='runs')
    ap.add_argument('--pattern', default='episode_*_frame_*.png')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise SystemExit(f"Runs dir not found: {runs_dir}")

    converted = 0
    for ed in find_eval_dirs(runs_dir):
        buckets = group_frames(ed, args.pattern)
        for key, files in buckets.items():
            out_gif = ed / f"{key}.gif"
            if out_gif.exists() and not args.overwrite:
                continue
            ok = save_gif(files, out_gif, args.fps)
            if ok:
                converted += 1
                print(f"[OK] {out_gif}")
    print(f"Converted {converted} episode(s) to GIFs.")

if __name__ == '__main__':
    main()

