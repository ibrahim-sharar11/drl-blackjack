import argparse
from pathlib import Path

def annotate_blackjack(img_path: Path, out_path: Path):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    W, H = img.size
    pad = 10
    # Semi-transparent panels with text
    def panel(x, y, text):
        box_w = min(420, int(W*0.6))
        box_h = 22 * len(text)
        overlay = Image.new('RGBA', (box_w, box_h), (0, 0, 0, 140))
        img.paste(overlay, (x, y), overlay)
        y_text = y + 6
        for line in text:
            draw.text((x + 8, y_text), line, fill=(250, 230, 90))
            y_text += 20
    panel(pad, pad, [
        'Blackjack Legend:',
        '- Top row: Dealer cards (first upcard visible; hole flips after STAND)',
        '- Bottom row: Player cards; appears when the agent HITS',
        '- Blue/Green bars: simple sum indicators',
        '- GIF ends when the hand resolves (Win/Lose/Push)'
    ])
    img.save(out_path)
    return True

def annotate_formflow(img_path: Path, out_path: Path):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    W, H = img.size
    pad = 10
    def panel(x, y, text):
        box_w = min(480, int(W*0.8))
        box_h = 22 * len(text)
        from PIL import Image as PILImage
        overlay = PILImage.new('RGBA', (box_w, box_h), (0, 0, 0, 140))
        img.paste(overlay, (x, y), overlay)
        y_text = y + 6
        for line in text:
            draw.text((x + 8, y_text), line, fill=(250, 230, 90))
            y_text += 20
    panel(pad, pad, [
        'FormFlow Legend:',
        '- Top bars: current page highlight as agent navigates',
        '- Squares: field_filled / field_valid / checkbox (red/green)',
        '- Yellow bar: latency bucket; Red bar: validation errors',
        '- GIF ends on submit (success) or time limit'
    ])
    img.save(out_path)
    return True

def first_frame_path(eval_dir: Path):
    # Prefer PNG frame if present; else use episode_1.gif
    frames = sorted(eval_dir.glob('episode_*_frame_*.png'))
    if frames:
        return frames[0]
    gif = eval_dir / 'episode_1.gif'
    if gif.exists():
        return gif
    return None

def main():
    ap = argparse.ArgumentParser(description='Generate annotated legend images for eval GIFs')
    ap.add_argument('--runs_dir', default='runs')
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue
        app = run.name.split('-')[0]
        eval_dir = run / 'eval'
        if not eval_dir.exists():
            continue
        src = first_frame_path(eval_dir)
        if not src:
            continue
        out = eval_dir / 'legend.png'
        ok = False
        if app == 'blackjack':
            ok = annotate_blackjack(src, out)
        elif app == 'formflow':
            ok = annotate_formflow(src, out)
        if ok:
            print(f'[OK] {out}')

if __name__ == '__main__':
    main()

