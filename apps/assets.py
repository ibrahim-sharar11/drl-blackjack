import os
import pygame


ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')

_img_cache = {}
_snd_cache = {}


def _try_load(path, convert=True):
    if not os.path.isfile(path):
        return None
    img = pygame.image.load(path)
    return img.convert_alpha() if convert else img


def _try_sound(path):
    if not os.path.isfile(path):
        return None
    try:
        return pygame.mixer.Sound(path)
    except Exception:
        return None


def card_image(rank: str, suit: str):
    """
    Try to resolve a card sprite for rank in {A,2..10,J,Q,K} and suit in {spades,hearts,diamonds,clubs}.
    Searches common naming patterns (Kenney, generic) under assets/cards/.
    Returns a pygame.Surface or None.
    """
    key = ('card', rank, suit)
    if key in _img_cache:
        return _img_cache[key]

    candidates = []
    base = os.path.join(ASSET_DIR, 'cards')
    suit_cap = suit.capitalize()
    suit_title = suit.capitalize()
    # Kenney: cardSpadesA.png, cardHearts10.png, etc.
    candidates.append(os.path.join(base, f"card{suit_title}{rank}.png"))
    # Alternative: rank_of_suit.png
    candidates.append(os.path.join(base, f"{rank}_of_{suit}.png"))
    # Short: AH.png, 10S.png
    suit_short = suit[0].upper()
    candidates.append(os.path.join(base, f"{rank}{suit_short}.png"))
    # Lowercase variants
    candidates.append(os.path.join(base, f"card{suit_title}{rank}.PNG"))
    candidates.append(os.path.join(base, f"{rank}_of_{suit}.PNG"))
    candidates.append(os.path.join(base, f"{rank}{suit_short}.PNG"))

    surf = None
    for p in candidates:
        surf = _try_load(p)
        if surf:
            break
    _img_cache[key] = surf
    return surf


def card_back_image():
    key = ('card_back',)
    if key in _img_cache:
        return _img_cache[key]
    base = os.path.join(ASSET_DIR, 'cards')
    candidates = [
        os.path.join(base, 'cardBack_blue2.png'),  # Kenney
        os.path.join(base, 'cardBack_red2.png'),
        os.path.join(base, 'back.png'),
        os.path.join(base, 'back_blue.png'),
    ]
    surf = None
    for p in candidates:
        surf = _try_load(p)
        if surf:
            break
    _img_cache[key] = surf
    return surf


def table_image():
    key = ('table',)
    if key in _img_cache:
        return _img_cache[key]
    base = os.path.join(ASSET_DIR, 'table')
    candidates = [
        os.path.join(base, 'table.png'),
        os.path.join(base, 'felt.png'),
        os.path.join(base, 'green_felt.png'),
    ]
    surf = None
    for p in candidates:
        surf = _try_load(p, convert=True)
        if surf:
            break
    _img_cache[key] = surf
    return surf


def chip_images():
    key = ('chips',)
    if key in _img_cache:
        return _img_cache[key]
    chips_dir = os.path.join(ASSET_DIR, 'chips')
    if not os.path.isdir(chips_dir):
        _img_cache[key] = []
        return []
    files = [f for f in os.listdir(chips_dir) if f.lower().endswith('.png')]
    files.sort()
    surfs = []
    for f in files:
        s = _try_load(os.path.join(chips_dir, f))
        if s:
            surfs.append(s)
    _img_cache[key] = surfs
    return surfs


def sound(name: str):
    key = ('sound', name)
    if key in _snd_cache:
        return _snd_cache[key]
    base = os.path.join(ASSET_DIR, 'sounds')
    candidates = [
        os.path.join(base, f"{name}.wav"),
        os.path.join(base, f"{name}.ogg"),
        os.path.join(base, f"{name}.mp3"),
    ]
    s = None
    for p in candidates:
        s = _try_sound(p)
        if s:
            break
    _snd_cache[key] = s
    return s

