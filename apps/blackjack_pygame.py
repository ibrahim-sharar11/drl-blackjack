import os
import sys
import time
import math
import argparse
from typing import Optional
import pygame
from apps import assets
from src.utils import load_configs
from src.make_env import make_env


def lerp(a, b, t):
    return a + (b - a) * t


class BlackjackViewer:
    def __init__(self, env, fps=60):
        self.env = env
        self.fps = fps
        pygame.init()
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
        except Exception:
            pass
        # Window
        self.W, self.H = 900, 600
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Blackjack Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20)
        self.bigfont = pygame.font.SysFont("arial", 28, bold=True)
        # Layout
        self.table_color = (32, 96, 64)
        self.deck_pos = (self.W - 120, self.H // 2 - 40)
        self.dealer_y = 120
        self.player_y = 360
        self.card_w, self.card_h = 80, 120
        self.spacing = 24
        # Cosmetics
        self.suits = ['♠', '♥', '♦', '♣']
        # State for animations
        self.player_cards_vis = []  # list of dicts {x,y,val}
        self.dealer_cards_vis = []
        self.last_info = {}
        self.confetti = []
        self.enable_betting = hasattr(env, 'bet_bins') and getattr(env, 'bet_bins', 0) > 0
        self.bet_bin_rects = []
        self.hole_revealed = False
        # Sounds
        self.snd_deal = assets.sound('deal')
        self.snd_flip = assets.sound('flip')
        self.snd_win = assets.sound('win')
        self.snd_lose = assets.sound('lose')
        self.snd_draw = assets.sound('draw')
        # Autoplay/model
        self.autoplay = False
        self.model = None
        self.autoplay_delay_ms = 500
        self.next_action_at = 0
        self.obs = None
        # Session scoreboard (wins, losses, draws)
        self.session_wins = 0
        self.session_losses = 0
        self.session_draws = 0
        # Decision/status for UX
        self.last_decision = None  # 'HIT'|'STAND'|'DOUBLE'
        self.pending_reveal = False
        self.reveal_delay_ms = 400
        self.reveal_at_ms = 0
        self._stand_prev_dealer_len = 0
        self._post_reveal_info = None
        # Recording (viewer-captured gameplay)
        self.recording_enabled = False
        self.record_frames = []
        self.record_out = None
        self.record_max_frames = 3000
        self.rounds_to_play = 0

    def draw_obs_panel(self):
        # Show exactly what the agent observes for Blackjack
        if self.obs is None or not isinstance(self.obs, (list, tuple)) and not hasattr(self.obs, '__len__'):
            return
        try:
            v = self.obs
            # Flatten if vector inside batch
            import numpy as np
            v = np.array(v).reshape(-1)
            # Keys per env observation order
            keys = [
                'player_sum_norm', 'dealer_upcard_norm', 'usable_ace', 'steps_left_norm',
                'bankroll_norm', 'bet_norm', 'rounds_left_norm', 'doubled_flag'
            ]
            lines = []
            for i, k in enumerate(keys):
                if i < len(v):
                    lines.append(f"{k}: {float(v[i]):.2f}")
            # Panel background
            pad = 8
            w = 260
            h = 14 * (len(lines) + 1)
            panel = pygame.Surface((w, h), pygame.SRCALPHA)
            panel.fill((20, 20, 20, 160))
            title = self.font.render('Agent Obs', True, (240, 240, 240))
            panel.blit(title, (pad, pad))
            y = pad + 18
            for t in lines:
                txt = self.font.render(t, True, (220, 220, 220))
                panel.blit(txt, (pad, y))
                y += 16
            self.screen.blit(panel, (20, 20))
        except Exception:
            pass

    def draw_table(self):
        table = assets.table_image()
        if table:
            # scale to fit while keeping aspect
            surf = pygame.transform.smoothscale(table, (self.W, self.H))
            self.screen.blit(surf, (0, 0))
        else:
            self.screen.fill(self.table_color)
            pygame.draw.circle(self.screen, (28, 84, 56), (self.W//2, self.H//2), 230, width=4)
        # Status banner overlay
        self._draw_status_banner()
        # Deck area
        back = assets.card_back_image()
        if back:
            b = pygame.transform.smoothscale(back, (self.card_w, self.card_h))
            self.screen.blit(b, self.deck_pos)
        else:
            pygame.draw.rect(self.screen, (60,60,60), (*self.deck_pos, self.card_w, self.card_h), border_radius=6)
            txt = self.font.render("DECK", True, (210, 210, 210))
            self.screen.blit(txt, (self.deck_pos[0]+10, self.deck_pos[1]+self.card_h+6))

    def draw_hud(self, info):
        # Titles
        dealer_lbl = self.bigfont.render("Dealer", True, (240, 240, 240))
        player_lbl = self.bigfont.render("Player", True, (240, 240, 240))
        self.screen.blit(dealer_lbl, (40, self.dealer_y - 60))
        self.screen.blit(player_lbl, (40, self.player_y - 60))

        # Sums
        dsum = info.get('dealer_sum', 0)
        psum = info.get('player_sum', 0)
        dmsg = self.font.render(f"Sum: {dsum}", True, (220, 220, 240))
        pmsg = self.font.render(f"Sum: {psum}", True, (220, 220, 240))
        self.screen.blit(dmsg, (40, self.dealer_y - 30))
        self.screen.blit(pmsg, (40, self.player_y - 30))

        # Bankroll/bet if present
        if 'bankroll' in info:
            bmsg = self.font.render(f"Bankroll: {int(info.get('bankroll', 0))}", True, (250, 230, 90))
            self.screen.blit(bmsg, (self.W - 260, 20))
        if 'bet' in info and info.get('bet', 0) > 0:
            betmsg = self.font.render(f"Bet: {int(info.get('bet', 0))}", True, (250, 230, 90))
            self.screen.blit(betmsg, (self.W - 260, 50))

        # Controls
        if self.enable_betting and getattr(self.env, 'phase', 'play') == 'bet':
            controls = "Choose Bet: Keys 1..{}  |  ESC: Quit".format(max(1, getattr(self.env, 'bet_bins', 1)))
        else:
            controls = "H: Hit  S: Stand  D: Double  SPACE: New Round  ESC: Quit"
        # Hide double suggestion if not allowed
        allow_double = bool(getattr(self.env, 'allow_double', False)) and int(getattr(self.env, 'bet_bins', 0)) > 0
        if not (self.enable_betting and getattr(self.env, 'phase', 'play') == 'bet'):
            controls = "H: Hit  S: Stand{}  SPACE: New Round  ESC: Quit".format("  D: Double" if allow_double else "")
        cmsg = self.font.render(controls, True, (230, 230, 230))
        self.screen.blit(cmsg, (40, self.H - 40))
        # Scoreboard (session): W-L-D
        score = f"W:{self.session_wins}  L:{self.session_losses}  D:{self.session_draws}"
        scmsg = self.font.render(score, True, (230, 230, 230))
        self.screen.blit(scmsg, (self.W - scmsg.get_width() - 20, self.H - 40))
        # Agent observation overlay (top-left)
        self.draw_obs_panel()
        # Status banner (top center)
        self._draw_status_banner()

    def _draw_status_banner(self):
        msg = None
        if self.pending_reveal:
            msg = "Revealing dealer…"
        elif self.last_decision:
            msg = f"Agent Decision: {self.last_decision}"
        if not msg:
            return
        surf = self.bigfont.render(msg, True, (250, 230, 90))
        pad = 8
        w = surf.get_width() + pad * 2
        h = surf.get_height() + pad * 2
        x = (self.W - w) // 2
        y = 60
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 140))
        panel.blit(surf, (pad, pad))
        self.screen.blit(panel, (x, y))

    def draw_card(self, x, y, val, face_up=True):
        rank = 'A' if val == 1 else ('K' if val == 13 else ('Q' if val == 12 else ('J' if val == 11 else str(val))))
        suit_name = ['spades','hearts','diamonds','clubs'][(val + 1) % 4]
        if face_up:
            surf = assets.card_image(rank, suit_name)
            if surf:
                card = pygame.transform.smoothscale(surf, (self.card_w, self.card_h))
                self.screen.blit(card, (x, y))
                return
        else:
            back = assets.card_back_image()
            if back:
                card = pygame.transform.smoothscale(back, (self.card_w, self.card_h))
                self.screen.blit(card, (x, y))
                return
        # Fallback procedural
        color = (245, 245, 245) if face_up else (120, 120, 120)
        pygame.draw.rect(self.screen, color, (x, y, self.card_w, self.card_h), border_radius=8)
        pygame.draw.rect(self.screen, (30, 30, 30), (x, y, self.card_w, self.card_h), 2, border_radius=8)
        if face_up:
            suit = self.suits[(val + 1) % 4]
            col = (200, 40, 40) if suit in ('♥', '♦') else (20, 20, 20)
            idx = (val + 1) % 4
            suit = {0:'♠',1:'♥',2:'♦',3:'♣'}.get(idx, '?')
            col = (200, 40, 40) if idx in (1, 2) else (20, 20, 20)
            rtxt = self.font.render('A' if val == 1 else str(val), True, col)
            stxt = self.font.render(suit, True, col)
            self.screen.blit(rtxt, (x + 8, y + 6))
            self.screen.blit(stxt, (x + self.card_w - 24, y + self.card_h - 26))

    def animate_deal(self, who='player', val=1, duration=0.25, face_up=True):
        # Slide a card from deck to the end of target row
        frames = max(1, int(self.fps * duration))
        start = self.deck_pos
        if who == 'player':
            row_y = self.player_y
            idx = len(self.player_cards_vis)
            target = (120 + idx * self.spacing, row_y)
        else:
            row_y = self.dealer_y
            idx = len(self.dealer_cards_vis)
            target = (120 + idx * self.spacing, row_y)
        for i in range(frames):
            t = (i + 1) / frames
            x = int(lerp(start[0], target[0], t))
            y = int(lerp(start[1], target[1], t))
            self.draw_table()
            # redraw existing
            self.redraw_cards()
            # draw animating on top
            self.draw_card(x, y, val, face_up=face_up)
            # HUD
            self.draw_hud(self.last_info)
            pygame.display.flip()
            self.clock.tick(self.fps)
            # Auto-start next round when autoplaying
            if self.autoplay and round_over and not self.pending_reveal and getattr(self, 'next_round_at', 0):
                now3 = pygame.time.get_ticks()
                if now3 >= self.next_round_at:
                    if self.rounds_to_play > 0:
                        self.rounds_to_play -= 1
                        if self.rounds_to_play <= 0:
                            running = False
                            continue
                    obs, info = self.env.reset()
                    self.obs = obs
                    outcome_msg = None
                    round_over = False
                    self.last_info = info
                    self.sync_from_env()
                    self.next_round_at = 0
            # Capture frame for recording
            if getattr(self, 'recording_enabled', False) and len(self.record_frames) < getattr(self, 'record_max_frames', 3000):
                try:
                    import numpy as np
                    arr = pygame.surfarray.array3d(self.screen)
                    frame = np.transpose(arr, (1, 0, 2))
                    self.record_frames.append(frame)
                except Exception:
                    pass
        # commit card to vis list
        if who == 'player':
            self.player_cards_vis.append({'x': target[0], 'y': target[1], 'val': val})
        else:
            self.dealer_cards_vis.append({'x': target[0], 'y': target[1], 'val': val})
        if self.snd_deal:
            try:
                self.snd_deal.play()
            except Exception:
                pass

    def redraw_cards(self):
        for c in self.dealer_cards_vis:
            # hide dealer second card until reveal
            idx = (c['x'] - 120) // self.spacing
            face = True
            if not self.hole_revealed and idx == 1:
                face = False
            self.draw_card(c['x'], c['y'], c['val'], face_up=face)
        for c in self.player_cards_vis:
            self.draw_card(c['x'], c['y'], c['val'], face_up=True)

    def sync_from_env(self):
        # Rebuild visual card arrays based on env state (no animation)
        self.player_cards_vis = []
        self.dealer_cards_vis = []
        for i, v in enumerate(getattr(self.env, 'dealer', [])):
            self.dealer_cards_vis.append({'x': 120 + i * self.spacing, 'y': self.dealer_y, 'val': int(v)})
        for i, v in enumerate(getattr(self.env, 'player', [])):
            self.player_cards_vis.append({'x': 120 + i * self.spacing, 'y': self.player_y, 'val': int(v)})
        # Reset hole state
        self.hole_revealed = False

    def animate_initial_deal(self):
        # Sequence: dealer up, player 1, dealer hole (face-down), player 2
        if len(self.env.dealer) >= 1:
            self.animate_deal('dealer', self.env.dealer[0], face_up=True)
        if len(self.env.player) >= 1:
            self.animate_deal('player', self.env.player[0], face_up=True)
        if len(self.env.dealer) >= 2:
            self.animate_deal('dealer', self.env.dealer[1], face_up=False)
        if len(self.env.player) >= 2:
            self.animate_deal('player', self.env.player[1], face_up=True)
        self.hole_revealed = False

    def reveal_hole_flip(self, duration=0.25):
        # Simple flip animation for dealer's hole card
        if len(self.dealer_cards_vis) < 2:
            return
        card = self.dealer_cards_vis[1]
        frames = max(1, int(self.fps * duration))
        for i in range(frames):
            t = (i + 1) / frames
            w_scale = max(0.2, abs(math.cos(t * math.pi)))
            # redraw
            self.draw_table()
            # dealer first card
            self.draw_card(self.dealer_cards_vis[0]['x'], card['y'], self.dealer_cards_vis[0]['val'], True)
            # flipping card as a thinner rect
            rect_w = max(4, int(self.card_w * w_scale))
            x = card['x'] + (self.card_w - rect_w)//2
            back = assets.card_back_image()
            if back:
                temp = pygame.transform.smoothscale(back, (rect_w, self.card_h))
                self.screen.blit(temp, (x, card['y']))
            else:
                pygame.draw.rect(self.screen, (120,120,120), (x, card['y'], rect_w, self.card_h), border_radius=8)
            # player row
            for c in self.player_cards_vis:
                self.draw_card(c['x'], c['y'], c['val'], True)
            self.draw_hud(self.last_info)
            pygame.display.flip()
            self.clock.tick(self.fps)
        self.hole_revealed = True
        if self.snd_flip:
            try:
                self.snd_flip.play()
            except Exception:
                pass

    def _bet_rects(self):
        # Compute clickable rects for bet bins
        bins = max(1, getattr(self.env, 'bet_bins', 1))
        width = min(self.W - 80, bins * 90)
        left = (self.W - width) // 2
        rects = []
        for i in range(bins):
            x = left + i * (width // bins)
            rects.append(pygame.Rect(x + 8, self.H - 110, 70, 70))
        return rects

    def draw_bet_ui(self):
        rects = self._bet_rects()
        bins = max(1, getattr(self.env, 'bet_bins', 1))
        title = self.bigfont.render("Select Bet", True, (240, 240, 240))
        self.screen.blit(title, (self.W//2 - title.get_width()//2, self.H - 160))
        chips = assets.chip_images()
        for i, r in enumerate(rects):
            if i < len(chips):
                chip = pygame.transform.smoothscale(chips[i], (r.w, r.h))
                self.screen.blit(chip, (r.x, r.y))
            else:
                pygame.draw.ellipse(self.screen, (230, 200, 80), r)
                pygame.draw.ellipse(self.screen, (40,40,40), r, 2)
            label = str(min(9, i+1))
            txt = self.bigfont.render(label, True, (20,20,20))
            self.screen.blit(txt, (r.centerx - txt.get_width()//2, r.centery - txt.get_height()//2))
        hint = self.font.render("Press 1..{} or click a chip".format(bins), True, (240, 240, 240))
        self.screen.blit(hint, (self.W//2 - hint.get_width()//2, self.H - 30))

    def spawn_confetti(self, n=80):
        import random
        self.confetti = []
        for _ in range(n):
            x = random.randint(0, self.W)
            y = random.randint(-50, 0)
            vx = random.uniform(-1.0, 1.0)
            vy = random.uniform(2.0, 5.0)
            color = (random.randint(120,255), random.randint(120,255), random.randint(120,255))
            size = random.randint(2,5)
            self.confetti.append([x,y,vx,vy,color,size])

    def update_confetti(self):
        for p in self.confetti:
            p[0] += p[2]
            p[1] += p[3]
            p[3] += 0.08
        self.confetti = [p for p in self.confetti if p[1] < self.H+10]
        for p in self.confetti:
            pygame.draw.rect(self.screen, p[4], (int(p[0]), int(p[1]), p[5], p[5]))

    def run(self):
        obs, info = self.env.reset()
        self.obs = obs
        self.last_info = info
        self.sync_from_env()
        running = True
        round_over = False
        outcome_msg = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if round_over:
                        if event.key == pygame.K_SPACE:
                            obs, info = self.env.reset()
                            self.obs = obs
                            outcome_msg = None
                            round_over = False
                            self.last_info = info
                            self.sync_from_env()
                    else:
                        if self.enable_betting and getattr(self.env, 'phase', 'play') == 'bet':
                            # bet selection keys 1..bet_bins
                            if pygame.K_1 <= event.key <= pygame.K_9:
                                choice = event.key - pygame.K_1
                                obs, r, term, trunc, info = self.env.step(choice)
                                self.last_info = info
                                # animate initial deal after choosing bet
                                self.animate_initial_deal()
                        elif event.key in (pygame.K_h, pygame.K_LEFT):
                            self.last_decision = 'HIT'
                            obs, r, term, trunc, info = self.env.step(0)
                            self.obs = obs
                            # animate last player card if drawn
                            if len(self.env.player) > len(self.player_cards_vis):
                                self.animate_deal('player', self.env.player[-1])
                            self.last_info = info
                            if term or trunc:
                                round_over = True
                                outcome_msg = self.outcome_text(info)
                                self._tally_outcome(info)
                                self._play_outcome_sound(info)
                                if self.autoplay:
                                    self.next_round_at = pygame.time.get_ticks() + self.auto_next_ms
                        elif event.key in (pygame.K_s, pygame.K_RIGHT):
                            # store prev dealer count to animate new cards and flip
                            self.last_decision = 'STAND'
                            prev_len = len(self.env.dealer)
                            obs, r, term, trunc, info = self.env.step(1)
                            self.obs = obs
                            # Defer reveal and dealer animations for a moment
                            self.pending_reveal = True
                            self.reveal_at_ms = pygame.time.get_ticks() + self.reveal_delay_ms
                            self._stand_prev_dealer_len = prev_len
                            self._post_reveal_info = (term, trunc, info)
                            self.last_info = info
                        elif event.key in (pygame.K_d, pygame.K_UP):
                            # double (action=2) if supported
                            try:
                                self.last_decision = 'DOUBLE'
                                obs, r, term, trunc, info = self.env.step(2)
                                self.obs = obs
                                # player took one card
                                if len(self.env.player) > len(self.player_cards_vis):
                                    self.animate_deal('player', self.env.player[-1])
                                self.last_info = info
                                if term or trunc:
                                    round_over = True
                                    outcome_msg = self.outcome_text(info)
                                    self._play_outcome_sound(info)
                                    if self.autoplay:
                                        self.next_round_at = pygame.time.get_ticks() + self.auto_next_ms
                            except Exception:
                                pass
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not round_over and self.enable_betting and getattr(self.env, 'phase', 'play') == 'bet':
                        mx, my = event.pos
                        for idx, rect in enumerate(self._bet_rects()):
                            if rect.collidepoint(mx, my):
                                obs, r, term, trunc, info = self.env.step(idx)
                                self.obs = obs
                                self.last_info = info
                                self.animate_initial_deal()
                                break

            # Autoplay decisions
            if self.autoplay and not round_over:
                now = pygame.time.get_ticks()
                if now >= self.next_action_at:
                    a = self._autoplay_action()
                    if a is not None:
                        prev_len = len(self.env.dealer)
                        obs, r, term, trunc, info = self.env.step(int(a))
                        self.obs = obs
                        # animate according to action and env changes
                        if getattr(self.env, 'phase', 'play') != 'bet':
                            if a == 0 and len(self.env.player) > len(self.player_cards_vis):
                                self.last_decision = 'HIT'
                                self.animate_deal('player', self.env.player[-1])
                            if a == 1:
                                self.last_decision = 'STAND'
                                self.pending_reveal = True
                                self.reveal_at_ms = now + self.reveal_delay_ms
                                self._stand_prev_dealer_len = prev_len
                                self._post_reveal_info = (term, trunc, info)
                            if a == 2:
                                self.last_decision = 'DOUBLE'
                        self.last_info = info
                        if term or trunc and a != 1:
                            round_over = True
                            outcome_msg = self.outcome_text(info)
                            self._tally_outcome(info)
                            self._play_outcome_sound(info)
                            if self.autoplay and self.rounds_to_play > 0:
                                self.rounds_to_play -= 1
                                if self.rounds_to_play <= 0:
                                    running = False
                            if self.autoplay:
                                self.next_round_at = pygame.time.get_ticks() + self.auto_next_ms
                        self.next_action_at = now + self.autoplay_delay_ms

            # Handle deferred reveal after STAND
            if self.pending_reveal:
                now2 = pygame.time.get_ticks()
                if now2 >= self.reveal_at_ms:
                    self.reveal_hole_flip()
                    new_len = len(self.env.dealer)
                    if new_len > self._stand_prev_dealer_len:
                        for j in range(self._stand_prev_dealer_len, new_len):
                            self.animate_deal('dealer', self.env.dealer[j], face_up=True)
                    self.sync_from_env()
                    if self._post_reveal_info is not None:
                        term, trunc, info = self._post_reveal_info
                        if term or trunc:
                            round_over = True
                            outcome_msg = self.outcome_text(info)
                            self._tally_outcome(info)
                            self._play_outcome_sound(info)
                            if self.autoplay:
                                self.next_round_at = pygame.time.get_ticks() + self.auto_next_ms
                        self.last_info = info
                    self.pending_reveal = False
                    self._post_reveal_info = None

            # Draw
            self.draw_table()
            self.redraw_cards()
            self.draw_hud(self.last_info)
            # Bet UI overlay if needed
            if self.enable_betting and getattr(self.env, 'phase', 'play') == 'bet':
                self.draw_bet_ui()
            if outcome_msg:
                self.draw_center_text(outcome_msg, (250, 230, 90))
                tip = self.font.render("Press SPACE for new round", True, (240,240,240))
                self.screen.blit(tip, (self.W//2 - tip.get_width()//2, self.H//2 + 30))
                # confetti on win
                if self.last_info.get('win', 0) and not self.confetti:
                    self.spawn_confetti()
            self.update_confetti()
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
        # Save recording if enabled
        if getattr(self, 'recording_enabled', False) and self.record_frames and self.record_out:
            try:
                import imageio
                imageio.mimsave(self.record_out, self.record_frames, fps=min(self.fps, 30))
            except Exception:
                pass

    def draw_center_text(self, text, color):
        surf = self.bigfont.render(text, True, color)
        self.screen.blit(surf, (self.W//2 - surf.get_width()//2, self.H//2 - surf.get_height()//2))

    @staticmethod
    def outcome_text(info):
        if info.get('win', 0):
            return "You Win!"
        if info.get('lose', 0):
            return "You Lose"
        if info.get('draw', 0):
            return "Push"
        # Fallback
        return "Round Over"

    def _play_outcome_sound(self, info):
        try:
            if info.get('win', 0) and self.snd_win:
                self.snd_win.play()
            elif info.get('lose', 0) and self.snd_lose:
                self.snd_lose.play()
            elif info.get('draw', 0) and self.snd_draw:
                self.snd_draw.play()
        except Exception:
            pass

    def _tally_outcome(self, info):
        if info.get('win', 0):
            self.session_wins += 1
        elif info.get('lose', 0):
            self.session_losses += 1
        elif info.get('draw', 0):
            self.session_draws += 1

    def enable_autoplay(self, model):
        self.autoplay = True
        self.model = model
        self.next_action_at = pygame.time.get_ticks() + self.autoplay_delay_ms
        self.autoplay_deterministic = True

    def _autoplay_action(self) -> Optional[int]:
        if self.model is None or self.obs is None:
            return None
        try:
            import numpy as np
            obs = self.obs
            det = getattr(self, 'autoplay_deterministic', True)
            a, _ = self.model.predict(obs, deterministic=det)
            a = int(a) if not isinstance(a, (list, tuple)) else int(a[0])
            # Clip bet-bin actions if needed
            if getattr(self.env, 'phase', 'play') == 'bet':
                bins = max(1, getattr(self.env, 'bet_bins', 1))
                a = max(0, min(bins-1, a))
            else:
                a = max(0, min(2, a))
            return a
        except Exception:
            return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--persona', default='survivor', choices=['survivor','explorer','speedrunner'])
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--fps', type=int, default=60)
    p.add_argument('--betting', action='store_true', help='Enable betting UI (experimental)')
    p.add_argument('--autoplay', action='store_true', help='Use a trained model to play automatically')
    p.add_argument('--stochastic', action='store_true', help='Sample actions (non-deterministic) in autoplay')
    p.add_argument('--record', action='store_true', help='Record viewer frames to a GIF')
    p.add_argument('--record_out', default=None, help='Output GIF path when recording is enabled')
    p.add_argument('--rounds', type=int, default=0, help='Autoplay this many rounds then exit (0=until quit)')
    p.add_argument('--algo', default='ppo', choices=['ppo','a2c'], help='Algo for autoplay model discovery')
    p.add_argument('--runs_dir', default='runs', help='Where trained runs are stored')
    p.add_argument('--model', default=None, help='Path to model.zip to use for autoplay')
    return p.parse_args()


def main():
    args = parse_args()
    # Load configs and override for interactive play (1 round per episode simplifies outcomes)
    cfg = load_configs(app='blackjack', algo='ppo', persona=args.persona)
    cfg['app']['seed'] = args.seed
    # In autoplay, keep app config (betting/rounds) as-is to better match trained model
    if not args.autoplay:
        cfg['app']['rounds_per_episode'] = 1
        if not args.betting:
            cfg['app']['bet_bins'] = 0
    # Build env
    env = make_env(cfg['app'], cfg['persona'])
    viewer = BlackjackViewer(env, fps=args.fps)
    if args.autoplay:
        # Load model (latest matching run if not provided)
        from stable_baselines3 import PPO, A2C
        Algo = PPO if args.algo == 'ppo' else A2C
        model_path = args.model
        if model_path is None:
            # auto-discover latest
            prefix = f"blackjack-{args.algo}-{args.persona}-seed{args.seed}"
            candidates = [d for d in os.listdir(args.runs_dir) if d.startswith(prefix)]
            if not candidates:
                print('No trained runs found matching', prefix)
                return
            candidates.sort()
            run_dir = os.path.join(args.runs_dir, candidates[-1])
            model_path = os.path.join(run_dir, 'model.zip')
        try:
            model = Algo.load(model_path, device='cpu')
            viewer.enable_autoplay(model)
            # set deterministic vs stochastic
            viewer.autoplay_deterministic = (not args.stochastic)
        except Exception as e:
            print('Failed to load model for autoplay:', e)
    # Recording config
    if args.record:
        viewer.recording_enabled = True
        viewer.record_out = args.record_out
    if args.autoplay and args.rounds > 0:
        viewer.rounds_to_play = int(args.rounds)
    viewer.run()


if __name__ == '__main__':
    main()
