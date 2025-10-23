import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class BlackjackEnv(gym.Env):
    """
    Blackjack environment (player vs dealer) with optional shoe, betting, and multi-round episodes.

    Actions (single Discrete space interpreted by phase):
      - Bet phase (if enabled via bet_bins>0): actions 0..(bet_bins-1) select bet bin
      - Play phase: 0=hit, 1=stand, 2=double (if allowed/first decision)

    Observation (vector of 8 floats in [0,1]):
      [player_sum_norm, dealer_upcard_norm, usable_ace,
       steps_left_norm, bankroll_norm, bet_norm, rounds_left_norm, doubled_flag]

    Rewards (weighted via persona weights with defaults here):
      step_cost (-0.001), bust_penalty (-1.0), win_reward (1.0), lose_penalty (-1.0),
      draw_bonus (0.0), blackjack_bonus (0.5), success (1.0 on reaching bankroll_target),
      speed_bonus (0.0) scaled by remaining steps on success.

    Notes:
      - Card values: 2..10 as face value; J/Q/K as 10; Ace as 1 or 11 (usable-ace logic).
      - Dealer hits until 17; soft-17 behavior configurable (dealer_hits_soft17).
      - Natural blackjack grants an additional bonus and uses payout_blackjack when bet mode on.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 max_steps=100,
                 seed=7,
                 reward_weights=None,
                 reward_scale=1.0,
                 # Advanced options
                 num_decks=1,
                 penetration=0.75,
                 rounds_per_episode=1,
                 bankroll_start=0,
                 bankroll_target=0,
                 bet_bins=0,
                 min_bet=1,
                 max_bet=10,
                 payout_blackjack=1.5,
                 dealer_hits_soft17=False,
                 allow_double=True,
                 bet_scaled_reward=False):
        super().__init__()
        self.rw = reward_weights or {}
        self.reward_scale = reward_scale
        self.max_steps = int(max_steps)
        self._rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Options
        self.num_decks = max(1, int(num_decks))
        self.penetration = float(penetration)
        self.rounds_per_episode = int(rounds_per_episode)
        self.bankroll_start = float(bankroll_start)
        self.bankroll_target = float(bankroll_target)
        self.bet_bins = int(bet_bins)
        self.min_bet = float(min_bet)
        self.max_bet = float(max_bet)
        self.payout_blackjack = float(payout_blackjack)
        self.dealer_hits_soft17 = bool(dealer_hits_soft17)
        self.allow_double = bool(allow_double)
        self.bet_scaled_reward = bool(bet_scaled_reward)

        # Action/Observation spaces
        self.n_actions = max(3, self.bet_bins)  # ensure space covers play phase (hit/stand/double)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

        # State
        self.player = []
        self.dealer = []
        self.steps = 0
        self.done = False
        self.natural = 0  # player natural blackjack flag
        self.player_bust = 0
        self.dealer_bust = 0
        self.phase = "play" if self.bet_bins == 0 else "bet"
        self.round_idx = 0
        self.bankroll = self.bankroll_start
        self.bet = 0.0
        self.first_decision = True
        self.doubled = False
        # Shoe (optional)
        self._shoe = []
        self._shoe_used = 0

    # --- Card mechanics ---
    def _build_shoe(self):
        deck = [1,2,3,4,5,6,7,8,9,10,10,10,10] * 4
        self._shoe = deck * self.num_decks
        self._rng.shuffle(self._shoe)
        self._shoe_used = 0

    def _maybe_shuffle(self):
        if not self._shoe or (self._shoe_used / len(self._shoe)) >= self.penetration:
            self._build_shoe()

    def _draw_card(self):
        if self.num_decks > 1 or self.penetration < 0.999:
            self._maybe_shuffle()
            card = self._shoe.pop()
            self._shoe_used += 1
            return card
        # single-draw fallback
        return self._rng.choice([1,2,3,4,5,6,7,8,9,10,10,10,10])

    @staticmethod
    def _usable_ace(hand):
        # Returns True if hand has an ace counted as 11 without busting
        if 1 in hand:
            if sum(hand) + 10 <= 21:
                return True
        return False

    @staticmethod
    def _hand_sum(hand):
        s = sum(hand)
        if 1 in hand and s + 10 <= 21:
            return s + 10
        return s

    # --- Env API ---
    def _obs(self):
        p_sum = self._hand_sum(self.player)
        d_up = self.dealer[0] if self.dealer else 0
        usable = 1.0 if self._usable_ace(self.player) else 0.0
        steps_left = max(0, self.max_steps - self.steps) / float(self.max_steps)
        rounds_left = 0.0
        if self.rounds_per_episode > 0:
            rounds_left = max(0, self.rounds_per_episode - self.round_idx) / float(max(1, self.rounds_per_episode))
        bankroll_norm = 0.0 if self.bankroll_start <= 0 else min(1.0, max(0.0, self.bankroll / max(1.0, self.bankroll_target if self.bankroll_target > 0 else self.bankroll_start * 2)))
        bet_norm = 0.0 if self.max_bet <= 0 else min(1.0, self.bet / self.max_bet)
        doubled_flag = 1.0 if self.doubled else 0.0
        obs = np.array([
            p_sum / 31.0,
            d_up / 11.0,
            usable,
            steps_left,
            bankroll_norm,
            bet_norm,
            rounds_left,
            doubled_flag,
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self.steps = 0
        self.done = False
        self.round_idx = 0
        self.bankroll = self.bankroll_start
        self._build_shoe() if (self.num_decks > 1 or self.penetration < 0.999) else None
        self._start_round()
        return self._obs(), {}

    def _start_round(self):
        # Start a fresh round; in betting mode, delay dealing until after bet is chosen
        self.player = []
        self.dealer = []
        self.player_bust = 0
        self.dealer_bust = 0
        self.natural = 0
        self.phase = "play" if self.bet_bins == 0 else "bet"
        self.first_decision = True
        self.doubled = False
        self.bet = 0.0 if self.bet_bins == 0 else self._default_bet()
        if self.bet_bins == 0:
            self._deal_initial()

    def _default_bet(self):
        if self.bet_bins <= 0:
            return 0.0
        # Default to minimum bet until agent selects otherwise
        return self.min_bet

    def _deal_initial(self):
        # Deal initial two cards each
        self.player = [self._draw_card(), self._draw_card()]
        self.dealer = [self._draw_card(), self._draw_card()]
        self.natural = 1 if self._hand_sum(self.player) == 21 and len(self.player) == 2 else 0

    def step(self, action: int):
        assert self.action_space.contains(action)
        if self.done or self.steps >= self.max_steps:
            info = self._info(finalize=False)
            return self._obs(), 0.0, True, False, info

        self.steps += 1
        shaped = 0.0
        shaped += self.rw.get("step_cost", -0.001)

        terminated = False
        truncated = False

        action_hit = action_stand = action_double = 0
        if self.phase == "bet" and self.bet_bins > 0:
            # Interpret action as bet bin selection
            bin_idx = max(0, min(self.bet_bins - 1, int(action)))
            if self.bet_bins == 1:
                frac = 1.0
            else:
                frac = (bin_idx + 1) / float(self.bet_bins)
            self.bet = max(self.min_bet, frac * self.max_bet)
            self.bet = min(self.bet, self.bankroll if self.bankroll > 0 else self.max_bet)
            self.phase = "play"
            # Now that bet is chosen, deal initial cards
            self._deal_initial()
        else:
            # Play phase actions: 0=hit, 1=stand, 2=double (if allowed & first decision)
            if action == 0:
                # hit
                prev_sum = self._hand_sum(self.player)
                self.player.append(self._draw_card())
                self.first_decision = False
                action_hit = 1
                # Shaping: reward moving closer to 21 without busting
                new_sum = self._hand_sum(self.player)
                if new_sum <= 21:
                    prev_gap = max(0, 21 - prev_sum)
                    new_gap = max(0, 21 - new_sum)
                    improvement = max(0.0, prev_gap - new_gap)
                    shaped += self.rw.get("approach_21_bonus", 0.02) * (improvement / 10.0)
                # Shaping: safe first hit on low totals (<=11) strongly encouraged
                if prev_sum <= 11 and self.steps <= 2:  # early in round
                    shaped += self.rw.get("safe_hit_bonus", 0.2)
                if self._hand_sum(self.player) > 21:
                    self.player_bust = 1
                    shaped += self._resolve_outcome()
                    terminated = self._advance_or_end()
            elif action == 2 and self.allow_double and self.first_decision and (self.bet_bins > 0):
                # double: double bet, take exactly one card, then stand
                add = min(self.bet, self.bankroll - self.bet) if self.bankroll > 0 else self.bet
                self.bet += max(0.0, add)
                self.player.append(self._draw_card())
                self.first_decision = False
                self.doubled = True
                action_double = 1
                # resolve immediately
                shaped += self._resolve_outcome()
                terminated = self._advance_or_end()
            elif action == 1:
                # stand -> resolve dealer
                action_stand = 1
                # Shaping: discourage very early stands (e.g., below 17)
                p_sum = self._hand_sum(self.player)
                if p_sum < 17:
                    shaped += self.rw.get("early_stand_penalty", -0.02) * ((17 - p_sum) / 17.0)
                shaped += self._resolve_outcome()
                terminated = self._advance_or_end()
            else:
                # invalid/no-op in this phase
                pass

        if self.steps >= self.max_steps and not terminated:
            truncated = True
            terminated = True
            self.done = True

        obs = self._obs()
        reward = float(shaped) * self.reward_scale
        info = self._info(finalize=terminated or truncated)
        # per-step action flags (aggregated later)
        info.update({
            "action_hit": action_hit,
            "action_stand": action_stand,
            "action_double": action_double,
        })
        return obs, reward, terminated, truncated, info

    def _dealer_play(self):
        # Dealer hits until threshold considering soft 17 rule
        while True:
            total = self._hand_sum(self.dealer)
            if total < 17:
                self.dealer.append(self._draw_card())
                continue
            if total == 17 and self.dealer_hits_soft17 and self._usable_ace(self.dealer):
                self.dealer.append(self._draw_card())
                continue
            break
        if self._hand_sum(self.dealer) > 21:
            self.dealer_bust = 1

    def _resolve_outcome(self):
        # Resolve dealer, compute shaped reward and bankroll change; return shaped reward
        shaped = 0.0
        if self._hand_sum(self.player) > 21:
            self.player_bust = 1
        else:
            self._dealer_play()

        p = self._hand_sum(self.player)
        d = self._hand_sum(self.dealer)

        bet_scale = (self.bet / self.max_bet) if (self.bet_bins > 0 and self.max_bet > 0) else 1.0

        if self.player_bust:
            shaped += self.rw.get("bust_penalty", -1.0) * (bet_scale if self.bet_scaled_reward else 1.0)
            pnl = -self.bet
        elif self.dealer_bust or p > d:
            # Win
            if self.bet_bins > 0:
                if self.natural and len(self.player) == 2:
                    win_amt = self.bet * self.payout_blackjack
                else:
                    win_amt = self.bet
                pnl = win_amt
            else:
                pnl = 0.0
            shaped += self.rw.get("win_reward", 1.0) * (bet_scale if self.bet_scaled_reward else 1.0)
            if self.natural:
                shaped += self.rw.get("blackjack_bonus", 0.5) * (bet_scale if self.bet_scaled_reward else 1.0)
        elif p < d:
            pnl = -self.bet if self.bet_bins > 0 else 0.0
            shaped += self.rw.get("lose_penalty", -1.0) * (bet_scale if self.bet_scaled_reward else 1.0)
        else:
            # draw (push)
            pnl = 0.0
            shaped += self.rw.get("draw_bonus", 0.0) * (bet_scale if self.bet_scaled_reward else 1.0)

        # Bankroll update
        if self.bet_bins > 0:
            self.bankroll += pnl
            # success on target reached
            if self.bankroll_target > 0 and self.bankroll >= self.bankroll_target:
                shaped += self.rw.get("success", 1.0)
                shaped += self.rw.get("speed_bonus", 0.0) * (self.max_steps - self.steps) / self.max_steps
                self.done = True

        return shaped

    def _advance_or_end(self):
        # Decide to end episode or move to next round; return terminated flag for Gym
        self.round_idx += 1
        # End if player bankrupt (cannot place min bet) or reached rounds or explicit success
        bankrupt = (self.bet_bins > 0 and self.bankroll <= 0)
        if bankrupt:
            # penalize bankruptcy a bit more
            pass
        if self.done or bankrupt or (self.rounds_per_episode > 0 and self.round_idx >= self.rounds_per_episode):
            self.done = True
            return True
        # otherwise, start the next round in same episode
        self._start_round()
        return False

    def _info(self, finalize: bool):
        p = self._hand_sum(self.player)
        d = self._hand_sum(self.dealer)
        win = int(finalize and not self.player_bust and (self.dealer_bust or (p <= 21 and p > d)))
        draw = int(finalize and p == d and p <= 21 and not self.dealer_bust)
        lose = int(finalize and (self.player_bust or (p < d and d <= 21)))
        return {
            "steps": self.steps,
            "player_sum": p,
            "dealer_sum": d,
            "dealer_upcard": self.dealer[0] if self.dealer else 0,
            "usable_ace": 1.0 if self._usable_ace(self.player) else 0.0,
            "natural": self.natural,
            "player_bust": self.player_bust,
            "dealer_bust": self.dealer_bust,
            "win": win,
            "draw": draw,
            "lose": lose,
            "success": float(win) if self.bankroll_target <= 0 else (1.0 if (self.bankroll >= self.bankroll_target) else 0.0),
            "bankroll": float(self.bankroll),
            "bet": float(self.bet),
            "round_idx": int(self.round_idx),
            "phase": self.phase,
        }

    def render(self):
        # Minimal RGB render: visualize cards and basic state
        h, w = 160, 320
        img = np.zeros((h, w, 3), dtype=np.uint8)

        def draw_card(x, y, val, face_up=True):
            col = (240, 240, 240) if face_up else (120, 120, 120)
            img[y:y+30, x:x+20] = col
            if face_up:
                # encode value by color tint
                tint = min(255, 40 + 10 * min(10, val))
                img[y+5:y+25, x+3:x+17] = (tint, 60, 60) if val == 1 else (60, 60, tint)

        # Draw dealer cards (first up, second down to mimic hidden card)
        dx = 20
        for i, c in enumerate(self.dealer):
            draw_card(20 + i*24, 20, c, face_up=(i == 0))
            dx = 20 + i*24
        # Dealer sum bar
        d = min(self._hand_sum(self.dealer), 31)
        dh = int((d / 31.0) * 60)
        img[10:10+dh, dx+40:dx+50] = (60, 120, 220)

        # Draw player cards (all up)
        px = 20
        for i, c in enumerate(self.player):
            draw_card(20 + i*24, 80, c, face_up=True)
            px = 20 + i*24
        # Player sum bar
        p = min(self._hand_sum(self.player), 31)
        ph = int((p / 31.0) * 60)
        img[70:70+ph, px+40:px+50] = (60, 200, 60)

        # Bet/bankroll indicators
        if self.bet_bins > 0:
            bn = 0 if self.max_bet <= 0 else int((min(self.bet, self.max_bet) / self.max_bet) * 100)
            img[130:135, 20:20+bn] = (200, 180, 60)
            bk = 0 if self.bankroll_start <= 0 else int((min(self.bankroll, max(self.bankroll_target, self.bankroll_start*2)) / max(1.0, max(self.bankroll_target, self.bankroll_start*2))) * (w-40))
            img[145:150, 20:20+bk] = (80, 200, 200)

        return img
