import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import defaultdict

class FormFlowEnv(gym.Env):
    """
    Simulated multi-page web form flow (no browser needed).
    Goals:
      - Reach final submit (success)
      - Discover validation errors (issue detection)
      - Cover distinct 'DOM selectors' (coverage proxy)
    Pages: [landing] -> [signup] -> [profile] -> [review] -> [submit]
    Actions:
      0: next_page, 1: prev_page, 2: type_input, 3: clear_input, 4: toggle_checkbox,
      5: click_random_selector, 6: submit_page
    Observation (vector):
      [page_id (onehot 5), field_filled(0/1), field_valid(0/1), checkbox(0/1),
       errors_on_page(0..3), latency_bucket(0..3), steps_left_norm(0..1)]
    Info metrics:
      steps, distinct_pages, distinct_selectors, validation_errors,
      softlock (flag if looped too long), latency_spike (flag)
    """
    metadata = {"render_modes": []}

    PAGES = ["landing", "signup", "profile", "review", "submit"]
    PAGE_TO_ID = {p:i for i,p in enumerate(PAGES)}
    NUM_PAGES = len(PAGES)
    ACTIONS = 7

    def __init__(self,
                 max_steps=150,
                 seed=7,
                 reward_weights=None,
                 reward_scale=1.0,
                 invalid_prob=0.2,
                 latency_spike_prob=0.05):
        super().__init__()
        self.rw = reward_weights or {}
        self.reward_scale = reward_scale
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.invalid_prob = invalid_prob
        self.latency_spike_prob = latency_spike_prob

        # observation: onehot(5) + 6 scalars = 11
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.ACTIONS)

        # State
        self.page = 0               # landing
        self.field_filled = 0
        self.field_valid = 0
        self.checkbox = 0
        self.errors_on_page = 0
        self.latency_bucket = 0
        self.steps = 0
        self.done = False

        # Metrics/coverage
        self.visited_pages = set()
        self.clicked_selectors = set()
        self.validation_errors = 0
        self.latency_spike = 0
        self.softlock = 0
        self._loop_detector = defaultdict(int)

    def _obs(self):
        onehot = np.zeros(self.NUM_PAGES, dtype=np.float32)
        onehot[self.page] = 1.0
        steps_left = max(0, self.max_steps - self.steps) / float(self.max_steps)
        vec = np.array([
            self.field_filled, self.field_valid, self.checkbox,
            min(self.errors_on_page, 3) / 3.0,
            self.latency_bucket / 3.0,
            steps_left
        ], dtype=np.float32)
        return np.concatenate([onehot, vec])

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self.page = 0
        self.field_filled = 0
        self.field_valid = 0
        self.checkbox = 0
        self.errors_on_page = 0
        self.latency_bucket = 0
        self.steps = 0
        self.done = False

        self.visited_pages = {self.page}
        self.clicked_selectors = set()
        self.validation_errors = 0
        self.latency_spike = 0
        self.softlock = 0
        self._loop_detector.clear()

        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        if self.done:
            return self._obs(), 0.0, True, False, {}

        self.steps += 1
        shaped = 0.0

        # base step cost to discourage dithering
        shaped += self.rw.get("step_cost", -0.001)

        # emulate page-specific validation & latency
        self.errors_on_page = 0
        self.latency_bucket = 0

        # probabilistic latency spike
        if self._rng.random() < self.latency_spike_prob:
            self.latency_bucket = self._rng.choice([1, 2, 3])
            if self.latency_bucket >= 2:
                self.latency_spike = 1
                shaped += self.rw.get("latency_penalty", -0.01) * self.latency_bucket

        # Actions
        action_type_input = 0
        action_click_selector = 0
        if action == 0:  # next_page
            if self.page < self.NUM_PAGES - 1:
                # gate: must have filled & valid & checkbox on signup/profile
                can_advance = True
                if self.PAGES[self.page] in ("signup", "profile"):
                    if not (self.field_filled and self.field_valid and self.checkbox):
                        can_advance = False
                        self.errors_on_page += 1
                        self.validation_errors += 1
                        shaped += self.rw.get("validation_error_bonus", 0.05)  # issue detection reward
                if can_advance:
                    self.page += 1
                    shaped += self.rw.get("page_progress", 0.01)
        elif action == 1:  # prev_page
            if self.page > 0:
                self.page -= 1
        elif action == 2:  # type_input
            self.field_filled = 1
            action_type_input = 1
            # chance of invalid entry when first typing
            if self._rng.random() < self.invalid_prob:
                self.field_valid = 0
                self.errors_on_page += 1
                self.validation_errors += 1
                shaped += self.rw.get("validation_error_bonus", 0.05)
            else:
                self.field_valid = 1
        elif action == 3:  # clear_input
            self.field_filled = 0
            self.field_valid = 0
        elif action == 4:  # toggle_checkbox
            self.checkbox = 1 - self.checkbox
        elif action == 5:  # click_random_selector (coverage proxy)
            sel = (self.page, self._rng.randint(0, 4))
            action_click_selector = 1
            if sel not in self.clicked_selectors:
                self.clicked_selectors.add(sel)
                shaped += self.rw.get("dom_coverage_bonus", 0.02)
        elif action == 6:  # submit_page
            # only meaningful on final page
            if self.page == self.PAGE_TO_ID["submit"]:
                self.done = True
                shaped += self.rw.get("success", 1.0)
                shaped += self.rw.get("speed_bonus", 0.05) * (self.max_steps - self.steps) / self.max_steps

        # coverage reward for first time visiting a page
        prev_len = len(self.visited_pages)
        self.visited_pages.add(self.page)
        if len(self.visited_pages) > prev_len:
            shaped += self.rw.get("page_coverage_bonus", 0.03)

        # softlock detection: repeating the same (page, field, check) too long
        sig = (self.page, self.field_filled, self.field_valid, self.checkbox)
        self._loop_detector[sig] += 1
        if self._loop_detector[sig] > 20:
            self.softlock = 1
            shaped += self.rw.get("softlock_penalty", -0.05)

        # time limit
        truncated = False
        if self.steps >= self.max_steps and not self.done:
            truncated = True

        obs = self._obs()
        reward = float(shaped) * self.reward_scale

        info = {
            "steps": self.steps,
            "distinct_pages": len(self.visited_pages),
            "distinct_selectors": len(self.clicked_selectors),
            "validation_errors": self.validation_errors,
            "latency_spike": self.latency_spike,
            "softlock": self.softlock,
            "success": 1.0 if self.done else 0.0,
            "page_id": int(self.page),
            "action_type_input": action_type_input,
            "action_click_selector": action_click_selector,
        }
        terminated = self.done
        return obs, reward, terminated, truncated, info

    def render(self):
        # Minimal RGB render: show page one-hot, field/valid/checkbox, latency, errors
        h, w = 120, 240
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # page bars
        bar_w = w // self.NUM_PAGES
        for i in range(self.NUM_PAGES):
            x0, x1 = i * bar_w, (i + 1) * bar_w
            color = (60, 60, 60)
            if i == self.page:
                color = (120, 200, 120)
            img[0:30, x0:x1] = color
        # field states (three squares)
        squares = [(20, 50, self.field_filled), (100, 50, self.field_valid), (180, 50, self.checkbox)]
        for x, y, v in squares:
            img[y:y+20, x:x+20] = (200, 50, 50) if v == 0 else (50, 200, 50)
        # latency bar
        lat_h = int(20 * (self.latency_bucket / 3.0))
        img[80:80+lat_h, 20:40] = (200, 200, 50)
        # errors count visualization (simple red bar)
        err_h = min(20, 5 * self.errors_on_page)
        img[80:80+err_h, 200:220] = (220, 60, 60)
        return img
