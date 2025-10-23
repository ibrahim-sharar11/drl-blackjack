import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class TetrisEnv(gym.Env):
    """
    Simple Tetris-like environment.
    - Board: 10x20 (width x height)
    - Actions: 0:noop, 1:left, 2:right, 3:rotate, 4:hard_drop
    - Observation: flattened board (200) + piece onehot (7) + [x_norm, y_norm] = 209 dims
    - Rewards (weights in persona):
        step_cost, line_clear_bonus, hole_penalty, height_penalty, topout_penalty,
        success, speed_bonus
    - Metrics (info): steps, lines_cleared_total, rows_cleared_last, holes_count,
      max_height, topout, success
    """

    metadata = {"render_modes": ["rgb_array"]}

    WIDTH = 10
    HEIGHT = 20
    ACTIONS = 5

    # Tetromino rotations (x,y) offsets for each rotation state
    # Origin is piece position; shapes defined to fit within board when placed appropriately
    TETROMINOES = {
        # I
        0: [np.array([[0, 0], [1, 0], [2, 0], [3, 0]]),
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]])],
        # O
        1: [np.array([[0, 0], [1, 0], [0, 1], [1, 1]])],
        # T
        2: [np.array([[0, 0], [1, 0], [2, 0], [1, 1]]),
            np.array([[1, 0], [0, 1], [1, 1], [1, 2]]),
            np.array([[1, 0], [0, 1], [1, 1], [2, 1]]),
            np.array([[0, 0], [0, 1], [1, 1], [0, 2]])],
        # L
        3: [np.array([[0, 0], [0, 1], [0, 2], [1, 2]]),
            np.array([[0, 0], [1, 0], [2, 0], [0, 1]]),
            np.array([[0, 0], [1, 0], [1, 1], [1, 2]]),
            np.array([[0, 1], [1, 1], [2, 1], [2, 0]])],
        # J
        4: [np.array([[1, 0], [1, 1], [1, 2], [0, 2]]),
            np.array([[0, 0], [1, 0], [2, 0], [2, 1]]),
            np.array([[0, 0], [0, 1], [0, 2], [1, 0]]),
            np.array([[0, 0], [0, 1], [1, 1], [2, 1]])],
        # S
        5: [np.array([[1, 0], [2, 0], [0, 1], [1, 1]]),
            np.array([[0, 0], [0, 1], [1, 1], [1, 2]])],
        # Z
        6: [np.array([[0, 0], [1, 0], [1, 1], [2, 1]]),
            np.array([[1, 0], [0, 1], [1, 1], [0, 2]])],
    }

    def __init__(self, max_steps=2000, seed=7, reward_weights=None, reward_scale=1.0, target_lines=10):
        super().__init__()
        self.rw = reward_weights or {}
        self.reward_scale = reward_scale
        self.max_steps = max_steps
        self.target_lines = target_lines
        self._rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(209,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.ACTIONS)

        # State
        self.board = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int8)
        self.steps = 0
        self.lines_cleared_total = 0
        self.topout = 0
        self.success = 0
        self.current_piece = None
        self.rotation = 0
        self.pos_x = 3
        self.pos_y = 0
        self._cell_px = 10

    def _sample_piece(self):
        piece_id = self._rng.randint(0, 6)
        rots = len(self.TETROMINOES[piece_id])
        rot = self._rng.randint(0, rots - 1)
        return piece_id, rot

    def _get_blocks(self, piece_id, rotation, x, y):
        blocks = self.TETROMINOES[piece_id][rotation]
        coords = blocks + np.array([x, y])
        return coords

    def _fits(self, piece_id, rotation, x, y):
        coords = self._get_blocks(piece_id, rotation, x, y)
        for cx, cy in coords:
            if cx < 0 or cx >= self.WIDTH or cy < 0 or cy >= self.HEIGHT:
                return False
            if self.board[cy, cx] == 1:
                return False
        return True

    def _lock_piece(self):
        for cx, cy in self._get_blocks(self.current_piece, self.rotation, self.pos_x, self.pos_y):
            if 0 <= cy < self.HEIGHT and 0 <= cx < self.WIDTH:
                self.board[cy, cx] = 1

    def _clear_lines(self):
        full_rows = [r for r in range(self.HEIGHT) if np.all(self.board[r] == 1)]
        if not full_rows:
            return 0
        # Remove full rows and add empty rows at top
        remaining = np.delete(self.board, full_rows, axis=0)
        add = np.zeros((len(full_rows), self.WIDTH), dtype=np.int8)
        self.board = np.vstack([add, remaining])
        return len(full_rows)

    def _holes_count(self):
        holes = 0
        for c in range(self.WIDTH):
            column = self.board[:, c]
            filled_seen = False
            for cell in column:
                if cell == 1:
                    filled_seen = True
                elif filled_seen and cell == 0:
                    holes += 1
        return holes

    def _max_height(self):
        heights = self.HEIGHT - np.argmax((self.board[::-1] == 1), axis=0)
        heights[~np.any(self.board == 1, axis=0)] = 0
        return int(np.max(heights))

    def _obs(self):
        board_flat = self.board.flatten().astype(np.float32)
        piece_oh = np.zeros(7, dtype=np.float32)
        if self.current_piece is not None:
            piece_oh[self.current_piece] = 1.0
        x_norm = np.float32(self.pos_x / (self.WIDTH - 1))
        y_norm = np.float32(self.pos_y / (self.HEIGHT - 1))
        return np.concatenate([board_flat, piece_oh, np.array([x_norm, y_norm], dtype=np.float32)])

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self.board[:] = 0
        self.steps = 0
        self.lines_cleared_total = 0
        self.topout = 0
        self.success = 0
        self.current_piece, self.rotation = self._sample_piece()
        self.pos_x = 3
        self.pos_y = 0
        # If initial placement already collides -> topout
        if not self._fits(self.current_piece, self.rotation, self.pos_x, self.pos_y):
            self.topout = 1
        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        if self.topout or self.steps >= self.max_steps:
            info = {
                "steps": self.steps,
                "lines_cleared_total": self.lines_cleared_total,
                "rows_cleared_last": 0,
                "holes_count": self._holes_count(),
                "max_height": self._max_height(),
                "topout": self.topout,
                "success": self.success,
            }
            return self._obs(), 0.0, True, False, info

        self.steps += 1
        shaped = 0.0
        shaped += self.rw.get("step_cost", -0.001)

        # Handle action
        if action == 1:  # left
            if self._fits(self.current_piece, self.rotation, self.pos_x - 1, self.pos_y):
                self.pos_x -= 1
        elif action == 2:  # right
            if self._fits(self.current_piece, self.rotation, self.pos_x + 1, self.pos_y):
                self.pos_x += 1
        elif action == 3:  # rotate
            new_rot = (self.rotation + 1) % len(self.TETROMINOES[self.current_piece])
            # simple wall kick: try x, x-1, x+1
            for dx in (0, -1, 1):
                if self._fits(self.current_piece, new_rot, self.pos_x + dx, self.pos_y):
                    self.rotation = new_rot
                    self.pos_x += dx
                    break
        elif action == 4:  # hard drop
            while self._fits(self.current_piece, self.rotation, self.pos_x, self.pos_y + 1):
                self.pos_y += 1

        # Gravity
        if self._fits(self.current_piece, self.rotation, self.pos_x, self.pos_y + 1):
            self.pos_y += 1
            rows_cleared = 0
        else:
            # Lock and spawn next
            self._lock_piece()
            rows_cleared = self._clear_lines()
            self.lines_cleared_total += rows_cleared
            # Rewards on lock
            if rows_cleared > 0:
                shaped += self.rw.get("line_clear_bonus", 0.5) * rows_cleared
            holes = self._holes_count()
            if holes > 0:
                shaped += self.rw.get("hole_penalty", -0.01) * holes
            h = self._max_height() / self.HEIGHT
            shaped += self.rw.get("height_penalty", -0.0) * h

            if self.lines_cleared_total >= self.target_lines:
                self.success = 1
                shaped += self.rw.get("success", 1.0)
                shaped += self.rw.get("speed_bonus", 0.0) * (self.max_steps - self.steps) / self.max_steps
                self.topout = 1  # end episode as success
            else:
                # Spawn next piece
                self.current_piece, self.rotation = self._sample_piece()
                self.pos_x = 3
                self.pos_y = 0
                if not self._fits(self.current_piece, self.rotation, self.pos_x, self.pos_y):
                    # topout
                    self.topout = 1
                    shaped += self.rw.get("topout_penalty", -1.0)

        truncated = False
        if self.steps >= self.max_steps and self.topout == 0:
            truncated = True

        obs = self._obs()
        reward = float(shaped) * self.reward_scale
        info = {
            "steps": self.steps,
            "lines_cleared_total": self.lines_cleared_total,
            "rows_cleared_last": rows_cleared,
            "holes_count": self._holes_count(),
            "max_height": self._max_height(),
            "topout": self.topout,
            "success": self.success,
        }
        terminated = bool(self.topout)
        return obs, reward, terminated, truncated, info

    def render(self):
        # RGB array render of the board and current piece overlay
        H = self.HEIGHT * self._cell_px
        W = self.WIDTH * self._cell_px
        img = np.zeros((H, W, 3), dtype=np.uint8)
        # draw locked cells
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                if self.board[y, x] == 1:
                    y0, y1 = y * self._cell_px, (y + 1) * self._cell_px
                    x0, x1 = x * self._cell_px, (x + 1) * self._cell_px
                    img[y0:y1, x0:x1] = (100, 200, 255)
        # draw current piece overlay
        if self.current_piece is not None:
            coords = self._get_blocks(self.current_piece, self.rotation, self.pos_x, self.pos_y)
            for cx, cy in coords:
                if 0 <= cx < self.WIDTH and 0 <= cy < self.HEIGHT:
                    y0, y1 = cy * self._cell_px, (cy + 1) * self._cell_px
                    x0, x1 = cx * self._cell_px, (cx + 1) * self._cell_px
                    img[y0:y1, x0:x1] = (255, 150, 50)
        return img
