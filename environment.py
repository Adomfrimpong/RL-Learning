"""
environment.py
Custom Frozen Lake environment built from first principles.
No Gymnasium, Gym, Stable Baselines, or RLlib are used.

Actions: 0 = Left, 1 = Down, 2 = Right, 3 = Up
State: single integer index, computed as row * ncols + col.
"""

import random


# Standard 8x8 Frozen Lake map.
DEFAULT_MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

# Action constants.
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


class FrozenLakeEnv:
    """Grid-world environment for the Frozen Lake task."""

    def __init__(self, grid_map=None, is_slippery=False, slip_prob=0.2, seed=None):
        """
        grid_map: list of equal-length strings describing the grid.
        is_slippery: when True, actions sometimes slip to a side direction.
        slip_prob: total probability the agent slips off the intended action.
        seed: optional integer for reproducible randomness.
        """
        self.grid = grid_map if grid_map is not None else DEFAULT_MAP
        self.nrows = len(self.grid)
        self.ncols = len(self.grid[0])
        self.n_states = self.nrows * self.ncols
        self.n_actions = 4
        self.is_slippery = is_slippery
        self.slip_prob = slip_prob
        self.rng = random.Random(seed)

        # Locate the start cell.
        self.start_state = self._find_start()
        self.state = self.start_state
        self.done = False

    def _find_start(self):
        """Return the integer index of the start cell."""
        for r in range(self.nrows):
            for c in range(self.ncols):
                if self.grid[r][c] == "S":
                    return self._to_index(r, c)
        return 0

    def _to_index(self, row, col):
        """Convert (row, col) to a single integer state index."""
        return row * self.ncols + col

    def _to_rowcol(self, index):
        """Convert a single integer state index to (row, col)."""
        return divmod(index, self.ncols)

    def _cell(self, index):
        """Return the character at a given state index."""
        row, col = self._to_rowcol(index)
        return self.grid[row][col]

    def reset(self):
        """Reset the agent to the start cell and return the start state."""
        self.state = self.start_state
        self.done = False
        return self.state

    def get_state(self):
        """Return the current integer state index."""
        return self.state

    def is_terminal(self, index=None):
        """Return True if the given state (or current state) is a hole or goal."""
        if index is None:
            index = self.state
        return self._cell(index) in ("H", "G")

    def _move(self, index, action):
        """Apply one deterministic action and return the resulting state index."""
        row, col = self._to_rowcol(index)
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.nrows - 1)
        elif action == RIGHT:
            col = min(col + 1, self.ncols - 1)
        elif action == UP:
            row = max(row - 1, 0)
        return self._to_index(row, col)

    def _resolve_action(self, action):
        """
        Return the action the environment executes.
        With slippery dynamics, the agent slips to a perpendicular direction
        with probability slip_prob, split evenly across the two sides.
        """
        if not self.is_slippery:
            return action
        roll = self.rng.random()
        if roll < self.slip_prob:
            # Perpendicular directions to the intended action.
            if action in (LEFT, RIGHT):
                return UP if roll < self.slip_prob / 2 else DOWN
            return LEFT if roll < self.slip_prob / 2 else RIGHT
        return action

    def step(self, action):
        """
        Apply an action. Return (next_state, reward, done).
        Reward is 1.0 at the goal and 0.0 everywhere else.
        """
        if self.done:
            return self.state, 0.0, True

        executed = self._resolve_action(action)
        next_state = self._move(self.state, executed)
        self.state = next_state

        cell = self._cell(next_state)
        if cell == "G":
            reward, self.done = 1.0, True
        elif cell == "H":
            reward, self.done = 0.0, True
        else:
            reward, self.done = 0.0, False

        return next_state, reward, self.done

    def render(self):
        """Print the grid with the agent position marked as A."""
        agent_row, agent_col = self._to_rowcol(self.state)
        lines = []
        for r in range(self.nrows):
            row_chars = []
            for c in range(self.ncols):
                if r == agent_row and c == agent_col:
                    row_chars.append("A")
                else:
                    row_chars.append(self.grid[r][c])
            lines.append(" ".join(row_chars))
        print("\n".join(lines))
        print()
