"""Grid environment used by the treasure-maze reinforcement-learning agent.

This module is based on the course-provided CS-370 starter environment and has
been corrected and documented for the maintained portfolio version.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


VISITED_MARK = 0.8
PIRATE_MARK = 0.5

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


class TreasureMaze:
    """Represent a two-dimensional maze and its reward-driven game state."""

    def __init__(
        self,
        maze: Sequence[Sequence[float]] | np.ndarray,
        pirate: tuple[int, int] = (0, 0),
    ) -> None:
        self._maze = np.asarray(maze, dtype=np.float32).copy()
        if self._maze.ndim != 2 or self._maze.size == 0:
            raise ValueError("maze must be a non-empty two-dimensional array")
        if not np.isin(self._maze, [0.0, 1.0]).all():
            raise ValueError("maze cells must contain only 0.0 or 1.0")

        nrows, ncols = self._maze.shape
        self.target = (nrows - 1, ncols - 1)
        if self._maze[self.target] == 0.0:
            raise ValueError("target cell cannot be blocked")

        all_free_cells = [
            (row, col)
            for row in range(nrows)
            for col in range(ncols)
            if self._maze[row, col] == 1.0
        ]
        self.free_cells = [cell for cell in all_free_cells if cell != self.target]
        self.reset(pirate)

    def _validate_pirate(self, pirate: tuple[int, int]) -> None:
        if pirate not in self.free_cells:
            raise ValueError("pirate must start on a non-target free cell")

    def reset(self, pirate: tuple[int, int]) -> None:
        """Reset the episode at a validated starting position."""
        self._validate_pirate(pirate)
        self.pirate = pirate
        self.maze = self._maze.copy()
        row, col = pirate
        self.maze[row, col] = PIRATE_MARK
        self.state = (row, col, "start")
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0.0
        self.visited: set[tuple[int, int]] = set()

    def update_state(self, action: int) -> None:
        """Apply a valid, invalid, or blocked movement to the current state."""
        pirate_row, pirate_col, _ = self.state
        next_row, next_col = pirate_row, pirate_col

        if self.maze[pirate_row, pirate_col] > 0.0:
            self.visited.add((pirate_row, pirate_col))

        valid_actions = self.valid_actions()
        if not valid_actions:
            next_mode = "blocked"
        elif action not in valid_actions:
            next_mode = "invalid"
        else:
            next_mode = "valid"
            if action == LEFT:
                next_col -= 1
            elif action == UP:
                next_row -= 1
            elif action == RIGHT:
                next_col += 1
            elif action == DOWN:
                next_row += 1

        self.state = (next_row, next_col, next_mode)

    def get_reward(self) -> float:
        """Return the reward associated with the current state."""
        pirate_row, pirate_col, mode = self.state
        if (pirate_row, pirate_col) == self.target:
            return 1.0
        if mode == "blocked":
            return self.min_reward - 1.0
        if mode == "invalid":
            return -0.75
        if (pirate_row, pirate_col) in self.visited:
            return -0.25
        if mode == "valid":
            return -0.04
        raise RuntimeError(f"unrecognized maze state: {mode}")

    def act(self, action: int) -> tuple[np.ndarray, float, str]:
        """Apply an action and return the next state, reward, and status."""
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        return self.observe(), reward, self.game_status()

    def observe(self) -> np.ndarray:
        """Return the flattened environment state expected by the network."""
        return self.draw_env().reshape((1, -1))

    def draw_env(self) -> np.ndarray:
        """Return a copy of the maze with the current pirate position marked."""
        canvas = self._maze.copy()
        row, col, _ = self.state
        canvas[row, col] = PIRATE_MARK
        return canvas

    def game_status(self) -> str:
        """Return win, lose, or not_over for the current episode."""
        if self.total_reward < self.min_reward:
            return "lose"
        pirate_row, pirate_col, _ = self.state
        if (pirate_row, pirate_col) == self.target:
            return "win"
        return "not_over"

    def valid_actions(self, cell: tuple[int, int] | None = None) -> list[int]:
        """Return actions that remain inside the maze and avoid blocked cells."""
        if cell is None:
            row, col, _ = self.state
        else:
            row, col = cell

        nrows, ncols = self._maze.shape
        if not (0 <= row < nrows and 0 <= col < ncols):
            raise ValueError("cell is outside the maze")
        if self._maze[row, col] == 0.0:
            return []

        actions = [LEFT, UP, RIGHT, DOWN]
        if row == 0 or self._maze[row - 1, col] == 0.0:
            actions.remove(UP)
        if row == nrows - 1 or self._maze[row + 1, col] == 0.0:
            actions.remove(DOWN)
        if col == 0 or self._maze[row, col - 1] == 0.0:
            actions.remove(LEFT)
        if col == ncols - 1 or self._maze[row, col + 1] == 0.0:
            actions.remove(RIGHT)
        return actions
