"""Experience-replay memory for the treasure-maze DQN.

This module is based on the course-provided CS-370 starter component and has
been validated and documented for the maintained portfolio version.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np


class GameExperience:
    """Store transitions and prepare DQN training targets."""

    def __init__(
        self,
        model: Any,
        target_model: Any,
        max_memory: int = 100,
        discount: float = 0.95,
    ) -> None:
        if model is None or target_model is None:
            raise ValueError("model and target_model are required")
        if max_memory <= 0:
            raise ValueError("max_memory must be positive")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be between 0 and 1")

        self.model = model
        self.target_model = target_model
        self.max_memory = max_memory
        self.discount = discount
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=max_memory
        )
        self.num_actions = int(model.output_shape[-1])

    @staticmethod
    def _state_array(state: Any) -> np.ndarray:
        array = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if array.shape[1] == 0:
            raise ValueError("state cannot be empty")
        return array

    @staticmethod
    def _model_output(model: Any, states: np.ndarray) -> np.ndarray:
        output = model(states, training=False)
        if hasattr(output, "numpy"):
            output = output.numpy()
        return np.asarray(output, dtype=np.float32)

    def remember(self, episode: Sequence[Any]) -> None:
        """Store one `(state, action, reward, next_state, done)` transition."""
        if len(episode) != 5:
            raise ValueError("episode must contain five values")

        state, action, reward, next_state, done = episode
        state_array = self._state_array(state)
        next_state_array = self._state_array(next_state)
        if state_array.shape != next_state_array.shape:
            raise ValueError("state and next_state must have matching shapes")

        action_index = int(action)
        if not 0 <= action_index < self.num_actions:
            raise ValueError("action is outside the model output range")

        self.memory.append(
            (
                state_array,
                action_index,
                float(reward),
                next_state_array,
                bool(done),
            )
        )

    def predict(self, envstate: Any) -> np.ndarray:
        """Return predicted Q-values for one environment state."""
        return self._model_output(self.model, self._state_array(envstate))[0]

    def sample(self, batch_size: int = 32) -> list[tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """Randomly sample a fixed-size batch, using replacement when needed."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.memory:
            raise ValueError("cannot sample an empty replay memory")

        indexes = np.random.choice(
            len(self.memory),
            size=batch_size,
            replace=len(self.memory) < batch_size,
        )
        return [self.memory[int(index)] for index in indexes]

    def get_data(self, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Build network inputs and Bellman targets from replay memory."""
        batch = self.sample(batch_size)
        states = np.vstack([episode[0] for episode in batch])
        next_states = np.vstack([episode[3] for episode in batch])

        q_values = self._model_output(self.model, states)
        q_next = self._model_output(self.target_model, next_states)
        expected_shape = (batch_size, self.num_actions)
        if q_values.shape != expected_shape or q_next.shape != expected_shape:
            raise ValueError("model output shape does not match replay batch")

        targets = q_values.copy()
        for index, (_, action, reward, _, done) in enumerate(batch):
            if done:
                targets[index, action] = reward
            else:
                targets[index, action] = (
                    reward + self.discount * float(np.max(q_next[index]))
                )

        return states, targets
