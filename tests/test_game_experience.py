"""Tests for DQN experience replay."""

import unittest

import numpy as np

from GameExperience import GameExperience


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def numpy(self) -> np.ndarray:
        return self.values


class FakeModel:
    output_shape = (None, 4)

    def __init__(self, q_values: list[float]) -> None:
        self.q_values = np.asarray(q_values, dtype=np.float32)

    def __call__(self, states: np.ndarray, training: bool = False) -> FakeTensor:
        del training
        values = np.tile(self.q_values, (len(states), 1))
        return FakeTensor(values)


class GameExperienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeModel([1.0, 2.0, 3.0, 4.0])
        self.target_model = FakeModel([4.0, 3.0, 2.0, 1.0])
        self.experience = GameExperience(
            self.model,
            self.target_model,
            max_memory=2,
            discount=0.95,
        )
        self.state = np.array([[1.0, 0.0, 0.5, 1.0]], dtype=np.float32)
        self.next_state = np.array([[1.0, 0.5, 1.0, 1.0]], dtype=np.float32)

    def test_predict_returns_one_q_value_per_action(self) -> None:
        prediction = self.experience.predict(self.state)

        np.testing.assert_array_equal(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            prediction,
        )

    def test_terminal_transition_uses_immediate_reward(self) -> None:
        self.experience.remember([self.state, 2, 1.0, self.next_state, True])

        inputs, targets = self.experience.get_data(batch_size=1)

        self.assertEqual((1, 4), inputs.shape)
        self.assertEqual(1.0, targets[0, 2])

    def test_nonterminal_transition_uses_target_network(self) -> None:
        self.experience.remember([self.state, 1, 0.5, self.next_state, False])

        _, targets = self.experience.get_data(batch_size=1)

        self.assertAlmostEqual(4.3, float(targets[0, 1]), places=5)

    def test_memory_is_bounded_and_empty_sampling_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.experience.sample()

        for reward in (0.0, 0.5, 1.0):
            self.experience.remember(
                [self.state, 0, reward, self.next_state, True]
            )

        self.assertEqual(2, len(self.experience.memory))


if __name__ == "__main__":
    unittest.main()
