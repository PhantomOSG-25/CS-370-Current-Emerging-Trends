"""Tests for the treasure-maze environment."""

import unittest

import numpy as np

from TreasureMaze import DOWN, LEFT, RIGHT, TreasureMaze


class TreasureMazeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.maze = np.array(
            [
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )

    def test_observation_has_one_flattened_state(self) -> None:
        environment = TreasureMaze(self.maze)

        self.assertEqual((1, 4), environment.observe().shape)
        self.assertEqual([RIGHT], environment.valid_actions())

    def test_invalid_action_receives_invalid_penalty_without_moving(self) -> None:
        environment = TreasureMaze(self.maze)

        _, reward, status = environment.act(LEFT)

        self.assertEqual((0, 0, "invalid"), environment.state)
        self.assertEqual(-0.75, reward)
        self.assertEqual("not_over", status)

    def test_valid_path_reaches_treasure(self) -> None:
        environment = TreasureMaze(self.maze)

        environment.act(RIGHT)
        _, reward, status = environment.act(DOWN)

        self.assertEqual(1.0, reward)
        self.assertEqual("win", status)

    def test_rejects_blocked_target_and_invalid_reset(self) -> None:
        with self.assertRaises(ValueError):
            TreasureMaze([[1.0, 1.0], [1.0, 0.0]])

        environment = TreasureMaze(self.maze)
        with self.assertRaises(ValueError):
            environment.reset((1, 0))


if __name__ == "__main__":
    unittest.main()
