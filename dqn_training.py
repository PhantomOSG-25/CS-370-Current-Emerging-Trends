"""Train and evaluate a deep Q-network on the treasure-maze environment."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from GameExperience import GameExperience
from TreasureMaze import TreasureMaze


NUM_ACTIONS = 4
DEFAULT_MAZE = np.array(
    [
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


def build_model(maze: np.ndarray) -> tf.keras.Model:
    """Build the fully connected Q-network used by the agent."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(maze.size,)),
            tf.keras.layers.Dense(maze.size),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(maze.size),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(NUM_ACTIONS),
        ]
    )


def create_train_step(model: tf.keras.Model):
    """Create a compiled training step bound to one model and optimizer."""
    loss_function = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(inputs: np.ndarray, targets: np.ndarray) -> tf.Tensor:
        with tf.GradientTape() as tape:
            q_values = model(inputs, training=True)
            loss = loss_function(targets, q_values)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    return train_step


def play_game(
    model: tf.keras.Model,
    environment: TreasureMaze,
    pirate_cell: tuple[int, int],
    max_steps: int | None = None,
) -> bool:
    """Run a greedy evaluation episode from one starting position."""
    environment.reset(pirate_cell)
    envstate = environment.observe()
    step_limit = max_steps or environment.maze.size * 4

    for _ in range(step_limit):
        state = np.asarray(envstate, dtype=np.float32).reshape(1, -1)
        q_values = model(state, training=False).numpy()[0]
        action = int(np.argmax(q_values))
        envstate, _, status = environment.act(action)
        if status == "win":
            return True
        if status == "lose":
            return False
    return False


def completion_check(
    model: tf.keras.Model,
    maze_or_environment: np.ndarray | TreasureMaze,
    max_steps: int | None = None,
) -> bool:
    """Confirm that the greedy policy wins from every valid starting cell."""
    environment = (
        maze_or_environment
        if isinstance(maze_or_environment, TreasureMaze)
        else TreasureMaze(maze_or_environment)
    )
    return all(
        play_game(model, environment, cell, max_steps=max_steps)
        for cell in environment.free_cells
        if environment.valid_actions(cell)
    )


def format_duration(seconds: float) -> str:
    if seconds < 400:
        return f"{seconds:.1f} seconds"
    if seconds < 4000:
        return f"{seconds / 60.0:.2f} minutes"
    return f"{seconds / 3600.0:.2f} hours"


def qtrain(
    model: tf.keras.Model,
    maze: np.ndarray,
    *,
    epochs: int = 15_000,
    max_memory: int = 1_000,
    batch_size: int = 50,
    target_update_frequency: int = 50,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the DQN and return summary metrics from the run."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    random_generator = random.Random(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    environment = TreasureMaze(maze)
    target_model = tf.keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())
    experience = GameExperience(model, target_model, max_memory=max_memory)
    train_step = create_train_step(model)

    win_history: list[int] = []
    history_window = environment.maze.size // 2
    win_rate = 0.0
    start_time = dt.datetime.now()
    completed_epoch = 0

    for epoch in range(epochs):
        completed_epoch = epoch + 1
        environment.reset(random_generator.choice(environment.free_cells))
        envstate = environment.observe()
        game_over = False
        episode_steps = 0
        loss = 0.0
        max_moves = environment.maze.size * 4

        while not game_over and episode_steps < max_moves:
            valid_actions = environment.valid_actions()
            if not valid_actions:
                win_history.append(0)
                break

            previous_state = envstate
            if np.random.random() < epsilon:
                action = random_generator.choice(valid_actions)
            else:
                action = int(np.argmax(experience.predict(previous_state)))

            envstate, reward, status = environment.act(action)
            episode_steps += 1
            timed_out = episode_steps >= max_moves and status == "not_over"
            game_over = status in {"win", "lose"} or timed_out

            if game_over:
                win_history.append(1 if status == "win" else 0)

            experience.remember(
                [previous_state, action, reward, envstate, game_over]
            )
            inputs, targets = experience.get_data(batch_size)
            loss = float(train_step(inputs, targets).numpy())

        if len(win_history) >= history_window:
            win_rate = sum(win_history[-history_window:]) / history_window

        if epoch % target_update_frequency == 0:
            target_model.set_weights(model.get_weights())

        elapsed = (dt.datetime.now() - start_time).total_seconds()
        print(
            f"Epoch {epoch:05d}/{epochs - 1} | loss {loss:.4f} | "
            f"steps {episode_steps:03d} | win rate {win_rate:.3f} | "
            f"{format_duration(elapsed)}"
        )

        epsilon = (
            epsilon_min
            if win_rate > 0.9
            else max(epsilon * epsilon_decay, epsilon_min)
        )

        if win_rate >= 0.999 and completion_check(model, environment):
            print(f"Reached full evaluation completion at epoch {epoch}")
            break

    elapsed = (dt.datetime.now() - start_time).total_seconds()
    return {
        "epochs_completed": completed_epoch,
        "win_rate": win_rate,
        "evaluation_complete": completion_check(model, environment),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("treasure_maze.keras"))
    args = parser.parse_args()

    model = build_model(DEFAULT_MAZE)
    metrics = qtrain(model, DEFAULT_MAZE, epochs=args.epochs, seed=args.seed)
    model.save(args.output)
    print(f"Saved model to {args.output}")
    print(metrics)


if __name__ == "__main__":
    main()
