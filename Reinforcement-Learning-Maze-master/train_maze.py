"""Command-line trainer for the grid maze reinforcement-learning examples."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np


MAZE_LAYOUT = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 2],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 2, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [1, 2, 0, 1, 0, 1, 2, 0],
        [2, 0, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("qtable", "qtrace", "sarsa", "sarsa-trace", "deep-q", "random"),
        default="qtable",
        help="RL model to train or run.",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--discount", type=float, default=0.90)
    parser.add_argument("--exploration-rate", type=float, default=0.10)
    parser.add_argument("--exploration-decay", type=float, default=0.995)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--eligibility-decay", type=float, default=0.80)
    parser.add_argument("--stop-at-convergence", action="store_true")
    parser.add_argument(
        "--render",
        choices=("nothing", "moves", "training"),
        default="nothing",
        help="Render no window, the played moves, or the training policy view.",
    )
    parser.add_argument(
        "--play-start",
        type=int,
        nargs=2,
        metavar=("COL", "ROW"),
        default=(4, 1),
        help="Start cell for the final demonstration game.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Optional path for the training reward/win-rate plot.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path for saving the trained model.",
    )
    parser.add_argument(
        "--save-bestmove",
        type=Path,
        default=None,
        help="Optional path for saving the learned best-action map.",
    )
    return parser.parse_args()


def build_model(model_name: str, game):
    import models

    if model_name == "random":
        return models.RandomModel(game)
    if model_name == "qtable":
        return models.QTableModel(game)
    if model_name == "qtrace":
        return models.QTableTraceModel(game)
    if model_name == "sarsa":
        return models.SarsaTableModel(game)
    if model_name == "sarsa-trace":
        return models.SarsaTableTraceModel(game)
    if model_name == "deep-q":
        if not hasattr(models, "QReplayNetworkModel"):
            raise RuntimeError(
                "deep-q requires tensorflow/keras. Install TensorFlow or use a tabular model."
            )
        return models.QReplayNetworkModel(game)
    raise ValueError(f"Unknown model: {model_name}")


def render_mode(name: str):
    from environment.maze import Render

    return {
        "nothing": Render.NOTHING,
        "moves": Render.MOVES,
        "training": Render.TRAINING,
    }[name]


def train_model(args: argparse.Namespace, model):
    if args.model == "random":
        return [], [], 0, None

    train_kwargs = {
        "discount": args.discount,
        "exploration_rate": args.exploration_rate,
        "exploration_decay": args.exploration_decay,
        "episodes": args.episodes,
    }
    if args.model in ("qtable", "qtrace", "sarsa", "sarsa-trace"):
        train_kwargs["learning_rate"] = args.learning_rate
    if args.model in ("qtrace", "sarsa-trace"):
        train_kwargs["eligibility_decay"] = args.eligibility_decay

    return model.train(stop_at_convergence=args.stop_at_convergence, **train_kwargs)


def save_training_plot(path: Path, model_name: str, reward_history, win_history) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True, figsize=(8, 6))
    fig.suptitle(model_name)
    ax1.plot(reward_history)
    ax1.set_xlabel("episode")
    ax1.set_ylabel("cumulative reward")
    if win_history:
        ax2.plot(*zip(*win_history))
    ax2.set_xlabel("episode")
    ax2.set_ylabel("win rate")
    fig.savefig(path, dpi=160)
    print(f"Saved training plot to {path}")


def save_model(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file)
    print(f"Saved trained model to {path}")


def save_bestmove_plot(path: Path, model, maze_layout: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from environment.maze import Action, Cell

    def clip(value: float) -> float:
        return max(min(value, 1.0), 0.0)

    path.parent.mkdir(parents=True, exist_ok=True)

    nrows, ncols = maze_layout.shape
    exit_cell = (ncols - 1, nrows - 1)
    empty = [
        (col, row)
        for col in range(ncols)
        for row in range(nrows)
        if maze_layout[row, col] == Cell.EMPTY and (col, row) != exit_cell
    ]
    traps = [
        (col, row)
        for col in range(ncols)
        for row in range(nrows)
        if maze_layout[row, col] == Cell.TRAP
    ]

    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 6))
    ax.set_title(f"{model.name} Best Move")
    ax.set_xticks(np.arange(0.5, nrows, step=1))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0.5, ncols, step=1))
    ax.set_yticklabels([])
    ax.grid(True)

    ax.plot(*exit_cell, "gs", markersize=30)
    ax.text(*exit_cell, "Exit", ha="center", va="center", color="white")
    for trap in traps:
        ax.plot(*trap, "s", color="#d95f02", markersize=30)
        ax.text(*trap, "Trap", ha="center", va="center", color="white", fontsize=8)

    for cell in empty:
        q_values = model.q(cell)
        actions = np.nonzero(q_values == np.max(q_values))[0]
        for action in actions:
            dx = 0.0
            dy = 0.0
            if action == Action.MOVE_LEFT:
                dx = -0.2
            if action == Action.MOVE_RIGHT:
                dx = 0.2
            if action == Action.MOVE_UP:
                dy = -0.2
            if action == Action.MOVE_DOWN:
                dy = 0.2

            color = clip((float(q_values[action]) + 1.0) / 2.0)
            ax.arrow(
                *cell,
                dx,
                dy,
                color=(1.0 - color, color, 0.0),
                head_width=0.2,
                head_length=0.1,
            )

    cmap = ListedColormap(["white", "black", "#f4a261", "#e76f51"])
    ax.imshow(maze_layout, cmap=cmap, vmin=Cell.EMPTY, vmax=Cell.CURRENT)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved best-move plot to {path}")


def main() -> None:
    args = parse_args()
    if args.render == "nothing":
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from environment.maze import Maze

    logging.basicConfig(
        format="%(levelname)-8s: %(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    game = Maze(MAZE_LAYOUT)
    game.render(render_mode(args.render))
    model = build_model(args.model, game)

    reward_history, win_history, trained_episodes, elapsed = train_model(args, model)
    if trained_episodes:
        print(f"Trained {trained_episodes} episodes in {elapsed}")

    status = game.play(model, start_cell=tuple(args.play_start))
    print(f"Final play from {tuple(args.play_start)}: {status.name}")

    if args.save_plot is not None and reward_history:
        save_training_plot(args.save_plot, model.name, reward_history, win_history)

    if args.save_model is not None and args.model != "random":
        save_model(args.save_model, model)

    if args.save_bestmove is not None and args.model != "random":
        save_bestmove_plot(args.save_bestmove, model, MAZE_LAYOUT)

    if args.render != "nothing":
        plt.show()


if __name__ == "__main__":
    main()
