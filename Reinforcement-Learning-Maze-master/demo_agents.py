"""Demo and evaluate trained Q-learning and DDQN maze agents."""

from __future__ import annotations

import argparse
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from environment.maze import Cell, Maze, Render, Status
from train_maze import MAZE_LAYOUT


@dataclass
class EpisodeResult:
    status: Status
    total_reward: float
    steps: int
    start_cell: tuple[int, int]
    final_cell: tuple[int, int]
    trajectory: list[tuple[int, int]]

    @property
    def trap(self) -> bool:
        col, row = self.final_cell
        return self.status == Status.LOSE and MAZE_LAYOUT[row, col] == Cell.TRAP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        choices=("both", "qlearning", "ddqn"),
        default="both",
        help="Which trained agent to evaluate.",
    )
    parser.add_argument(
        "--q-model",
        type=Path,
        default=Path("models/qtable_trap_model.pkl"),
        help="Path to the trained Q-learning pickle model.",
    )
    parser.add_argument(
        "--ddqn-model",
        type=Path,
        default=Path("models/dqn_trap_model.pt"),
        help="Path to the trained DDQN torch checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of random-start episodes. Use 0 to test every legal start once.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--start",
        type=int,
        nargs=2,
        metavar=("COL", "ROW"),
        default=None,
        help="Evaluate only one start cell instead of all/random starts.",
    )
    parser.add_argument(
        "--render-demo",
        choices=("none", "qlearning", "ddqn"),
        default="none",
        help="Show one rendered MOVES demo after evaluation.",
    )
    parser.add_argument(
        "--render-start",
        type=int,
        nargs=2,
        metavar=("COL", "ROW"),
        default=(4, 1),
        help="Start cell for --render-demo and --save-path-plot.",
    )
    parser.add_argument(
        "--save-path-plot",
        type=Path,
        default=None,
        help="Optional image path for the rendered trajectory of --render-demo.",
    )
    return parser.parse_args()


def load_qlearning_model(path: Path):
    with path.open("rb") as file:
        model = pickle.load(file)
    model.environment = Maze(MAZE_LAYOUT)
    return model


def load_ddqn_model(path: Path):
    import torch

    from train_dqn_maze import DQNAgent, QNetwork

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    maze_layout = np.asarray(checkpoint.get("maze_layout", MAZE_LAYOUT))
    game = Maze(maze_layout)
    state_dim = int(checkpoint.get("state_dim", game.maze.size))
    action_dim = int(checkpoint.get("action_dim", len(game.actions)))
    network = QNetwork(state_dim, action_dim)
    network.load_state_dict(checkpoint["model_state"])
    network.eval()
    agent = DQNAgent(game, network)
    agent.name = "DDQNModel"
    return agent


def legal_start_cells() -> list[tuple[int, int]]:
    nrows, ncols = MAZE_LAYOUT.shape
    exit_cell = (ncols - 1, nrows - 1)
    return [
        (col, row)
        for col in range(ncols)
        for row in range(nrows)
        if MAZE_LAYOUT[row, col] == Cell.EMPTY and (col, row) != exit_cell
    ]


def build_start_cells(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.start is not None:
        return [tuple(args.start)]

    starts = legal_start_cells()
    if args.episodes <= 0:
        return starts

    return [random.choice(starts) for _ in range(args.episodes)]


def run_episode(
    model,
    start_cell: tuple[int, int],
    *,
    render: Render = Render.NOTHING,
    max_steps: int = 200,
) -> EpisodeResult:
    game = Maze(MAZE_LAYOUT)
    game.render(render)
    model.environment = game

    state = game.reset(start_cell)
    total_reward = 0.0
    trajectory = [start_cell]
    status = Status.PLAYING

    for step in range(1, max_steps + 1):
        action = model.predict(state)
        state, reward, status = game.step(action)
        total_reward += reward
        current_cell = tuple(int(value) for value in state.flatten())
        trajectory.append(current_cell)
        if status in (Status.WIN, Status.LOSE):
            return EpisodeResult(
                status=status,
                total_reward=total_reward,
                steps=step,
                start_cell=start_cell,
                final_cell=current_cell,
                trajectory=trajectory,
            )

    return EpisodeResult(
        status=Status.LOSE,
        total_reward=total_reward,
        steps=max_steps,
        start_cell=start_cell,
        final_cell=trajectory[-1],
        trajectory=trajectory,
    )


def evaluate_agent(name: str, model, starts: list[tuple[int, int]], max_steps: int) -> list[EpisodeResult]:
    results = [run_episode(model, start, max_steps=max_steps) for start in starts]
    wins = sum(result.status == Status.WIN for result in results)
    losses = len(results) - wins
    traps = sum(result.trap for result in results)
    avg_reward = float(np.mean([result.total_reward for result in results]))
    avg_steps = float(np.mean([result.steps for result in results]))
    max_steps_used = max(result.steps for result in results)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"episodes      : {len(results)}")
    print(f"wins / losses : {wins} / {losses}")
    print(f"trap losses   : {traps}")
    print(f"win rate      : {wins / len(results):.3f}")
    print(f"avg reward    : {avg_reward:.3f}")
    print(f"avg steps     : {avg_steps:.2f}")
    print(f"max steps     : {max_steps_used}")

    failed = [result for result in results if result.status != Status.WIN]
    if failed:
        examples = ", ".join(
            f"{result.start_cell}->{result.final_cell}/{result.status.name}"
            for result in failed[:8]
        )
        print(f"failed starts : {examples}")
    else:
        print("failed starts : none")

    return results


def save_path_plot(path: Path, result: EpisodeResult, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    path.parent.mkdir(parents=True, exist_ok=True)
    nrows, ncols = MAZE_LAYOUT.shape
    exit_cell = (ncols - 1, nrows - 1)
    traps = [
        (col, row)
        for col in range(ncols)
        for row in range(nrows)
        if MAZE_LAYOUT[row, col] == Cell.TRAP
    ]

    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 6))
    ax.set_title(title)
    ax.set_xticks(np.arange(0.5, nrows, step=1))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0.5, ncols, step=1))
    ax.set_yticklabels([])
    ax.grid(True)
    ax.imshow(
        MAZE_LAYOUT,
        cmap=ListedColormap(["white", "black", "#f4a261", "#e76f51"]),
        vmin=Cell.EMPTY,
        vmax=Cell.CURRENT,
    )
    ax.plot(*exit_cell, "gs", markersize=28)
    ax.text(*exit_cell, "Exit", ha="center", va="center", color="white")
    for trap in traps:
        ax.plot(*trap, "s", color="#d95f02", markersize=28)
        ax.text(*trap, "Trap", ha="center", va="center", color="white", fontsize=8)

    xs = [cell[0] for cell in result.trajectory]
    ys = [cell[1] for cell in result.trajectory]
    ax.plot(xs, ys, "o-", color="#0072b2", linewidth=2.0, markersize=5)
    ax.plot(*result.start_cell, "rs", markersize=24)
    ax.text(*result.start_cell, "Start", ha="center", va="center", color="white", fontsize=8)
    ax.plot(*result.final_cell, "o", color="#cc79a7", markersize=14)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved demo path plot to {path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.render_demo == "none":
        import matplotlib

        matplotlib.use("Agg")

    starts = build_start_cells(args)
    loaded_agents = {}

    if args.agent in ("both", "qlearning"):
        loaded_agents["qlearning"] = load_qlearning_model(args.q_model)
    if args.agent in ("both", "ddqn"):
        loaded_agents["ddqn"] = load_ddqn_model(args.ddqn_model)

    for name, model in loaded_agents.items():
        evaluate_agent(name, model, starts, args.max_steps)

    if args.render_demo != "none":
        import matplotlib.pyplot as plt

        model = loaded_agents.get(args.render_demo)
        if model is None:
            raise ValueError(f"--render-demo {args.render_demo} requires --agent both or {args.render_demo}")
        render_start = tuple(args.render_start)
        result = run_episode(
            model,
            render_start,
            render=Render.MOVES,
            max_steps=args.max_steps,
        )
        print(
            f"\nRendered {args.render_demo} from {render_start}: "
            f"{result.status.name}, reward={result.total_reward:.2f}, steps={result.steps}"
        )
        if args.save_path_plot is not None:
            save_path_plot(
                args.save_path_plot,
                result,
                f"{args.render_demo} demo: {result.status.name}",
            )
        plt.show()
    elif args.save_path_plot is not None:
        name = "qlearning" if "qlearning" in loaded_agents else "ddqn"
        result = run_episode(
            loaded_agents[name],
            tuple(args.render_start),
            max_steps=args.max_steps,
        )
        save_path_plot(args.save_path_plot, result, f"{name} demo: {result.status.name}")


if __name__ == "__main__":
    main()
