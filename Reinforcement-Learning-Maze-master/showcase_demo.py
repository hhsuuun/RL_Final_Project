"""Live showcase demo for Q-learning, DQN, and DDQN agents.

Run this script when you need to demonstrate the agents walking through the maze.
It can show both the fixed trap maze and dynamic mazes where walls, traps, and
rewards change between rounds.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path

if "--no-live-render" not in sys.argv:
    os.environ["RL_MAZE_LIVE_RENDER"] = "1"

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import matplotlib

if os.environ.get("RL_MAZE_LIVE_RENDER") == "1":
    for backend in ("MacOSX", "TkAgg", "QtAgg"):
        try:
            matplotlib.use(backend, force=True)
            break
        except Exception:
            continue
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from dynamic_maze_compare import (
    DynamicDDQNAgent,
    PositionQAgent,
    QNetwork as DynamicQNetwork,
    apply_reward_config,
    sample_dynamic_episode,
    valid_action_mask,
    valid_actions,
)
from environment.maze import Cell, Maze, Status
from train_dqn_maze import DQNAgent, QNetwork as FixedQNetwork
from train_maze import MAZE_LAYOUT


@dataclass
class RoundSpec:
    layout: np.ndarray
    start_cell: tuple[int, int]
    reward_config: object | None = None


@dataclass
class LiveState:
    game: Maze
    cell: tuple[int, int]
    total_reward: float
    trajectory: list[tuple[int, int]]
    status: Status
    done: bool
    ended_in_trap: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("fixed", "dynamic", "both"),
        default="both",
        help="Which demo to show.",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--live-render", dest="live_render", action="store_true", default=True)
    parser.add_argument("--no-live-render", dest="live_render", action="store_false")
    parser.add_argument("--live-delay", type=float, default=0.04)
    parser.add_argument("--live-step-stride", type=int, default=3)
    parser.add_argument("--round-pause", type=float, default=0.80)
    parser.add_argument("--fixed-q-model", type=Path, default=Path("models/qtable_trap_model.pkl"))
    parser.add_argument("--fixed-dqn-model", type=Path, default=Path("models/dqn_test_model.pt"))
    parser.add_argument("--fixed-ddqn-model", type=Path, default=Path("models/dqn_trap_model.pt"))
    parser.add_argument("--dynamic-q-model", type=Path, default=Path("models/dynamic_qlearning.pkl"))
    parser.add_argument("--dynamic-dqn-model", type=Path, default=Path("models/dynamic_dqn.pt"))
    parser.add_argument("--dynamic-ddqn-model", type=Path, default=Path("models/dynamic_ddqn.pt"))
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--wall-prob", type=float, default=0.18)
    parser.add_argument("--trap-prob", type=float, default=0.10)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.fixed_q_model = project_path(args.fixed_q_model)
    args.fixed_dqn_model = project_path(args.fixed_dqn_model)
    args.fixed_ddqn_model = project_path(args.fixed_ddqn_model)
    args.dynamic_q_model = project_path(args.dynamic_q_model)
    args.dynamic_dqn_model = project_path(args.dynamic_dqn_model)
    args.dynamic_ddqn_model = project_path(args.dynamic_ddqn_model)
    return args


def load_pickle_model(path: Path):
    cached = sys.modules.get("models")
    if cached is not None:
        cached_file = str(getattr(cached, "__file__", ""))
        if str(PROJECT_DIR / "models") not in cached_file:
            for name in list(sys.modules):
                if name == "models" or name.startswith("models."):
                    sys.modules.pop(name, None)
    with path.open("rb") as file:
        return pickle.load(file)


def load_fixed_dqn_agent(path: Path, label: str) -> DQNAgent:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    game = Maze(MAZE_LAYOUT)
    state_dim = int(checkpoint.get("state_dim", game.maze.size))
    action_dim = int(checkpoint.get("action_dim", len(game.actions)))
    network = FixedQNetwork(state_dim, action_dim)
    network.load_state_dict(checkpoint["model_state"])
    network.eval()
    agent = DQNAgent(game, network)
    agent.name = label
    return agent


def load_dynamic_deep_agent(path: Path, label: str) -> DynamicDDQNAgent:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    size = int(checkpoint.get("size", 8))
    state_dim = int(checkpoint.get("state_dim", size * size * 4 + 7))
    action_dim = int(checkpoint.get("action_dim", len(Maze.actions)))
    network = DynamicQNetwork(state_dim, action_dim)
    network.load_state_dict(checkpoint["model_state"])
    network.eval()
    agent = DynamicDDQNAgent(size, size, network)
    agent.name = label
    return agent


def load_fixed_agents(args: argparse.Namespace) -> dict[str, object]:
    q_agent = load_pickle_model(args.fixed_q_model)
    q_agent.environment = Maze(MAZE_LAYOUT)
    return {
        "Q-learning": q_agent,
        "DQN": load_fixed_dqn_agent(args.fixed_dqn_model, "DQN"),
        "DDQN": load_fixed_dqn_agent(args.fixed_ddqn_model, "DDQN"),
    }


def load_dynamic_agents(args: argparse.Namespace) -> dict[str, object]:
    q_agent = load_pickle_model(args.dynamic_q_model)
    dqn_path = args.dynamic_dqn_model
    dqn_label = "DQN"
    if not dqn_path.exists():
        dqn_path = args.dynamic_ddqn_model
        dqn_label = "DQN fallback"
        print(
            "Note: models/dynamic_dqn.pt was not found; "
            "using models/dynamic_ddqn.pt for the DQN display column."
        )
    return {
        "Q-learning": q_agent,
        dqn_label: load_dynamic_deep_agent(dqn_path, dqn_label),
        "DDQN": load_dynamic_deep_agent(args.dynamic_ddqn_model, "DDQN"),
    }


def fixed_rounds(count: int) -> list[RoundSpec]:
    nrows, ncols = MAZE_LAYOUT.shape
    exit_cell = (ncols - 1, nrows - 1)
    starts = [
        (col, row)
        for row in range(nrows)
        for col in range(ncols)
        if MAZE_LAYOUT[row, col] == Cell.EMPTY and (col, row) != exit_cell
    ]
    preferred = [(4, 1), (0, 0), (2, 4), (6, 0), (0, 7)]
    ordered = [cell for cell in preferred if cell in starts]
    ordered += [cell for cell in starts if cell not in ordered]
    return [RoundSpec(MAZE_LAYOUT.copy(), ordered[idx % len(ordered)]) for idx in range(count)]


def dynamic_rounds(args: argparse.Namespace) -> list[RoundSpec]:
    rounds = []
    for _ in range(args.rounds):
        spec = sample_dynamic_episode(args.size, args.wall_prob, args.trap_prob)
        rounds.append(RoundSpec(spec.layout, spec.start_cell, spec.reward_config))
    return rounds


def init_live_state(spec: RoundSpec) -> LiveState:
    if spec.reward_config is not None:
        apply_reward_config(spec.reward_config)
    game = Maze(spec.layout, start_cell=spec.start_cell)
    state = game.reset(spec.start_cell)
    cell = tuple(int(value) for value in state.flatten())
    return LiveState(
        game=game,
        cell=cell,
        total_reward=0.0,
        trajectory=[cell],
        status=Status.PLAYING,
        done=False,
    )


def infer_action(mode: str, label: str, agent, spec: RoundSpec, live: LiveState) -> int:
    if mode == "fixed":
        agent.environment = live.game
        return int(agent.predict(np.array([[*live.cell]])))

    if label == "Q-learning":
        return int(agent.predict(live.cell, valid_actions(spec.layout, live.cell)))

    state = agent.encode(spec.layout, live.cell, spec.reward_config)
    mask = valid_action_mask(spec.layout, live.cell)
    return int(agent.predict(state, mask))


def step_agent(mode: str, label: str, agent, spec: RoundSpec, live: LiveState) -> None:
    if live.done:
        return

    action = infer_action(mode, label, agent, spec, live)
    next_state, reward, status = live.game.step(action)
    next_cell = tuple(int(value) for value in next_state.flatten())
    live.cell = next_cell
    live.total_reward += reward
    live.trajectory.append(next_cell)
    live.status = status

    if status in (Status.WIN, Status.LOSE):
        live.done = True
        col, row = next_cell
        live.ended_in_trap = status == Status.LOSE and spec.layout[row, col] == Cell.TRAP


def setup_axis(ax, spec: RoundSpec, title: str):
    cmap = ListedColormap(["white", "black", "#f4a261", "#e76f51"])
    ax.clear()
    ax.imshow(spec.layout, cmap=cmap, vmin=Cell.EMPTY, vmax=Cell.CURRENT)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(0.5, spec.layout.shape[1], step=1))
    ax.set_yticks(np.arange(0.5, spec.layout.shape[0], step=1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    exit_cell = (spec.layout.shape[1] - 1, spec.layout.shape[0] - 1)
    ax.plot(*exit_cell, "gs", markersize=18)
    ax.text(*exit_cell, "Exit", ha="center", va="center", color="white", fontsize=8)
    ax.plot(*spec.start_cell, "rs", markersize=18)
    ax.text(*spec.start_cell, "Start", ha="center", va="center", color="white", fontsize=7)
    for row in range(spec.layout.shape[0]):
        for col in range(spec.layout.shape[1]):
            if spec.layout[row, col] == Cell.TRAP:
                ax.text(col, row, "Trap", ha="center", va="center", color="white", fontsize=7)
    line = ax.plot([], [], "o-", linewidth=2.0, markersize=5)[0]
    head = ax.plot([], [], "o", color="#cc79a7", markersize=12)[0]
    return line, head


def result_text(live: LiveState) -> str:
    if live.status == Status.WIN:
        status_name = "WIN"
        reason = "win"
    elif live.ended_in_trap:
        status_name = "LOSE"
        reason = "trap"
    else:
        status_name = "LOSE"
        reason = "timeout"
    return f"{status_name} / {reason} / reward={live.total_reward:.2f} / steps={len(live.trajectory) - 1}"


def run_live_section(
    mode: str,
    title: str,
    agents: dict[str, object],
    rounds: list[RoundSpec],
    args: argparse.Namespace,
) -> None:
    colors = {
        "Q-learning": "#0072b2",
        "DQN": "#e69f00",
        "DQN fallback": "#e69f00",
        "DDQN": "#cc79a7",
    }
    labels = list(agents.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(5.4 * len(labels), 5.5), tight_layout=True)
    axes = np.atleast_1d(axes)
    if hasattr(fig.canvas, "manager"):
        fig.canvas.manager.set_window_title(title)

    for round_idx, spec in enumerate(rounds, start=1):
        lives = {label: init_live_state(spec) for label in labels}
        lines = {}
        heads = {}
        for ax, label in zip(axes, labels):
            line, head = setup_axis(ax, spec, f"{title} | Round {round_idx}/{len(rounds)}\n{label}")
            line.set_color(colors.get(label, "#0072b2"))
            lines[label] = line
            heads[label] = head

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(max(args.live_delay, 0.01))

        for step_idx in range(1, args.max_steps + 1):
            for label in labels:
                step_agent(mode, label, agents[label], spec, lives[label])

            if (
                step_idx % max(1, args.live_step_stride) == 0
                or all(live.done for live in lives.values())
                or step_idx == args.max_steps
            ):
                for ax, label in zip(axes, labels):
                    live = lives[label]
                    xs = [cell[0] for cell in live.trajectory]
                    ys = [cell[1] for cell in live.trajectory]
                    lines[label].set_data(xs, ys)
                    heads[label].set_data([xs[-1]], [ys[-1]])
                    ax.set_title(
                        f"{title} | Round {round_idx}/{len(rounds)}\n"
                        f"{label}: step={step_idx}, reward={live.total_reward:.2f}",
                        fontsize=10,
                    )
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(max(args.live_delay, 0.01))

            if all(live.done for live in lives.values()):
                break

        for ax, label in zip(axes, labels):
            ax.set_title(
                f"{title} | Round {round_idx}/{len(rounds)}\n{label}: {result_text(lives[label])}",
                fontsize=10,
            )
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(max(args.round_pause, 0.01))

        print(f"\n{title} round {round_idx}")
        for label in labels:
            print(f"{label:12s}: {result_text(lives[label])}")

    print(f"\n{title} finished. Close the window to continue.")
    plt.show()


def main() -> None:
    args = normalize_paths(parse_args())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.live_render:
        print("Live render is disabled because --no-live-render was used.")
    else:
        print(f"Opening live walking-path demo window with matplotlib backend: {matplotlib.get_backend()}")

    if args.mode in ("fixed", "both"):
        run_live_section(
            "fixed",
            "Fixed Maze Demo",
            load_fixed_agents(args),
            fixed_rounds(args.rounds),
            args,
        )

    if args.mode in ("dynamic", "both"):
        run_live_section(
            "dynamic",
            "Dynamic Maze Demo",
            load_dynamic_agents(args),
            dynamic_rounds(args),
            args,
        )


if __name__ == "__main__":
    main()
