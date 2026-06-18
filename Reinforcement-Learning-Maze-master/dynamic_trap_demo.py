"""Run a 10-round demo where trap positions change every round.

The same sampled round is given to Q-learning and DDQN, so their results are
directly comparable.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace

if "--live-render" in sys.argv:
    os.environ["RL_MAZE_LIVE_RENDER"] = "1"

import matplotlib

if os.environ.get("RL_MAZE_LIVE_RENDER") != "1":
    matplotlib.use("Agg")
import numpy as np
import torch

from dynamic_maze_compare import (
    DynamicDDQNAgent,
    DynamicEpisode,
    EvalResult,
    PositionQAgent,
    QNetwork,
    RewardConfig,
    reachable_safe_cells,
    run_ddqn_episode,
    run_q_episode,
    sample_dynamic_episode,
    sample_reward_config,
    summarize,
)
from environment.maze import Cell, Maze, Status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("both", "qlearning", "ddqn"), default="both")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--q-model", type=Path, default=Path("models/dynamic_qlearning.pkl"))
    parser.add_argument("--ddqn-model", type=Path, default=Path("models/dynamic_ddqn.pt"))
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--wall-prob", type=float, default=0.18)
    parser.add_argument("--trap-prob", type=float, default=0.10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--dynamic-walls",
        action="store_true",
        help="Also change wall positions each round. Default changes trap positions only.",
    )
    parser.add_argument(
        "--fixed-reward",
        action="store_true",
        help="Use one reward setting for all rounds instead of changing reward each round.",
    )
    parser.add_argument(
        "--live-render",
        action="store_true",
        help="Open a live matplotlib window and animate each round.",
    )
    parser.add_argument(
        "--live-delay",
        type=float,
        default=0.04,
        help="Seconds between animation steps for --live-render.",
    )
    parser.add_argument(
        "--live-step-stride",
        type=int,
        default=4,
        help="Render every Nth trajectory step so long timeout episodes do not look frozen.",
    )
    parser.add_argument(
        "--round-pause",
        type=float,
        default=0.60,
        help="Seconds to pause after each rendered round.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=Path("plots/dynamic_trap_10round_demo.png"),
    )
    parser.add_argument(
        "--save-round-dir",
        type=Path,
        default=None,
        help="Directory for one rendered image per round. Use 'none' to disable.",
    )
    parser.add_argument("--save-csv", type=Path, default=Path("plots/dynamic_trap_10round_demo.csv"))
    return parser.parse_args()


def load_qlearning_model(path: Path) -> PositionQAgent:
    # PositionQAgent is imported into this __main__ module so pickle files saved
    # from dynamic_maze_compare.py can still resolve the class name.
    with path.open("rb") as file:
        return pickle.load(file)


def load_ddqn_model(path: Path) -> DynamicDDQNAgent:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    size = int(checkpoint.get("size", 8))
    state_dim = int(checkpoint.get("state_dim", size * size * 4 + 7))
    action_dim = int(checkpoint.get("action_dim", len(Maze.actions)))
    network = QNetwork(state_dim, action_dim)
    network.load_state_dict(checkpoint["model_state"])
    network.eval()
    return DynamicDDQNAgent(size, size, network)


def sample_wall_layout(size: int, wall_prob: float) -> np.ndarray:
    for _ in range(200):
        layout = np.zeros((size, size), dtype=np.int8)
        exit_cell = (size - 1, size - 1)
        for row in range(size):
            for col in range(size):
                if (col, row) == exit_cell:
                    continue
                if random.random() < wall_prob:
                    layout[row, col] = Cell.OCCUPIED
        if len(reachable_safe_cells(layout)) >= max(4, size // 2):
            return layout
    raise RuntimeError("Could not sample a solvable wall layout. Lower --wall-prob.")


def sample_traps_on_layout(base_layout: np.ndarray, trap_prob: float) -> np.ndarray:
    size = base_layout.shape[0]
    exit_cell = (size - 1, size - 1)
    for _ in range(200):
        layout = base_layout.copy()
        for row in range(size):
            for col in range(size):
                if (col, row) == exit_cell or layout[row, col] == Cell.OCCUPIED:
                    continue
                if random.random() < trap_prob:
                    layout[row, col] = Cell.TRAP
        safe_starts = reachable_safe_cells(layout)
        if safe_starts:
            return layout
    raise RuntimeError("Could not sample traps while keeping a safe path. Lower --trap-prob.")


def sample_rounds(args: argparse.Namespace) -> list[DynamicEpisode]:
    reward_config = sample_reward_config()
    base_layout = None if args.dynamic_walls else sample_wall_layout(args.size, args.wall_prob)
    rounds = []

    for _ in range(args.rounds):
        if args.dynamic_walls:
            episode = sample_dynamic_episode(args.size, args.wall_prob, args.trap_prob)
            if args.fixed_reward:
                episode.reward_config = reward_config
        else:
            layout = sample_traps_on_layout(base_layout, args.trap_prob)
            safe_starts = reachable_safe_cells(layout)
            episode = DynamicEpisode(
                layout=layout,
                start_cell=random.choice(safe_starts),
                reward_config=reward_config if args.fixed_reward else sample_reward_config(),
            )
        rounds.append(episode)

    return rounds


def run_demo_agent(agent_name: str, agent, rounds: list[DynamicEpisode], max_steps: int):
    results = []
    ddqn_args = SimpleNamespace(max_steps=max_steps, reward_scale=30.0, batch_size=64, learning_starts=512, gamma=0.95)

    for episode in rounds:
        if agent_name == "qlearning":
            result = run_q_episode(
                agent,
                episode,
                max_steps=max_steps,
                epsilon=0.0,
                learning_rate=0.10,
                gamma=0.95,
                train=False,
            )
        else:
            result, _ = run_ddqn_episode(
                agent,
                episode,
                replay=None,
                q_net=None,
                target_net=None,
                optimizer=None,
                args=ddqn_args,
                epsilon=0.0,
                train=False,
            )
        results.append(result)
    return results


def infer_q_step(agent: PositionQAgent, cell: tuple[int, int], episode: DynamicEpisode) -> int:
    from dynamic_maze_compare import valid_actions

    return agent.predict(cell, valid_actions(episode.layout, cell))


def infer_ddqn_step(agent: DynamicDDQNAgent, cell: tuple[int, int], episode: DynamicEpisode) -> int:
    from dynamic_maze_compare import valid_action_mask

    state = agent.encode(episode.layout, cell, episode.reward_config)
    mask = valid_action_mask(episode.layout, cell)
    return agent.predict(state, mask)


def print_results(name: str, results) -> None:
    summary = summarize(results)
    print(f"\n{name}")
    print("-" * len(name))
    for idx, result in enumerate(results, start=1):
        reason = "trap" if result.ended_in_trap else ("win" if result.status == Status.WIN else "timeout")
        print(
            f"round {idx:02d}: {result.status.name:4s} | {reason:7s} | "
            f"reward={result.total_reward:7.2f} | steps={result.steps:3d} | "
            f"start={result.start_cell} | final={result.final_cell}"
        )
    print(
        "summary : "
        f"win_rate={summary['win_rate']:.3f}, "
        f"wins={summary['wins']:.0f}, "
        f"trap_losses={summary['trap_losses']:.0f}, "
        f"timeouts={summary['timeout_losses']:.0f}, "
        f"avg_reward={summary['avg_reward']:.2f}, "
        f"avg_steps={summary['avg_steps']:.2f}"
    )


def save_csv(path: Path, all_results: dict[str, list]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "agent",
                "round",
                "status",
                "ended_in_trap",
                "total_reward",
                "steps",
                "start_cell",
                "final_cell",
            ],
        )
        writer.writeheader()
        for agent_name, results in all_results.items():
            for idx, result in enumerate(results, start=1):
                writer.writerow(
                    {
                        "agent": agent_name,
                        "round": idx,
                        "status": result.status.name,
                        "ended_in_trap": result.ended_in_trap,
                        "total_reward": result.total_reward,
                        "steps": result.steps,
                        "start_cell": result.start_cell,
                        "final_cell": result.final_cell,
                    }
                )
    print(f"\nSaved round metrics to {path}")


def save_plot(path: Path, rounds: list[DynamicEpisode], all_results: dict[str, list]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    path.parent.mkdir(parents=True, exist_ok=True)
    ncols = 2
    nrows = int(np.ceil(len(rounds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.6 * nrows), tight_layout=True)
    axes = np.atleast_1d(axes).flatten()
    cmap = ListedColormap(["white", "black", "#f4a261", "#e76f51"])

    for idx, episode in enumerate(rounds):
        ax = axes[idx]
        ax.imshow(episode.layout, cmap=cmap, vmin=Cell.EMPTY, vmax=Cell.CURRENT)
        ax.set_title(f"Round {idx + 1}: trap positions changed")
        ax.set_xticks(np.arange(0.5, episode.layout.shape[1], step=1))
        ax.set_yticks(np.arange(0.5, episode.layout.shape[0], step=1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        ax.plot(episode.layout.shape[1] - 1, episode.layout.shape[0] - 1, "gs", markersize=18)
        ax.plot(*episode.start_cell, "rs", markersize=16)

        if "qlearning" in all_results:
            result = all_results["qlearning"][idx]
            xs = [cell[0] for cell in result.trajectory]
            ys = [cell[1] for cell in result.trajectory]
            ax.plot(xs, ys, "o-", color="#0072b2", linewidth=1.8, markersize=3, label="Q-learning")

        if "ddqn" in all_results:
            result = all_results["ddqn"][idx]
            xs = [cell[0] for cell in result.trajectory]
            ys = [cell[1] for cell in result.trajectory]
            ax.plot(xs, ys, "o-", color="#cc79a7", linewidth=1.8, markersize=3, label="DDQN")

        ax.legend(loc="upper left", fontsize=8)

    for ax in axes[len(rounds) :]:
        ax.axis("off")

    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved 10-round demo plot to {path}")


def result_label(result) -> str:
    reason = "trap" if result.ended_in_trap else ("win" if result.status == Status.WIN else "timeout")
    return f"{result.status.name} / {reason} / reward={result.total_reward:.2f} / steps={result.steps}"


def draw_single_agent_round(ax, episode: DynamicEpisode, result, title: str, color: str) -> None:
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["white", "black", "#f4a261", "#e76f51"])
    ax.imshow(episode.layout, cmap=cmap, vmin=Cell.EMPTY, vmax=Cell.CURRENT)
    ax.set_title(f"{title}\n{result_label(result)}", fontsize=10)
    ax.set_xticks(np.arange(0.5, episode.layout.shape[1], step=1))
    ax.set_yticks(np.arange(0.5, episode.layout.shape[0], step=1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    exit_cell = (episode.layout.shape[1] - 1, episode.layout.shape[0] - 1)
    ax.plot(*exit_cell, "gs", markersize=18)
    ax.text(*exit_cell, "Exit", ha="center", va="center", color="white", fontsize=8)
    ax.plot(*episode.start_cell, "rs", markersize=18)
    ax.text(*episode.start_cell, "Start", ha="center", va="center", color="white", fontsize=7)

    for row in range(episode.layout.shape[0]):
        for col in range(episode.layout.shape[1]):
            if episode.layout[row, col] == Cell.TRAP:
                ax.text(col, row, "Trap", ha="center", va="center", color="white", fontsize=7)

    xs = [cell[0] for cell in result.trajectory]
    ys = [cell[1] for cell in result.trajectory]
    ax.plot(xs, ys, "o-", color=color, linewidth=2.0, markersize=4)
    ax.plot(*result.final_cell, "o", color="#cc79a7", markersize=12)


def save_round_images(path: Path, rounds: list[DynamicEpisode], all_results: dict[str, list]) -> None:
    import matplotlib.pyplot as plt

    path.mkdir(parents=True, exist_ok=True)
    for idx, episode in enumerate(rounds):
        active_agents = list(all_results.keys())
        fig, axes = plt.subplots(1, len(active_agents), figsize=(5.6 * len(active_agents), 5.4), tight_layout=True)
        axes = np.atleast_1d(axes)
        for ax, agent_name in zip(axes, active_agents):
            color = "#0072b2" if agent_name == "qlearning" else "#cc79a7"
            draw_single_agent_round(
                ax,
                episode,
                all_results[agent_name][idx],
                f"Round {idx + 1} - {agent_name}",
                color,
            )
        fig.savefig(path / f"round_{idx + 1:02d}.png", dpi=160)
        plt.close(fig)
    print(f"Saved per-round render images to {path}")


def setup_live_axis(ax, episode: DynamicEpisode, title: str):
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["white", "black", "#f4a261", "#e76f51"])
    ax.clear()
    ax.imshow(episode.layout, cmap=cmap, vmin=Cell.EMPTY, vmax=Cell.CURRENT)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(0.5, episode.layout.shape[1], step=1))
    ax.set_yticks(np.arange(0.5, episode.layout.shape[0], step=1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    exit_cell = (episode.layout.shape[1] - 1, episode.layout.shape[0] - 1)
    ax.plot(*exit_cell, "gs", markersize=18)
    ax.text(*exit_cell, "Exit", ha="center", va="center", color="white", fontsize=8)
    ax.plot(*episode.start_cell, "rs", markersize=18)
    ax.text(*episode.start_cell, "Start", ha="center", va="center", color="white", fontsize=7)

    for row in range(episode.layout.shape[0]):
        for col in range(episode.layout.shape[1]):
            if episode.layout[row, col] == Cell.TRAP:
                ax.text(col, row, "Trap", ha="center", va="center", color="white", fontsize=7)

    return ax.plot([], [], "o-", linewidth=2.0, markersize=5)[0], ax.plot([], [], "o", color="#cc79a7", markersize=12)[0]


def frame_indices(length: int, stride: int) -> list[int]:
    if length <= 1:
        return [1]
    stride = max(1, stride)
    indices = list(range(1, length + 1, stride))
    if indices[-1] != length:
        indices.append(length)
    return indices


def live_render(
    rounds: list[DynamicEpisode],
    all_results: dict[str, list],
    delay: float,
    round_pause: float,
    step_stride: int,
) -> None:
    import matplotlib.pyplot as plt

    active_agents = list(all_results.keys())
    colors = {"qlearning": "#0072b2", "ddqn": "#cc79a7"}
    plt.ion()
    fig, axes = plt.subplots(1, len(active_agents), figsize=(5.8 * len(active_agents), 5.6), tight_layout=True)
    axes = np.atleast_1d(axes)

    for round_idx, episode in enumerate(rounds):
        lines = {}
        heads = {}
        max_len = 0
        for ax, agent_name in zip(axes, active_agents):
            result = all_results[agent_name][round_idx]
            line, head = setup_live_axis(
                ax,
                episode,
                f"Round {round_idx + 1} - {agent_name}\nplaying...",
            )
            line.set_color(colors.get(agent_name, "#0072b2"))
            lines[agent_name] = line
            heads[agent_name] = head
            max_len = max(max_len, len(result.trajectory))

        fig.canvas.manager.set_window_title("Dynamic Trap RL Demo")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(max(delay, 0.01))

        for step_idx in frame_indices(max_len, step_stride):
            for ax, agent_name in zip(axes, active_agents):
                result = all_results[agent_name][round_idx]
                path = result.trajectory[: min(step_idx, len(result.trajectory))]
                xs = [cell[0] for cell in path]
                ys = [cell[1] for cell in path]
                lines[agent_name].set_data(xs, ys)
                heads[agent_name].set_data([xs[-1]], [ys[-1]])
                if step_idx >= len(result.trajectory):
                    ax.set_title(
                        f"Round {round_idx + 1} - {agent_name}\n{result_label(result)}",
                        fontsize=10,
                    )
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(max(delay, 0.01))

        plt.pause(max(round_pause, 0.01))

    plt.ioff()
    print("\nLive render finished. Close the matplotlib window to return to terminal.")
    plt.show()


def live_inference_demo(
    agents: dict[str, object],
    rounds: list[DynamicEpisode],
    max_steps: int,
    delay: float,
    round_pause: float,
    step_stride: int,
) -> dict[str, list]:
    import matplotlib.pyplot as plt

    from dynamic_maze_compare import apply_reward_config

    active_agents = list(agents.keys())
    colors = {"qlearning": "#0072b2", "ddqn": "#cc79a7"}
    all_results = {agent_name: [] for agent_name in active_agents}

    plt.ion()
    fig, axes = plt.subplots(1, len(active_agents), figsize=(5.8 * len(active_agents), 5.6), tight_layout=True)
    axes = np.atleast_1d(axes)
    if hasattr(fig.canvas, "manager"):
        fig.canvas.manager.set_window_title("Dynamic Trap RL Demo - Live Inference")

    for round_idx, episode in enumerate(rounds):
        games = {}
        states = {}
        trajectories = {}
        totals = {}
        statuses = {}
        finished = {}
        finish_steps = {}
        lines = {}
        heads = {}

        apply_reward_config(episode.reward_config)
        for ax, agent_name in zip(axes, active_agents):
            games[agent_name] = Maze(episode.layout, start_cell=episode.start_cell)
            state = games[agent_name].reset(episode.start_cell)
            states[agent_name] = tuple(int(value) for value in state.flatten())
            trajectories[agent_name] = [states[agent_name]]
            totals[agent_name] = 0.0
            statuses[agent_name] = Status.PLAYING
            finished[agent_name] = False
            finish_steps[agent_name] = max_steps
            line, head = setup_live_axis(
                ax,
                episode,
                f"Round {round_idx + 1} - {agent_name}\ninference...",
            )
            line.set_color(colors.get(agent_name, "#0072b2"))
            lines[agent_name] = line
            heads[agent_name] = head

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(max(delay, 0.01))

        last_step = 0
        for step_idx in range(1, max_steps + 1):
            last_step = step_idx
            for ax, agent_name in zip(axes, active_agents):
                if finished[agent_name]:
                    continue

                cell = states[agent_name]
                if agent_name == "qlearning":
                    action = infer_q_step(agents[agent_name], cell, episode)
                else:
                    action = infer_ddqn_step(agents[agent_name], cell, episode)

                next_state, reward, status = games[agent_name].step(action)
                next_cell = tuple(int(value) for value in next_state.flatten())
                states[agent_name] = next_cell
                trajectories[agent_name].append(next_cell)
                totals[agent_name] += reward
                statuses[agent_name] = status

                if status in (Status.WIN, Status.LOSE):
                    finished[agent_name] = True
                    finish_steps[agent_name] = step_idx

            if step_idx % max(1, step_stride) == 0 or all(finished.values()) or step_idx == max_steps:
                for ax, agent_name in zip(axes, active_agents):
                    path = trajectories[agent_name]
                    xs = [cell[0] for cell in path]
                    ys = [cell[1] for cell in path]
                    lines[agent_name].set_data(xs, ys)
                    heads[agent_name].set_data([xs[-1]], [ys[-1]])
                    ax.set_title(
                        f"Round {round_idx + 1} - {agent_name}\n"
                        f"inference step={step_idx}, reward={totals[agent_name]:.2f}",
                        fontsize=10,
                    )
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(max(delay, 0.01))

            if all(finished.values()):
                break

        for ax, agent_name in zip(axes, active_agents):
            final_cell = states[agent_name]
            col, row = final_cell
            ended_in_trap = statuses[agent_name] == Status.LOSE and episode.layout[row, col] == Cell.TRAP
            result = EvalResult(
                statuses[agent_name] if finished[agent_name] else Status.LOSE,
                totals[agent_name],
                finish_steps[agent_name],
                episode.start_cell,
                final_cell,
                trajectories[agent_name],
                ended_in_trap,
            )
            all_results[agent_name].append(result)
            ax.set_title(f"Round {round_idx + 1} - {agent_name}\n{result_label(result)}", fontsize=10)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(max(round_pause, 0.01))

    plt.ioff()
    print("\nLive inference finished. Close the matplotlib window to return to terminal.")
    plt.show()
    return all_results


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rounds = sample_rounds(args)
    agents = {}
    if args.agent in ("both", "qlearning"):
        agents["qlearning"] = load_qlearning_model(args.q_model)
    if args.agent in ("both", "ddqn"):
        agents["ddqn"] = load_ddqn_model(args.ddqn_model)

    if args.live_render:
        all_results = live_inference_demo(
            agents,
            rounds,
            args.max_steps,
            args.live_delay,
            args.round_pause,
            args.live_step_stride,
        )
    else:
        all_results = {
            agent_name: run_demo_agent(agent_name, agent, rounds, args.max_steps)
            for agent_name, agent in agents.items()
        }

    print(
        f"Demo rounds: {args.rounds} | "
        f"trap changes every round | "
        f"walls={'dynamic' if args.dynamic_walls else 'fixed'} | "
        f"reward={'fixed' if args.fixed_reward else 'dynamic'}"
    )
    for agent_name, results in all_results.items():
        print_results(agent_name, results)

    if args.save_csv is not None:
        save_csv(args.save_csv, all_results)
    if args.save_plot is not None:
        save_plot(args.save_plot, rounds, all_results)
    if args.save_round_dir is not None and str(args.save_round_dir).lower() != "none":
        save_round_images(args.save_round_dir, rounds, all_results)


if __name__ == "__main__":
    main()
