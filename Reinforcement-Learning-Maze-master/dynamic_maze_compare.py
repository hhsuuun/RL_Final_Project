"""Compare tabular Q-learning and DDQN on dynamic maze layouts.

This script intentionally uses a harder setting than train_maze.py:

* walls can change between episodes
* trap locations can change between episodes
* reward values can change between episodes

The tabular Q-learning baseline only observes the agent position. DDQN observes
the full map and current reward settings, so it can react to dynamic traps/maps.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from environment.maze import Action, Cell, Maze, Status
from train_maze import moving_average


@dataclass(frozen=True)
class RewardConfig:
    exit_reward: float
    trap_penalty: float
    move_penalty: float
    visited_penalty: float
    wall_penalty: float
    closer_reward: float
    farther_penalty: float

    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.exit_reward / 30.0,
                self.trap_penalty / 30.0,
                self.move_penalty / 0.20,
                self.visited_penalty / 1.00,
                self.wall_penalty / 2.00,
                self.closer_reward / 0.20,
                self.farther_penalty / 0.20,
            ],
            dtype=np.float32,
        )


@dataclass
class DynamicEpisode:
    layout: np.ndarray
    start_cell: tuple[int, int]
    reward_config: RewardConfig


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_valid_mask: np.ndarray


@dataclass
class EvalResult:
    status: Status
    total_reward: float
    steps: int
    start_cell: tuple[int, int]
    final_cell: tuple[int, int]
    trajectory: list[tuple[int, int]]
    ended_in_trap: bool = False


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(1)
        next_masks = torch.as_tensor(np.stack([t.next_valid_mask for t in batch]), dtype=torch.bool)
        return states, actions, rewards, next_states, dones, next_masks

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DynamicDDQNAgent:
    name = "DynamicDDQN"

    def __init__(self, nrows: int, ncols: int, network: QNetwork) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.network = network

    @property
    def state_dim(self) -> int:
        return self.nrows * self.ncols * 4 + 7

    def encode(
        self,
        layout: np.ndarray,
        cell: tuple[int, int],
        reward_config: RewardConfig,
    ) -> np.ndarray:
        col, row = cell
        agent = np.zeros_like(layout, dtype=np.float32)
        walls = (layout == Cell.OCCUPIED).astype(np.float32)
        traps = (layout == Cell.TRAP).astype(np.float32)
        exit_layer = np.zeros_like(layout, dtype=np.float32)
        agent[row, col] = 1.0
        exit_layer[-1, -1] = 1.0
        return np.concatenate(
            [
                agent.flatten(),
                walls.flatten(),
                traps.flatten(),
                exit_layer.flatten(),
                reward_config.vector(),
            ]
        ).astype(np.float32)

    def q(self, state: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.network(tensor).squeeze(0).numpy()
        masked = np.full_like(q_values, -1e6)
        masked[valid_mask] = q_values[valid_mask]
        return masked

    def predict(self, state: np.ndarray, valid_mask: np.ndarray) -> int:
        q_values = self.q(state, valid_mask)
        actions = np.nonzero(q_values == np.max(q_values))[0]
        return int(random.choice(actions))


class PositionQAgent:
    name = "PositionQ"

    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions
        self.Q: dict[tuple[tuple[int, int], int], float] = {}

    def q(self, cell: tuple[int, int]) -> np.ndarray:
        return np.array([self.Q.get((cell, int(action)), 0.0) for action in self.actions])

    def predict(self, cell: tuple[int, int], valid_actions: list[int]) -> int:
        q_values = self.q(cell)
        masked = np.full_like(q_values, -1e6)
        masked[valid_actions] = q_values[valid_actions]
        actions = np.nonzero(masked == np.max(masked))[0]
        return int(random.choice(actions))

    def update(
        self,
        cell: tuple[int, int],
        action: int,
        reward: float,
        next_cell: tuple[int, int],
        done: bool,
        learning_rate: float,
        gamma: float,
    ) -> None:
        key = (cell, int(action))
        old_value = self.Q.get(key, 0.0)
        next_value = 0.0 if done else float(np.max(self.q(next_cell)))
        self.Q[key] = old_value + learning_rate * (reward + gamma * next_value - old_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("both", "qlearning", "ddqn"), default="both")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--wall-prob", type=float, default=0.18)
    parser.add_argument("--trap-prob", type=float, default=0.10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--q-lr", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20_000)
    parser.add_argument("--learning-starts", type=int, default=512)
    parser.add_argument("--ddqn-lr", type=float, default=5e-4)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--target-update", type=int, default=100)
    parser.add_argument("--reward-scale", type=float, default=30.0)
    parser.add_argument("--save-plot", type=Path, default=Path("plots/dynamic_compare_curve.png"))
    parser.add_argument("--save-csv", type=Path, default=Path("plots/dynamic_compare_metrics.csv"))
    parser.add_argument("--save-q-model", type=Path, default=Path("models/dynamic_qlearning.pkl"))
    parser.add_argument("--save-ddqn-model", type=Path, default=Path("models/dynamic_ddqn.pt"))
    return parser.parse_args()


def sample_reward_config() -> RewardConfig:
    closer = random.uniform(0.02, 0.10)
    return RewardConfig(
        exit_reward=random.uniform(15.0, 30.0),
        trap_penalty=-random.uniform(12.0, 30.0),
        move_penalty=-random.uniform(0.02, 0.12),
        visited_penalty=-random.uniform(0.20, 0.60),
        wall_penalty=-random.uniform(0.80, 2.00),
        closer_reward=closer,
        farther_penalty=-closer,
    )


def apply_reward_config(config: RewardConfig) -> None:
    Maze.reward_exit = config.exit_reward
    Maze.penalty_trap = config.trap_penalty
    Maze.penalty_move = config.move_penalty
    Maze.penalty_visited = config.visited_penalty
    Maze.penalty_impossible_move = config.wall_penalty
    Maze.reward_closer_to_exit = config.closer_reward
    Maze.penalty_farther_from_exit = config.farther_penalty


def valid_actions(layout: np.ndarray, cell: tuple[int, int]) -> list[int]:
    col, row = cell
    nrows, ncols = layout.shape
    candidates = (
        (Action.MOVE_LEFT, col - 1, row),
        (Action.MOVE_RIGHT, col + 1, row),
        (Action.MOVE_UP, col, row - 1),
        (Action.MOVE_DOWN, col, row + 1),
    )
    return [
        int(action)
        for action, next_col, next_row in candidates
        if (
            0 <= next_col < ncols
            and 0 <= next_row < nrows
            and layout[next_row, next_col] != Cell.OCCUPIED
        )
    ]


def valid_action_mask(layout: np.ndarray, cell: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(4, dtype=bool)
    mask[valid_actions(layout, cell)] = True
    return mask


def reachable_safe_cells(layout: np.ndarray) -> list[tuple[int, int]]:
    nrows, ncols = layout.shape
    exit_cell = (ncols - 1, nrows - 1)
    queue = deque([exit_cell])
    visited = {exit_cell}

    while queue:
        col, row = queue.popleft()
        for action in valid_actions(layout, (col, row)):
            if action == Action.MOVE_LEFT:
                next_cell = (col - 1, row)
            elif action == Action.MOVE_RIGHT:
                next_cell = (col + 1, row)
            elif action == Action.MOVE_UP:
                next_cell = (col, row - 1)
            else:
                next_cell = (col, row + 1)
            next_col, next_row = next_cell
            if layout[next_row, next_col] != Cell.EMPTY:
                continue
            if next_cell not in visited:
                visited.add(next_cell)
                queue.append(next_cell)

    return [cell for cell in visited if cell != exit_cell]


def sample_dynamic_episode(size: int, wall_prob: float, trap_prob: float) -> DynamicEpisode:
    for _ in range(200):
        layout = np.zeros((size, size), dtype=np.int8)
        exit_cell = (size - 1, size - 1)

        for row in range(size):
            for col in range(size):
                if (col, row) == exit_cell:
                    continue
                if random.random() < wall_prob:
                    layout[row, col] = Cell.OCCUPIED

        for row in range(size):
            for col in range(size):
                if (col, row) == exit_cell or layout[row, col] == Cell.OCCUPIED:
                    continue
                if random.random() < trap_prob:
                    layout[row, col] = Cell.TRAP

        safe_starts = reachable_safe_cells(layout)
        if len(safe_starts) >= max(4, size // 2):
            return DynamicEpisode(
                layout=layout,
                start_cell=random.choice(safe_starts),
                reward_config=sample_reward_config(),
            )

    raise RuntimeError("Could not sample a solvable dynamic maze. Lower wall/trap probabilities.")


def step_cell(cell: tuple[int, int], action: int) -> tuple[int, int]:
    col, row = cell
    if action == Action.MOVE_LEFT:
        return col - 1, row
    if action == Action.MOVE_RIGHT:
        return col + 1, row
    if action == Action.MOVE_UP:
        return col, row - 1
    return col, row + 1


def train_qlearning(args: argparse.Namespace) -> tuple[PositionQAgent, list[float], list[float]]:
    agent = PositionQAgent(Maze.actions)
    returns: list[float] = []
    win_rates: list[float] = []
    epsilon = args.epsilon_start

    for episode_idx in range(1, args.episodes + 1):
        spec = sample_dynamic_episode(args.size, args.wall_prob, args.trap_prob)
        result = run_q_episode(
            agent,
            spec,
            max_steps=args.max_steps,
            epsilon=epsilon,
            learning_rate=args.q_lr,
            gamma=args.gamma,
            train=True,
        )
        returns.append(result.total_reward)
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if episode_idx % 50 == 0:
            win_rate = evaluate_qlearning(agent, args, episodes=50)["win_rate"]
            win_rates.append(win_rate)
            print(f"qlearning episode {episode_idx}/{args.episodes} | eval win rate={win_rate:.3f}")

    return agent, returns, win_rates


def run_q_episode(
    agent: PositionQAgent,
    spec: DynamicEpisode,
    *,
    max_steps: int,
    epsilon: float,
    learning_rate: float,
    gamma: float,
    train: bool,
) -> EvalResult:
    apply_reward_config(spec.reward_config)
    game = Maze(spec.layout, start_cell=spec.start_cell)
    state = game.reset(spec.start_cell)
    cell = tuple(int(value) for value in state.flatten())
    total_reward = 0.0
    trajectory = [cell]

    for step in range(1, max_steps + 1):
        actions = valid_actions(spec.layout, cell)
        if train and random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = agent.predict(cell, actions)

        next_state, reward, status = game.step(action)
        next_cell = tuple(int(value) for value in next_state.flatten())
        done = status in (Status.WIN, Status.LOSE)
        if train:
            agent.update(cell, action, reward, next_cell, done, learning_rate, gamma)
        total_reward += reward
        trajectory.append(next_cell)
        if done:
            col, row = next_cell
            ended_in_trap = status == Status.LOSE and spec.layout[row, col] == Cell.TRAP
            return EvalResult(
                status,
                total_reward,
                step,
                spec.start_cell,
                next_cell,
                trajectory,
                ended_in_trap,
            )
        cell = next_cell

    return EvalResult(Status.LOSE, total_reward, max_steps, spec.start_cell, cell, trajectory)


def optimize_ddqn(
    q_net: QNetwork,
    target_net: QNetwork,
    replay: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float,
) -> float:
    states, actions, rewards, next_states, dones, next_masks = replay.sample(batch_size)
    q_values = q_net(states).gather(1, actions)
    with torch.no_grad():
        online_next_q = q_net(next_states).masked_fill(~next_masks, -1e6)
        best_next_actions = online_next_q.argmax(dim=1, keepdim=True)
        target_next_q = target_net(next_states).gather(1, best_next_actions)
        targets = rewards + gamma * (1.0 - dones) * target_next_q

    loss = F.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def train_ddqn(args: argparse.Namespace) -> tuple[DynamicDDQNAgent, list[float], list[float]]:
    state_dim = args.size * args.size * 4 + 7
    action_dim = len(Maze.actions)
    q_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.ddqn_lr)
    replay = ReplayBuffer(args.buffer_size)
    agent = DynamicDDQNAgent(args.size, args.size, q_net)
    returns: list[float] = []
    win_rates: list[float] = []
    epsilon = args.epsilon_start

    for episode_idx in range(1, args.episodes + 1):
        spec = sample_dynamic_episode(args.size, args.wall_prob, args.trap_prob)
        result, losses = run_ddqn_episode(
            agent,
            spec,
            replay,
            q_net,
            target_net,
            optimizer,
            args,
            epsilon=epsilon,
            train=True,
        )
        returns.append(result.total_reward)
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if episode_idx % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if episode_idx % 50 == 0:
            win_rate = evaluate_ddqn(agent, args, episodes=50)["win_rate"]
            win_rates.append(win_rate)
            mean_loss = float(np.mean(losses)) if losses else 0.0
            print(
                f"ddqn episode {episode_idx}/{args.episodes} | "
                f"eval win rate={win_rate:.3f} | loss={mean_loss:.4f}"
            )

    return agent, returns, win_rates


def run_ddqn_episode(
    agent: DynamicDDQNAgent,
    spec: DynamicEpisode,
    replay: ReplayBuffer | None,
    q_net: QNetwork | None,
    target_net: QNetwork | None,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    *,
    epsilon: float,
    train: bool,
) -> tuple[EvalResult, list[float]]:
    apply_reward_config(spec.reward_config)
    game = Maze(spec.layout, start_cell=spec.start_cell)
    raw_state = tuple(int(value) for value in game.reset(spec.start_cell).flatten())
    state = agent.encode(spec.layout, raw_state, spec.reward_config)
    total_reward = 0.0
    trajectory = [raw_state]
    losses: list[float] = []

    for step in range(1, args.max_steps + 1):
        mask = valid_action_mask(spec.layout, raw_state)
        if train and random.random() < epsilon:
            action = int(random.choice(np.nonzero(mask)[0]))
        else:
            action = agent.predict(state, mask)

        next_state_array, reward, status = game.step(action)
        next_raw_state = tuple(int(value) for value in next_state_array.flatten())
        next_state = agent.encode(spec.layout, next_raw_state, spec.reward_config)
        next_mask = valid_action_mask(spec.layout, next_raw_state)
        done = status in (Status.WIN, Status.LOSE)
        total_reward += reward
        trajectory.append(next_raw_state)

        if train and replay is not None:
            replay.push(
                Transition(
                    state=state,
                    action=action,
                    reward=reward / max(args.reward_scale, 1e-6),
                    next_state=next_state,
                    done=done,
                    next_valid_mask=next_mask,
                )
            )
            if (
                q_net is not None
                and target_net is not None
                and optimizer is not None
                and len(replay) >= max(args.batch_size, args.learning_starts)
            ):
                losses.append(
                    optimize_ddqn(q_net, target_net, replay, optimizer, args.batch_size, args.gamma)
                )

        if done:
            col, row = next_raw_state
            ended_in_trap = status == Status.LOSE and spec.layout[row, col] == Cell.TRAP
            return (
                EvalResult(
                    status,
                    total_reward,
                    step,
                    spec.start_cell,
                    next_raw_state,
                    trajectory,
                    ended_in_trap,
                ),
                losses,
            )

        raw_state = next_raw_state
        state = next_state

    return EvalResult(Status.LOSE, total_reward, args.max_steps, spec.start_cell, raw_state, trajectory), losses


def summarize(results: list[EvalResult]) -> dict[str, float]:
    wins = sum(result.status == Status.WIN for result in results)
    losses = len(results) - wins
    trap_losses = sum(result.ended_in_trap for result in results)
    timeout_losses = losses - trap_losses
    return {
        "episodes": float(len(results)),
        "wins": float(wins),
        "losses": float(losses),
        "trap_losses": float(trap_losses),
        "timeout_losses": float(timeout_losses),
        "win_rate": wins / len(results),
        "avg_reward": float(np.mean([result.total_reward for result in results])),
        "avg_steps": float(np.mean([result.steps for result in results])),
    }


def evaluate_qlearning(agent: PositionQAgent, args: argparse.Namespace, episodes: int | None = None) -> dict[str, float]:
    results = [
        run_q_episode(
            agent,
            sample_dynamic_episode(args.size, args.wall_prob, args.trap_prob),
            max_steps=args.max_steps,
            epsilon=0.0,
            learning_rate=args.q_lr,
            gamma=args.gamma,
            train=False,
        )
        for _ in range(episodes or args.eval_episodes)
    ]
    return summarize(results)


def evaluate_ddqn(agent: DynamicDDQNAgent, args: argparse.Namespace, episodes: int | None = None) -> dict[str, float]:
    results = [
        run_ddqn_episode(
            agent,
            sample_dynamic_episode(args.size, args.wall_prob, args.trap_prob),
            replay=None,
            q_net=None,
            target_net=None,
            optimizer=None,
            args=args,
            epsilon=0.0,
            train=False,
        )[0]
        for _ in range(episodes or args.eval_episodes)
    ]
    return summarize(results)


def save_plot(path: Path, series: dict[str, list[float]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 4))
    for name, returns in series.items():
        if not returns:
            continue
        values = np.asarray(returns, dtype=np.float32)
        smoothed = moving_average(values, min(50, len(values)))
        ax.plot(values, alpha=0.18, linewidth=0.8)
        ax.plot(
            np.arange(len(smoothed)) + 1,
            smoothed,
            linewidth=2.0,
            label=f"{name} moving average",
        )
    ax.set_title("Dynamic map/trap/reward training returns")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved training comparison plot to {path}")


def save_metrics(path: Path, metrics: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "agent",
        "episodes",
        "wins",
        "losses",
        "trap_losses",
        "timeout_losses",
        "win_rate",
        "avg_reward",
        "avg_steps",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for agent_name, values in metrics.items():
            writer.writerow({"agent": agent_name, **values})
    print(f"Saved evaluation metrics to {path}")


def save_q_model(path: Path, agent: PositionQAgent) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(agent, file)
    print(f"Saved dynamic Q-learning model to {path}")


def save_ddqn_model(path: Path, agent: DynamicDDQNAgent, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": agent.network.state_dict(),
            "state_dim": agent.state_dim,
            "action_dim": len(Maze.actions),
            "size": args.size,
            "config": vars(args),
        },
        path,
    )
    print(f"Saved dynamic DDQN model to {path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    original_rewards = {
        "reward_exit": Maze.reward_exit,
        "penalty_move": Maze.penalty_move,
        "penalty_visited": Maze.penalty_visited,
        "penalty_impossible_move": Maze.penalty_impossible_move,
        "penalty_trap": Maze.penalty_trap,
        "reward_closer_to_exit": Maze.reward_closer_to_exit,
        "penalty_farther_from_exit": Maze.penalty_farther_from_exit,
    }

    returns: dict[str, list[float]] = {}
    metrics: dict[str, dict[str, float]] = {}

    try:
        if args.agent in ("both", "qlearning"):
            q_agent, q_returns, _ = train_qlearning(args)
            returns["qlearning"] = q_returns
            metrics["qlearning"] = evaluate_qlearning(q_agent, args)
            save_q_model(args.save_q_model, q_agent)

        if args.agent in ("both", "ddqn"):
            ddqn_agent, ddqn_returns, _ = train_ddqn(args)
            returns["ddqn"] = ddqn_returns
            metrics["ddqn"] = evaluate_ddqn(ddqn_agent, args)
            save_ddqn_model(args.save_ddqn_model, ddqn_agent, args)
    finally:
        for name, value in original_rewards.items():
            setattr(Maze, name, value)

    for agent_name, values in metrics.items():
        print(f"\n{agent_name}")
        print("-" * len(agent_name))
        for key, value in values.items():
            if key == "win_rate":
                print(f"{key:23s}: {value:.3f}")
            else:
                print(f"{key:23s}: {value:.2f}")

    if args.save_plot is not None:
        save_plot(args.save_plot, returns)
    if args.save_csv is not None:
        save_metrics(args.save_csv, metrics)


if __name__ == "__main__":
    main()
