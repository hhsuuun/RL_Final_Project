"""Train a DQN agent for the discrete ball-in-maze control task."""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import MODEL_DIR, PLOT_DIR
from maze_env import BallMazeEnv
from training_plots import save_training_curve


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


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
        next_states = torch.as_tensor(
            np.stack([t.next_state for t in batch]), dtype=torch.float32
        )
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=80_000)
    parser.add_argument("--target-update", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default=str(MODEL_DIR / "dqn_ball_maze.pt"))
    parser.add_argument("--plot-path", type=str, default=str(PLOT_DIR / "dqn_training_curve.png"))
    parser.add_argument("--csv-path", type=str, default=str(PLOT_DIR / "dqn_training_log.csv"))
    return parser.parse_args()


def epsilon_by_episode(episode: int, total_episodes: int) -> float:
    start, end = 1.0, 0.05
    decay_fraction = min(1.0, episode / max(1, total_episodes * 0.75))
    return end + (start - end) * (1.0 - decay_fraction)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = BallMazeEnv(seed=args.seed)
    q_net = QNetwork(env.state_dim, env.discrete_action_dim)
    target_net = QNetwork(env.state_dim, env.discrete_action_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size)

    best_return = -float("inf")
    MODEL_DIR.mkdir(exist_ok=True)
    returns: list[float] = []
    epsilons: list[float] = []

    for episode in range(1, args.episodes + 1):
        state = env.reset(randomize=True)
        episode_return = 0.0
        epsilon = epsilon_by_episode(episode, args.episodes)

        for _ in range(env.config.max_steps):
            if random.random() < epsilon:
                action = random.randrange(env.discrete_action_dim)
            else:
                with torch.no_grad():
                    q_values = q_net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = int(q_values.argmax(dim=1).item())

            result = env.step_discrete(action)
            replay.push(
                Transition(
                    state=state,
                    action=action,
                    reward=result.reward,
                    next_state=result.state,
                    done=result.done,
                )
            )
            state = result.state
            episode_return += result.reward

            if len(replay) >= args.batch_size:
                states, actions, rewards, next_states, dones = replay.sample(args.batch_size)
                q_values = q_net(states).gather(1, actions)
                with torch.no_grad():
                    next_q = target_net(next_states).max(dim=1, keepdim=True).values
                    targets = rewards + args.gamma * (1.0 - dones) * next_q
                loss = F.smooth_l1_loss(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
                optimizer.step()

            if result.done:
                break

        if episode % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        returns.append(episode_return)
        epsilons.append(epsilon)

        if episode_return > best_return:
            best_return = episode_return
            torch.save(
                {
                    "model_state": q_net.state_dict(),
                    "episode": episode,
                    "return": episode_return,
                    "config": vars(args),
                },
                args.save_path,
            )

        if episode == 1 or episode % 10 == 0:
            print(
                f"episode={episode:04d} return={episode_return:8.2f} "
                f"epsilon={epsilon:.3f} best={best_return:8.2f}"
            )

    save_training_curve(
        returns=returns,
        metric_values=epsilons,
        metric_name="epsilon",
        title="DQN Training Curve",
        csv_path=Path(args.csv_path),
        figure_path=Path(args.plot_path),
    )
    print(f"Saved best DQN checkpoint to {args.save_path}")
    print(f"Saved DQN training curve to {args.plot_path}")
    print(f"Saved DQN training log to {args.csv_path}")


if __name__ == "__main__":
    main()
