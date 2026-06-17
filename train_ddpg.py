"""Train a DDPG agent for continuous board-tilt control."""

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

from config import CONFIG, MODEL_DIR, PLOT_DIR
from maze_env import BallMazeEnv
from training_plots import save_training_curve
from training_visualizer import TrainingStatus, TrainingVisualizer


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat([state, action], dim=1))


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
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
        actions = torch.as_tensor(np.stack([t.action for t in batch]), dtype=torch.float32)
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
    parser.add_argument("--episodes", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=120_000)
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1_000,
        help="Collect this many transitions before updating actor and critic.",
    )
    parser.add_argument(
        "--initial-policy",
        choices=("random", "goal-biased"),
        default="goal-biased",
        help="Policy used while filling the replay buffer before learning starts.",
    )
    parser.add_argument(
        "--initial-noise-std",
        type=float,
        default=0.25,
        help="Gaussian noise added to goal-directed initial continuous actions.",
    )
    parser.add_argument("--noise-std", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--save-path", type=str, default=str(MODEL_DIR / "ddpg_ball_maze.pt"))
    parser.add_argument("--plot-path", type=str, default=str(PLOT_DIR / "ddpg_training_curve.png"))
    parser.add_argument("--csv-path", type=str, default=str(PLOT_DIR / "ddpg_training_log.csv"))
    parser.add_argument(
        "--render-training",
        action="store_true",
        help="Show a live game view and training curve during training.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render every N episodes when --render-training is enabled.",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=30,
        help="Maximum FPS for the live training monitor.",
    )
    parser.add_argument(
        "--render-hold-seconds",
        type=float,
        default=5.0,
        help="Keep the live monitor open for N seconds after training.",
    )
    return parser.parse_args()


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def initial_continuous_action(env: BallMazeEnv, policy: str, noise_std: float) -> np.ndarray:
    if policy == "random":
        return np.random.uniform(
            -1.0,
            1.0,
            size=env.continuous_action_dim,
        ).astype(np.float32)

    direction = env.goal_pos - env.pos
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        action_norm = np.zeros(env.continuous_action_dim, dtype=np.float32)
    else:
        action_norm = (direction / norm).astype(np.float32)

    action_norm += np.random.normal(
        0.0,
        max(0.0, noise_std),
        size=env.continuous_action_dim,
    )
    return np.clip(action_norm, -1.0, 1.0).astype(np.float32)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = BallMazeEnv(seed=args.seed)
    actor = Actor(env.state_dim, env.continuous_action_dim)
    actor_target = Actor(env.state_dim, env.continuous_action_dim)
    critic = Critic(env.state_dim, env.continuous_action_dim)
    critic_target = Critic(env.state_dim, env.continuous_action_dim)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    replay = ReplayBuffer(args.buffer_size)

    best_return = -float("inf")
    MODEL_DIR.mkdir(exist_ok=True)
    returns: list[float] = []
    noise_values: list[float] = []
    visualizer = (
        TrainingVisualizer("ddpg", args.episodes, fps=args.render_fps)
        if args.render_training
        else None
    )

    try:
        for episode in range(1, args.episodes + 1):
            state = env.reset(randomize=True)
            episode_return = 0.0
            noise_scale = args.noise_std * max(0.08, 1.0 - episode / args.episodes)
            last_event = "running"

            for _ in range(env.config.max_steps):
                if len(replay) < args.learning_starts:
                    action_norm = initial_continuous_action(
                        env,
                        policy=args.initial_policy,
                        noise_std=args.initial_noise_std,
                    )
                else:
                    with torch.no_grad():
                        action_norm = actor(
                            torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                        ).squeeze(0).numpy()
                    action_norm += np.random.normal(
                        0.0,
                        noise_scale,
                        size=env.continuous_action_dim,
                    )
                    action_norm = np.clip(action_norm, -1.0, 1.0).astype(np.float32)
                result = env.step_continuous(action_norm * CONFIG.max_tilt_deg)

                replay.push(
                    Transition(
                        state=state,
                        action=action_norm,
                        reward=result.reward,
                        next_state=result.state,
                        done=result.done,
                    )
                )
                state = result.state
                episode_return += result.reward
                last_event = result.info["event"]

                if visualizer is not None and episode % max(1, args.render_every) == 0:
                    best_so_far = max(best_return, episode_return)
                    keep_rendering = visualizer.update(
                        env,
                        returns,
                        TrainingStatus(
                            algorithm="ddpg",
                            episode=episode,
                            total_episodes=args.episodes,
                            episode_return=episode_return,
                            best_return=best_so_far,
                            metric_name="noise",
                            metric_value=noise_scale,
                            event=last_event,
                        ),
                    )
                    if not keep_rendering:
                        visualizer = None

                if len(replay) >= max(args.batch_size, args.learning_starts):
                    states, actions, rewards, next_states, dones = replay.sample(args.batch_size)

                    with torch.no_grad():
                        next_actions = actor_target(next_states)
                        target_q = critic_target(next_states, next_actions)
                        targets = rewards + args.gamma * (1.0 - dones) * target_q

                    q_values = critic(states, actions)
                    critic_loss = F.mse_loss(q_values, targets)
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
                    critic_optimizer.step()

                    actor_loss = -critic(states, actor(states)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
                    actor_optimizer.step()

                    soft_update(actor, actor_target, args.tau)
                    soft_update(critic, critic_target, args.tau)

                if result.done:
                    break

            returns.append(episode_return)
            noise_values.append(noise_scale)

            if episode_return > best_return:
                best_return = episode_return
                torch.save(
                    {
                        "actor_state": actor.state_dict(),
                        "critic_state": critic.state_dict(),
                        "episode": episode,
                        "return": episode_return,
                        "config": vars(args),
                    },
                    args.save_path,
                )

            if visualizer is not None and episode % max(1, args.render_every) == 0:
                keep_rendering = visualizer.update(
                    env,
                    returns,
                    TrainingStatus(
                        algorithm="ddpg",
                        episode=episode,
                        total_episodes=args.episodes,
                        episode_return=episode_return,
                        best_return=best_return,
                        metric_name="noise",
                        metric_value=noise_scale,
                        event=f"finished: {last_event}",
                    ),
                )
                if not keep_rendering:
                    visualizer = None

            if episode == 1 or episode % 10 == 0:
                print(
                    f"episode={episode:04d} return={episode_return:8.2f} "
                    f"noise={noise_scale:.3f} best={best_return:8.2f}"
                )
    finally:
        if visualizer is not None:
            visualizer.hold(args.render_hold_seconds)
            visualizer.close()

    save_training_curve(
        returns=returns,
        metric_values=noise_values,
        metric_name="noise_scale",
        title="DDPG Training Curve",
        csv_path=Path(args.csv_path),
        figure_path=Path(args.plot_path),
    )
    print(f"Saved best DDPG checkpoint to {args.save_path}")
    print(f"Saved DDPG training curve to {args.plot_path}")
    print(f"Saved DDPG training log to {args.csv_path}")


if __name__ == "__main__":
    main()
