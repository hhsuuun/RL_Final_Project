"""Train a DQN agent on the trap maze environment."""

from __future__ import annotations

import argparse
import logging
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from environment.maze import Maze, Render, Status
from train_maze import MAZE_LAYOUT, save_bestmove_plot, save_training_plot


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
            np.stack([t.next_state for t in batch]),
            dtype=torch.float32,
        )
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


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


class DQNAgent:
    name = "DQNModel"

    def __init__(self, game: Maze, network: QNetwork) -> None:
        self.environment = game
        self.network = network
        self.nrows, self.ncols = game.maze.shape
        self.state_dim = self.nrows * self.ncols

    def encode_state(self, state) -> np.ndarray:
        if isinstance(state, tuple):
            col, row = state
        else:
            col, row = np.asarray(state).flatten()
        encoded = np.zeros(self.state_dim, dtype=np.float32)
        encoded[int(row) * self.ncols + int(col)] = 1.0
        return encoded

    def valid_actions(self, state) -> list[int]:
        if isinstance(state, tuple):
            col, row = state
        else:
            col, row = np.asarray(state).flatten()
        col = int(col)
        row = int(row)

        actions = []
        candidates = (
            (0, col - 1, row),
            (1, col + 1, row),
            (2, col, row - 1),
            (3, col, row + 1),
        )
        for action, next_col, next_row in candidates:
            if (
                0 <= next_col < self.ncols
                and 0 <= next_row < self.nrows
                and self.environment.maze[next_row, next_col] != 1
            ):
                actions.append(action)
        return actions

    def state_from_encoded(self, encoded: np.ndarray) -> tuple[int, int]:
        index = int(np.argmax(encoded))
        return index % self.ncols, index // self.ncols

    def q(self, state) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(self.encode_state(state), dtype=torch.float32).unsqueeze(0)
            q_values = self.network(tensor).squeeze(0).numpy()
        masked = np.full_like(q_values, -1e6)
        valid_actions = self.valid_actions(state)
        masked[valid_actions] = q_values[valid_actions]
        return masked

    def predict(self, state) -> int:
        q_values = self.q(state)
        actions = np.nonzero(q_values == np.max(q_values))[0]
        return int(random.choice(actions))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=20.0,
        help="Divide rewards by this value for neural-network updates.",
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.992)
    parser.add_argument("--target-update", type=int, default=50)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stop-at-convergence", action="store_true")
    parser.add_argument(
        "--render",
        choices=("nothing", "moves", "training"),
        default="nothing",
    )
    parser.add_argument(
        "--play-start",
        type=int,
        nargs=2,
        metavar=("COL", "ROW"),
        default=(4, 1),
    )
    parser.add_argument("--save-plot", type=Path, default=None)
    parser.add_argument("--save-bestmove", type=Path, default=None)
    parser.add_argument("--save-model", type=Path, default=None)
    return parser.parse_args()


def render_mode(name: str) -> Render:
    return {
        "nothing": Render.NOTHING,
        "moves": Render.MOVES,
        "training": Render.TRAINING,
    }[name]


def optimize(
    q_net: QNetwork,
    target_net: QNetwork,
    replay: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float,
    agent: DQNAgent,
) -> float:
    states, actions, rewards, next_states, dones = replay.sample(batch_size)
    q_values = q_net(states).gather(1, actions)
    with torch.no_grad():
        online_next_q = q_net(next_states)
        target_next_q = target_net(next_states)
        next_mask = torch.zeros_like(online_next_q, dtype=torch.bool)
        for row_idx, encoded in enumerate(next_states.numpy()):
            cell = agent.state_from_encoded(encoded)
            next_mask[row_idx, agent.valid_actions(cell)] = True
        online_next_q = online_next_q.masked_fill(~next_mask, -1e6)
        best_next_actions = online_next_q.argmax(dim=1, keepdim=True)
        next_q = target_next_q.gather(1, best_next_actions)
        targets = rewards + gamma * (1.0 - dones) * next_q

    loss = F.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def train(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    game = Maze(MAZE_LAYOUT)
    game.render(render_mode(args.render))

    state_dim = game.maze.size
    action_dim = len(game.actions)
    q_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size)
    agent = DQNAgent(game, q_net)

    epsilon = args.epsilon_start
    reward_history: list[float] = []
    win_history: list[tuple[int, float]] = []
    best_win_rate = 0.0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in q_net.state_dict().items()
    }
    start_list: list[tuple[int, int]] = []
    start_time = datetime.now()

    for episode in range(1, args.episodes + 1):
        if not start_list:
            start_list = game.empty.copy()
        start_cell = random.choice(start_list)
        start_list.remove(start_cell)

        raw_state = game.reset(start_cell)
        state = agent.encode_state(raw_state)
        episode_reward = 0.0
        losses: list[float] = []

        while True:
            if random.random() < epsilon:
                action = int(random.choice(agent.valid_actions(raw_state)))
            else:
                action = agent.predict(raw_state)

            next_raw_state, reward, status = game.step(action)
            next_state = agent.encode_state(next_raw_state)
            done = status in (Status.WIN, Status.LOSE)
            scaled_reward = reward / max(args.reward_scale, 1e-6)
            replay.push(
                Transition(
                    state=state,
                    action=action,
                    reward=scaled_reward,
                    next_state=next_state,
                    done=done,
                )
            )

            episode_reward += reward

            if len(replay) >= max(args.batch_size, args.learning_starts):
                losses.append(
                    optimize(
                        q_net,
                        target_net,
                        replay,
                        optimizer,
                        args.batch_size,
                        args.gamma,
                        agent,
                    )
                )

            if done:
                break

            raw_state = next_raw_state
            state = next_state
            if args.render == "training":
                game.render_q(agent)

        reward_history.append(episode_reward)
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if episode % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        mean_loss = float(np.mean(losses)) if losses else 0.0
        logging.info(
            "episode: %d/%d | status: %-4s | return: %.2f | loss: %.4f | e: %.5f",
            episode,
            args.episodes,
            status.name,
            episode_reward,
            mean_loss,
            epsilon,
        )

        if episode % args.check_every == 0:
            won_all, win_rate = game.check_win_all(agent)
            win_history.append((episode, win_rate))
            if win_rate >= best_win_rate:
                best_win_rate = win_rate
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in q_net.state_dict().items()
                }
            if won_all and args.stop_at_convergence:
                logging.info("won from all start cells, stop learning")
                break

    q_net.load_state_dict(best_state)
    target_net.load_state_dict(best_state)
    elapsed = datetime.now() - start_time
    logging.info("episodes: %d | time spent: %s", episode, elapsed)
    return game, agent, reward_history, win_history, episode, elapsed, best_win_rate


def save_dqn_model(path: Path, agent: DQNAgent, args: argparse.Namespace, best_win_rate: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": agent.network.state_dict(),
            "maze_layout": MAZE_LAYOUT,
            "state_dim": agent.state_dim,
            "action_dim": len(agent.environment.actions),
            "best_win_rate": best_win_rate,
            "config": vars(args),
        },
        path,
    )
    print(f"Saved trained DQN model to {path}")


def main() -> None:
    args = parse_args()
    if args.render == "nothing":
        import matplotlib

        matplotlib.use("Agg")

    logging.basicConfig(
        format="%(levelname)-8s: %(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    game, agent, reward_history, win_history, episodes, elapsed, best_win_rate = train(args)
    print(f"Trained {episodes} episodes in {elapsed}; best win rate={best_win_rate:.5f}")

    status = game.play(agent, start_cell=tuple(args.play_start))
    print(f"Final play from {tuple(args.play_start)}: {status.name}")

    if args.save_plot is not None:
        save_training_plot(
            args.save_plot,
            agent.name,
            reward_history,
            win_history,
            cumulative_rewards=False,
        )
    if args.save_bestmove is not None:
        save_bestmove_plot(args.save_bestmove, agent, MAZE_LAYOUT)
    if args.save_model is not None:
        save_dqn_model(args.save_model, agent, args, best_win_rate)


if __name__ == "__main__":
    main()
