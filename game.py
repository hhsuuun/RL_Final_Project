from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame

from config import CONFIG, MODEL_DIR
from maze_env import BallMazeEnv

COLORS = {
    "background": (237, 239, 232),
    "board": (248, 248, 241),
    "wall": (53, 69, 81),
    "wall_shadow": (31, 42, 51),
    "trap": (28, 31, 35),
    "trap_ring": (192, 71, 69),
    "goal": (67, 150, 104),
    "goal_ring": (31, 110, 72),
    "ball": (240, 207, 82),
    "ball_edge": (160, 116, 27),
    "text": (41, 47, 55),
    "muted": (101, 111, 120),
    "trail": (95, 153, 185),
}


class DQNPolicy:
    def __init__(self, model_path: Path) -> None:
        import torch

        from train_dqn import QNetwork

        self.torch = torch
        self.net = QNetwork(BallMazeEnv.state_dim, BallMazeEnv.discrete_action_dim)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.net.load_state_dict(checkpoint["model_state"])
        self.net.eval()

    def act(self, state: np.ndarray) -> int:
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(state, dtype=self.torch.float32).unsqueeze(0)
            return int(self.net(tensor).argmax(dim=1).item())


class DDPGPolicy:
    def __init__(self, model_path: Path) -> None:
        import torch

        from train_ddpg import Actor

        self.torch = torch
        self.net = Actor(BallMazeEnv.state_dim, BallMazeEnv.continuous_action_dim)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.net.load_state_dict(checkpoint["actor_state"])
        self.net.eval()

    def act(self, state: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(state, dtype=self.torch.float32).unsqueeze(0)
            normalized = self.net(tensor).squeeze(0).numpy()
            return normalized * CONFIG.max_tilt_deg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("manual", "random", "dqn", "ddpg"),
        default="manual",
        help="Control mode for the demo.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to a trained model checkpoint.",
    )
    return parser.parse_args()


def draw_scene(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    env: BallMazeEnv,
    mode: str,
    total_reward: float,
    status: str,
    trail: list[tuple[int, int]],
) -> None:
    screen.fill(COLORS["background"])
    board = pygame.Rect(18, 18, CONFIG.width - 36, CONFIG.height - 36)
    pygame.draw.rect(screen, COLORS["board"], board, border_radius=8)
    pygame.draw.rect(screen, (203, 208, 198), board, width=2, border_radius=8)

    for tx, ty in trail[-120:]:
        pygame.draw.circle(screen, COLORS["trail"], (tx, ty), 2)

    for wall in env.walls:
        x, y, w, h = [int(v) for v in wall]
        pygame.draw.rect(
            screen, COLORS["wall_shadow"], (x + 3, y + 3, w, h), border_radius=3
        )
        pygame.draw.rect(screen, COLORS["wall"], (x, y, w, h), border_radius=3)

    for trap in env.traps:
        center = tuple(np.round(trap).astype(int))
        pygame.draw.circle(screen, COLORS["trap_ring"], center, CONFIG.trap_radius + 4)
        pygame.draw.circle(screen, COLORS["trap"], center, CONFIG.trap_radius)

    goal = tuple(np.round(env.goal_pos).astype(int))
    pygame.draw.circle(screen, COLORS["goal_ring"], goal, CONFIG.goal_radius + 5)
    pygame.draw.circle(screen, COLORS["goal"], goal, CONFIG.goal_radius)
    pygame.draw.circle(screen, (232, 246, 236), goal, 10)

    ball = tuple(np.round(env.pos).astype(int))
    pygame.draw.circle(
        screen, COLORS["ball_edge"], (ball[0] + 2, ball[1] + 3), CONFIG.ball_radius
    )
    pygame.draw.circle(screen, COLORS["ball"], ball, CONFIG.ball_radius)
    pygame.draw.circle(screen, (255, 243, 154), (ball[0] - 4, ball[1] - 4), 4)

    title = font.render("Automated Ball-in-Maze Control", True, COLORS["text"])
    screen.blit(title, (42, CONFIG.height + 12))
    info = small_font.render(
        f"mode={mode}  steps={env.steps}/{CONFIG.max_steps}  reward={total_reward:7.2f}  {status}",
        True,
        COLORS["muted"],
    )
    screen.blit(info, (42, CONFIG.height + 48))

    hint = "Arrow keys/WASD tilt the board. R resets. Esc quits."
    if mode != "manual":
        hint = "R resets. Esc quits. Use trained checkpoints from models/."
    hint_surface = small_font.render(hint, True, COLORS["muted"])
    screen.blit(
        hint_surface, (CONFIG.width - hint_surface.get_width() - 42, CONFIG.height + 48)
    )


def manual_action(keys: pygame.key.ScancodeWrapper) -> np.ndarray:
    x = float(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - float(
        keys[pygame.K_LEFT] or keys[pygame.K_a]
    )
    y = float(keys[pygame.K_DOWN] or keys[pygame.K_s]) - float(
        keys[pygame.K_UP] or keys[pygame.K_w]
    )
    return np.array([x, y], dtype=np.float32) * CONFIG.max_tilt_deg


def main() -> None:
    args = parse_args()
    model_path = args.model
    if model_path is None and args.mode == "dqn":
        model_path = MODEL_DIR / "dqn_ball_maze.pt"
    if model_path is None and args.mode == "ddpg":
        model_path = MODEL_DIR / "ddpg_ball_maze.pt"

    policy = None
    if args.mode == "dqn":
        policy = DQNPolicy(model_path)
    elif args.mode == "ddpg":
        policy = DDPGPolicy(model_path)

    pygame.init()
    pygame.display.set_caption("Automated Ball-in-Maze Control")
    screen = pygame.display.set_mode((CONFIG.width, CONFIG.height + 86))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24, bold=True)
    small_font = pygame.font.SysFont("arial", 16)

    env = BallMazeEnv(seed=7)
    state = env.reset(randomize=False)
    total_reward = 0.0
    status = "running"
    episode_done = False
    trail: list[tuple[int, int]] = []
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    state = env.reset(randomize=False)
                    total_reward = 0.0
                    status = "running"
                    episode_done = False
                    trail.clear()

        keys = pygame.key.get_pressed()
        if not episode_done:
            if args.mode == "manual":
                result = env.step_continuous(manual_action(keys))
            elif args.mode == "random":
                result = env.step_discrete(
                    int(env.rng.integers(0, env.discrete_action_dim))
                )
            elif args.mode == "dqn":
                result = env.step_discrete(policy.act(state))
            else:
                result = env.step_continuous(policy.act(state))

            state = result.state
            total_reward += result.reward
            status = result.info["event"]
            if result.done:
                episode_done = True
                status = f"finished: {status}"
            trail.append(tuple(np.round(env.pos).astype(int)))

        draw_scene(
            screen, font, small_font, env, args.mode, total_reward, status, trail
        )
        pygame.display.flip()
        clock.tick(CONFIG.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
