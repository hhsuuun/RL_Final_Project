"""Optional live visualization for training runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CONFIG
from training_plots import moving_average


@dataclass
class TrainingStatus:
    algorithm: str
    episode: int
    total_episodes: int
    episode_return: float
    best_return: float
    metric_name: str
    metric_value: float
    event: str


class TrainingVisualizer:
    """Pygame window showing the maze and a live return curve."""

    def __init__(self, algorithm: str, total_episodes: int, fps: int = 30) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for live training visualization. "
                "Install it with: python3 -m pip install pygame"
            ) from exc

        from game import COLORS

        self.pygame = pygame
        self.colors = COLORS
        self.algorithm = algorithm
        self.total_episodes = total_episodes
        self.fps = fps
        self.plot_width = 460
        self.width = CONFIG.width + self.plot_width
        self.height = CONFIG.height + 86
        self.trail: list[tuple[int, int]] = []
        self.last_episode = 0
        self.enabled = True

        pygame.init()
        pygame.display.set_caption(f"{algorithm.upper()} Training Monitor")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24, bold=True)
        self.small_font = pygame.font.SysFont("arial", 16)
        self.tiny_font = pygame.font.SysFont("arial", 13)

    def update(
        self,
        env,
        returns: list[float],
        status: TrainingStatus,
    ) -> bool:
        if not self.enabled:
            return False

        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                return False

        if status.episode != self.last_episode:
            self.trail.clear()
            self.last_episode = status.episode
        self.trail.append(tuple(np.round(env.pos).astype(int)))

        from game import draw_scene

        draw_scene(
            self.screen,
            self.font,
            self.small_font,
            env,
            f"{status.algorithm} train",
            status.episode_return,
            status.event,
            self.trail,
        )
        self._draw_plot(returns, status)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self) -> None:
        if self.enabled:
            self.enabled = False
            self.pygame.display.quit()

    def _draw_plot(self, returns: list[float], status: TrainingStatus) -> None:
        pygame = self.pygame
        x0 = CONFIG.width
        panel = pygame.Rect(x0, 0, self.plot_width, self.height)
        pygame.draw.rect(self.screen, (247, 248, 244), panel)
        pygame.draw.line(self.screen, (203, 208, 198), (x0, 0), (x0, self.height), 2)

        title = self.font.render("Training Plot", True, self.colors["text"])
        self.screen.blit(title, (x0 + 28, 24))

        lines = [
            f"episode {status.episode}/{status.total_episodes}",
            f"return {status.episode_return:8.2f}",
            f"best   {status.best_return:8.2f}",
            f"{status.metric_name} {status.metric_value:.3f}",
            "Esc closes monitor; training continues.",
        ]
        for idx, line in enumerate(lines):
            text = self.small_font.render(line, True, self.colors["muted"])
            self.screen.blit(text, (x0 + 30, 62 + idx * 22))

        chart = pygame.Rect(x0 + 36, 190, self.plot_width - 70, 380)
        pygame.draw.rect(self.screen, (255, 255, 252), chart, border_radius=6)
        pygame.draw.rect(self.screen, (198, 204, 195), chart, width=1, border_radius=6)

        if len(returns) >= status.episode:
            all_returns = returns
        else:
            all_returns = returns + [status.episode_return]
        if len(all_returns) < 2:
            empty = self.small_font.render(
                "Curve appears after the first episode.",
                True,
                self.colors["muted"],
            )
            self.screen.blit(empty, (chart.x + 34, chart.centery - 10))
            return

        low = min(all_returns)
        high = max(all_returns)
        if abs(high - low) < 1e-6:
            low -= 1.0
            high += 1.0
        margin = (high - low) * 0.08
        low -= margin
        high += margin

        self._draw_axes(chart, low, high)
        self._draw_curve(chart, all_returns, low, high, (95, 153, 185), 2)

        smooth_window = min(25, len(all_returns))
        smoothed = moving_average(all_returns, smooth_window)
        if len(smoothed) >= 2:
            smooth_values = smoothed.tolist()
            self._draw_curve(chart, smooth_values, low, high, (200, 79, 75), 3)

        legend_y = chart.bottom + 18
        self._legend_item(chart.x, legend_y, (95, 153, 185), "episode return")
        self._legend_item(chart.x + 165, legend_y, (200, 79, 75), "moving average")

    def _draw_axes(self, chart, low: float, high: float) -> None:
        pygame = self.pygame
        axis_color = (116, 126, 132)
        grid_color = (225, 229, 221)
        for idx in range(5):
            y = chart.bottom - int(chart.height * idx / 4)
            pygame.draw.line(self.screen, grid_color, (chart.x, y), (chart.right, y), 1)
            value = low + (high - low) * idx / 4
            label = self.tiny_font.render(f"{value:.0f}", True, self.colors["muted"])
            self.screen.blit(label, (chart.x + 6, y - 16))
        pygame.draw.line(self.screen, axis_color, chart.bottomleft, chart.bottomright, 1)
        pygame.draw.line(self.screen, axis_color, chart.bottomleft, chart.topleft, 1)

    def _draw_curve(
        self,
        chart,
        values: list[float],
        low: float,
        high: float,
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        if len(values) < 2:
            return
        pygame = self.pygame
        points = []
        denom = max(1, len(values) - 1)
        for idx, value in enumerate(values):
            x = chart.x + int(chart.width * idx / denom)
            normalized = (value - low) / (high - low)
            y = chart.bottom - int(chart.height * normalized)
            points.append((x, y))
        pygame.draw.lines(self.screen, color, False, points, width)

    def _legend_item(
        self,
        x: int,
        y: int,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        pygame = self.pygame
        pygame.draw.line(self.screen, color, (x, y + 8), (x + 28, y + 8), 3)
        text = self.tiny_font.render(label, True, self.colors["muted"])
        self.screen.blit(text, (x + 36, y))
