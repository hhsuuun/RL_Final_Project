"""Physics environment for the automated ball-in-maze control task.

State follows the proposal:
    [x, y, vx, vy, d_trap] in R^5

The environment intentionally has no Pygame dependency so training can run
headlessly. The screen renderer in game.py uses this same class.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, sin, sqrt
from typing import Iterable

import numpy as np

from config import CONFIG, GOAL_POS, START_POS, TRAPS, WALLS, MazeConfig


DISCRETE_ACTIONS = {
    0: (0.0, -1.0),  # Up
    1: (0.0, 1.0),   # Down
    2: (-1.0, 0.0),  # Left
    3: (1.0, 0.0),   # Right
    4: (0.0, 0.0),   # Stay
}


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict


class BallMazeEnv:
    """A compact RL environment for the ball-in-maze proposal."""

    state_dim = 5
    discrete_action_dim = 5
    continuous_action_dim = 2

    def __init__(
        self,
        config: MazeConfig = CONFIG,
        start_pos: tuple[float, float] = START_POS,
        goal_pos: tuple[float, float] = GOAL_POS,
        traps: Iterable[tuple[float, float]] = TRAPS,
        walls: Iterable[tuple[float, float, float, float]] = WALLS,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.traps = tuple(np.array(t, dtype=np.float32) for t in traps)
        self.walls = tuple(tuple(float(v) for v in wall) for wall in walls)
        self.rng = np.random.default_rng(seed)
        self.pos = self.start_pos.copy()
        self.vel = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.prev_goal_dist = self._distance_to_goal()

    def reset(self, randomize: bool = True) -> np.ndarray:
        jitter = self.rng.uniform(-8.0, 8.0, size=2) if randomize else 0.0
        self.pos = self.start_pos + jitter
        self.vel = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.prev_goal_dist = self._distance_to_goal()
        return self._state()

    def step_discrete(self, action: int) -> StepResult:
        if action not in DISCRETE_ACTIONS:
            raise ValueError(f"Invalid discrete action {action}")
        unit = np.array(DISCRETE_ACTIONS[action], dtype=np.float32)
        tilt = unit * self.config.max_tilt_deg
        return self.step_continuous(tilt)

    def step_continuous(self, action: np.ndarray | tuple[float, float]) -> StepResult:
        self.steps += 1
        action_arr = np.asarray(action, dtype=np.float32)
        action_arr = np.clip(
            action_arr, -self.config.max_tilt_deg, self.config.max_tilt_deg
        )

        accel = np.array(
            [
                self.config.gravity_accel * sin(radians(float(action_arr[0]))),
                self.config.gravity_accel * sin(radians(float(action_arr[1]))),
            ],
            dtype=np.float32,
        )
        self.vel = (self.vel + accel * self.config.dt) * self.config.friction
        speed = float(np.linalg.norm(self.vel))
        if speed > self.config.max_speed:
            self.vel *= self.config.max_speed / speed

        old_pos = self.pos.copy()
        self.pos = self.pos + self.vel * self.config.dt

        wall_hit = self._resolve_wall_collisions(old_pos)
        goal_dist = self._distance_to_goal()
        progress = self.prev_goal_dist - goal_dist
        self.prev_goal_dist = goal_dist

        reward = self.config.time_penalty + self.config.progress_scale * progress
        done = False
        event = "running"

        if wall_hit:
            reward += self.config.wall_penalty
            event = "wall"

        if self._in_trap():
            reward += self.config.trap_penalty
            done = True
            event = "trap"
        elif goal_dist <= self.config.goal_success_radius:
            reward += self.config.goal_reward
            done = True
            event = "goal"
        elif self.steps >= self.config.max_steps:
            done = True
            event = "timeout"

        return StepResult(
            state=self._state(),
            reward=float(reward),
            done=done,
            info={
                "event": event,
                "steps": self.steps,
                "goal_distance": goal_dist,
                "nearest_trap_distance": self._nearest_trap_distance(),
                "wall_hit": wall_hit,
            },
        )

    def _state(self) -> np.ndarray:
        return np.array(
            [
                self.pos[0] / self.config.width,
                self.pos[1] / self.config.height,
                self.vel[0] / self.config.max_speed,
                self.vel[1] / self.config.max_speed,
                self._nearest_trap_distance()
                / sqrt(self.config.width**2 + self.config.height**2),
            ],
            dtype=np.float32,
        )

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.pos - self.goal_pos))

    def _nearest_trap_distance(self) -> float:
        if not self.traps:
            return sqrt(self.config.width**2 + self.config.height**2)
        return min(float(np.linalg.norm(self.pos - trap)) for trap in self.traps)

    def _in_trap(self) -> bool:
        return any(
            float(np.linalg.norm(self.pos - trap))
            <= self.config.trap_radius + self.config.ball_radius * 0.45
            for trap in self.traps
        )

    def _resolve_wall_collisions(self, old_pos: np.ndarray) -> bool:
        hit = False
        for wall in self.walls:
            x, y, w, h = wall
            closest_x = float(np.clip(self.pos[0], x, x + w))
            closest_y = float(np.clip(self.pos[1], y, y + h))
            delta = self.pos - np.array([closest_x, closest_y], dtype=np.float32)
            dist = float(np.linalg.norm(delta))
            if dist < self.config.ball_radius:
                hit = True
                normal = self._collision_normal(delta, old_pos, wall)
                self.pos = np.array([closest_x, closest_y], dtype=np.float32)
                self.pos += normal * (self.config.ball_radius + 0.5)
                velocity_into_wall = float(np.dot(self.vel, normal))
                if velocity_into_wall < 0:
                    self.vel -= 1.65 * velocity_into_wall * normal
                self.vel *= 0.72
        return hit

    def _collision_normal(
        self,
        delta: np.ndarray,
        old_pos: np.ndarray,
        wall: tuple[float, float, float, float],
    ) -> np.ndarray:
        dist = float(np.linalg.norm(delta))
        if dist > 1e-6:
            return delta / dist

        x, y, w, h = wall
        distances = np.array(
            [
                abs(old_pos[0] - x),
                abs(old_pos[0] - (x + w)),
                abs(old_pos[1] - y),
                abs(old_pos[1] - (y + h)),
            ],
            dtype=np.float32,
        )
        side = int(np.argmin(distances))
        normals = (
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        )
        return normals[side]
