"""Shared configuration for the ball-in-maze project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
PLOT_DIR = ROOT_DIR / "plots"


@dataclass(frozen=True)
class MazeConfig:
    width: int = 900
    height: int = 620
    margin: int = 36
    ball_radius: int = 11
    trap_radius: int = 25
    goal_radius: int = 28
    goal_success_radius: int = 16
    max_steps: int = 700
    fps: int = 60

    # Physics. The action is treated as a board tilt angle in degrees.
    max_tilt_deg: float = 10.0
    gravity_accel: float = 1250.0
    friction: float = 0.985
    max_speed: float = 620.0
    dt: float = 1.0 / 60.0

    # RL reward values from the proposal, with distance shaping added.
    goal_reward: float = 100.0
    trap_penalty: float = -50.0
    wall_penalty: float = -5.0
    time_penalty: float = -0.01
    progress_scale: float = 0.10


CONFIG = MazeConfig()


START_POS = (84.0, 536.0)
GOAL_POS = (812.0, 82.0)

TRAPS = (
    (210.0, 430.0),
    (340.0, 210.0),
    (525.0, 470.0),
    (640.0, 260.0),
    (745.0, 390.0),
)

# Wall rectangles: x, y, width, height.
WALLS = (
    (0.0, 0.0, CONFIG.width, CONFIG.margin),
    (0.0, CONFIG.height - CONFIG.margin, CONFIG.width, CONFIG.margin),
    (0.0, 0.0, CONFIG.margin, CONFIG.height),
    (CONFIG.width - CONFIG.margin, 0.0, CONFIG.margin, CONFIG.height),
    (150.0, 120.0, 24.0, 365.0),
    (150.0, 120.0, 260.0, 24.0),
    (270.0, 250.0, 24.0, 300.0),
    (270.0, 250.0, 250.0, 24.0),
    (430.0, 75.0, 24.0, 260.0),
    (560.0, 180.0, 24.0, 360.0),
    (560.0, 180.0, 200.0, 24.0),
    (690.0, 325.0, 24.0, 210.0),
)
