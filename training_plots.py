"""Utilities for saving RL training curves."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


def moving_average(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float32)
    window = max(1, min(window, len(values)))
    weights = np.ones(window, dtype=np.float32) / window
    return np.convolve(np.asarray(values, dtype=np.float32), weights, mode="valid")


def save_training_curve(
    returns: list[float],
    metric_values: list[float],
    metric_name: str,
    title: str,
    csv_path: Path,
    figure_path: Path,
    average_window: int = 25,
) -> None:
    csv_path.parent.mkdir(exist_ok=True)
    figure_path.parent.mkdir(exist_ok=True)

    smoothed = moving_average(returns, average_window)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "return", metric_name])
        for episode, (episode_return, metric_value) in enumerate(
            zip(returns, metric_values), start=1
        ):
            writer.writerow([episode, episode_return, metric_value])

    try:
        cache_dir = figure_path.parent / ".matplotlib_cache"
        cache_dir.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to draw training curves. "
            "Install it with: python3 -m pip install matplotlib"
        ) from exc

    episodes = np.arange(1, len(returns) + 1)
    plt.figure(figsize=(10, 5.6))
    plt.plot(episodes, returns, color="#7aa6c2", alpha=0.35, linewidth=1.0, label="Episode return")
    if len(smoothed) > 0:
        smooth_episodes = np.arange(average_window, average_window + len(smoothed))
        if len(returns) < average_window:
            smooth_episodes = np.arange(1, len(smoothed) + 1)
        plt.plot(
            smooth_episodes,
            smoothed,
            color="#c84f4b",
            linewidth=2.2,
            label=f"{average_window}-episode moving average",
        )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
