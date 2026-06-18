"""
bonus.py
Bonus task, Option B: visualize training performance with graphs.

The script reads the training statistics saved by train.py and plots
reward, success rate, and epsilon decay over the training episodes.
The figure is saved to results/training_performance.png.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"


def moving_average(values, window):
    """Return a simple moving average over the given window."""
    averaged = []
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
            averaged.append(running / window)
        else:
            averaged.append(running / (i + 1))
    return averaged


def plot_training(stats, window=200):
    """Plot reward trend, success rate, and epsilon decay."""
    rewards = stats["episode_rewards"]
    success = stats["success_flags"]
    epsilon = stats["epsilon_history"]
    episodes = list(range(1, len(rewards) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(episodes, moving_average(rewards, window), color="#1f77b4")
    axes[0].set_title("Average Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward (moving avg)")

    axes[1].plot(
        episodes,
        [100 * v for v in moving_average(success, window)],
        color="#2ca02c",
    )
    axes[1].set_title("Success Rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success rate %")

    axes[2].plot(episodes, epsilon, color="#d62728")
    axes[2].set_title("Epsilon Decay")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_performance.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "training_stats.json"), "r") as f:
        stats = json.load(f)

    path = plot_training(stats)
    print("Saved training graphs to", path)


if __name__ == "__main__":
    main()
