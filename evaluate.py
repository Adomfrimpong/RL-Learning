"""
evaluate.py
Evaluate a trained Q-Learning agent.
Loads the saved Q-table and runs greedy episodes.
Reports success rate, average reward, failures, and successful runs.
"""

import json
import os

from environment import FrozenLakeEnv
from agent import QLearningAgent

RESULTS_DIR = "results"


def load_agent(env):
    """Build an agent and load the trained Q-table from disk."""
    path = os.path.join(RESULTS_DIR, "q_table.json")
    with open(path, "r") as f:
        q_table = json.load(f)
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    agent.q_table = q_table
    agent.epsilon = 0.0
    return agent


def evaluate(agent, env, episodes=100, max_steps=200):
    """Run greedy episodes and return evaluation metrics."""
    successes = 0
    total_reward = 0.0

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = agent.greedy_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                if reward > 0:
                    successes += 1
                break

    failures = episodes - successes
    success_rate = 100.0 * successes / episodes
    average_reward = total_reward / episodes

    return {
        "episodes": episodes,
        "success_rate": success_rate,
        "average_reward": average_reward,
        "failures": failures,
        "successful_runs": successes,
    }


def main():
    env = FrozenLakeEnv(is_slippery=False, seed=7)
    agent = load_agent(env)
    metrics = evaluate(agent, env, episodes=100)

    print("Evaluation over", metrics["episodes"], "episodes")
    print(f"Success Rate: {metrics['success_rate']:.2f}%")
    print(f"Average Reward: {metrics['average_reward']:.4f}")
    print(f"Number of Failures: {metrics['failures']}")
    print(f"Number of Successful Runs: {metrics['successful_runs']}")

    with open(os.path.join(RESULTS_DIR, "evaluation.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
