"""
train.py
Train the Q-Learning agent on Frozen Lake.
Records episode rewards, success rate, successful episodes, and epsilon.
Extracts and prints the learned policy.
Saves the Q-table and training statistics to the results folder.
"""

import json
import os

from environment import FrozenLakeEnv
from agent import QLearningAgent

RESULTS_DIR = "results"

# Action symbols for the policy grid.
ARROWS = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}


def train(
    episodes=20000,
    max_steps=200,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.9995,
    is_slippery=False,
    seed=42,
    verbose=True,
):
    """Run training and return the agent, environment, and statistics."""
    env = FrozenLakeEnv(is_slippery=is_slippery, seed=seed)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    episode_rewards = []
    epsilon_history = []
    success_flags = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay()
        episode_rewards.append(total_reward)
        epsilon_history.append(agent.epsilon)
        success_flags.append(1 if total_reward > 0 else 0)

        if verbose and (episode + 1) % 2000 == 0:
            window = success_flags[-2000:]
            rate = 100.0 * sum(window) / len(window)
            print(
                f"Episode {episode + 1:>6} | "
                f"epsilon {agent.epsilon:.3f} | "
                f"success rate last 2000: {rate:5.1f}%"
            )

    successful_episodes = sum(success_flags)
    overall_rate = 100.0 * successful_episodes / episodes

    stats = {
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "is_slippery": is_slippery,
        "successful_episodes": successful_episodes,
        "overall_success_rate": overall_rate,
        "episode_rewards": episode_rewards,
        "epsilon_history": epsilon_history,
        "success_flags": success_flags,
    }
    return agent, env, stats


def extract_policy(agent, env):
    """Return a list of best actions per state from the Q-table."""
    policy = []
    for s in range(env.n_states):
        if env.is_terminal(s):
            policy.append(None)
        else:
            policy.append(agent.greedy_action(s))
    return policy


def render_policy(policy, env):
    """Return the policy as a printable grid of arrows and symbols."""
    lines = []
    for r in range(env.nrows):
        row_symbols = []
        for c in range(env.ncols):
            index = r * env.ncols + c
            cell = env.grid[r][c]
            if cell == "H":
                row_symbols.append("H")
            elif cell == "G":
                row_symbols.append("G")
            else:
                row_symbols.append(ARROWS[policy[index]])
        lines.append(" ".join(row_symbols))
    return "\n".join(lines)


def save_results(agent, stats, policy_text):
    """Save the Q-table, statistics, and policy grid to the results folder."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "q_table.json"), "w") as f:
        json.dump(agent.q_table, f)

    with open(os.path.join(RESULTS_DIR, "training_stats.json"), "w") as f:
        json.dump(stats, f)

    with open(os.path.join(RESULTS_DIR, "policy.txt"), "w") as f:
        f.write(policy_text + "\n")


def main():
    agent, env, stats = train()

    policy = extract_policy(agent, env)
    policy_text = render_policy(policy, env)

    print("\nLearned policy grid:")
    print(policy_text)
    print(f"\nSuccessful training episodes: {stats['successful_episodes']}")
    print(f"Overall training success rate: {stats['overall_success_rate']:.2f}%")

    save_results(agent, stats, policy_text)
    print(f"\nSaved Q-table, stats, and policy to the {RESULTS_DIR} folder.")


if __name__ == "__main__":
    main()
