"""
agent.py
Q-Learning agent built from first principles.

Update rule:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
"""

import random


class QLearningAgent:
    """Tabular Q-Learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        decay_epsilon=True,
        seed=None,
    ):
        """
        n_states: number of discrete states.
        n_actions: number of discrete actions.
        alpha: learning rate.
        gamma: discount factor.
        epsilon: starting exploration rate.
        epsilon_min: lower bound on epsilon.
        epsilon_decay: multiplicative decay factor applied each episode.
        decay_epsilon: when False, epsilon stays fixed (pure epsilon-greedy).
        seed: optional integer for reproducible action selection.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_decay = decay_epsilon
        self.rng = random.Random(seed)

        # Q-table initialized to zeros.
        self.q_table = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]

    def _argmax(self, values):
        """Return the index of the highest value, breaking ties at random."""
        best = max(values)
        candidates = [i for i, v in enumerate(values) if v == best]
        return self.rng.choice(candidates)

    def choose_action(self, state):
        """Select an action using the epsilon-greedy rule."""
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        return self._argmax(self.q_table[state])

    def greedy_action(self, state):
        """Select the highest-value action with no exploration."""
        return self._argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """Apply the Q-Learning update for one transition."""
        current = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] = current + self.alpha * (target - current)

    def decay(self):
        """Decay epsilon toward its minimum after an episode."""
        if self.use_decay:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
