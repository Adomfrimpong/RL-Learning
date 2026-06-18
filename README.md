# Frozen Lake from First Principles Using Q-Learning

Name: Isaac Frimpong Asante
Student ID: 22424646
GitHub Repository: https://github.com/Adomfrimpong/My-Reinforcement-Learning-Hub
Report Link: https://github.com/Adomfrimpong/My-Reinforcement-Learning-Hub/blob/main/Report.pdf

This project delivers a complete Reinforcement Learning solution for the 8x8 Frozen Lake problem. The environment, the agent, the training loop, and the evaluation run from scratch in plain Python. The code uses no Gymnasium, Gym, Stable Baselines, or RLlib.

## Introduction

### What is Reinforcement Learning?

Reinforcement Learning trains an agent to make decisions through trial and error. The agent observes a state, picks an action, and receives a reward. Over many episodes the agent learns a policy. A policy maps each state to the action with the highest long-term value.

### What is Frozen Lake?

Frozen Lake is a grid-world. The agent starts at the top-left cell and walks to the goal at the bottom-right cell. Frozen cells are safe. Holes end the episode with no reward. The goal ends the episode with a reward of 1. The map is 8x8:

```
SFFFFFFF
FFFFFFFF
FFFHFFFF
FFFHFFFF
FFFHFFFF
FHHFFFHF
FHFFHFHF
FFFHFFFG
```

S is the start. F is frozen. H is a hole. G is the goal.

## Environment Design

### State representation

The environment uses a single integer index per state. The index equals `row * 8 + col`. The grid holds 64 states numbered 0 to 63.

### Action representation

Four actions drive the agent:

```
0 = Left
1 = Down
2 = Right
3 = Up
```

The environment enforces grid boundaries. An action into a wall keeps the agent in place.

### Reward structure

- Reach the goal: reward 1.0 and the episode ends.
- Fall into a hole: reward 0.0 and the episode ends.
- Step onto a frozen cell: reward 0.0 and the episode continues.

## Q-Learning Algorithm

### Description of Q-Learning

Q-Learning learns a table of action values. Each entry `Q(s, a)` estimates the expected return of taking action `a` in state `s` and then acting greedily. The agent starts the table at zero and refines each entry from experience.

### The update equation

The agent implements the rule exactly as required:

```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
```

- alpha is the learning rate. A higher alpha shifts estimates faster.
- gamma is the discount factor. A higher gamma values future reward more.
- The bracket term is the temporal-difference error. The agent moves each estimate toward the observed target.

### Exploration strategy

The agent uses epsilon-greedy exploration. With probability epsilon the agent picks a random action. Otherwise the agent picks the highest-value action. Epsilon starts at 1.0 and decays toward 0.01. Early episodes explore the grid. Later episodes exploit the learned policy.

## Training Procedure

### Hyperparameters used

```
Episodes:        20000
Max steps:       200
Learning rate:   0.10
Discount factor: 0.99
Epsilon start:   1.00
Epsilon min:     0.01
Epsilon decay:   0.9995 per episode
```

### Training loop

Each episode resets the agent to the start. The agent acts, observes the reward and next state, and updates the Q-table on every step. Epsilon decays after each episode. The training loop records episode rewards, success flags, and epsilon over time.

## Results

### Final success rate

The greedy evaluation over 100 episodes on the deterministic map reaches a success rate of 100 percent with an average reward of 1.0 and zero failures. Across the full 20000 training episodes the success rate climbs above 99 percent once epsilon settles near its minimum.

### Learned policy

```
↓ → ↓ ↓ ↓ ↓ ↓ ↓
→ → → → → → ↓ ↓
↑ ↑ ↑ H → → → ↓
↑ ↑ ↑ H ↑ → → ↓
↑ ↑ ↑ H → → → ↓
↑ H H → ↑ ↑ H ↓
↑ H ← → H ↑ H ↓
→ ↓ ← H ↓ ↑ ↓ G
```

The policy routes the agent right along the top row, then down the right edge, then into the goal. The arrows steer the agent away from every hole.

### Discussion of performance

The agent learns a stable path. Reward and success rate rise together as epsilon falls. The agent reaches near-perfect play once exploration ends.

## Bonus Task

This project completes Option B, visualize training performance with graphs. The bonus script reads the saved training statistics and plots reward, success rate, and epsilon decay across the training episodes. The figure saves to `results/training_performance.png`. Reward and success rate climb together as epsilon falls, and the agent reaches near-perfect play once exploration ends.

## Repository Structure

```
frozen-lake-qlearning/
├── environment.py      Custom Frozen Lake environment (Part A)
├── agent.py            Q-Learning agent (Part B)
├── train.py            Training, stats, policy extraction (Parts C and D)
├── evaluate.py         Evaluation over 100+ episodes (Part E)
├── bonus.py            Bonus task, Option B training graphs
├── requirements.txt
├── README.md
├── report.pdf
└── results/
    ├── q_table.json
    ├── training_stats.json
    ├── policy.txt
    ├── evaluation.json
    └── training_performance.png
```

## Execution Instructions

Install the one dependency for the graphs:

```
pip install -r requirements.txt
```

Train the agent and extract the policy:

```
python train.py
```

Evaluate the trained agent over 100 episodes:

```
python evaluate.py
```

Generate the training performance graph:

```
python bonus.py
```

The core agent and environment use the Python standard library only. Matplotlib drives the bonus graphs.
