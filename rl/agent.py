"""
Tabular Q-learning agent for SignalScope.

Trains on all 6 ticker episodes (train split) using ε-greedy exploration.
Saves/loads the Q-table as a JSON file for reproducibility.
"""

import json
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

from rl.environment import load_env, N_ACTIONS, ACTIONS

Q_TABLE_PATH = Path(__file__).parent / "qtable.json"


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q: dict[tuple, list[float]] = defaultdict(lambda: [0.0] * N_ACTIONS)

    # ------------------------------------------------------------------
    def act(self, state: str, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        return int(np.argmax(self.q[state]))

    def update(self, state: str, action: int, reward: float, next_state: str, done: bool):
        target = reward if done else reward + self.gamma * max(self.q[next_state])
        self.q[state][action] += self.alpha * (target - self.q[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    def save(self, path: Path = Q_TABLE_PATH):
        serializable = {str(k): v for k, v in self.q.items()}
        path.write_text(json.dumps(serializable, indent=2))
        print(f"Q-table saved to {path}  ({len(self.q)} states)")

    def load(self, path: Path = Q_TABLE_PATH):
        data = json.loads(path.read_text())
        self.q = defaultdict(lambda: [0.0] * N_ACTIONS, {k: v for k, v in data.items()})
        print(f"Q-table loaded from {path}  ({len(self.q)} states)")


# ------------------------------------------------------------------
def train(n_passes: int = 200, seed: int = 42) -> QLearningAgent:
    """
    Train the portfolio env for `n_passes` full sweeps over all trading dates.
    Each pass runs all tickers simultaneously for every date in the train split.
    Per-ticker rewards are used for Q-updates; shared cash enforces capital constraint.
    """
    random.seed(seed)
    np.random.seed(seed)

    env   = load_env("train")
    agent = QLearningAgent()

    for pass_idx in range(n_passes):
        states      = env.reset()
        done        = False
        pass_reward = 0.0

        while not done:
            actions = {t: agent.act(states[t], explore=True) for t in env.tickers}
            next_states, ticker_rewards, total_reward, done = env.step(actions)

            for t in env.tickers:
                agent.update(states[t], actions[t], ticker_rewards[t],
                             next_states[t], done)

            states       = next_states
            pass_reward += total_reward

        agent.decay_epsilon()

        if (pass_idx + 1) % 20 == 0:
            print(
                f"Pass {pass_idx+1:3d}/{n_passes}  "
                f"reward={pass_reward:.4f}  "
                f"eps={agent.epsilon:.3f}  "
                f"states_seen={len(agent.q)}"
            )

    agent.save()
    return agent


# ------------------------------------------------------------------
def policy_summary(agent: QLearningAgent):
    """Print the greedy policy for every known state."""
    print(f"\nPolicy summary ({len(agent.q)} states):")
    action_names = list(ACTIONS.values())
    counts = {name: 0 for name in action_names}
    for state, vals in sorted(agent.q.items()):
        best = action_names[int(np.argmax(vals))]
        counts[best] += 1
    for a, c in counts.items():
        print(f"  {a}: {c} states ({100*c/len(agent.q):.1f}%)")


if __name__ == "__main__":
    agent = train(n_passes=200)
    policy_summary(agent)
