"""
Tabular Q-learning agent for SignalScope.

Trains on the portfolio env (train split) using ε-greedy exploration.
Supports two optional dynamic lot-sizing features, toggled independently:

  use_confidence  — scales lot size by sigmoid(Q[BUY] − Q[HOLD]),
                    so the agent buys/sells more when it is more certain
  use_vol         — scales lot size by a volatility multiplier parsed
                    from the state string: low→1.5, medium→1.0, high→0.5

Domain randomization (randomize=True):
  Runs n_episodes_per_pass independent episodes per training pass, each
  with a randomly sampled starting cash and base lot size. Forces the
  agent to learn a policy robust to different account sizes and position
  granularities. Backtest always evaluates at the fixed defaults.

All flags default to False / 1 (backward-compatible).
"""

import json
import math
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

from rl.environment import (
    load_env, N_ACTIONS, ACTIONS, LOT_SIZE, MIN_LOT, MAX_LOT, STARTING_CASH,
)

Q_TABLE_PATH = Path(__file__).parent.parent / "models" / "qtable.json"

_VOL_MULTIPLIER  = {"low": 1.5, "medium": 1.0, "high": 0.5}

# Domain-randomization option pools (sampled per episode when randomize=True)
CASH_OPTIONS = [5_000, 7_500, 10_000, 15_000, 20_000]
LOT_OPTIONS  = [10, 15, 25, 40, 50]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _parse_volatility(state: str) -> str:
    """Extract volatility bucket from state string."""
    for part in state.split("|"):
        if part.startswith("volatility="):
            return part.split("=", 1)[1]
    return "medium"


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q: dict[str, list[float]] = defaultdict(lambda: [0.0] * N_ACTIONS)

    # ------------------------------------------------------------------
    def _lot_size(
        self,
        state:          str,
        use_confidence: bool,
        use_vol:        bool,
        base_lot:       int = LOT_SIZE,
    ) -> int:
        """Compute dynamic lot size for the given state."""
        multiplier = 1.0

        if use_vol:
            vol = _parse_volatility(state)
            multiplier *= _VOL_MULTIPLIER.get(vol, 1.0)

        if use_confidence:
            q_buy, q_hold = self.q[state][2], self.q[state][1]
            confidence = _sigmoid(q_buy - q_hold)
            multiplier *= (1.0 + confidence)

        return int(np.clip(base_lot * multiplier, MIN_LOT, MAX_LOT))

    def act(
        self,
        state:          str,
        explore:        bool = True,
        use_confidence: bool = False,
        use_vol:        bool = False,
        base_lot:       int  = LOT_SIZE,
    ) -> tuple[int, int]:
        """
        Returns (action, lot_size).
        base_lot sets the episode lot size (randomized during domain-randomization
        training; fixed LOT_SIZE at backtest time).
        """
        if explore and random.random() < self.epsilon:
            action = random.randint(0, N_ACTIONS - 1)
        else:
            action = int(np.argmax(self.q[state]))

        lot = self._lot_size(state, use_confidence, use_vol, base_lot)
        return action, lot

    def update(self, state: str, action: int, reward: float, next_state: str, done: bool):
        target = reward if done else reward + self.gamma * max(self.q[next_state])
        self.q[state][action] += self.alpha * (target - self.q[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    def save(self, path: Path = Q_TABLE_PATH):
        path.write_text(json.dumps({k: v for k, v in self.q.items()}, indent=2))
        print(f"Q-table saved to {path}  ({len(self.q)} states)")

    def load(self, path: Path = Q_TABLE_PATH):
        data = json.loads(path.read_text())
        self.q = defaultdict(lambda: [0.0] * N_ACTIONS, data)
        print(f"Q-table loaded from {path}  ({len(self.q)} states)")


# ------------------------------------------------------------------
def train(
    n_passes:            int  = 200,
    seed:                int  = 42,
    use_confidence:      bool = False,
    use_vol:             bool = False,
    n_episodes_per_pass: int  = 1,
    randomize:           bool = False,
) -> QLearningAgent:
    """
    Train the portfolio env for `n_passes` full sweeps.

    use_confidence      : confidence-based lot sizing
    use_vol             : volatility-adjusted lot sizing
    n_episodes_per_pass : episodes per pass (default 1; set 4-5 with randomize=True)
    randomize           : domain randomization — each episode samples a random
                          starting_cash from CASH_OPTIONS and base lot from LOT_OPTIONS
    """
    random.seed(seed)
    np.random.seed(seed)

    env   = load_env("train")
    agent = QLearningAgent()

    for pass_idx in range(n_passes):
        pass_reward = 0.0

        for _ep in range(n_episodes_per_pass):
            cash = random.choice(CASH_OPTIONS) if randomize else STARTING_CASH
            lot  = random.choice(LOT_OPTIONS)  if randomize else LOT_SIZE

            states = env.reset(starting_cash=cash, lot_size=lot)
            done   = False

            while not done:
                action_lots = {
                    t: agent.act(states[t], explore=True,
                                 use_confidence=use_confidence,
                                 use_vol=use_vol, base_lot=lot)
                    for t in env.tickers
                }
                actions   = {t: al[0] for t, al in action_lots.items()}
                lot_sizes = {t: al[1] for t, al in action_lots.items()}

                next_states, ticker_rewards, total_reward, done = env.step(actions, lot_sizes)

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
def train_with_eval(
    n_passes:            int  = 5000,
    eval_interval:       int  = 500,
    use_confidence:      bool = False,
    use_vol:             bool = False,
    n_episodes_per_pass: int  = 1,
    randomize:           bool = False,
    train_start:         str | None = None,
    seed:                int  = 42,
) -> tuple[QLearningAgent, list[dict]]:
    """Train and eval at every eval_interval passes. Returns (agent, checkpoints)."""
    from rl.backtest import run_backtest  # local import — avoids circular dependency

    random.seed(seed)
    np.random.seed(seed)

    env   = load_env("train", train_start=train_start)
    agent = QLearningAgent()
    checkpoints: list[dict] = []

    for pass_idx in range(n_passes):
        for _ep in range(n_episodes_per_pass):
            cash = random.choice(CASH_OPTIONS) if randomize else STARTING_CASH
            lot  = random.choice(LOT_OPTIONS)  if randomize else LOT_SIZE

            states = env.reset(starting_cash=cash, lot_size=lot)
            done   = False

            while not done:
                action_lots = {
                    t: agent.act(states[t], explore=True,
                                 use_confidence=use_confidence,
                                 use_vol=use_vol, base_lot=lot)
                    for t in env.tickers
                }
                actions   = {t: al[0] for t, al in action_lots.items()}
                lot_sizes = {t: al[1] for t, al in action_lots.items()}

                next_states, ticker_rewards, _, done = env.step(actions, lot_sizes)

                for t in env.tickers:
                    agent.update(states[t], actions[t], ticker_rewards[t],
                                 next_states[t], done)
                states = next_states

        agent.decay_epsilon()

        if (pass_idx + 1) % 100 == 0:
            pct = (pass_idx + 1) / n_passes * 100
            print(f"  pass {pass_idx+1:4d}/{n_passes}  ({pct:.0f}%)  eps={agent.epsilon:.3f}  states={len(agent.q)}", flush=True)

        if (pass_idx + 1) % eval_interval == 0:
            df_eval, _ = run_backtest(agent=agent,
                                      use_confidence=use_confidence,
                                      use_vol=use_vol)
            ql = df_eval[df_eval["strategy"] == "Q-Learning"].iloc[0]
            cp = {
                "passes": pass_idx + 1,
                "return": round(ql["total_return"] * 100, 2),
                "sharpe": round(ql["sharpe"], 3),
                "max_dd": round(ql["max_drawdown"] * 100, 2),
            }
            checkpoints.append(cp)
            print(f"  [Eval @ {pass_idx+1}]  return={cp['return']:.1f}%  sharpe={cp['sharpe']:.3f}  max_dd={cp['max_dd']:.1f}%", flush=True)

    return agent, checkpoints


# ------------------------------------------------------------------
def policy_summary(agent: QLearningAgent):
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
