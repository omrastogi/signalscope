"""
Backtesting framework for SignalScope.

Simulates three strategies on the test split (2024) and prints a
performance comparison table: Q-learning, buy-and-hold, random.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path

from rl.environment import load_env, N_ACTIONS
from rl.agent import QLearningAgent, Q_TABLE_PATH

STARTING_CAPITAL = 10_000.0


# ------------------------------------------------------------------
def _run_episode(env, agent_fn, ep_idx: int) -> dict:
    """
    Run a single episode with the given action function.
    Returns a dict with performance metrics and the daily equity series.
    """
    state = env.reset(ep_idx)
    done = False
    equity = STARTING_CAPITAL
    equity_curve = [equity]
    daily_returns = []

    while not done:
        action = agent_fn(state)
        next_state, reward, done = env.step(action)
        # reward = action * daily_return, so pct_return = reward
        equity *= 1.0 + reward
        equity_curve.append(equity)
        daily_returns.append(reward)
        state = next_state

    total_return = (equity - STARTING_CAPITAL) / STARTING_CAPITAL
    sharpe = _sharpe(daily_returns)
    max_dd = _max_drawdown(equity_curve)

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
    }


def _sharpe(daily_returns: list[float], annualize: int = 252) -> float:
    r = np.array(daily_returns)
    if r.std() == 0:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(annualize))


def _max_drawdown(equity_curve: list[float]) -> float:
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak
    return float(drawdown.min())


# ------------------------------------------------------------------
def run_backtest(seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    env = load_env("test")
    tickers = env.episode_tickers()

    agent = QLearningAgent()
    agent.load(Q_TABLE_PATH)
    agent.epsilon = 0.0  # pure greedy

    results = []
    equity_curves: dict[str, dict] = {}

    for ep_idx, ticker in enumerate(tickers):
        # --- Q-learning ---
        ql = _run_episode(env, lambda s: agent.act(s, explore=False), ep_idx)

        # --- Buy-and-hold: BUY on day 0, HOLD every day after ---
        _bh_step = [0]
        def _bh_action(s, _counter=_bh_step):
            act = 2 if _counter[0] == 0 else 1  # BUY first, then HOLD
            _counter[0] += 1
            return act
        bh = _run_episode(env, _bh_action, ep_idx)

        # --- Random ---
        env.reset(ep_idx)
        rnd = _run_episode(env, lambda s: random.randint(0, N_ACTIONS - 1), ep_idx)

        equity_curves[ticker] = {
            "Q-Learning": ql["equity_curve"],
            "Buy-and-Hold": bh["equity_curve"],
            "Random": rnd["equity_curve"],
        }

        for strat_name, metrics in [
            ("Q-Learning", ql),
            ("Buy-and-Hold", bh),
            ("Random", rnd),
        ]:
            results.append(
                {
                    "ticker": ticker,
                    "strategy": strat_name,
                    "total_return": metrics["total_return"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )

    df = pd.DataFrame(results)
    return df, equity_curves


# ------------------------------------------------------------------
def print_summary(df: pd.DataFrame):
    print("\n=== Test Period (2024) Performance ===\n")
    for strat in ["Q-Learning", "Buy-and-Hold", "Random"]:
        sub = df[df["strategy"] == strat]
        print(f"  {strat}")
        print(f"    Avg return  : {sub['total_return'].mean()*100:.1f}%")
        print(f"    Avg Sharpe  : {sub['sharpe'].mean():.2f}")
        print(f"    Avg max DD  : {sub['max_drawdown'].mean()*100:.1f}%")
        print()

    print("--- Per-ticker breakdown ---")
    pivot = df.pivot(index=["ticker"], columns="strategy", values="total_return") * 100
    print(pivot.round(1).to_string())


if __name__ == "__main__":
    df, curves = run_backtest()
    print_summary(df)
    df.to_csv(
        Path(__file__).parent.parent / "data" / "processed" / "backtest_results.csv",
        index=False,
    )
    print("\nResults saved to data/processed/backtest_results.csv")
