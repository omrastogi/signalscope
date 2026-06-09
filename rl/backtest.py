"""
Backtesting framework for SignalScope portfolio environment.

Runs three strategies over the test split (2024) on the full 6-ticker
portfolio. agent_fn_map functions return (action, lot_size) tuples so
dynamic lot sizing flows through cleanly.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path

from rl.environment import load_env, N_ACTIONS, MAX_LOTS, LOT_SIZE
from rl.agent import QLearningAgent, Q_TABLE_PATH

STARTING_CAPITAL = 10_000.0


# ------------------------------------------------------------------
def _run_episode(env, agent_fn_map: dict) -> dict:
    """
    Run one full episode with per-ticker action functions.
    agent_fn_map: {ticker: callable(state_str) -> (action_int, lot_size_int)}
    """
    states = env.reset()
    done   = False
    equity_curve  = [env.portfolio_value]
    daily_returns = []

    while not done:
        action_lots = {t: agent_fn_map[t](states[t]) for t in env.tickers}
        actions     = {t: al[0] for t, al in action_lots.items()}
        lot_sizes   = {t: al[1] for t, al in action_lots.items()}

        next_states, _, total_reward, done = env.step(actions, lot_sizes)
        equity_curve.append(env.portfolio_value)
        daily_returns.append(total_reward)
        states = next_states

    total_return = (equity_curve[-1] - STARTING_CAPITAL) / STARTING_CAPITAL
    return {
        "total_return":  total_return,
        "sharpe":        _sharpe(daily_returns),
        "max_drawdown":  _max_drawdown(equity_curve),
        "equity_curve":  equity_curve,
        "daily_returns": daily_returns,
    }


def _sharpe(daily_returns: list[float], annualize: int = 252) -> float:
    r = np.array(daily_returns)
    if r.std() == 0:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(annualize))


def _max_drawdown(equity_curve: list[float]) -> float:
    eq   = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


# ------------------------------------------------------------------
def run_backtest(
    seed:           int  = 42,
    use_confidence: bool = False,
    use_vol:        bool = False,
) -> tuple[pd.DataFrame, dict]:
    random.seed(seed)
    np.random.seed(seed)

    env   = load_env("test")
    agent = QLearningAgent()
    agent.load(Q_TABLE_PATH)
    agent.epsilon = 0.0  # pure greedy

    # --- Q-learning (with dynamic lot sizing flags) ---
    ql_map = {
        t: (lambda s, t=t: agent.act(s, explore=False,
                                     use_confidence=use_confidence,
                                     use_vol=use_vol))
        for t in env.tickers
    }
    ql = _run_episode(env, ql_map)

    # --- Buy-and-hold: accumulate MAX_LOTS then hold (fixed lot size) ---
    bh_counts = {t: 0 for t in env.tickers}
    def _bh_fn(ticker):
        def _act(s):
            if bh_counts[ticker] < MAX_LOTS:
                bh_counts[ticker] += 1
                return 2, LOT_SIZE
            return 1, LOT_SIZE
        return _act
    bh_map = {t: _bh_fn(t) for t in env.tickers}
    bh = _run_episode(env, bh_map)

    # --- Random (fixed lot size) ---
    rnd_map = {t: (lambda s: (random.randint(0, N_ACTIONS - 1), LOT_SIZE))
               for t in env.tickers}
    rnd = _run_episode(env, rnd_map)

    results = []
    equity_curves = {}
    for strat_name, metrics in [("Q-Learning", ql), ("Buy-and-Hold", bh), ("Random", rnd)]:
        results.append({
            "strategy":     strat_name,
            "total_return": metrics["total_return"],
            "sharpe":       metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
        })
        equity_curves[strat_name] = metrics["equity_curve"]

    return pd.DataFrame(results), equity_curves


# ------------------------------------------------------------------
def print_summary(df: pd.DataFrame):
    print("\n=== Test Period (2024) Portfolio Performance ===\n")
    for _, row in df.iterrows():
        final = STARTING_CAPITAL * (1 + row["total_return"])
        print(f"  {row['strategy']}")
        print(f"    Final value  : ${final:,.0f}  ({row['total_return']*100:+.1f}%)")
        print(f"    Sharpe       : {row['sharpe']:.2f}")
        print(f"    Max drawdown : {row['max_drawdown']*100:.1f}%")
        print()


# ------------------------------------------------------------------
if __name__ == "__main__":
    df, curves = run_backtest()
    print_summary(df)
    df.to_csv(
        Path(__file__).parent.parent / "data" / "processed" / "backtest_results.csv",
        index=False,
    )
    print("Results saved to data/processed/backtest_results.csv")
