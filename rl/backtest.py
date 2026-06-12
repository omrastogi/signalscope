"""
Backtesting framework for SignalScope portfolio environment.

Runs three strategies over the test split (2025) on the full 6-ticker
portfolio. agent_fn_map functions return (action, lot_size) tuples so
dynamic lot sizing flows through cleanly.

Extended params:
  eval_capital     -- starting cash for backtest (default 10k)
  eval_lot         -- base lot size for backtest (default 25)
  agent            -- QLearningAgent to evaluate (required)
  track_per_ticker -- if True, accumulate per-ticker cumulative rewards
  use_stop_loss    -- toggle stop-loss risk control (default True)
  use_liquidation  -- toggle liquidation risk control (default True)
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path

from rl.environment import load_env, N_ACTIONS, MAX_LOTS, LOT_SIZE
from rl.agent import QLearningAgent

STARTING_CAPITAL = 10_000.0


# ------------------------------------------------------------------
def _run_episode(
    env,
    agent_fn_map: dict,
    starting_cash: float = STARTING_CAPITAL,
    lot_size: int = LOT_SIZE,
    track_per_ticker: bool = False,
) -> dict:
    states = env.reset(starting_cash=starting_cash, lot_size=lot_size)
    done          = False
    equity_curve  = [env.portfolio_value]
    daily_returns = []
    ticker_cumulative = {t: 0.0 for t in env.tickers} if track_per_ticker else None

    while not done:
        action_lots = {t: agent_fn_map[t](states[t]) for t in env.tickers}
        actions     = {t: al[0] for t, al in action_lots.items()}
        lot_sizes   = {t: al[1] for t, al in action_lots.items()}

        next_states, ticker_rewards, total_reward, done = env.step(actions, lot_sizes)
        equity_curve.append(env.portfolio_value)
        daily_returns.append(total_reward)

        if track_per_ticker:
            for t in env.tickers:
                ticker_cumulative[t] += ticker_rewards[t]

        states = next_states

    result = {
        "total_return":  (equity_curve[-1] - starting_cash) / starting_cash,
        "sharpe":        _sharpe(daily_returns),
        "max_drawdown":  _max_drawdown(equity_curve),
        "equity_curve":  equity_curve,
        "daily_returns": daily_returns,
    }
    if track_per_ticker:
        result["ticker_returns"] = ticker_cumulative
    return result


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
    agent:            QLearningAgent,
    seed:             int   = 42,
    use_confidence:   bool  = False,
    use_vol:          bool  = False,
    eval_capital:     float = STARTING_CAPITAL,
    eval_lot:         int   = LOT_SIZE,
    track_per_ticker: bool  = False,
    use_stop_loss:    bool  = True,
    use_liquidation:  bool  = True,
) -> tuple[pd.DataFrame, dict]:
    random.seed(seed)
    np.random.seed(seed)

    env = load_env("test", use_stop_loss=use_stop_loss, use_liquidation=use_liquidation)
    agent.epsilon = 0.0  # pure greedy

    # --- Q-learning ---
    ql_map = {
        t: (lambda s, t=t: agent.act(s, explore=False,
                                     use_confidence=use_confidence,
                                     use_vol=use_vol,
                                     base_lot=eval_lot))
        for t in env.tickers
    }
    ql = _run_episode(env, ql_map,
                      starting_cash=eval_capital,
                      lot_size=eval_lot,
                      track_per_ticker=track_per_ticker)

    # --- Buy-and-hold: accumulate MAX_LOTS then hold ---
    bh_counts = {t: 0 for t in env.tickers}
    def _bh_fn(ticker):
        def _act(s):
            if bh_counts[ticker] < MAX_LOTS:
                bh_counts[ticker] += 1
                return 2, eval_lot
            return 1, eval_lot
        return _act
    bh_map = {t: _bh_fn(t) for t in env.tickers}
    bh = _run_episode(env, bh_map, starting_cash=eval_capital, lot_size=eval_lot)

    # --- Random ---
    rnd_map = {t: (lambda s: (random.randint(0, N_ACTIONS - 1), eval_lot))
               for t in env.tickers}
    rnd = _run_episode(env, rnd_map, starting_cash=eval_capital, lot_size=eval_lot)

    results = []
    equity_curves: dict = {}
    for name, metrics in [("Q-Learning", ql), ("Buy-and-Hold", bh), ("Random", rnd)]:
        results.append({
            "strategy":     name,
            "total_return": metrics["total_return"],
            "sharpe":       metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
        })
        equity_curves[name] = metrics["equity_curve"]

    if track_per_ticker:
        equity_curves["ticker_returns"] = ql.get("ticker_returns", {})

    return pd.DataFrame(results), equity_curves


# ------------------------------------------------------------------
def print_summary(df: pd.DataFrame):
    print("\n=== Test Period (2025) Portfolio Performance ===\n")
    for _, row in df.iterrows():
        final = STARTING_CAPITAL * (1 + row["total_return"])
        print(f"  {row['strategy']}")
        print(f"    Final value  : ${final:,.0f}  ({row['total_return']*100:+.1f}%)")
        print(f"    Sharpe       : {row['sharpe']:.2f}")
        print(f"    Max drawdown : {row['max_drawdown']*100:.1f}%")
        print()


# ------------------------------------------------------------------
if __name__ == "__main__":
    from rl.agent import Q_TABLE_PATH
    _agent = QLearningAgent()
    _agent.load(Q_TABLE_PATH)
    df, curves = run_backtest(_agent)
    print_summary(df)
    df.to_csv(
        Path(__file__).parent.parent / "data" / "processed" / "experiments" / "backtest_results.csv",
        index=False,
    )
    print("Results saved to data/processed/backtest_results.csv")
