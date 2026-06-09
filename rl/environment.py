"""
Portfolio trading environment for SignalScope RL agent.

All 6 tickers run simultaneously on the same calendar dates with one
shared $10,000 account. Each day the agent takes one action per ticker
(BUY/HOLD/SELL a fixed lot of 25 shares). Cash is shared — buying AAPL
reduces cash available for AMZN.

State per ticker: signal buckets + current lot count for that ticker.
Reward per ticker: (shares × price × daily_return[t+1]) / STARTING_CASH
  — dollar P&L from held position normalised by starting capital.
  — tomorrow's return used to avoid same-day look-ahead bias.

Risk controls (applied each step after agent actions):
  Stop-loss   : auto-sell all lots of a ticker when price falls ≥ STOP_LOSS_PCT
                below the average entry price for that ticker.
  Liquidation : if total portfolio value drops below LIQUIDATION_THRESHOLD × STARTING_CASH,
                force-sell ALL positions to cash immediately.
  Bankruptcy  : if portfolio value is below BANKRUPTCY_THRESHOLD after all defensive
                actions, the episode ends with a proportional penalty reward.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED   = Path(__file__).parent.parent / "data" / "processed"
RL_PATH     = PROCESSED / "rl_ready_dataset.csv"
TRENDS_PATH = PROCESSED / "google_trends_features.csv"
TRAIN_END   = "2024-12-31"

ACTIONS       = {0: "SELL", 1: "HOLD", 2: "BUY"}
N_ACTIONS     = 3
LOT_SIZE      = 25        # shares transacted per BUY or SELL action
MAX_LOTS      = 4         # max lots held per ticker (100 shares)
STARTING_CASH = 10_000.0  # shared account balance at episode start

# Risk controls
STOP_LOSS_PCT         = 0.10   # auto-sell ticker if price < avg_entry × (1 − 0.10)
LIQUIDATION_THRESHOLD = 0.50   # force-sell all positions if portfolio < 50 % of start
BANKRUPTCY_THRESHOLD  = 500.0  # end episode if portfolio < $500 after defensive actions

# Dynamic lot sizing bounds (used when lot_sizes are passed to step())
MIN_LOT = 5
MAX_LOT = 100


def _bucket(series: pd.Series, train_mask: pd.Series, n: int = 3) -> pd.Series:
    """Quantile-bucket a series using only training rows to set boundaries."""
    labels = [f"q{i}" for i in range(1, n + 1)] if n > 3 else ["low", "medium", "high"][:n]
    boundaries = series[train_mask].quantile(
        [i / n for i in range(1, n)], interpolation="midpoint"
    ).tolist()
    unique_bounds = sorted(set(boundaries))
    if len(unique_bounds) < n - 1:
        boundaries = unique_bounds
        labels = labels[: len(boundaries) + 1]
    return pd.cut(series, bins=[-np.inf] + boundaries + [np.inf], labels=labels)


def _build_dataset() -> pd.DataFrame:
    """
    Load market + trends data and produce a clean daily feature DataFrame.
    Trend values are forward-filled from weekly to daily with merge_asof.
    Bucket boundaries are computed on the training period only (no leakage).
    Dates where any ticker is missing daily_return or close are dropped so
    all tickers are perfectly aligned.
    """
    rl = pd.read_csv(RL_PATH, parse_dates=["date"])
    trends = pd.read_csv(TRENDS_PATH, parse_dates=["date"])

    trends = trends[["date", "ai_trend_score", "ai_trend_bucket"]].sort_values("date")

    tickers = rl["ticker"].unique()
    frames = []
    for ticker in tickers:
        sub = rl[rl["ticker"] == ticker].sort_values("date").copy()
        sub = pd.merge_asof(sub, trends, on="date", direction="backward",
                            suffixes=("_old", ""))
        sub = sub.drop(columns=[c for c in sub.columns if c.endswith("_old")], errors="ignore")
        frames.append(sub)

    df = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])

    # Drop dates where ANY ticker is missing daily_return or close (only day 1)
    valid_dates = set.intersection(*[
        set(sub.dropna(subset=["daily_return", "close"])["date"])
        for _, sub in df.groupby("ticker")
    ])
    df = df[df["date"].isin(valid_dates)].copy()

    train_mask = df["date"] <= TRAIN_END

    df["trend_bucket"]         = _bucket(df["ai_trend_score"].fillna(0),  train_mask, n=3)
    df["volume_bucket"]        = _bucket(df["volume_zscore"],            train_mask, n=3)
    df["volatility_bucket"]    = _bucket(df["rolling_volatility_20d"],   train_mask, n=3)
    df["recent_return_bucket"] = _bucket(df["daily_return"],             train_mask, n=3)
    df["sec_ai_bucket"]        = _bucket(df["sec_ai_density"],           train_mask, n=3)

    def make_signal_state(row):
        return (
            f"trend={row['trend_bucket']}"
            f"|volume={row['volume_bucket']}"
            f"|volatility={row['volatility_bucket']}"
            f"|recent_return={row['recent_return_bucket']}"
            f"|sec_ai={row['sec_ai_bucket']}"
        )

    df["signal_state"] = df.apply(make_signal_state, axis=1)
    return df


def load_env(split: str = "train") -> "TradingEnv":
    df = _build_dataset()
    if split == "train":
        df = df[df["date"] <= TRAIN_END]
    else:
        df = df[df["date"] > TRAIN_END]
    return TradingEnv(df)


# ------------------------------------------------------------------
class TradingEnv:
    """
    Portfolio environment: all tickers trade simultaneously each day.

    reset() → dict[ticker, state_str]
    step(actions: dict[ticker, int]) → (next_states, ticker_rewards, total_reward, done)
    """

    def __init__(self, data: pd.DataFrame):
        self.tickers = sorted(data["ticker"].unique())
        self.dates   = sorted(data["date"].unique())

        # Per-ticker DataFrames indexed by date for O(1) lookup
        self.data_by_ticker: dict[str, pd.DataFrame] = {
            t: data[data["ticker"] == t].set_index("date")
            for t in self.tickers
        }

        self.starting_cash = STARTING_CASH
        self.lot_size      = LOT_SIZE
        self.cash          = STARTING_CASH
        self.shares:    dict[str, int]   = {t: 0   for t in self.tickers}
        self.avg_cost:  dict[str, float] = {t: 0.0 for t in self.tickers}
        self._step_idx = 0

    # ------------------------------------------------------------------
    def _make_state(self, ticker: str) -> str:
        date = self.dates[self._step_idx]
        sig  = self.data_by_ticker[ticker].loc[date, "signal_state"]
        lots = self.shares[ticker] // self.lot_size
        return f"{sig}|lots={lots}"

    @property
    def portfolio_value(self) -> float:
        idx  = min(self._step_idx, len(self.dates) - 1)
        date = self.dates[idx]
        stock_val = sum(
            self.shares[t] * float(self.data_by_ticker[t].loc[date, "close"])
            for t in self.tickers
        )
        return self.cash + stock_val

    # ------------------------------------------------------------------
    def reset(
        self,
        starting_cash: float | None = None,
        lot_size:      int   | None = None,
    ) -> dict[str, str]:
        self.starting_cash = starting_cash if starting_cash is not None else STARTING_CASH
        self.lot_size      = lot_size      if lot_size      is not None else LOT_SIZE
        self.cash          = self.starting_cash
        self.shares        = {t: 0   for t in self.tickers}
        self.avg_cost      = {t: 0.0 for t in self.tickers}
        self._step_idx     = 0
        return {t: self._make_state(t) for t in self.tickers}

    def step(
        self,
        actions:   dict[str, int],
        lot_sizes: dict[str, int] | None = None,
    ) -> tuple[dict[str, str], dict[str, float], float, bool]:
        """
        lot_sizes: optional per-ticker lot size (shares per transaction).
                   Falls back to LOT_SIZE when None.
        """
        date   = self.dates[self._step_idx]
        prices = {t: float(self.data_by_ticker[t].loc[date, "close"]) for t in self.tickers}

        # 1. Apply agent actions (BUY/SELL/HOLD) with capital and lot constraints
        for ticker in self.tickers:
            action = actions[ticker]
            price  = prices[ticker]
            lot    = lot_sizes[ticker] if lot_sizes else self.lot_size

            if action == 2:  # BUY
                max_shares  = MAX_LOTS * self.lot_size     # hard cap scales with lot
                can_buy     = max(0, max_shares - self.shares[ticker])
                actual_lot  = min(lot, can_buy)
                cost        = actual_lot * price
                if actual_lot > 0 and self.cash >= cost:
                    old = self.shares[ticker]
                    self.avg_cost[ticker] = (
                        (old * self.avg_cost[ticker] + actual_lot * price)
                        / (old + actual_lot)
                    )
                    self.shares[ticker] += actual_lot
                    self.cash           -= cost
            elif action == 0:  # SELL
                actual_lot = min(lot, self.shares[ticker])
                if actual_lot > 0:
                    self.shares[ticker] -= actual_lot
                    self.cash           += actual_lot * price
                    if self.shares[ticker] == 0:
                        self.avg_cost[ticker] = 0.0
            # HOLD (1): no change

        # 2. Stop-loss: auto-sell a ticker's entire position if price fell ≥ 10 %
        #    below the average entry price
        for ticker in self.tickers:
            if self.shares[ticker] > 0 and self.avg_cost[ticker] > 0:
                if prices[ticker] < self.avg_cost[ticker] * (1.0 - STOP_LOSS_PCT):
                    self.cash           += self.shares[ticker] * prices[ticker]
                    self.shares[ticker]  = 0
                    self.avg_cost[ticker] = 0.0

        # 3. Liquidation: if total portfolio value < 50 % of starting capital,
        #    force-sell everything to preserve remaining cash
        pv = self.cash + sum(self.shares[t] * prices[t] for t in self.tickers)
        if pv < self.starting_cash * LIQUIDATION_THRESHOLD:
            for ticker in self.tickers:
                if self.shares[ticker] > 0:
                    self.cash            += self.shares[ticker] * prices[ticker]
                    self.shares[ticker]   = 0
                    self.avg_cost[ticker] = 0.0

        # 4. Bankruptcy: if portfolio is still below threshold, end the episode
        pv = self.cash + sum(self.shares[t] * prices[t] for t in self.tickers)
        if pv < BANKRUPTCY_THRESHOLD:
            penalty      = (pv - self.starting_cash) / self.starting_cash   # e.g. −0.95
            per_ticker   = penalty / len(self.tickers)
            last_states  = {t: self._make_state_at(t, date) for t in self.tickers}
            return last_states, {t: per_ticker for t in self.tickers}, penalty, True

        # 5. Advance to next day and compute rewards
        self._step_idx += 1
        done = self._step_idx >= len(self.dates)

        if done:
            last_states = {t: self._make_state_at(t, date) for t in self.tickers}
            return last_states, {t: 0.0 for t in self.tickers}, 0.0, True

        next_date = self.dates[self._step_idx]
        ticker_rewards: dict[str, float] = {}
        total_reward = 0.0

        for ticker in self.tickers:
            daily_return = float(self.data_by_ticker[ticker].loc[next_date, "daily_return"])
            r = (self.shares[ticker] * prices[ticker] * daily_return) / self.starting_cash
            ticker_rewards[ticker] = r
            total_reward          += r

        next_states = {t: self._make_state(t) for t in self.tickers}
        return next_states, ticker_rewards, total_reward, done

    def _make_state_at(self, ticker: str, date) -> str:
        """State string for a specific date (used at terminal step)."""
        sig  = self.data_by_ticker[ticker].loc[date, "signal_state"]
        lots = self.shares[ticker] // self.lot_size
        return f"{sig}|lots={lots}"

    # ------------------------------------------------------------------
    @property
    def n_dates(self) -> int:
        return len(self.dates)


# ------------------------------------------------------------------
if __name__ == "__main__":
    import random

    env = load_env("train")
    print(f"Tickers : {env.tickers}")
    print(f"Train dates: {env.n_dates}")

    df = _build_dataset()
    train_df = df[df["date"] <= TRAIN_END]
    n_sig = train_df["signal_state"].nunique()
    print(f"Unique signal states (train): {n_sig}")
    print(f"Unique full states with lots (×{MAX_LOTS+1}): {n_sig * (MAX_LOTS + 1)}")
    print()

    # Run one pass with random policy
    states = env.reset()
    done   = False
    total_reward = 0.0
    steps = 0
    while not done:
        actions = {t: random.randint(0, N_ACTIONS - 1) for t in env.tickers}
        states, ticker_rewards, reward, done = env.step(actions)
        total_reward += reward
        steps += 1

    print(f"Steps: {steps}  total_reward={total_reward:.4f}")
    print(f"Final cash: {env.cash:.0f}")
    for t in env.tickers:
        print(f"  {t}: {env.shares[t]} shares")
    print(f"Portfolio value: {env.portfolio_value:.0f}")
