"""
Trading environment for SignalScope RL agent.

3-action formulation: SELL (0), HOLD (1), BUY (2).
The agent tracks its current position (0=cash, 1=stock) and chooses a
transition each day. Position is appended to the state string so the agent
can learn different policies depending on whether it is currently invested.

Reward = position_after_action * daily_return[t+1] — tomorrow's return
is used (not today's) to avoid same-day look-ahead bias.

Google Trends (weekly) are forward-filled onto daily market rows via
merge_asof to fix the broken exact-date join in the original pipeline.
Volume, volatility, recent_return, sec_ai are bucketed into 10 quantiles
(q1–q10); trend stays at 3 buckets (low/medium/high).
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED   = Path(__file__).parent.parent / "data" / "processed"
RL_PATH     = PROCESSED / "rl_ready_dataset.csv"
TRENDS_PATH = PROCESSED / "google_trends_features.csv"
TRAIN_END   = "2023-12-31"

ACTIONS   = {0: "SELL", 1: "HOLD", 2: "BUY"}
N_ACTIONS = 3


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
    train_mask = df["date"] <= TRAIN_END

    df["trend_bucket"]         = _bucket(df["ai_trend_score"],          train_mask, n=3)
    df["volume_bucket"]        = _bucket(df["volume_zscore"],            train_mask, n=3)
    df["volatility_bucket"]    = _bucket(df["rolling_volatility_20d"],   train_mask, n=3)
    df["recent_return_bucket"] = _bucket(df["daily_return"],             train_mask, n=3)
    df["sec_ai_bucket"]        = _bucket(df["sec_ai_density"],           train_mask, n=3)

    # state string WITHOUT position — position is appended at runtime by the env
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
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self._episodes: list[pd.DataFrame] = []
        self._ep_idx = 0
        self._position = 0  # 0=cash, 1=stock

        for _, grp in data.groupby("ticker", sort=True):
            grp = grp.sort_values("date").reset_index(drop=True)
            grp = grp.dropna(subset=["daily_return", "signal_state"]).reset_index(drop=True)
            if len(grp) > 1:
                self._episodes.append(grp)

        self._step_idx = 0
        self._current_ep: pd.DataFrame = pd.DataFrame()

    def _make_state(self, signal_state: str) -> str:
        return f"{signal_state}|position={self._position}"

    # ------------------------------------------------------------------
    def reset(self, episode_idx: int | None = None) -> str:
        if episode_idx is not None:
            self._ep_idx = episode_idx % len(self._episodes)
        else:
            self._ep_idx = (self._ep_idx + 1) % len(self._episodes)
        self._current_ep = self._episodes[self._ep_idx]
        self._step_idx = 0
        self._position = 0  # always start in cash
        return self._make_state(self._current_ep.iloc[0]["signal_state"])

    def step(self, action: int) -> tuple[str, float, bool]:
        # Apply action to update position
        if action == 2:    # BUY
            self._position = 1
        elif action == 0:  # SELL
            self._position = 0
        # HOLD (1) leaves position unchanged

        # Advance to next day for the reward
        self._step_idx += 1
        done = self._step_idx >= len(self._current_ep)

        if done:
            last_sig = self._current_ep.iloc[-1]["signal_state"]
            return self._make_state(last_sig), 0.0, True

        next_row = self._current_ep.iloc[self._step_idx]
        reward = self._position * float(next_row["daily_return"])
        next_state = self._make_state(next_row["signal_state"])
        return next_state, reward, done

    # ------------------------------------------------------------------
    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    def episode_tickers(self) -> list[str]:
        return [ep["ticker"].iloc[0] for ep in self._episodes]


# ------------------------------------------------------------------
if __name__ == "__main__":
    import random

    env = load_env("train")
    print(f"Train episodes: {env.n_episodes}  tickers: {env.episode_tickers()}")

    df = _build_dataset()
    train_df = df[df["date"] <= TRAIN_END]
    print(f"Unique signal states (train, no position): {train_df['signal_state'].nunique()}")
    print(f"Unique full states with position:          {train_df['signal_state'].nunique() * 2}")
    print()

    total_reward = 0.0
    for ep in range(env.n_episodes):
        state = env.reset(ep)
        done = False
        ep_reward, steps = 0.0, 0
        while not done:
            action = random.randint(0, N_ACTIONS - 1)
            state, reward, done = env.step(action)
            ep_reward += reward
            steps += 1
        print(f"  Episode {ep} ({env.episode_tickers()[ep]}): {steps} steps, reward={ep_reward:.4f}")
        total_reward += ep_reward

    print(f"\nTotal reward (random policy, train): {total_reward:.4f}")
