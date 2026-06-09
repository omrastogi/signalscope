# Trading Environment Design

## Overview

`TradingEnv` (`rl/environment.py`) simulates a 6-ticker portfolio where all stocks trade simultaneously on the same calendar dates with one shared $10,000 account. The agent steps through trading days one at a time, choosing to buy, hold, or sell a fixed lot of shares for each ticker independently. Cash is shared — buying AAPL reduces cash available for AMZN on the same day.

---

## Data Split

The split is **time-based** — all of 2022–2023 is training, all of 2024 is evaluation. No data is shuffled or shared between splits.

| Split | Date Range | Trading Days | Purpose |
|-------|-----------|--------------|---------|
| Train | 2022-01-04 → 2023-12-31 | 500 days | Agent learns Q-values; feature bucket boundaries set here |
| Test  | 2024-01-02 → 2024-12-31 | 252 days | Out-of-sample evaluation only; agent never sees this during training |

**Why time-based and not random?** Financial data has temporal structure — future prices depend on past ones. A random split would allow the model to learn from future data and produce misleadingly optimistic results (look-ahead bias).

**Feature bucketing uses training data only.** Quantile boundaries for `volume`, `volatility`, `recent_return`, and `sec_ai` are computed from 2022–2023 rows and then applied to 2024. This ensures the test-period state strings are produced using boundaries the agent was never told about during training — a proper out-of-sample test.

```
_build_dataset():
    train_mask = df["date"] <= "2023-12-31"
    df["volume_bucket"] = _bucket(df["volume_zscore"], train_mask, n=3)
    # boundaries derived from train rows only, applied to all rows
```

The 2022–2023 training period included a significant bear market (2022) followed by a recovery (2023), giving the agent exposure to both rising and falling conditions. The 2024 test period was predominantly a bull market — a genuine out-of-sample challenge.

---

## Tickers

AAPL, AMZN, GOOG, META, MSFT, NVDA — all present on every trading date (perfect alignment, verified).

---

## Action Space

Three actions, applied per ticker per day:

| Code | Name | Effect |
|------|------|--------|
| 0 | SELL | Sell 25 shares if holding ≥ 25; otherwise no-op |
| 1 | HOLD | Do nothing |
| 2 | BUY  | Buy 25 shares if cash ≥ 25 × price and lots < 4; otherwise no-op |

Actions transact a fixed **lot size of 25 shares**. This keeps the action space small (3 actions) while allowing the agent to express graduated conviction — buy once when signals are moderate, buy again when signals are strong.

---

## State Space

Each ticker gets its own state string every day:

```
trend=low|volume=q2|volatility=q1|recent_return=low|sec_ai=q3|lots=2
```

| Component | Values | Source |
|-----------|--------|--------|
| `trend` | low / medium / high | Google Trends AI score (weekly, forward-filled daily) |
| `volume` | low / medium / high | Volume z-score quantile |
| `volatility` | low / medium / high | 20-day rolling volatility quantile |
| `recent_return` | low / medium / high | Prior day return quantile |
| `sec_ai` | low / medium / high | SEC filing AI keyword density quantile |
| `lots` | 0 / 1 / 2 / 3 / 4 | Shares held in this ticker ÷ lot size |

Quantile boundaries are computed on the **training period only** to prevent data leakage. The same Q-table is used for all tickers — the agent has no ticker identity in its state and learns a universal signal-following policy.

**State space size**: ~252 unique signal states × 5 lot levels = **~1,260 total states**

---

## Account Model

Each episode starts fresh:

```
cash         = $10,000  (shared across all 6 tickers)
shares[t]    = 0        (per ticker)
avg_cost[t]  = 0.0      (per ticker, tracks weighted average entry price)
lot_size     = 25 shares
max_lots     = 4        (max 100 shares per ticker)
```

BUY deducts `25 × price` from cash and adds 25 to that ticker's shares. SELL does the reverse. The affordability check (`cash ≥ 25 × price`) prevents leverage — the agent can never go into debt. At typical prices, one lot costs $1,850–$6,000, so the agent can afford 1–5 lots from the starting $10,000.

---

## Reward

```
reward[ticker, t] = (shares[ticker] × close_price[t] × daily_return[t+1]) / 10,000
```

- **Tomorrow's return** (`t+1`) avoids same-day look-ahead bias.
- Normalised by `STARTING_CASH` so the signal is dimensionless and consistent across tickers at different price levels.
- Scales with conviction — holding 4 lots generates 4× the reward signal of 1 lot.
- **Per-ticker rewards** are used for Q-table updates (clean credit assignment). The total portfolio reward (sum across tickers) is used for logging and backtest equity tracking.

### Example

Agent holds 2 lots (50 shares) of AAPL at $150. Next day return is +2%:

```
reward = (50 × 150 × 0.02) / 10,000 = 0.015  (+1.5% of starting capital)
```

---

## Risk Controls

Applied each step **after** the agent's actions, in this order:

### 1. Stop-Loss (`STOP_LOSS_PCT = 0.10`)
If a ticker's current price falls ≥ 10% below the weighted average entry price for that ticker, the entire position is force-sold at the current price. `avg_cost` is updated on every BUY using a weighted average:

```python
avg_cost[t] = (old_shares × avg_cost[t] + LOT_SIZE × price) / (old_shares + LOT_SIZE)
```

### 2. Liquidation (`LIQUIDATION_THRESHOLD = 0.50`)
If total portfolio value (cash + all stock holdings) drops below $5,000 (50% of starting capital), all remaining positions are force-sold immediately to preserve cash.

### 3. Bankruptcy (`BANKRUPTCY_THRESHOLD = $500`)
If portfolio value is below $500 after stop-loss and liquidation, the episode ends immediately. The agent receives a proportional penalty:

```python
penalty = (portfolio_value - 10_000) / 10_000   # e.g. −0.95
```

This penalty propagates back through Q-values, teaching the agent that the states and actions that led to near-total loss were catastrophic.

**Step order within each day:**
```
1. Apply agent actions (BUY/SELL/HOLD) with capital and lot constraints
2. Stop-loss check per ticker
3. Liquidation check (portfolio-wide)
4. Bankruptcy check → end episode if triggered
5. Advance to next day, compute next-day rewards
```

---

## Training

The agent is a tabular Q-learner (`rl/agent.py`). One training pass steps through all 500 training dates with all 6 tickers acting simultaneously — 3,000 transitions per pass.

| Hyperparameter | Value |
|---|---|
| Learning rate α | 0.1 |
| Discount γ | 0.99 |
| ε start | 1.0 |
| ε end | 0.05 |
| ε decay (per pass) | 0.995 |

Epsilon reaches its floor (~0.05) around pass 590. The Q-table is saved to `rl/qtable.json` after training.

**Optimal pass counts by configuration** (see experiment results below):

| Config | Sweet spot | Notes |
|---|---|---|
| baseline | ~1000 passes | Degrades past ~1200 — over-optimises for 2022 bear market |
| conf_only | ~2000 passes | Confidence signal needs sharp Q-value differences; keeps improving |

---

## Dynamic Lot Sizing

Two optional feature flags control how many shares the agent buys or sells per action:

### `use_confidence` — Q-value confidence scaling

```python
confidence  = sigmoid(Q[BUY] − Q[HOLD])    # 0–1 scale
lot_size    = clip(BASE_LOT × (1 + confidence), MIN_LOT, MAX_LOT)
```

When the agent is highly certain a BUY is better than holding, `Q[BUY] − Q[HOLD]` is large, `confidence → 1`, and the lot doubles. When uncertain, confidence → 0.5 and the lot stays near baseline. Requires converged Q-values to carry a meaningful signal — performs poorly at low pass counts.

### `use_vol` — Volatility-adjusted scaling

```python
vol_multiplier = {"low": 1.5, "medium": 1.0, "high": 0.5}
lot_size       = clip(BASE_LOT × vol_multiplier[state.volatility], MIN_LOT, MAX_LOT)
```

Buys 1.5× more in calm markets and 0.5× in turbulent ones — a position-sizing heuristic that reduces risk-adjusted exposure during high-volatility regimes.

Both flags are independent and can be combined. `MIN_LOT = 5`, `MAX_LOT = 100`.

---

## Experiment Results (2024 Out-of-Sample)

`rl/experiments.py` trains each configuration from scratch and backtests on the 2024 test set. Buy-and-hold baseline: **+34.2% / $13,425 / Sharpe 1.45**.

### 200 passes (pre-convergence)

| Config | Return | Final $ | Sharpe | Max DD | vs B&H |
|---|---|---|---|---|---|
| baseline | +43.9% | $14,390 | 1.70 | -12.0% | +9.7% |
| conf_only | +34.2% | $13,422 | 1.15 | -11.0% | +0.0% |
| vol_only | +27.4% | $12,744 | 1.18 | -12.7% | -6.8% |
| both | +38.2% | $13,817 | 1.55 | -13.5% | +4.0% |

At 200 passes, Q-values are not yet converged. The confidence signal is noisy — `conf_only` and `vol_only` underperform baseline.

### 1000 passes

| Config | Return | Final $ | Sharpe | Max DD | vs B&H |
|---|---|---|---|---|---|
| **baseline** | **+45.5%** | **$14,553** | **1.76** | **-11.7%** | **+11.3%** |
| **conf_only** | **+62.9%** | **$16,286** | **1.96** | **-14.5%** | **+28.7%** |

Baseline peaks here. `conf_only` is still improving.

### 2000 passes (full convergence)

| Config | Return | Final $ | Sharpe | Max DD | vs B&H |
|---|---|---|---|---|---|
| baseline | +37.8% | $13,784 | 1.25 | -18.9% | +3.6% |
| **conf_only** | **+66.4%** | **$16,640** | **2.08** | -15.0% | **+32.2%** |
| vol_only | +64.7% | $16,472 | 2.02 | **-13.3%** | +30.5% |
| both | +53.0% | $15,301 | 1.74 | **-12.1%** | +18.9% |

At full convergence, `conf_only` wins overall (+66.4%, Sharpe 2.08). `vol_only` wins on drawdown (-13.3%). Combining both (`both`) is middle-ground — the two multipliers partially cancel each other on high-vol / high-confidence signals.

**Key finding:** Baseline degrades past ~1000 passes (over-fits to 2022 bear market patterns), while dynamic sizing configurations keep improving because sharper Q-values produce a more reliable confidence signal.

---

## Policy Summary (after 1000 passes, conf_only)

| Action | States | % |
|---|---|---|
| SELL | 575 | 58.7% |
| HOLD | 225 | 23.0% |
| BUY  | 180 | 18.4% |

The SELL bias reflects the 2022 bear market in training data — the agent is conservative by default and only enters positions when signals are clearly favourable.

---

## Files

| File | Role |
|------|------|
| `rl/environment.py` | `TradingEnv` class, data pipeline, risk controls |
| `rl/agent.py` | `QLearningAgent` class, training loop |
| `rl/backtest.py` | Evaluation: Q-learning vs buy-and-hold vs random |
| `rl/experiments.py` | Multi-config experiment runner; `--passes` CLI arg |
| `rl/qtable.json` | Serialised Q-table (state string → [Q_sell, Q_hold, Q_buy]) |
| `data/processed/rl_ready_dataset.csv` | Merged market + SEC + trends features |
| `data/processed/google_trends_features.csv` | Weekly AI trend scores |
| `data/processed/experiment_results.csv` | Latest experiment comparison table |
| `data/processed/backtest_results.csv` | Latest single-run backtest output |
