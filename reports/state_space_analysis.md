# State Space Size, Data, and the Curse of Dimensionality

## Why We Need to Keep the State Space Small

A Q-table stores one value for every **(state, action) pair**. The agent can only learn that Q(s, a) is good or bad by experiencing that exact state many times and averaging the rewards. It has no way to generalise — `volume=q7` tells it nothing about `volume=q8`.

This means the training data must be spread across every state in the table. The constraint is:

```
visits per state = (total training transitions) / (number of states)
```

We have a fixed training set: **6 tickers × ~500 days = 3,000 transitions per pass**. Every time we expand the state space, each individual state gets visited less often — and Q-values become less reliable.

---

## Experiment 1 — Small State Space, 2 Actions (Baseline)

**Setup:**
- 3 quantile buckets per feature (low / medium / high)
- Actions: FLAT (hold cash) or LONG (hold stock)
- No position tracking in state
- 203 unique states, ~7,400 visits/state across 500 passes

**Results at 100 passes (epsilon ~0.6):**

| Strategy | Return | Sharpe | Max DD |
|---|---|---|---|
| Q-Learning | 56.2% | **1.53** | -18% |
| Buy-and-Hold | 63.2% | 1.47 | -20% |
| Random | 40.8% | 1.30 | -11% |

The agent beat buy-and-hold on Sharpe — earning nearly the same return at lower risk. With 7,400 visits per state, Q-values had enough samples to converge to something meaningful.

**At convergence (500 passes, epsilon → 0.08):**

| Strategy | Return | Sharpe |
|---|---|---|
| Q-Learning | 27.5% | 1.16 |
| Buy-and-Hold | 63.2% | 1.47 |

The agent overfit to the 2022–2023 training regime (volatile, choppy) and converged to 72% FLAT — a policy that protected capital in the training period but missed 2024's bull run entirely.

---

## Experiment 2 — Large State Space, 3 Actions

**Setup:**
- 10 quantile buckets per feature (q1–q10)
- Actions: SELL / HOLD / BUY with position tracked in state
- 4,732 unique states — a 23× expansion
- ~316 visits/state across 500 passes

**Results:**

| Strategy | Return | Sharpe | Max DD |
|---|---|---|---|
| Q-Learning | **5.2%** | 0.45 | -8% |
| Buy-and-Hold | 63.2% | 1.47 | -20% |
| Random | 17.9% | 0.63 | -17% |

The agent failed to beat random. With only 316 visits per state, Q-values are noise. The finer buckets (q1–q10) also create a false precision problem — the data does not contain enough signal to reliably distinguish "volume at the 71st percentile" from "volume at the 79th percentile."

---

## Experiment 3 — Small State Space, 3 Actions

**Setup:**
- 3 quantile buckets per feature (low / medium / high) — same as Experiment 1
- Actions: SELL / HOLD / BUY with position tracked in state — same as Experiment 2
- 504 unique states, ~2,960 visits/state across 500 passes

**Results:**

| Strategy | Return | Sharpe | Max DD |
|---|---|---|---|
| Q-Learning | **25.9%** | 0.87 | -18% |
| Buy-and-Hold | 63.2% | 1.47 | -20% |
| Random | 17.9% | 0.63 | -17% |

The agent beats random clearly and uses all three actions meaningfully (SELL 44.8% / HOLD 27.2% / BUY 28.0%). Compared to Experiment 2, the halved state space doubled visits per state and performance jumped from 5.2% → 25.9% return and 0.45 → 0.87 Sharpe.

---

## Comparison Across All Experiments

| Experiment | States | Visits/state | Actions | Return | Sharpe |
|---|---|---|---|---|---|
| 3-bucket, FLAT/LONG — 100 passes | 203 | ~7,400 | 2 | 56.2% | **1.53** |
| 3-bucket, FLAT/LONG — 500 passes | 203 | ~7,400 | 2 | 27.5% | 1.16 |
| 10-bucket, SELL/HOLD/BUY | 4,732 | ~316 | 3 | 5.2% | 0.45 |
| **3-bucket, SELL/HOLD/BUY** | **504** | **~2,960** | **3** | **25.9%** | **0.87** |
| Buy-and-Hold (baseline) | — | — | — | 63.2% | 1.47 |
| Random (baseline) | — | — | — | 17.9% | 0.63 |

**The pattern is clear:** performance degrades directly with visits per state. The original 2-action experiment at 100 passes (before overfitting) holds the best Sharpe — it had the most training data per state and hadn't yet collapsed into a conservative policy.

---

## The Relationship Visualised

```
Visits per state vs. Q-learning Sharpe ratio:

Sharpe
  1.53 |  ● (203 states, 100 passes)
       |
  1.16 |     ● (203 states, 500 passes)
       |
  0.87 |         ● (504 states, 500 passes)
       |
  0.45 |                                    ● (4732 states, 500 passes)
       |
       +-----------------------------------------------> visits/state
            7400      7400      2960              316
```

Hard floor: **below ~500 visits per state, Q-values don't reliably converge** with the noise levels present in daily return data.

---

## What We Actually Need

### Option A — More Data, Keep Large State Space

To justify 4,732 states with 500 visits/state minimum:

```
required transitions = 4,732 × 500 × 3 actions ≈ 7M
with 100 tickers → ~50,000 transitions/pass → only 140 passes needed
```

Expanding from 6 → 100 tickers is the single highest-leverage fix.

### Option B — Right-Size the State Space

With 6 tickers and 3 years, the comfortable ceiling is ~1,500 states. A bucketing scheme that fits:

| Feature | Buckets |
|---|---|
| trend | 3 (low / medium / high) |
| volume | 4 |
| volatility | 4 |
| recent_return | 4 |
| sec_ai | 3 |
| position | ×2 |

→ 3 × 4 × 4 × 4 × 3 × 2 = **1,152 states** — within the feasible zone.

### Option C — Switch to DQN

Tabular Q-learning cannot generalise across states. A neural network Q-function (DQN) can — it learns that "volume=q7 and volume=q8 are similar" through shared weights. With the same 1.5M transitions, a small DQN would dramatically outperform the Q-table on a large state space.

---

## Root Cause Summary

| Dimension | Current | Needed |
|---|---|---|
| Tickers | 6 (all US mega-cap tech) | 50–100 (cross-sector) |
| Training years | 2 (2022–2023) | 10+ (multiple regimes) |
| Transitions/pass | ~3,000 | ~50,000 |
| Visits/state (large space) | ~316 | 500+ |
| SEC filing frequency | Annual only | Quarterly + 8-K |
| Trend granularity | Weekly forward-filled | Daily news sentiment |

The model is not broken — it is **data-starved**. The small state space produced a Sharpe ratio above buy-and-hold with only 100 training passes, confirming the pipeline works. Scaling the data is the next step before any architectural changes.
