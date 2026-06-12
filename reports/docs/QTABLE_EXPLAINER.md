# SignalScope RL: Q-Table, Actions, and Reward

## The Q-Table

The Q-table maps every (state, action) pair to an expected cumulative reward.

```
Q-table: { state_string -> [Q(s, FLAT), Q(s, LONG)] }
```

**Example entries from a trained table:**

| State | Q(FLAT) | Q(LONG) | Greedy action |
|---|---|---|---|
| `trend=missing\|volume=high\|volatility=high\|recent_return=high\|sec_ai=low` | 0.838 | 0.909 | **LONG** |
| `trend=missing\|volume=medium\|volatility=nan\|recent_return=medium\|sec_ai=nan` | 0.723 | 0.610 | **FLAT** |
| `trend=missing\|volume=medium\|volatility=low\|recent_return=low\|sec_ai=medium` | 0.250 | 0.185 | **FLAT** |

- 108 unique states are observed in the training data (2022–2023)
- Higher Q value = "the agent expects more cumulative return by taking this action in this state"
- Stored as `rl/qtable.json`

---

## State Encoding

Each state is a `|`-separated string of bucketed signal features:

```
trend=<bucket> | volume=<bucket> | volatility=<bucket> | recent_return=<bucket> | sec_ai=<bucket>
```

| Feature | Source | Buckets |
|---|---|---|
| `trend` | Google Trends score (weekly, forward-filled to daily) | `low / medium / high` |
| `volume` | Volume z-score vs 20-day avg | `low / medium / high / nan` |
| `volatility` | 20-day rolling return std | `low / medium / high / nan` |
| `recent_return` | Yesterday's daily return | `low / medium / high` |
| `sec_ai` | AI keyword density in latest 10-K filing | `low / medium / high / nan` |

> Google Trends is collected weekly. The original pipeline did an exact-date join, so all daily rows got `NaN` → `missing`.
> The environment now uses `merge_asof` (backward direction) to forward-fill the most recent weekly reading onto each trading day.
> This expands the observable state space from 108 → 204 unique states.

Bucket boundaries are computed on the **training set only** (pre-2024) and then applied to the test set, preventing any look-ahead bias from the bucketing step.

---

## Actions

The agent chooses one of two actions each trading day:

| Action | Name | Meaning |
|---|---|---|
| `0` | FLAT | Hold cash — no market exposure |
| `1` | LONG | Hold the stock — fully invested |

There is no shorting or partial sizing in this version.

---

## Reward

```
reward[t] = action[t] * daily_return[t+1]
```

- `daily_return[t+1]` = `(close[t+1] - close[t]) / close[t]` — tomorrow's price change
- If `action=LONG`: reward = tomorrow's actual stock return (positive if up, negative if down)
- If `action=FLAT`: reward = 0 (no market exposure, no gain or loss)

**Why t+1?** The agent observes state[t] — signals available at the close of day t (including today's return, volume, volatility). It then chooses a position to hold *starting tomorrow*. The reward is therefore tomorrow's return, not today's. Using today's return as the reward would let the agent see the answer before deciding, which inflates performance dramatically (in testing, this leakage produced ~470% returns vs ~60% for buy-and-hold — a clear red flag that triggered the fix).

---

## Q-Learning Update Rule

At each step the agent observes (state, action, reward, next_state) and updates:

```
Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
```

| Hyperparameter | Value | Role |
|---|---|---|
| `α` (learning rate) | 0.1 | How fast Q-values update |
| `γ` (discount) | 0.99 | How much future rewards matter |
| `ε` start | 1.0 | Full exploration at start |
| `ε` end | 0.05 | 5% random exploration at convergence |
| `ε` decay | 0.995 per pass | Gradual shift to exploitation |

---

## Realistic Expectations

The EDA showed weak correlations (R ≈ 0.02–0.05) between the available signals and forward returns. The trained agent reflects this:

| Strategy | Avg Return (2024 test) | Avg Sharpe |
|---|---|---|
| Q-Learning | ~56% | 1.53 |
| Buy-and-Hold | ~63% | 1.47 |
| Random | ~41% | 1.30 |

Q-learning beats random but slightly trails buy-and-hold on raw returns, while offering a marginally better Sharpe.

**Fixed**: Google Trends was always `missing` due to a weekly→daily join bug in the original pipeline. `environment.py` now uses `merge_asof` forward-fill, expanding the state space from 108 → 204 unique states and giving the agent access to the AI-attention signal the project is built around.
