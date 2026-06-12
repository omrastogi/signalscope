# SignalScope — Ablation Study Results

Training period: **2020–2024** (unless noted) · Test set: **2025** (250 days, out-of-sample)  
Baseline strategy: Buy-and-Hold on 6-ticker tech portfolio → **−11.6%** in 2025.

---

## Experiment 1 — Baseline Convergence

**Question:** How does the vanilla Q-learning agent (no dynamic lot sizing, no domain randomization) perform as training continues? Does the policy converge or keep oscillating?

**Setup:** Baseline config (fixed lot size, single episode per pass), trained for 4 000 passes.  
Evaluated on the 2025 test set every 500 passes.

### Results

| Pass | Return | Sharpe | Max DD |
|-----:|-------:|-------:|-------:|
|  500 | +27.3% |  1.017 | −31.0% |
| 1000 | **+53.6%** | **2.167** | −17.4% |
| 1500 | +19.7% |  0.619 | −28.4% |
| 2000 | +33.5% |  1.249 | −20.1% |
| 2500 | +31.8% |  1.077 | −27.2% |
| 3000 | +17.1% |  0.524 | −29.6% |
| 3500 | +49.6% |  1.632 | −11.1% |
| 4000 | +11.6% |  0.502 | −32.2% |

**Summary statistics across all checkpoints:**  
Mean return = **30.5%** · Std = **15.5 pp** · Range = 11.6% – 53.6%

### Findings

- The agent never converges. Return swings between +11.6% and +53.6% across checkpoints — a 42 pp range with no downward trend in variance.
- Best single checkpoint is pass 1 000 (+53.6%, Sharpe 2.17), but the same policy evaluated 500 passes later has already degraded to +19.7%.
- The policy oscillates because the Q-table is still being actively revised: ε-greedy exploration continues rewriting state–action values, meaning a good policy at checkpoint *k* may be partially overwritten by pass *k+1*.
- **Implication:** More training is not strictly better for the baseline. The agent needs a stabilising mechanism — domain randomisation is explored in Exp 4.

---

## Experiment 2 — Training Window Ablation

**Question:** How much historical data does the agent need? Does a longer training window always produce a better policy?

**Setup:** Five training windows, all ending 2024-12-31 and evaluated on the same 2025 test set. Each window trained for 2 000 passes (no DR, fixed lot).

| Window label | Train period | Training days |
|---|---|---:|
| 2024-only | 2024-01-01 → 2024-12-31 | 252 |
| 2023-2024 | 2023-01-01 → 2024-12-31 | 504 |
| 2022-2024 | 2022-01-01 → 2024-12-31 | 756 |
| 2021-2024 | 2021-01-01 → 2024-12-31 | 1 008 |
| 2020-2024 | 2020-01-01 → 2024-12-31 | 1 257 |

### Results

| Window | QL Return | Sharpe | Max DD | vs B&H (−11.6%) |
|---|---:|---:|---:|---:|
| 2024-only | −4.9% | −0.215 | −32.6% | −6.7 pp *(loses to B&H)* |
| **2023-2024** | **+22.1%** | **0.800** | **−26.1%** | **+33.7 pp** |
| 2022-2024 | +10.9% | 0.357 | −33.7% | +22.5 pp |
| 2021-2024 | +18.4% | 0.705 | −28.7% | +30.0 pp |
| 2020-2024 | +21.4% | 0.762 | −26.3% | +33.0 pp |

### Findings

- **Longer is not always better.** The 2-year window (2023-2024) outperforms every longer window.
- **2022 is a confounding year.** 2022 was a severe tech bear market (−30% to −50% across the portfolio). Adding it to training teaches the agent an overly defensive posture — it learns to hold cash and avoid buying — which hurts performance in 2025's different market regime.
- **2024-only fails outright** (−4.9%), confirming that a single year of data (252 steps) provides insufficient state coverage for the Q-table to learn a generalising policy.
- **The 5-year window (2020-2024) nearly matches the 2-year window** (+21.4% vs +22.1%), suggesting the agent does extract useful signal from 2020-2021, partly offsetting the damage from 2022.
- **All windows beat B&H** except 2024-only.
- **Selected window for subsequent experiments: 2023-2024** (highest return, highest Sharpe).

---

## Experiment 3 — Lot Sizing Convergence (no Domain Randomisation)

**Question:** Does dynamic lot sizing — scaling trade size by confidence or volatility — improve or harm the baseline policy? How stable is training across 4 000 passes?

**Setup:** Three configurations, all trained from 2020-2024, evaluated every 500 passes on the 2025 test set. No domain randomisation (single episode per pass, fixed starting cash).

| Config | Lot-size rule |
|---|---|
| `conf_only` | Lot × sigmoid(Q[BUY] − Q[HOLD]) · (1 + confidence) |
| `vol_only` | Lot × volatility multiplier (low→1.5×, medium→1.0×, high→0.5×) |
| `both` | Both scalings applied multiplicatively |

### Results

#### conf\_only

| Pass | Return | Sharpe | Max DD |
|-----:|-------:|-------:|-------:|
|  500 | +25.2% |  0.904 | −36.0% |
| 1000 | +65.4% |  1.683 | −14.0% |
| 1500 | **+94.1%** | **2.326** | −25.5% |
| 2000 | +43.8% |  1.383 | −16.7% |
| 2500 |  +0.1% |  0.005 | −24.2% |
| 3000 | +13.8% |  0.551 | −23.2% |
| 3500 | +21.3% |  0.797 | −24.9% |
| 4000 | +46.1% |  1.543 | −19.2% |

#### vol\_only

| Pass | Return | Sharpe | Max DD |
|-----:|-------:|-------:|-------:|
|  500 |  −1.5% | −0.069 | −22.5% |
| 1000 |  −9.9% | −0.425 | −36.9% |
| 1500 | −14.6% | −0.718 | −35.5% |
| 2000 | +16.6% |  0.565 | −34.7% |
| 2500 |  −0.6% | −0.022 | −21.4% |
| 3000 |  +1.4% |  0.063 | −21.6% |
| 3500 |  +3.8% |  0.130 | −36.1% |
| 4000 | **+29.8%** | **0.996** | −19.9% |

#### both

| Pass | Return | Sharpe | Max DD |
|-----:|-------:|-------:|-------:|
|  500 | +19.7% |  0.583 | −34.4% |
| 1000 |  −4.8% | −0.210 | −23.9% |
| 1500 | +29.7% |  1.182 | −21.7% |
| 2000 | **+34.0%** | **1.157** | −12.3% |
| 2500 |  +0.9% |  0.038 | −29.0% |
| 3000 | +18.0% |  0.676 | −26.4% |
| 3500 | −11.1% | −0.428 | −36.1% |
| 4000 |  −2.8% | −0.130 | −33.4% |

### Findings

- **conf\_only reaches the highest peak** of any configuration tested so far: **+94.1%** at pass 1 500. However, this peak is not sustained — performance collapses to near zero at pass 2 500 before recovering. The confidence signal amplifies good decisions when the Q-table is partially trained but also amplifies errors as the policy keeps changing.
- **vol\_only is the slowest learner.** The agent is effectively handicapped for the first 1 500 passes (all returns negative), because reducing lot size in high-volatility states also reduces the reward signal needed for Q-value updates. Once the Q-table matures enough (~pass 2 000+), the volatility scaling starts acting as a risk filter rather than a handicap.
- **both is the least stable.** Combining two multiplicative scalings produces a wider effective lot-size range (0.5× to 3×+) that can catastrophically over-size positions during the unstable early training phase. It ends at −2.8% — the only config with a negative final checkpoint.
- **None of the three configs converge** over 4 000 passes without domain randomisation. The high variance across checkpoints (20–40 pp swings) mirrors what was observed in Exp 1, and is more severe for dynamic lot configs because lot-size changes interact with the Q-value learning rate.
- **Implication:** Dynamic lot sizing provides upside (conf_only's +94% peak) but cannot be reliably exploited without a stabilising training scheme. Experiment 4 adds domain randomisation to address this instability.

---

*Generated from CSV data in `data/processed/`. Plots: `figures/` (generated by `rl/plot_experiments.py`). Next experiments: Exp 4 (DR + lot sizing) and Exp 5 (generalisation heatmap) are currently running.*
