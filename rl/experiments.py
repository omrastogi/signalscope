"""
Experiment runner: compare four dynamic lot-sizing configurations.

  baseline  : fixed LOT_SIZE=25 (no dynamic sizing)
  conf_only : confidence-based sizing only
  vol_only  : volatility-adjusted sizing only
  both      : confidence + volatility combined

Each configuration trains from scratch and is evaluated on the 2024 test split.
Results are printed as a comparison table.

CLI args:
  --passes          training passes per config (default 200)
  --n-episodes      episodes per pass (default 1; use 4-5 with --randomize)
  --randomize       enable domain randomization (random cash + lot per episode)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from rl.agent import train, policy_summary
from rl.backtest import run_backtest

RESULTS_PATH = Path(__file__).parent.parent / "data" / "processed" / "experiment_results.csv"

CONFIGS = [
    ("baseline",  False, False),
    ("conf_only", True,  False),
    ("vol_only",  False, True),
    ("both",      True,  True),
]


def run_experiments(
    n_passes:            int  = 200,
    n_episodes_per_pass: int  = 1,
    randomize:           bool = False,
) -> pd.DataFrame:
    rows = []

    for label, use_conf, use_vol in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Config: {label}  (confidence={use_conf}, volatility={use_vol})")
        if randomize:
            print(f"  Domain randomization ON  ({n_episodes_per_pass} episodes/pass)")
        print(f"{'='*60}")

        agent = train(
            n_passes=n_passes,
            use_confidence=use_conf,
            use_vol=use_vol,
            n_episodes_per_pass=n_episodes_per_pass,
            randomize=randomize,
        )
        policy_summary(agent)

        df, curves = run_backtest(use_confidence=use_conf, use_vol=use_vol)
        ql = df[df["strategy"] == "Q-Learning"].iloc[0]
        bh = df[df["strategy"] == "Buy-and-Hold"].iloc[0]

        final_ql = 10_000 * (1 + ql["total_return"])
        final_bh = 10_000 * (1 + bh["total_return"])

        rows.append({
            "config":          label,
            "use_confidence":  use_conf,
            "use_vol":         use_vol,
            "ql_return_pct":   round(ql["total_return"] * 100, 1),
            "ql_final_value":  round(final_ql, 0),
            "ql_sharpe":       round(ql["sharpe"], 2),
            "ql_max_dd_pct":   round(ql["max_drawdown"] * 100, 1),
            "bh_return_pct":   round(bh["total_return"] * 100, 1),
        })

    return pd.DataFrame(rows)


def print_experiment_table(results: pd.DataFrame):
    print("\n" + "="*75)
    print("  EXPERIMENT RESULTS — Q-Learning vs Buy-and-Hold (2024 test set)")
    print("="*75)
    print(f"  {'Config':<12} {'Return':>8} {'Final $':>9} {'Sharpe':>7} {'Max DD':>8} {'vs B&H':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*7} {'-'*8} {'-'*8}")
    for _, row in results.iterrows():
        vs_bh = row["ql_return_pct"] - row["bh_return_pct"]
        print(
            f"  {row['config']:<12} "
            f"{row['ql_return_pct']:>7.1f}% "
            f"${row['ql_final_value']:>8,.0f} "
            f"{row['ql_sharpe']:>7.2f} "
            f"{row['ql_max_dd_pct']:>7.1f}% "
            f"{vs_bh:>+7.1f}%"
        )
    print("="*75)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes",     type=int,  default=200)
    parser.add_argument("--n-episodes", type=int,  default=1,     dest="n_episodes")
    parser.add_argument("--randomize",  action="store_true", default=False)
    args = parser.parse_args()

    results = run_experiments(
        n_passes=args.passes,
        n_episodes_per_pass=args.n_episodes,
        randomize=args.randomize,
    )
    print_experiment_table(results)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")
