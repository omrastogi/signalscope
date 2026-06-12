"""
Ablation experiment suite for SignalScope.

Experiments:
  1  Baseline convergence          -- baseline config, eval every N passes
  2  Training window ablation      -- vary train_start, eval always on 2025
  3  Lot sizing convergence        -- conf/vol/both, no DR, eval every N passes
  4  DR + lot sizing convergence   -- same 3 configs with domain randomization
  5  Generalization heatmap        -- best config x 5 lot sizes x 5 cash amounts
  6  Final ablations               -- risk controls, per-ticker, equity curve

Usage:
  python -m rl.run_experiments --exp all --passes 2500 --eval-interval 500
  python -m rl.run_experiments --exp 1,2 --passes 2500
  python -m rl.run_experiments --exp 5,6 --best-config vol_only_dr
"""

import argparse
from pathlib import Path
from urllib import request as urllib_request

import numpy as np
import pandas as pd

from rl.agent import QLearningAgent, train_with_eval
from rl.backtest import run_backtest
from rl.environment import LOT_SIZE

NTFY_CHANNEL = "claude-om-notify"


def _notify(message: str):
    """Fire-and-forget push to ntfy.sh. Silently ignores failures."""
    try:
        req = urllib_request.Request(
            f"https://ntfy.sh/{NTFY_CHANNEL}",
            data=message.encode("utf-8"),
            method="POST",
        )
        req.add_header("Title", "SignalScope Experiment")
        urllib_request.urlopen(req, timeout=5)
    except Exception:
        pass

OUT = Path(__file__).parent.parent / "data" / "processed" / "experiments"

# Training windows for exp2 (label, train_start)
WINDOWS = [
    ("2024-only", "2024-01-01"),
    ("2023-2024", "2023-01-01"),
    ("2022-2024", "2022-01-01"),
    ("2021-2024", "2021-01-01"),
    ("2020-2024", "2020-01-01"),
]

# Lot-sizing configs for exp3 / exp4
LOT_CONFIGS = [
    ("conf_only", True,  False),
    ("vol_only",  False, True),
    ("both",      True,  True),
]

# Generalization grid for exp5
EVAL_LOTS = [10, 15, 25, 40, 50]
EVAL_CASH = [5_000, 7_500, 10_000, 15_000, 20_000]

# Risk-control combos for exp6a
RISK_CONFIGS = [
    ("no_controls",      False, False),
    ("stop_loss_only",   True,  False),
    ("liquidation_only", False, True),
    ("both (baseline)",  True,  True),
]

# Known-best configs (use_confidence, use_vol, randomize)
BEST_CONFIGS = {
    "vol_only":    (False, True,  False),
    "vol_only_dr": (False, True,  True),
    "conf_only":   (True,  False, False),
    "both_dr":     (True,  True,  True),
}


# ------------------------------------------------------------------
def exp1_baseline_convergence(max_passes: int = 2500, eval_interval: int = 500) -> pd.DataFrame:
    print("\n=== EXP 1: Baseline Convergence ===")
    _, checkpoints = train_with_eval(
        n_passes=max_passes,
        eval_interval=eval_interval,
        use_confidence=False,
        use_vol=False,
    )
    df = pd.DataFrame(checkpoints)
    df["config"] = "baseline"
    return df


# ------------------------------------------------------------------
def exp2_training_window(passes: int = 2000) -> pd.DataFrame:
    print("\n=== EXP 2: Training Window Ablation ===")
    rows = []
    for label, start in WINDOWS:
        print(f"  Window: {label}  (train_start={start})")
        agent, _ = train_with_eval(
            n_passes=passes,
            eval_interval=passes,   # single checkpoint at the end
            train_start=start,
        )
        df_bt, _ = run_backtest(agent=agent)
        ql = df_bt[df_bt["strategy"] == "Q-Learning"].iloc[0]
        bh = df_bt[df_bt["strategy"] == "Buy-and-Hold"].iloc[0]
        rows.append({
            "window":      label,
            "train_start": start,
            "return":      round(ql["total_return"] * 100, 2),
            "sharpe":      round(ql["sharpe"], 3),
            "max_dd":      round(ql["max_drawdown"] * 100, 2),
            "bh_return":   round(bh["total_return"] * 100, 2),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
def exp3_lot_sizing_convergence(
    max_passes: int = 2500,
    eval_interval: int = 500,
    only_config: str | None = None,
) -> pd.DataFrame:
    print("\n=== EXP 3: Lot Sizing Convergence (no DR) ===")
    frames = []
    configs = [c for c in LOT_CONFIGS if only_config is None or c[0] == only_config]
    for label, use_conf, use_vol in configs:
        print(f"  Config: {label}", flush=True)
        _, checkpoints = train_with_eval(
            n_passes=max_passes,
            eval_interval=eval_interval,
            use_confidence=use_conf,
            use_vol=use_vol,
        )
        df = pd.DataFrame(checkpoints)
        df["config"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------
def exp4_dr_convergence(
    max_passes: int = 2500,
    eval_interval: int = 500,
    only_config: str | None = None,
) -> pd.DataFrame:
    print("\n=== EXP 4: DR + Lot Sizing Convergence ===")
    frames = []
    configs = [c for c in LOT_CONFIGS if only_config is None or c[0] == only_config]
    for label, use_conf, use_vol in configs:
        print(f"  Config: {label} + DR", flush=True)
        _, checkpoints = train_with_eval(
            n_passes=max_passes,
            eval_interval=eval_interval,
            use_confidence=use_conf,
            use_vol=use_vol,
            n_episodes_per_pass=3,
            randomize=True,
        )
        df = pd.DataFrame(checkpoints)
        df["config"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------
def exp5_generalization(agent: QLearningAgent, use_confidence: bool, use_vol: bool) -> pd.DataFrame:
    print("\n=== EXP 5: Generalization Heatmap (5 lots x 5 cash) ===")
    rows = []
    for lot in EVAL_LOTS:
        for cash in EVAL_CASH:
            df_bt, _ = run_backtest(
                agent=agent,
                use_confidence=use_confidence,
                use_vol=use_vol,
                eval_lot=lot,
                eval_capital=cash,
            )
            ql = df_bt[df_bt["strategy"] == "Q-Learning"].iloc[0]
            rows.append({
                "lot_size":      lot,
                "starting_cash": cash,
                "return":        round(ql["total_return"] * 100, 2),
                "sharpe":        round(ql["sharpe"], 3),
            })
            print(f"  lot={lot:3d}  cash={cash:6,.0f}  return={rows[-1]['return']:+.1f}%", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
def exp6a_risk_controls(agent: QLearningAgent, use_confidence: bool, use_vol: bool) -> pd.DataFrame:
    print("\n=== EXP 6a: Risk Controls Ablation ===")
    rows = []
    for label, sl, liq in RISK_CONFIGS:
        df_bt, _ = run_backtest(
            agent=agent,
            use_confidence=use_confidence,
            use_vol=use_vol,
            use_stop_loss=sl,
            use_liquidation=liq,
        )
        ql = df_bt[df_bt["strategy"] == "Q-Learning"].iloc[0]
        rows.append({
            "config": label,
            "return": round(ql["total_return"] * 100, 2),
            "sharpe": round(ql["sharpe"], 3),
            "max_dd": round(ql["max_drawdown"] * 100, 2),
        })
        print(f"  {label:<22}  return={rows[-1]['return']:+.1f}%  sharpe={rows[-1]['sharpe']:.3f}  max_dd={rows[-1]['max_dd']:.1f}%")
    return pd.DataFrame(rows)


def exp6b_per_ticker(agent: QLearningAgent, use_confidence: bool, use_vol: bool) -> pd.DataFrame:
    print("\n=== EXP 6b: Per-Ticker Breakdown ===")
    df_bt, equity_curves = run_backtest(
        agent=agent,
        use_confidence=use_confidence,
        use_vol=use_vol,
        track_per_ticker=True,
    )
    ticker_returns = equity_curves.get("ticker_returns", {})
    rows = [{"ticker": t, "cumulative_reward": v}
            for t, v in sorted(ticker_returns.items())]
    for r in rows:
        print(f"  {r['ticker']}  cumulative_reward={r['cumulative_reward']:.4f}")
    return pd.DataFrame(rows)


def exp6c_equity_curve(agent: QLearningAgent, use_confidence: bool, use_vol: bool) -> pd.DataFrame:
    print("\n=== EXP 6c: Equity Curve ===")
    df_bt, equity_curves = run_backtest(
        agent=agent,
        use_confidence=use_confidence,
        use_vol=use_vol,
    )
    rows = []
    for name, curve in equity_curves.items():
        if not isinstance(curve, list):
            continue
        for day, val in enumerate(curve):
            rows.append({"strategy": name, "day": day, "value": val})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
def _train_best(use_confidence: bool, use_vol: bool, randomize: bool,
                max_passes: int, qtable_path: Path) -> QLearningAgent:
    print(f"\nTraining best config (conf={use_confidence}, vol={use_vol}, dr={randomize}) ...")
    agent, _ = train_with_eval(
        n_passes=max_passes,
        eval_interval=max_passes,
        use_confidence=use_confidence,
        use_vol=use_vol,
        n_episodes_per_pass=3 if randomize else 1,
        randomize=randomize,
    )
    agent.save(qtable_path)
    return agent


# ------------------------------------------------------------------
def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="SignalScope ablation experiments")
    parser.add_argument("--exp", default="all",
                        help="Comma-separated experiment numbers or 'all' (e.g. 1,2,3)")
    parser.add_argument("--passes", type=int, default=2500,
                        help="Max training passes for convergence experiments (default 2500)")
    parser.add_argument("--eval-interval", type=int, default=500, dest="eval_interval",
                        help="Eval every N passes (default 500)")
    parser.add_argument("--window-passes", type=int, default=2000, dest="window_passes",
                        help="Passes for training window ablation (default 2000)")
    parser.add_argument("--best-config", default="vol_only_dr",
                        choices=list(BEST_CONFIGS.keys()),
                        help="Config key for exp5/exp6 (default: vol_only_dr)")
    parser.add_argument("--only-config", default=None, dest="only_config",
                        choices=["conf_only", "vol_only", "both"],
                        help="Run only one lot-sizing config for exp3/exp4 (enables parallel launches)")
    args = parser.parse_args()

    exps = set(args.exp.split(",")) if args.exp != "all" else {"1", "2", "3", "4", "5", "6"}

    if "1" in exps:
        df = exp1_baseline_convergence(args.passes, args.eval_interval)
        df.to_csv(OUT / "exp1_baseline_convergence.csv", index=False)
        best = df.loc[df["return"].idxmax()]
        msg = f"Exp1 done: best return={best['return']:+.1f}% @ pass {int(best['passes'])}"
        print(f"  {msg}", flush=True)
        _notify(msg)

    if "2" in exps:
        df = exp2_training_window(args.window_passes)
        df.to_csv(OUT / "exp2_training_window.csv", index=False)
        best = df.loc[df["return"].idxmax()]
        msg = f"Exp2 done: best window={best['window']} return={best['return']:+.1f}%"
        print(f"  {msg}", flush=True)
        _notify(msg)

    if "3" in exps:
        df = exp3_lot_sizing_convergence(args.passes, args.eval_interval, args.only_config)
        suffix = f"_{args.only_config}" if args.only_config else ""
        out3 = OUT / f"exp3_lot_sizing_convergence{suffix}.csv"
        df.to_csv(out3, index=False)
        best = df.loc[df["return"].idxmax()]
        msg = f"Exp3 done ({args.only_config or 'all'}): best={best['config']} return={best['return']:+.1f}% @ pass {int(best['passes'])}"
        print(f"  {msg}", flush=True)
        _notify(msg)

    if "4" in exps:
        df = exp4_dr_convergence(args.passes, args.eval_interval, args.only_config)
        suffix = f"_{args.only_config}" if args.only_config else ""
        out4 = OUT / f"exp4_dr_convergence{suffix}.csv"
        df.to_csv(out4, index=False)
        best = df.loc[df["return"].idxmax()]
        msg = f"Exp4+DR done ({args.only_config or 'all'}): best={best['config']} return={best['return']:+.1f}% @ pass {int(best['passes'])}"
        print(f"  {msg}", flush=True)
        _notify(msg)

    if exps & {"5", "6"}:
        use_conf, use_vol, randomize = BEST_CONFIGS[args.best_config]
        qtable_path = Path(__file__).parent.parent / "models" / "qtable_best.json"
        agent = _train_best(use_conf, use_vol, randomize, args.passes, qtable_path)

        if "5" in exps:
            df = exp5_generalization(agent, use_conf, use_vol)
            df.to_csv(OUT / "exp5_generalization.csv", index=False)
            best = df.loc[df["return"].idxmax()]
            msg = f"Exp5 done: best lot={int(best['lot_size'])} cash=${int(best['starting_cash']):,} return={best['return']:+.1f}%"
            print(f"  {msg}", flush=True)
            _notify(msg)

        if "6" in exps:
            df6a = exp6a_risk_controls(agent, use_conf, use_vol)
            df6a.to_csv(OUT / "exp6a_risk_controls.csv", index=False)

            df6b = exp6b_per_ticker(agent, use_conf, use_vol)
            df6b.to_csv(OUT / "exp6b_per_ticker.csv", index=False)

            df6c = exp6c_equity_curve(agent, use_conf, use_vol)
            df6c.to_csv(OUT / "exp6c_equity_curve.csv", index=False)

            _notify("Exp6 done: risk controls, per-ticker, equity curve saved")
            print("  Exp6 complete — risk controls, per-ticker, equity curve saved", flush=True)

    print("\nAll requested experiments complete.", flush=True)


if __name__ == "__main__":
    main()
