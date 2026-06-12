"""
Plot all 6 ablation experiment results.

Reads CSVs from data/processed/exp*.csv and writes PNGs to figures/.

Usage:
  python -m rl.plot_experiments          # all plots
  python -m rl.plot_experiments --exp 1  # single plot
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "processed" / "experiments"
FIG  = Path(__file__).parent.parent / "figures" / "experiments"
FIG.mkdir(exist_ok=True)

COLORS = {
    "baseline":  "#555555",
    "conf_only": "#2196F3",
    "vol_only":  "#4CAF50",
    "both":      "#FF9800",
}
DR_STYLE   = "--"
NODR_STYLE = "-"

plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":   150,
})


# ------------------------------------------------------------------
def plot_exp1():
    """Baseline convergence: return and Sharpe vs passes."""
    df = pd.read_csv(DATA / "exp1_baseline_convergence.csv")

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.plot(df["passes"], df["return"], color=COLORS["baseline"],
             linewidth=2, marker="o", markersize=5, label="Return (%)")
    ax2.plot(df["passes"], df["sharpe"], color=COLORS["baseline"],
             linewidth=2, linestyle="--", marker="s", markersize=5, alpha=0.7, label="Sharpe")

    ax1.set_xlabel("Training Passes")
    ax1.set_ylabel("Return (%, 2025 test)")
    ax2.set_ylabel("Sharpe Ratio", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")

    ax1.set_title("Exp 1 — Baseline Convergence (eval every 500 passes)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    path = FIG / "exp1_baseline_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------
def plot_exp2():
    """Training window ablation: return & Sharpe by training window."""
    df = pd.read_csv(DATA / "exp2_training_window.csv")

    x     = np.arange(len(df))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars1 = ax.bar(x - width / 2, df["return"],  width, label="Q-Learning Return (%)",
                   color="#4CAF50", alpha=0.85)
    bars2 = ax.bar(x + width / 2, df["bh_return"], width, label="Buy-and-Hold Return (%)",
                   color="#BDBDBD", alpha=0.85)

    ax2 = ax.twinx()
    ax2.plot(x, df["sharpe"], color="#2196F3", marker="D", linewidth=2,
             markersize=6, label="Sharpe")
    ax2.set_ylabel("Sharpe Ratio", color="#2196F3")
    ax2.tick_params(axis="y", labelcolor="#2196F3")

    ax.set_xticks(x)
    ax.set_xticklabels(df["window"], rotation=15, ha="right")
    ax.set_ylabel("Return (%, 2025 test)")
    ax.set_title("Exp 2 — Training Window Ablation (eval always on 2025)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:+.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = FIG / "exp2_training_window.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------
def _load_exp3() -> pd.DataFrame:
    """Load exp3 CSV — merges per-config files if present, else reads combined."""
    per_config = [DATA / f"exp3_lot_sizing_convergence_{c}.csv"
                  for c in ["conf_only", "vol_only", "both"]]
    if all(p.exists() for p in per_config):
        return pd.concat([pd.read_csv(p) for p in per_config], ignore_index=True)
    return pd.read_csv(DATA / "exp3_lot_sizing_convergence.csv")


def _load_exp4() -> pd.DataFrame:
    """Load exp4 CSV — merges per-config files if present, else reads combined."""
    per_config = [DATA / f"exp4_dr_convergence_{c}.csv"
                  for c in ["conf_only", "vol_only", "both"]]
    if all(p.exists() for p in per_config):
        return pd.concat([pd.read_csv(p) for p in per_config], ignore_index=True)
    return pd.read_csv(DATA / "exp4_dr_convergence.csv")


def plot_exp3():
    """Lot sizing convergence (no DR): 3 configs + baseline overlay."""
    df3 = _load_exp3()

    baseline_csv = DATA / "exp1_baseline_convergence.csv"
    has_baseline = baseline_csv.exists()
    if has_baseline:
        df1 = pd.read_csv(baseline_csv)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if has_baseline:
        ax.plot(df1["passes"], df1["return"], color=COLORS["baseline"],
                linewidth=1.8, linestyle=":", marker="o", markersize=4,
                label="baseline")

    for config, color in [(c, COLORS[c]) for c in ["conf_only", "vol_only", "both"]]:
        sub = df3[df3["config"] == config]
        ax.plot(sub["passes"], sub["return"], color=color,
                linewidth=2.2, marker="o", markersize=5, label=config)

    ax.set_xlabel("Training Passes")
    ax.set_ylabel("Return (%, 2025 test)")
    ax.set_title("Exp 3 — Lot Sizing Convergence (no Domain Randomization)")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")

    fig.tight_layout()
    path = FIG / "exp3_lot_sizing_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------
def plot_exp4():
    """DR convergence: 3 configs with DR, overlay best no-DR line."""
    df4 = _load_exp4()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    # Left: Return
    ax = axes[0]
    for config, color in [(c, COLORS[c]) for c in ["conf_only", "vol_only", "both"]]:
        sub3_csv = DATA / "exp3_lot_sizing_convergence.csv"
        if sub3_csv.exists():
            df3 = pd.read_csv(sub3_csv)
            sub3 = df3[df3["config"] == config]
            ax.plot(sub3["passes"], sub3["return"], color=color,
                    linewidth=1.4, linestyle=":", alpha=0.5, label=f"{config} (no DR)")

        sub4 = df4[df4["config"] == config]
        ax.plot(sub4["passes"], sub4["return"], color=color,
                linewidth=2.2, marker="o", markersize=5, label=f"{config} + DR")

    ax.set_xlabel("Training Passes")
    ax.set_ylabel("Return (%, 2025 test)")
    ax.set_title("Return — no DR (dotted) vs DR (solid)")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.legend(fontsize=8)

    # Right: Sharpe
    ax2 = axes[1]
    for config, color in [(c, COLORS[c]) for c in ["conf_only", "vol_only", "both"]]:
        sub4 = df4[df4["config"] == config]
        ax2.plot(sub4["passes"], sub4["sharpe"], color=color,
                 linewidth=2.2, marker="s", markersize=5, label=f"{config} + DR")

    ax2.set_xlabel("Training Passes")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Sharpe — with Domain Randomization")
    ax2.legend(fontsize=9)

    fig.suptitle("Exp 4 — Domain Randomization + Lot Sizing Convergence", fontweight="bold")
    fig.tight_layout()
    path = FIG / "exp4_dr_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------
def plot_exp5():
    """Generalization heatmap: 5 lot sizes x 5 cash amounts."""
    df = pd.read_csv(DATA / "exp5_generalization.csv")

    lots  = sorted(df["lot_size"].unique())
    cash  = sorted(df["starting_cash"].unique())
    grid  = df.pivot(index="lot_size", columns="starting_cash", values="return")
    grid  = grid.reindex(index=lots, columns=cash)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid.values, cmap="RdYlGn", aspect="auto",
                   vmin=grid.values.min(), vmax=grid.values.max())

    ax.set_xticks(range(len(cash)))
    ax.set_xticklabels([f"${c:,.0f}" for c in cash])
    ax.set_yticks(range(len(lots)))
    ax.set_yticklabels([f"{l} shares" for l in lots])
    ax.set_xlabel("Starting Cash")
    ax.set_ylabel("Lot Size")
    ax.set_title("Exp 5 — Generalization: Return (%) by Lot Size × Starting Cash\n(Best config trained at $10k / 25 shares)")

    for i in range(len(lots)):
        for j in range(len(cash)):
            val = grid.values[i, j]
            color = "white" if abs(val) > 0.6 * abs(grid.values).max() else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Return (%)")
    fig.tight_layout()
    path = FIG / "exp5_generalization_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------
def plot_exp6a():
    """Risk controls ablation: grouped bar for return, Sharpe, max DD."""
    df = pd.read_csv(DATA / "exp6a_risk_controls.csv")

    x     = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(x - width, df["return"],      width, label="Return (%)",  color="#4CAF50", alpha=0.85)
    ax.bar(x,         df["sharpe"] * 10, width, label="Sharpe ×10",  color="#2196F3", alpha=0.85)
    ax.bar(x + width, df["max_dd"],      width, label="Max DD (%)",  color="#F44336", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=12, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Value")
    ax.set_title("Exp 6a — Risk Controls Ablation (best config on 2025 test)")
    ax.legend()

    fig.tight_layout()
    path = FIG / "exp6a_risk_controls.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------
def plot_exp6b():
    """Per-ticker cumulative reward breakdown."""
    df = pd.read_csv(DATA / "exp6b_per_ticker.csv")
    df = df.sort_values("cumulative_reward", ascending=True)

    colors = ["#F44336" if v < 0 else "#4CAF50" for v in df["cumulative_reward"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(df["ticker"], df["cumulative_reward"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cumulative Normalised Reward (P&L / starting_cash)")
    ax.set_title("Exp 6b — Per-Ticker Return Contribution (2025 test)")

    for bar, val in zip(bars, df["cumulative_reward"]):
        ax.text(val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=9)

    fig.tight_layout()
    path = FIG / "exp6b_per_ticker.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------
def plot_exp6c():
    """Equity curve: Q-Learning vs Buy-and-Hold vs Random over 2025 test."""
    df = pd.read_csv(DATA / "exp6c_equity_curve.csv")

    strategy_styles = {
        "Q-Learning":   ("#4CAF50", "-",  2.5),
        "Buy-and-Hold": ("#F44336", "--", 1.8),
        "Random":       ("#9E9E9E", ":",  1.5),
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for strat, (color, ls, lw) in strategy_styles.items():
        sub = df[df["strategy"] == strat].sort_values("day")
        if sub.empty:
            continue
        ax.plot(sub["day"], sub["value"], color=color, linestyle=ls,
                linewidth=lw, label=strat)

    ax.axhline(10_000, color="black", linewidth=0.7, linestyle=":", alpha=0.5,
               label="Starting capital ($10k)")
    ax.set_xlabel("Trading Day (2025)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Exp 6c — Equity Curve: 2025 Out-of-Sample Test")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()

    fig.tight_layout()
    path = FIG / "exp6c_equity_curve.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------
PLOT_FNS = {
    "1": plot_exp1,
    "2": plot_exp2,
    "3": plot_exp3,
    "4": plot_exp4,
    "5": plot_exp5,
    "6a": plot_exp6a,
    "6b": plot_exp6b,
    "6c": plot_exp6c,
}


def main():
    parser = argparse.ArgumentParser(description="Plot SignalScope ablation results")
    parser.add_argument("--exp", default="all",
                        help="Experiments to plot: 1,2,3,4,5,6a,6b,6c or 'all'")
    args = parser.parse_args()

    keys = list(PLOT_FNS.keys()) if args.exp == "all" else args.exp.split(",")

    for key in keys:
        key = key.strip()
        if key not in PLOT_FNS:
            print(f"Unknown experiment key: {key}  (valid: {list(PLOT_FNS.keys())})")
            continue
        csv_map = {
            "1":  "exp1_baseline_convergence.csv",
            "2":  "exp2_training_window.csv",
            "3":  "exp3_lot_sizing_convergence.csv",
            "4":  "exp4_dr_convergence.csv",
            "5":  "exp5_generalization.csv",
            "6a": "exp6a_risk_controls.csv",
            "6b": "exp6b_per_ticker.csv",
            "6c": "exp6c_equity_curve.csv",
        }
        # exp3/4 may be split into per-config files
        if key == "3":
            per = [DATA / f"exp3_lot_sizing_convergence_{c}.csv" for c in ["conf_only","vol_only","both"]]
            if not any(p.exists() for p in per) and not (DATA / csv_map["3"]).exists():
                print("  [skip] exp3 CSVs not found — run experiments first.")
                continue
        elif key == "4":
            per = [DATA / f"exp4_dr_convergence_{c}.csv" for c in ["conf_only","vol_only","both"]]
            if not any(p.exists() for p in per) and not (DATA / csv_map["4"]).exists():
                print("  [skip] exp4 CSVs not found — run experiments first.")
                continue
        else:
            csv_path = DATA / csv_map[key]
            if not csv_path.exists():
                print(f"  [skip] {csv_path.name} not found — run experiments first.")
                continue
        PLOT_FNS[key]()

    print(f"\nAll plots saved to {FIG}/")


if __name__ == "__main__":
    main()
