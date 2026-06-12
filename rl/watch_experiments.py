"""
Passive watcher — monitors experiment CSVs and fires ntfy notifications.
Run alongside experiments; does NOT interfere with them.

Usage: python -u -m rl.watch_experiments
"""

import time
import pandas as pd
from pathlib import Path
from urllib import request as urllib_request

NTFY = "claude-om-notify"
OUT  = Path(__file__).parent.parent / "data" / "processed"

WATCH = {
    "Exp1 baseline":        OUT / "exp1_baseline_convergence.csv",
    "Exp2 windows":         OUT / "exp2_training_window.csv",
    "Exp3 conf_only":       OUT / "exp3_lot_sizing_convergence_conf_only.csv",
    "Exp3 vol_only":        OUT / "exp3_lot_sizing_convergence_vol_only.csv",
    "Exp3 both":            OUT / "exp3_lot_sizing_convergence_both.csv",
    "Exp4 conf_only+DR":    OUT / "exp4_dr_convergence_conf_only.csv",
    "Exp4 vol_only+DR":     OUT / "exp4_dr_convergence_vol_only.csv",
    "Exp4 both+DR":         OUT / "exp4_dr_convergence_both.csv",
    "Exp5 heatmap":         OUT / "exp5_generalization.csv",
    "Exp6a risk controls":  OUT / "exp6a_risk_controls.csv",
    "Exp6b per-ticker":     OUT / "exp6b_per_ticker.csv",
    "Exp6c equity curve":   OUT / "exp6c_equity_curve.csv",
}


def _notify(msg: str):
    try:
        req = urllib_request.Request(
            f"https://ntfy.sh/{NTFY}",
            data=msg.encode("utf-8"),
            method="POST",
        )
        req.add_header("Title", "SignalScope")
        urllib_request.urlopen(req, timeout=5)
        print(f"  [ntfy] {msg}", flush=True)
    except Exception as e:
        print(f"  [ntfy FAILED] {e}", flush=True)


def _summary(label: str, csv: Path) -> str:
    try:
        df = pd.read_csv(csv)
        if "return" in df.columns:
            best = df.loc[df["return"].idxmax()]
            extra = f"best return={best['return']:+.1f}%"
            if "passes" in df.columns:
                extra += f" @ pass {int(best['passes'])}"
            elif "window" in df.columns:
                extra += f" window={best['window']}"
            elif "lot_size" in df.columns:
                extra += f" lot={int(best['lot_size'])} cash=${int(best['starting_cash']):,}"
            return f"{label} DONE — {extra}"
        return f"{label} DONE ({len(df)} rows)"
    except Exception:
        return f"{label} DONE"


def main():
    seen   = set()
    pending = dict(WATCH)

    print(f"Watching {len(pending)} experiment outputs...", flush=True)
    for label, path in pending.items():
        status = "EXISTS" if path.exists() else "waiting"
        print(f"  {label:<25} {status}", flush=True)
        if path.exists():
            seen.add(label)

    print(flush=True)

    while pending:
        time.sleep(15)
        for label in list(pending):
            if label in seen:
                del pending[label]
                continue
            path = WATCH[label]
            if path.exists():
                seen.add(label)
                del pending[label]
                msg = _summary(label, path)
                print(msg, flush=True)
                _notify(msg)

        remaining = len(pending)
        if remaining:
            print(f"  [{time.strftime('%H:%M:%S')}] {remaining} experiments still running...", flush=True)

    print("All experiments complete.", flush=True)
    _notify("SignalScope: ALL experiments complete!")


if __name__ == "__main__":
    main()
