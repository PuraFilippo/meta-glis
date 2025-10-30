#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# USER CONFIG
# ----------------------------
# Which deviations to expect (percent)
DEVIATIONS = [0, 10, 20, 40, 60, 80]
# Where to look ('.' = current directory)
LOG_DIR = "robot_experiments/Ki50"
# Threshold for failure (3 degrees)
ERR_THRESH = 2.9 * np.pi / 180.0
# Output artifacts
OUT_CSV = "out/sweep_metrics.csv"
OUT_PNG = "out/sweep_summary.png"
OUT_PDF = "out/sweep_summary.pdf"
TITLE = "Performance vs Gain Deviation — Real vs Gazebo"
# ----------------------------

# Recognize filenames like: real+10.csv, real-20.csv, gazebo+40.csv, gazebo-80.csv
FILE_RE = re.compile(r'^(real|gazebo)([+-])(\d+)\.csv$', re.IGNORECASE)

JOINTS = list(range(7))

def load_log(path):
    """Load a CSV log and return time (np), q_err matrix (N,7).
       If q_err* not present, compute from q_d* - q* if available."""
    df = pd.read_csv(path)

    # Ensure there is a time column named "time"
    if "time" not in df.columns:
        # try case-insensitive fallback
        for c in df.columns:
            if c.strip().lower() == "time":
                df.rename(columns={c: "time"}, inplace=True)
                break
    if "time" not in df.columns:
        raise ValueError(f"{path}: no 'time' column.")

    df = df.copy()
    df["time"] = df["time"].astype(float)
    df.sort_values("time", inplace=True)

    # try to find q_err*
    qerr_cols = [f"q_err{j}" for j in JOINTS]
    have_qerr = all(col in df.columns for col in qerr_cols)

    if have_qerr:
        qerr = np.column_stack([df[f"q_err{j}"].astype(float).values for j in JOINTS])
    else:
        # Compute from q_d - q
        qd_cols = [f"q_d{j}" for j in JOINTS]
        q_cols  = [f"q{j}"   for j in JOINTS]
        if not all(c in df.columns for c in qd_cols + q_cols):
            raise ValueError(f"{path}: neither q_err* nor (q_d* and q*) present.")
        qd = np.column_stack([df[f"q_d{j}"].astype(float).values for j in JOINTS])
        q  = np.column_stack([df[f"q{j}"].astype(float).values   for j in JOINTS])
        qerr = qd - q

    t = df["time"].values
    # de-offset time (not strictly necessary for integral, but nice)
    if len(t) > 0:
        t = t - t[0]

    return t, qerr

def performance_score(t, qerr):
    """Integral of sum(|q_err|) dt over the run (lower is better)."""
    if len(t) < 2:
        return np.nan
    series = np.sum(np.abs(qerr), axis=1)  # shape (N,)
    # trapezoidal integration
    return float(np.trapz(series, t))

def failure_flag(qerr, thresh):
    """True if any joint exceeds |error| >= thresh at any time."""
    if qerr.size == 0:
        return False
    return bool(np.any(np.max(np.abs(qerr), axis=1) >= thresh))

def collect_runs():
    """Walk LOG_DIR, parse expected files, compute metrics."""
    rows = []
    for fname in os.listdir(LOG_DIR):
        m = FILE_RE.match(fname)
        if not m:
            continue
        kind = m.group(1).lower()       # 'real' or 'gazebo'
        sign = m.group(2)               # '+' or '-'
        pct  = int(m.group(3))          # 10,20,40,80
        if pct not in DEVIATIONS:
            continue

        signed_pct = pct if sign == '+' else -pct
        fpath = os.path.join(LOG_DIR, fname)

        try:
            t, qerr = load_log(fpath)
            score   = performance_score(t, qerr)
            failed  = failure_flag(qerr, ERR_THRESH)

            # some extras you might find useful
            max_abs_err = float(np.max(np.abs(qerr))) if qerr.size else np.nan
            mean_abs_err = float(np.mean(np.abs(qerr))) if qerr.size else np.nan

            rows.append(dict(
                file=fname,
                kind=kind,                   # 'real'/'gazebo'
                deviation=signed_pct,        # -80 .. +80
                score=score,
                failed=failed,
                max_abs_err=max_abs_err,
                mean_abs_err=mean_abs_err,
                duration=float(t[-1] if len(t) else 0.0),
                n_samples=int(len(t)),
            ))
        except Exception as e:
            rows.append(dict(
                file=fname,
                kind=kind,
                deviation=signed_pct,
                score=np.nan,
                failed=True,   # mark as failed if unreadable
                max_abs_err=np.nan,
                mean_abs_err=np.nan,
                duration=0.0,
                n_samples=0,
                error=str(e),
            ))
    df = pd.DataFrame(rows)
    # Keep only deviations we care about, in sorted order
    if not df.empty:
        df = df[df["deviation"].isin([*DEVIATIONS, *[-d for d in DEVIATIONS]])]
        df.sort_values(["kind", "deviation"], inplace=True)
    return df

def plot_summary(df):
    """Plot Real vs Gazebo score vs deviation with fail markers."""
    if df.empty:
        print("No valid runs found.")
        return

    # Prepare data for both series on aligned x
    x_all = sorted({int(d) for d in df["deviation"].unique()})

    def series(kind):
        sub = df[df["kind"] == kind]
        y = []
        fail = []
        for x in x_all:
            row = sub[sub["deviation"] == x]
            if row.empty:
                y.append(np.nan)
                fail.append(False)
            else:
                y.append(float(row["score"].values[0]))
                fail.append(bool(row["failed"].values[0]))
        return np.array(y, dtype=float), np.array(fail, dtype=bool)

    y_real, f_real = series("real")
    y_gz,   f_gz   = series("gazebo")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_all, y_real, "-o", label="Real", linewidth=2, alpha=0.9)
    ax.plot(x_all, y_gz,   "-s", label="Gazebo", linewidth=2, alpha=0.9)

    # Highlight failures: overlay red X on those points
    for x, y, fail in zip(x_all, y_real, f_real):
        if np.isfinite(y) and fail:
            ax.plot(x, y, "x", color="red", markersize=10, mew=2, label="_nolegend_")
    for x, y, fail in zip(x_all, y_gz, f_gz):
        if np.isfinite(y) and fail:
            ax.plot(x, y, "x", color="red", markersize=10, mew=2, label="_nolegend_")

    ax.set_title(TITLE)
    ax.set_xlabel("Gain deviation from defaults [%]  (negative = lower gains)")
    ax.set_ylabel("Performance (∫ Σ |q_err| dt)  [rad·s]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"Saved plot to {OUT_PNG} and {OUT_PDF}")

def main():
    df = collect_runs()
    # Save metrics table
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved metrics to {OUT_CSV}")
    # Human-readable preview
    if not df.empty:
        print(df[["file","kind","deviation","score","failed","max_abs_err","duration","n_samples"]])
    plot_summary(df)

if __name__ == "__main__":
    main()
