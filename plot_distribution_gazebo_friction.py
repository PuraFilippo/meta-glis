#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# USER CONFIG
# ----------------------------
# Deviations to expect (percent magnitudes). Script will use ± of these.
DEVIATIONS = [0, 10, 20, 40, 60, 80]

# Folder containing all CSV logs
LOG_DIR = "robot_experiments/friction_perturbations_0.8"

# Failure threshold (radians)
ERR_THRESH = 2.9 * np.pi / 180.0

# Output
OUT_DIR = "out"
OUT_CSV = os.path.join(OUT_DIR, "sweep_metrics.csv")
OUT_PNG = os.path.join(OUT_DIR, "sweep_summary.png")
OUT_PDF = os.path.join(OUT_DIR, "sweep_summary.pdf")

TITLE = "Performance vs Gain Deviation — Real vs Gazebo (friction sets)"

SHOW_LINES = True
# ----------------------------

os.makedirs(OUT_DIR, exist_ok=True)
JOINTS = list(range(7))

# Old "real/gazebo ±XX.csv" pattern
FILE_RE_SIMPLE = re.compile(r'^(real|gazebo)([+-])(\d+)\.csv$', re.IGNORECASE)

# New Gazebo pattern: custom_pid_log_gazebo_fricXX_fac+YY.csv
FILE_RE_GZ = re.compile(
    r'^custom_pid_log_gazebo_fric(?P<fric>\d+)_fac(?P<sign>[+-])(?P<pct>\d+)\.csv$',
    re.IGNORECASE
)

def load_log(path):
    """Load CSV and return t, q_err."""
    df = pd.read_csv(path)

    if "time" not in df.columns:
        for c in df.columns:
            if c.strip().lower() == "time":
                df.rename(columns={c: "time"}, inplace=True)
                break
    if "time" not in df.columns:
        raise ValueError(f"{path}: no 'time' column.")

    df["time"] = df["time"].astype(float)
    df.sort_values("time", inplace=True)

    qerr_cols = [f"q_err{j}" for j in JOINTS]
    if all(c in df.columns for c in qerr_cols):
        qerr = np.column_stack([df[c].astype(float).values for c in qerr_cols])
    else:
        qd_cols = [f"q_d{j}" for j in JOINTS]
        q_cols  = [f"q{j}"   for j in JOINTS]
        if not all(c in df.columns for c in qd_cols + q_cols):
            raise ValueError(f"{path}: neither q_err* nor (q_d* and q*) present.")
        qd = np.column_stack([df[f"q_d{j}"].values for j in JOINTS])
        q  = np.column_stack([df[f"q{j}"].values   for j in JOINTS])
        qerr = qd - q

    t = df["time"].values
    if len(t) > 0:
        t = t - t[0]
    return t, qerr

def performance_score(t, qerr):
    """Integral of sum(|q_err|) dt (lower = better)."""
    if len(t) < 2 or qerr.size == 0:
        return np.nan
    return float(np.trapz(np.sum(np.abs(qerr), axis=1), t))

def failure_flag(qerr, thresh):
    """True if any joint exceeds |error| >= thresh."""
    return bool(qerr.size and np.any(np.max(np.abs(qerr), axis=1) >= thresh))

def collect_runs():
    """Scan LOG_DIR, parse filenames, compute metrics."""
    rows = []
    for fname in os.listdir(LOG_DIR):
        path = os.path.join(LOG_DIR, fname)

        # --- Gazebo new pattern ---
        m_gz = FILE_RE_GZ.match(fname)
        if m_gz:
            fric = int(m_gz.group("fric"))
            sign = m_gz.group("sign")
            pct  = int(m_gz.group("pct"))
            if pct not in DEVIATIONS:
                continue
            signed_pct = pct if sign == '+' else -pct
            series = f"Gazebo fric{fric:02d}"
            try:
                t, qerr = load_log(path)
                score = performance_score(t, qerr)
                failed = failure_flag(qerr, ERR_THRESH)
                if failed:
                    score = 0.0
                rows.append(dict(
                    file=fname, series=series, kind="gazebo",
                    friction_set=fric, deviation=signed_pct,
                    score=score, failed=failed,
                    max_abs_err=np.max(np.abs(qerr)) if qerr.size else np.nan,
                    mean_abs_err=np.mean(np.abs(qerr)) if qerr.size else np.nan,
                    duration=(t[-1] if len(t) else 0.0),
                    n_samples=len(t)
                ))
            except Exception as e:
                rows.append(dict(file=fname, series=series, kind="gazebo",
                                 friction_set=fric, deviation=signed_pct,
                                 score=np.nan, failed=True, error=str(e)))
            continue

        # --- Real/simple pattern ---
        m_s = FILE_RE_SIMPLE.match(fname)
        if m_s:
            origin = m_s.group(1).lower()
            sign   = m_s.group(2)
            pct    = int(m_s.group(3))
            if pct not in DEVIATIONS:
                continue
            signed_pct = pct if sign == '+' else -pct
            series = "Real" if origin == "real" else "Gazebo"
            try:
                t, qerr = load_log(path)
                score = performance_score(t, qerr)
                failed = failure_flag(qerr, ERR_THRESH)
                if failed:
                    score = 0.0
                rows.append(dict(
                    file=fname, series=series, kind=origin,
                    friction_set=None, deviation=signed_pct,
                    score=score, failed=failed,
                    max_abs_err=np.max(np.abs(qerr)) if qerr.size else np.nan,
                    mean_abs_err=np.mean(np.abs(qerr)) if qerr.size else np.nan,
                    duration=(t[-1] if len(t) else 0.0),
                    n_samples=len(t)
                ))
            except Exception as e:
                rows.append(dict(file=fname, series=series, kind=origin,
                                 deviation=signed_pct, score=np.nan, failed=True, error=str(e)))

    df = pd.DataFrame(rows)
    if not df.empty:
        keep = [*DEVIATIONS, *[-d for d in DEVIATIONS]]
        df = df[df["deviation"].isin(keep)]
        df.sort_values(["series", "deviation"], inplace=True)
    return df

def plot_summary(df):
    if df.empty:
        print("No valid runs found.")
        return

    x_all = sorted(df["deviation"].unique())
    real_name = "Real"
    gazebo_series = [s for s in sorted(df["series"].unique()) if s != real_name]

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Plot REAL ---
    if real_name in df["series"].unique():
        sub = df[df["series"] == real_name]
        y, fail = [], []
        for x in x_all:
            row = sub[sub["deviation"] == x]
            if row.empty:
                y.append(np.nan)
                fail.append(False)
            else:
                y.append(float(row["score"].values[0]))
                fail.append(bool(row["failed"].values[0]))

        plot_style = "-o" if SHOW_LINES else "o"
        ax.plot(x_all, y, plot_style, linewidth=3 if SHOW_LINES else 0,
                color="black", markerfacecolor="white", markersize=6, label="Real")
        for xi, yi, f in zip(x_all, y, fail):
            if np.isfinite(yi) and f:
                ax.plot(xi, yi, "x", color="black", markersize=9, mew=2)

    # --- Plot Gazebo series ---
    import matplotlib.colors as mcolors
    base_color = np.array(mcolors.to_rgb("#2ca02c"))
    n_gz = max(1, len(gazebo_series))
    colors = [
        tuple(base_color * (0.4 + 0.6 * (i / max(1, n_gz - 1))))
        for i in range(n_gz)
    ]

    for idx, series in enumerate(gazebo_series):
        sub = df[df["series"] == series]
        y, fail = [], []
        for x in x_all:
            row = sub[sub["deviation"] == x]
            if row.empty:
                y.append(np.nan)
                fail.append(False)
            else:
                y.append(float(row["score"].values[0]))
                fail.append(bool(row["failed"].values[0]))

        color = colors[idx]
        plot_style = "-" if SHOW_LINES else "o"
        ax.plot(x_all, y, plot_style, linewidth=1.5 if SHOW_LINES else 0,
                alpha=0.4, color=color, marker="o", markersize=4,
                markerfacecolor="none", label="_nolegend_")
        for xi, yi, f in zip(x_all, y, fail):
            if np.isfinite(yi) and f:
                ax.plot(xi, yi, "x", color="red", markersize=7, mew=1.8)

    ax.set_title(TITLE)
    ax.set_xlabel("Gain deviation from defaults [%]  (negative = lower gains)")
    ax.set_ylabel("Performance (∫ Σ |q_err| dt)  [rad·s]")
    ax.grid(True, which="both", alpha=0.3)

    from matplotlib.lines import Line2D
    gz_proxy = Line2D([0], [0], color="#2ca02c", lw=2, alpha=0.7)
    real_proxy = Line2D([0], [0], color="black", lw=3)
    ax.legend([real_proxy, gz_proxy], ["Real", "Gazebo friction sets"], loc="best")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"Saved plots to {OUT_PNG} and {OUT_PDF}")

def main():
    df = collect_runs()
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved metrics to {OUT_CSV}")
    if not df.empty:
        cols = ["file", "series", "deviation", "score", "failed", "max_abs_err", "duration"]
        print(df[cols].to_string(index=False))
    plot_summary(df)

if __name__ == "__main__":
    main()
