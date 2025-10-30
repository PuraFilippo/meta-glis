#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------
# USER CONFIGURATION
# ----------------------------
GAZEBO_LOG = "robot_experiments/Ki50/gazebo+00.csv"
REAL_LOG   = "robot_experiments/Ki50/real+00.csv"
# GAZEBO_LOG = "custom_pid_log_default_gazebo.csv"
# REAL_LOG   = "custom_pid_log_default.csv"

OUTDIR = "out/"
TITLE = "Gazebo vs Real PID Comparison"
DT = 0.001  # seconds (1 kHz interpolation)
ERR_THRESH = 2.8*np.pi/180  # 3 degrees in radians
THRESH_MARGIN = 0.0         # set >0 (e.g., 0.002) if you want a slightly easier crossing
# ----------------------------

JOINTS = list(range(7))
PREFIX_ORDER = [
    ("q_d",      "Desired position"),
    ("q",        "Measured position"),
    ("q_err",    "Position error"),
    ("dq_err",   "Velocity error"),
    ("int_err",  "Integral error"),
    ("coriolis", "Coriolis"),
    ("tau_cmd",  "Commanded torque (pre-sat)"),
    # ("tau_J_d",  "Last commanded torque (robot state)"),
]

def find_cols(df, prefix):
    cols = []
    for j in JOINTS:
        name = f"{prefix}{j}"
        if name in df.columns:
            cols.append(name)
        else:
            matches = [c for c in df.columns if c.strip().lower() == name.lower()]
            if matches:
                cols.append(matches[0])
            else:
                raise KeyError(f"Column {name} not found in CSV. Available: {list(df.columns)}")
    return cols

def load_and_prepare(path):
    df = pd.read_csv(path)
    if "time" not in df.columns:
        for c in df.columns:
            if c.strip().lower() == "time":
                df.rename(columns={c: "time"}, inplace=True)
                break
    df["time"] = df["time"].astype(float)
    df.sort_values("time", inplace=True)
    df["time"] -= df["time"].iloc[0]
    return df

def interp_df(df, t, columns):
    out = pd.DataFrame({"time": t})
    for c in columns:
        out[c] = np.interp(t, df["time"], df[c], left=np.nan, right=np.nan)
    return out.dropna()

def compute_metrics(gz, real, prefix):
    rows = []
    for j in JOINTS:
        c = f"{prefix}{j}"
        if c not in gz.columns or c not in real.columns:
            continue
        diff = real[c].values - gz[c].values
        rmse = np.sqrt(np.mean(diff**2))
        mean = np.mean(diff)
        rows.append({"signal": prefix, "joint": j, "rmse": rmse, "mean": mean})
    return rows

def first_crossing(t, qerr_mat, thr, margin=0.0):
    """
    Return (t_cross, idx) where any |qerr| >= (thr - margin) first occurs.
    If never crosses or qerr_mat is None, return (None, None).
    """
    if qerr_mat is None:
        return None, None
    thresh = max(thr - margin, 0.0)
    max_abs = np.max(np.abs(qerr_mat), axis=1)  # [N]
    mask = max_abs >= thresh
    if not np.any(mask):
        return None, None
    idx = int(np.argmax(mask))
    return t[idx], idx

def end_time(t):
    return t[-1] if t is not None and len(t) else None

def maybe_plot(axs, t, sig_gz, sig_rl, title):
    if sig_gz is None and sig_rl is None:
        return
    axs[0].set_title(title)
    for j in range(7):
        if sig_gz is not None:
            axs[j].plot(t['gazebo'], sig_gz[:, j], alpha=0.8, color="blue", label="Gazebo")
        if sig_rl is not None:
            axs[j].plot(t['real'], sig_rl[:, j], alpha=0.8, color="orange", label="Real")
        axs[j].set_ylabel(f"J{j}")
        axs[j].grid(True)
        if j == 6:
            axs[j].set_xlabel("Time [s]")
    axs[0].legend()

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    df_gz   = load_and_prepare(GAZEBO_LOG)
    df_real = load_and_prepare(REAL_LOG)

    # Common time vector
    tmin = max(df_gz["time"].min(), df_real["time"].min())
    tmax = min(df_gz["time"].max(), df_real["time"].max())
    t_common = np.arange(tmin, tmax, DT)

    # Build list of common columns
    cols = ["time"]
    for p, _ in PREFIX_ORDER:
        try:
            gz_cols = find_cols(df_gz, p)
            real_cols = find_cols(df_real, p)
            common = sorted(list(set(gz_cols) & set(real_cols)))
            cols.extend(common)
        except Exception:
            pass
    cols = list(dict.fromkeys(cols))  # dedup

    gz_i = interp_df(df_gz, t_common, cols)
    real_i = interp_df(df_real, t_common, cols)

    n = min(len(gz_i), len(real_i))
    gz_i, real_i = gz_i.iloc[:n], real_i.iloc[:n]
    t = gz_i["time"].values

    # Prepare q_err matrices (prefer logged q_err*, else compute q_d - q if available)
    def get_mat(df, prefix):
        try:
            cols = [f"{prefix}{j}" for j in JOINTS]
            if all(c in df.columns for c in cols):
                return np.column_stack([df[c].values for c in cols])
        except Exception:
            pass
        return None

    gz_qerr = get_mat(gz_i, "q_err")
    rl_qerr = get_mat(real_i, "q_err")

    # Fallback if q_err not present: compute from q_d - q
    if gz_qerr is None:
        qd_cols = [f"q_d{j}" for j in JOINTS]
        q_cols  = [f"q{j}" for j in JOINTS]
        if all(c in gz_i.columns for c in qd_cols) and all(c in gz_i.columns for c in q_cols):
            gz_qerr = np.column_stack([gz_i[qd_cols[j]].values - gz_i[q_cols[j]].values for j in JOINTS])
    if rl_qerr is None:
        qd_cols = [f"q_d{j}" for j in JOINTS]
        q_cols  = [f"q{j}" for j in JOINTS]
        if all(c in real_i.columns for c in qd_cols) and all(c in real_i.columns for c in q_cols):
            rl_qerr = np.column_stack([real_i[qd_cols[j]].values - real_i[q_cols[j]].values for j in JOINTS])

    # Find first crossings
    gz_cross_t, _ = first_crossing(t, gz_qerr, ERR_THRESH, THRESH_MARGIN)
    rl_cross_t, _ = first_crossing(t, rl_qerr, ERR_THRESH, THRESH_MARGIN)

    # End times
    gz_end = end_time(t)
    rl_end = end_time(t)

    pdf_path = os.path.join(OUTDIR, "gazebo_real_compare.pdf")
    metrics_path = os.path.join(OUTDIR, "summary_metrics.csv")

    def add_vlines_all(axs, gz_t, rl_t):
        for ax in axs:
            if gz_t is not None:
                ax.axvline(gz_t, color="blue", linestyle="--", linewidth=0.9)
            if rl_t is not None:
                ax.axvline(rl_t, color="orange", linestyle="--", linewidth=0.9)

    with PdfPages(pdf_path) as pdf:
        # Plot each signal family with vertical bars on every subplot
        for prefix, label in PREFIX_ORDER:
            needed = [f"{prefix}{j}" for j in JOINTS]
            if not all(c in gz_i.columns and c in real_i.columns for c in needed):
                continue

            # assemble matrices for plotting
            gz_mat = np.column_stack([gz_i[f"{prefix}{j}"].values for j in JOINTS])
            rl_mat = np.column_stack([real_i[f"{prefix}{j}"].values for j in JOINTS])

            fig, axes = plt.subplots(7, 1, figsize=(10, 16), sharex=True)
            fig.suptitle(f"{TITLE} — {label}")

            # plot lines
            Tmap = {'gazebo': t, 'real': t}  # both aligned
            maybe_plot(axes, Tmap, gz_mat, rl_mat, label)

            # vertical markers on ALL subplots
            add_vlines_all(axes, gz_cross_t, rl_cross_t)

            # make legend appear only once (top axis) and be clean
            handles, leglabels = axes[0].get_legend_handles_labels()
            # ensure both exceed labels appear if present
            axes[0].legend(handles, leglabels)

            axes[-1].set_xlabel("Time [s]")
            pdf.savefig(fig); plt.close(fig)

    # Metrics CSV
    all_rows = []
    for prefix, _ in PREFIX_ORDER:
        all_rows.extend(compute_metrics(gz_i, real_i, prefix))
    pd.DataFrame(all_rows).to_csv(metrics_path, index=False)

    print(f"Saved plots to {pdf_path}")
    print(f"Saved summary metrics to {metrics_path}")
    print(f"First exceed (>= {ERR_THRESH - THRESH_MARGIN:.6f} rad): Gazebo @ {gz_cross_t if gz_cross_t is not None else '—'}, Real @ {rl_cross_t if rl_cross_t is not None else '—'}")
    print(f"Common time window: 0 … {min(gz_end, rl_end):.3f}s")

if __name__ == "__main__":
    main()
