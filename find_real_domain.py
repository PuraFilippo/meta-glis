from skopt import dump, load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

opt = load('skopt_optimizer.pkl')

# ---- 1) Extract from skopt Optimizer ----
# Replace `opt` with your Optimizer object
Xi = np.array(opt.Xi, dtype=float)  # shape (n_runs, 21)
yi = np.array(opt.yi, dtype=float)  # shape (n_runs,)

# New column names: Kp1–7, Ki1–7, Kd1–7
Kp_cols = [f"Kp{i}" for i in range(1, 8)]
Ki_cols = [f"Ki{i}" for i in range(1, 8)]
Kd_cols = [f"Kd{i}" for i in range(1, 8)]
Kcols = Kp_cols + Ki_cols + Kd_cols

# DataFrame
df = pd.DataFrame(Xi, columns=Kcols)
df["error"] = yi

# ---- 2) Identify feasible vs infeasible ----
feasible = ~np.isclose(df["error"].values, 1.0, atol=1e-12)
df["feasible"] = feasible

# ---- 3) Summarize feasible domain ----
def summarize_feasible_domain(frame, cols):
    out = []
    f = frame[frame["feasible"]]
    for c in cols:
        vals = f[c].values
        if len(vals) == 0:
            out.append((c, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            out.append((
                c,
                np.min(vals),
                np.percentile(vals, 5),
                np.median(vals),
                np.percentile(vals, 95),
                np.max(vals),
            ))
    summ = pd.DataFrame(out, columns=["Gain", "min", "p05", "median", "p95", "max"])
    return summ

summary = summarize_feasible_domain(df, Kcols)
print("Feasible-domain summary (min / 5th / median / 95th / max):")
print(summary.to_string(index=False))

# ---- 4) Visualizations ----
groups = {
    "Kp Gains": Kp_cols,
    "Ki Gains": Ki_cols,
    "Kd Gains": Kd_cols,
}

# (A) Histograms of feasible values
def hist_grid_for_group(frame, cols, title):
    f = frame[frame["feasible"]]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2*len(cols), 3))
    if len(cols) == 1:
        axes = [axes]
    for ax, c in zip(axes, cols):
        vals = f[c].values
        if len(vals):
            ax.hist(vals, bins=20)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    fig.suptitle(f"{title} (Feasible Only)", fontsize=12, y=1.05)
    plt.tight_layout()
    plt.show()

for title, cols in groups.items():
    hist_grid_for_group(df, cols, title)

# (B) Scatter plots: gain vs error (feasible vs infeasible)
def scatter_error_per_K(frame, cols, title):
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2*len(cols), 3.4))
    if len(cols) == 1:
        axes = [axes]
    for ax, c in zip(axes, cols):
        x = frame[c].values
        y = frame["error"].values
        m_feas = frame["feasible"].values
        ax.scatter(x[m_feas], y[m_feas], s=16, label="Feasible", alpha=0.8)
        ax.scatter(x[~m_feas], y[~m_feas], s=16, marker="x", label="Infeasible", alpha=0.8)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("Gain Value")
        ax.set_ylabel("Error")
        ax.set_yscale("log")  # show low feasible errors clearly
        ax.grid(True, which="both", ls=":")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"{title}: Gain vs Error (log-scale y)", fontsize=12, y=1.05)
    plt.tight_layout()
    plt.show()

for title, cols in groups.items():
    scatter_error_per_K(df, cols, title)


