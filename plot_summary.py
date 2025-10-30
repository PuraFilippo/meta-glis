import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("summary.csv")

# Define column groups
kp_cols = [f"K{i}" for i in range(0, 7)]
ki_cols = [f"K{i}" for i in range(7, 14)]
kd_cols = [f"K{i}" for i in range(14, 21)]

# Helper to plot scatter
def scatter_group(df, cols, title, color):
    plt.figure(figsize=(10, 6))
    for c in cols:
        plt.scatter(df["run_id"], df[c], label=c, alpha=0.7, s=40)
    plt.title(f"{title} Gains per Run")
    plt.xlabel("Run ID")
    plt.ylabel("Gain Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot each group separately
scatter_group(df, kp_cols, "Kp", "tab:blue")
scatter_group(df, ki_cols, "Ki", "tab:green")
scatter_group(df, kd_cols, "Kd", "tab:red")