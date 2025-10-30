import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.rcParams['axes.grid']=True
plt.rcParams['axes.xmargin']=0

n_DoFs = 7
loc = '../data/robot/'
ext_f = '_friction-high'
ext_compare = '_friction'  # Set to None if you don't want to compare

# --- Load main experiment ---
q_r = np.load(loc + 'q_r' + ext_f + '.npy')
q_measured = np.load(loc + 'q_measured' + ext_f + '.npy')
runs_outputs = np.load(loc + 'runs_outputs' + ext_f + '.npy')
runs_targets = np.load(loc + 'runs_targets' + ext_f + '.npy')
robot_masses = np.load(loc + 'robot_masses' + ext_f + '.npy')

# --- Load comparison data if provided ---
if ext_compare:
    runs_targets_cmp = np.load(loc + 'runs_targets' + ext_compare + '.npy')

# --- Best params extraction (only for main experiment) ---
best_params = []
for q_m, r_o, r_t in zip(q_measured, runs_outputs, runs_targets):
    best_idx = np.argmin(r_t) if 'glis' in ext_f else np.argmax(r_t)
    best_target = r_t[best_idx]
    best_output = r_o[best_idx]
    best_params.append((best_target, best_output))

best_params_tgt = np.array([x[0] for x in best_params])
mean_best_target = np.mean(best_params_tgt)
std_best_target = np.std(best_params_tgt)
best_params_out = np.array([x[1] for x in best_params])

clip = -2
def plot_outs(outs, targets):
    tgt = targets.ravel()
    tgt = np.clip(tgt, clip, None)

    fig, axes = plt.subplots(nrows=3, ncols=6,
                             figsize=(18, 9),
                             sharex=False, sharey=False)

    groups = ['Kd', 'Ki', 'Kp']
    offsets = [0, 7, 14]

    # Original bounds
    # bounds = {
    #     'Kd': (0, 50),
    #     'Ki': (0, 500),
    #     'Kp': (0, 400)
    # }

    bounds = {
        'Kd': (0, 500),
        'Ki': (0, 200),
        'Kp': (0, 15000)
    }

    def add_margin(lim, margin=0.05):
        lower, upper = lim
        range_ = upper - lower
        return (lower - margin * range_, upper + margin * range_)

    limits = {k: add_margin(v) for k, v in bounds.items()}

    for row, (group, base) in enumerate(zip(groups, offsets)):
        for col in range(6):
            i, j = base + col, base + col + 1
            ax = axes[row, col]

            x = outs[..., i].ravel()
            y = outs[..., j].ravel()

            sc = ax.scatter(x, y, c=tgt,
                            cmap='plasma_r',
                            s=20, alpha=0.3)

            ax.set_title(f'{group}{col + 1} vs {group}{col + 2}', fontsize=9)
            if col == 0:
                ax.set_ylabel(group)
            if row == 2:
                ax.set_xlabel(f'{group}{col + 1}')
            ax.grid(True, lw=0.3, alpha=0.4)

            # Set axis limits with margin
            ax.set_xlim(limits[group])
            ax.set_ylim(limits[group])

    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cax,
                 label=f'objective value -Log(J)')

    plt.show()


# --- Incumbent transformation and statistics ---
runs_targets_clipped = -np.clip(runs_targets, clip, None)
runs_targets_clipped = np.minimum.accumulate(runs_targets_clipped, axis=1)
mean_target = np.mean(runs_targets_clipped, axis=0)
std_target = np.std(runs_targets_clipped, axis=0)

if ext_compare:
    runs_targets_cmp_clipped = -np.clip(runs_targets_cmp, clip, None)
    runs_targets_cmp_clipped = np.minimum.accumulate(runs_targets_cmp_clipped, axis=1)
    mean_target_cmp = np.mean(runs_targets_cmp_clipped, axis=0)
    std_target_cmp = np.std(runs_targets_cmp_clipped, axis=0)

# --- Plot mean ± std (with optional comparison) ---
n_iterations = runs_targets_clipped.shape[1]
iterations = np.arange(n_iterations)

plt.figure(figsize=(12, 7))
plt.plot(iterations, mean_target, color='blue', label='Baseline Mean Incumbent')
plt.fill_between(iterations, mean_target - std_target, mean_target + std_target,
                 color='blue', alpha=0.3, label='Baseline ±1 Std Dev')

if ext_compare:
    n_iterations_ext = runs_targets_cmp_clipped.shape[1]
    iterations_ext = np.arange(n_iterations_ext)
    plt.plot(iterations_ext, mean_target_cmp, color='green', label='Latent Mean Incumbent')
    plt.fill_between(iterations_ext, mean_target_cmp - std_target_cmp, mean_target_cmp + std_target_cmp,
                     color='green', alpha=0.3, label='Latent ±1 Std Dev')


plt.xlabel("Iteration")
plt.ylabel("Objective value (log J)")
plt.title("Optimization Progress (Mean Incumbent ± Std Dev)")
plt.grid(True)
plt.legend()
plt.show()

# --- Plot all incumbents across experiments ---
n_experiments = runs_targets_clipped.shape[0]

plt.figure(figsize=(12, 7))

for i in range(n_experiments):
    plt.plot(range(n_iterations), runs_targets_clipped[i], color='tab:blue', alpha=0.2, lw=2)

if ext_compare:
    for i in range(runs_targets_cmp_clipped.shape[0]):
        plt.plot(range(n_iterations_ext), runs_targets_cmp_clipped[i], color='tab:green', alpha=0.2, lw=2)

plt.xlabel("Iteration")
plt.ylabel("Objective value (Log J)")
plt.title("Incumbent Optimization Progress Across Experiments")
plt.grid(True)

# Legend placeholders
plt.plot([], [], color='tab:blue', label='Baseline')
if ext_compare:
    plt.plot([], [], color='tab:green', label='Latent Space')

plt.legend()
plt.show()

#plot_outs(runs_outputs, runs_targets)

plot_outs(best_params_out, best_params_tgt)

fig, axs = plt.subplots(n_DoFs, 1, figsize=(10, 2.5 * n_DoFs), sharex=True)
for i in range(n_DoFs):
    for j in range(len(q_measured)):
        axs[i].plot(np.degrees(q_r[:, i] - q_measured[j, :, i]), c='black', alpha=0.3)
    # axs[i].plot(q_r[:, i], c='r', label=f'q_r{i+1}')
    axs[i].set_ylabel("Angle [deg]")
    axs[i].set_title(f"Joint {i+1}")
    axs[i].legend()
    axs[i].grid(True)
plt.show()


