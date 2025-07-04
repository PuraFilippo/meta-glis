import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from glis.solvers import GLISp
from matlab.obj_PID_panda import obj_PID_panda
from matlab.panda_robot import panda_robot
from dataset import RobotDataset
from models import Autoencoder


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.rcParams['axes.grid']=True
plt.rcParams['axes.xmargin']=0

np.random.seed(42)

ts = 1e-3
Tsim = 4.0
time = np.arange(0, Tsim, ts)
n_DoFs = 7
friction = np.array([2]*n_DoFs)
masses = np.array([1, 0, 3, 0, 5, 0, 2.5])

randomized_masses = [
    m + (np.random.uniform(-1, 1)) if m != 0 else m + (np.random.uniform(0, 1)) for m in masses
]
randomized_masses[-1] = masses[-1] + np.random.uniform(-1, 3)

Robot = panda_robot(randomized_masses)

A = 15*np.pi/180
f = 1
q_0 = np.array([-0.7160, -0.5850, 0.3504, -1.5666, 0.2241, -2.1201, -2.8398])
q_r = [q_0 - A]
dq_r = [np.zeros(n_DoFs)]
ddq_r = [np.zeros(n_DoFs)]

for t in time:
    q_r.append(q_0 + A * np.sin(2 * np.pi * f * t - np.pi / 2))
    dq_r.append((q_r[-1] - q_r[-2])/ts)
    ddq_r.append((dq_r[-1] - dq_r[-2])/ts)

r = dict(q_r=np.array(q_r), dq_r=np.array(dq_r), ddq_r=np.array(ddq_r))
const = dict(Ts=ts, Tsim=Tsim, time=time, n_DoFs=n_DoFs,
             r=r, Robot=Robot, Robot_friction=friction, q_0=q_0,
             toll_qerr=10*np.pi/180, masses=randomized_masses)


def plot_params(outs, targets):
    clip = 2
    tgt = targets.ravel()
    tgt = np.clip(tgt, None, clip)

    fig, axes = plt.subplots(nrows=3, ncols=6,
                             figsize=(18, 9),
                             sharex=False, sharey=False)

    groups = ['Kd', 'Ki', 'Kp']
    offsets = [0, 7, 14]
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
                            cmap='viridis',
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
                 label=f'objective value (>= {str(clip)} shown as {str(clip)})')

    plt.show()

def pref_fun(x, xbest, plot=False):
    def to_param_dict(param_array):
        param_dict = {}
        groups = ['Kd', 'Ki', 'Kp']
        idx = 0
        for group in groups:
            for i in range(1, 8):  # from 1 to 7
                key = f"{group}{i}"
                param_dict[key] = param_array[idx]
                idx += 1
        return param_dict

    cost_best, q_msr_best = obj_PID_panda(to_param_dict(xbest), const, return_trace=True)
    cost, q_msr = obj_PID_panda(to_param_dict(x), const, return_trace=True)
    q_r = r['q_r']

    q_r = q_r[:len(time)]
    q_msr_best = q_msr_best[:len(time)]
    q_msr = q_msr[:len(time)]

    if plot:
        fig, axs = plt.subplots(n_DoFs, 1, figsize=(10, 2.5 * n_DoFs), sharex=True)
        for i in range(n_DoFs):
            axs[i].plot(q_msr_best[:, i], c='green', alpha=0.9)
            axs[i].plot(q_msr[:, i], c='blue', alpha=0.9)
            axs[i].plot(q_r[:, i], c='black', label=f'q_r{i + 1}')
            axs[i].set_ylabel("Angle [rad]")
            axs[i].set_title(f"Joint {i + 1}")
            axs[i].legend()
            axs[i].grid(True)
        plt.show()

    return -1 if cost < cost_best else 1, cost, q_msr, cost_best, q_msr_best


def main():
    """
    Main function for testing the model and evaluating optimization performance.
    """
    # Configuration
    input_dim = 21
    latent_dim = 10
    alpha = 0
    batch_size = 128

    home_dir = os.path.expanduser('~')
    model_dir = '../out/robot'
    model_path = os.path.join(model_dir, f'model_{input_dim}d_{latent_dim}l_300iter_exp_decay_alpha_{alpha:02}.pt')

    # Choose device
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_bounds = (
            [(0, 500)] * 7 +
            [(0, 200)] * 7 +
            [(0, 15000)] * 7
    )

    # Load the trained Autoencoder model
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, out_bounds=x_bounds)

    try:
        checkpoint = torch.load(model_path, weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        autoencoder.load_state_dict(state_dict)
    except:
        "Could not load pretrained model"
        try:
            state_dict = torch.load(model_path, weights_only=True)
            autoencoder.load_state_dict(state_dict)
        except:
            raise Exception("Could not load pretrained model")

    autoencoder.to(device)
    autoencoder.eval()

    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    encoded_bounds = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    template = np.array([-1, 0, -1, 1, 1, 1, -1, -1, -1, 1])
    idx_glisp = template < 0

    lb = np.zeros((len(template[idx_glisp]), 1)).flatten()
    ub = lb.copy()
    for i in range(0, len(lb)):
        lb[i] = 0 # x_bounds[i][0]
        ub[i] = 1 # x_bounds[i][1]

    prob = GLISp(bounds=(lb, ub), n_initial_random=5)  # initialize GLISp object
    xbest_encoded, x_encoded = prob.initialize()

    xbest_full_encoded = template.copy().astype(float)
    x_full_encoded = template.copy().astype(float)
    xbest_full_encoded[idx_glisp] = xbest_encoded
    x_full_encoded[idx_glisp] = x_encoded

    x_arr = []
    costs = []
    q_mesures = []
    with torch.no_grad():
        xbest = decoder(torch.tensor(xbest_full_encoded).float().to(device)).cpu().numpy()
        x = decoder(torch.tensor(x_full_encoded).float().to(device)).cpu().numpy()
        initial_x = x.copy()
        for k in range(10):
            pref, cost, q_msr, _, _ = pref_fun(x, xbest)  # evaluate preference

            x_arr.append(x)
            costs.append(cost)
            q_mesures.append(q_msr)

            x_encoded = prob.update(pref)
            xbest_encoded = prob.xbest

            xbest_full_encoded = template.copy().astype(float)
            x_full_encoded = template.copy().astype(float)
            xbest_full_encoded[idx_glisp] = xbest_encoded
            x_full_encoded[idx_glisp] = x_encoded
            xbest = decoder(torch.tensor(xbest_full_encoded).float().to(device)).cpu().numpy()
            x = decoder(torch.tensor(x_full_encoded).float().to(device)).cpu().numpy()

        xopt = xbest

    _, cost, _, cost_best, _ = pref_fun(initial_x, xopt, plot=False)
    print(cost, cost_best)
    plot_params(np.array(x_arr), np.array(costs))

    q_r = r['q_r']
    q_r = q_r[:len(time)]
    q_mesures = np.array(q_mesures)

    fig, axs = plt.subplots(n_DoFs, 1, figsize=(10, 2.5 * n_DoFs), sharex=True)
    for i in range(n_DoFs):
        for j in range(len(q_mesures)):
            axs[i].plot(q_mesures[j, :, i], c='black', alpha=0.3)
        axs[i].plot(q_r[:, i], c='r', label=f'q_r{i + 1}')
        axs[i].set_ylabel("Angle [rad]")
        axs[i].set_title(f"Joint {i + 1}")
        axs[i].legend()
        axs[i].grid(True)
    plt.show()

if __name__ == "__main__":
    main()
