import time
import numpy as np
import numba as nb
import torch
import os
import sys
sys.path.append('/home/fpura/projects/meta-glis/')
from panda_robot import panda_robot
from obj_PID_panda import obj_PID_panda
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from glis.solvers import GLIS
from robot.models import Autoencoder
from multiprocessing import Pool


def to_robot_dict(x):
    kd_values = x[0:7]
    ki_values = x[7:14]
    kp_values = x[14:21]

    x_dict = {}
    for i in range(7):
        x_dict[f"Kp{i + 1}"] = kp_values[i]
        x_dict[f"Ki{i + 1}"] = ki_values[i]
        x_dict[f"Kd{i + 1}"] = kd_values[i]

    return x_dict


@torch.no_grad()
def run_single_experiment(i, seed=134566):
    np.random.seed(seed + i)

    latent_space = False
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


    # Configuration
    input_dim = 21
    latent_dim = 10
    alpha = 0

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
        checkpoint = torch.load(model_path, map_location=device)  # <- maps to CPU
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        autoencoder.load_state_dict(state_dict)
    except Exception as e:
        raise Exception(f"Could not load pretrained model: {e}")

    autoencoder.to(device)
    autoencoder.eval()

    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    if latent_space:
        bounds = {}
        for i in range(10):
            bounds[f"Lat{i}"] = (0, 1)
    else:
        bounds = {}
        for j in range(1, 8):
            bounds[f"Kp{j}"] = (0, 15000)
            bounds[f"Ki{j}"] = (0, 200)
            bounds[f"Kd{j}"] = (0, 500)


    def black_box_function(**kwargs):
        if latent_space:
            x_encoded = [*kwargs.values()]
            x = to_robot_dict(decoder(torch.tensor(x_encoded).float().to(device)).cpu().numpy())

            return -np.log(obj_PID_panda(x, const))

        return -np.log(obj_PID_panda(kwargs, const))

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=bounds,
        verbose=2,
        random_state=seed + i,
    )

    x0 = [0., 0., 23.20590318, 0., 0., 26.04577232, 0., 17.2236932, 189.31349122, 50.36133633, 32.21360127, 0., 0.,
          145.51627347, 9241.47318728, 13002.09987407, 12296.04897835, 10326.88976112, 7350.71219842, 8132.5718691,
          10527.77205599]
    # TODO: encoder into decoder

    if latent_space:
        x0_encoded = encoder(torch.tensor(x0).float().to(device)).cpu().numpy()

        x0_dict = {}
        for i in range(10):
            x0_dict[f"Lat{i}"] = x0_encoded[i]

    else:
        # Split x0 into kd, ki, kp
        kd_values = x0[0:7]
        ki_values = x0[7:14]
        kp_values = x0[14:21]

        # Build param dictionary in order: Kp1, Ki1, Kd1, Kp2, ...
        x0_dict = {}
        for i in range(7):
            x0_dict[f"Kp{i + 1}"] = kp_values[i]
            x0_dict[f"Ki{i + 1}"] = ki_values[i]
            x0_dict[f"Kd{i + 1}"] = kd_values[i]

    # optimizer.probe(params=x0_dict, lazy=True)

    optimizer.maximize(
        init_points=1,
        n_iter=0,
        acquisition_function=UtilityFunction(kind='ei')
    )

    best_params = optimizer.max['params']
    if latent_space:
        best_params = to_robot_dict(decoder(torch.tensor([*best_params.values()]).float().to(device)).cpu().numpy())

    iters = optimizer.res
    outs = [list(it['params'].values()) for it in iters]
    if latent_space:
        outs = decoder(torch.tensor(outs).float().to(device)).cpu().numpy()

    targets = [it['target'] for it in iters]

    cost, q_msr = obj_PID_panda(best_params, const, return_trace=True)
    q_r = r['q_r']

    q_r = q_r[:len(time)]
    q_msr = q_msr[:len(time)]

    t = const['time']
    q_err = q_r - q_msr

    return dict(
        experiment=i,
        best_params=best_params,
        outs=outs,
        targets=targets,
        q_r=q_r,
        q_measured=q_msr,
        masses=randomized_masses
    )

# Parallel execution
if __name__ == '__main__':
    n_experiments = 100
    max_concurrent_processes = 10

    start = time.time()
    with Pool(processes=max_concurrent_processes) as pool:
        results = pool.map(run_single_experiment, range(n_experiments))

    runs_outputs = []
    runs_targets = []
    q_measured = []
    robot_masses = []
    # Results is a list of dicts for each experiment
    for res in results:
        runs_outputs.append(res['outs'])
        runs_targets.append(res['targets'])
        q_measured.append(res['q_measured'])
        robot_masses.append(res['masses'])
        print(f"Experiment {res['experiment']} - Best Params: {res['best_params']}")

    q_r = np.array(results[0]['q_r'])
    q_measured = np.array(q_measured)
    runs_outputs = np.array(runs_outputs)
    runs_targets = np.array(runs_targets)
    robot_masses = np.array(robot_masses)
    print("--- %s seconds ---" % (time.time() - start))
    np.save('q_r_test.npy', q_r)
    np.save('q_measured_test.npy', q_measured)
    np.save('runs_outputs_test.npy', runs_outputs)
    np.save('runs_targets_test.npy', runs_targets)
    np.save('robot_masses_test.npy', robot_masses)
