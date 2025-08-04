import numpy as np
from panda_robot import panda_robot
from obj_PID_panda import obj_PID_panda
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from glis.solvers import GLIS
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import numba as nb
import imageio
import sys
import os


seed = 42
np.random.seed(seed)

bo = True
n_experiments = 5

# Constants
ts = 1e-3
Tsim = 4.0
time = np.arange(0, Tsim, ts)
n_DoFs = 7
friction = np.array([2]*n_DoFs)
masses = np.array([1, 0, 3, 0, 5, 0, 2.5])
# variation_factor = 0.3  # ±30%

q_measured = []
runs_outputs = []
runs_targets = []
for i in range(n_experiments):

    randomized_masses = np.array([
        m + (np.random.uniform(-0.5, 0.5)) if m != 0 else m + (np.random.uniform(0, 0.5)) for m in masses
    ])

    print("Original masses:     ", masses)
    print("Randomized masses:   ", randomized_masses.round(3))

    rm = [randomized_masses[0]]
    rm.extend(list(masses[1:]))
    Robot = panda_robot(randomized_masses)

    # R_0 = np.diag([-1, 1, 1]) #np.eye(3)
    # x_0 = np.array([0.3, 0.0, 0.5])
    # x_f = np.array([0.4, 0.0, 0.6])
    # dx_0 = dx_f = ddx_0 = ddx_f = np.zeros(3)
    #
    # T_0 = np.eye(4)
    # T_0[:3, :3] = R_0
    # T_0[:3, 3] = x_0

    # q_guess = np.array([-0.7160, -0.5850, 0.3504, -1.5666, 0.2241, -2.1201, -2.8398])
    # q_0 = Robot.ikine_LM(T_0, q0=q_guess).q
    #
    # t0 = 0
    # tf = 2.
    #
    # # Trajectory interpolation
    # C = np.vstack([
    #     [1, t, t**2, t**3, t**4, t**5] for t in [t0, tf]
    # ] + [
    #     [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4] for t in [t0, tf]
    # ] + [
    #     [0, 0, 2, 6*t, 12*t**2, 20*t**3] for t in [t0, tf]
    # ])
    #
    # xM = np.vstack([x_0, x_f, dx_0, dx_f, ddx_0, ddx_f])
    # a = np.linalg.solve(C, xM)
    #
    # x_r = []
    # dx_r = []
    # ddx_r = []
    # q_r = [q_0]
    # dq_r = [np.zeros(n_DoFs)]
    # ddq_r = [np.zeros(n_DoFs)]
    #
    # for t in time:
    #     if t <= tf:
    #         xt = np.array([np.polyval(a[::-1, i], t) for i in range(3)])
    #         dxt = np.array([np.polyval(np.polyder(a[::-1, i]), t) for i in range(3)])
    #         ddxt = np.array([np.polyval(np.polyder(a[::-1, i], 2), t) for i in range(3)])
    #         x_r.append(xt)
    #         dx_r.append(dxt)
    #         ddx_r.append(ddxt)
    #
    #         T_i = np.eye(4)
    #         T_i[:3, :3] = R_0
    #         T_i[:3, 3] = xt
    #
    #         q_i = Robot.ikine_LM(T_i, q0=q_r[-1]).q
    #         q_r.append(q_i)
    #         dq_r.append((q_i - q_r[-2])/ts)
    #         ddq_r.append((dq_r[-1] - dq_r[-2])/ts)
    #     else:
    #         x_r.append(x_r[-1])
    #         dx_r.append(np.zeros(3))
    #         ddx_r.append(np.zeros(3))
    #         q_r.append(q_r[-1])
    #         dq_r.append(np.zeros(n_DoFs))
    #         ddq_r.append(np.zeros(n_DoFs))

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

    # Robot.plot(np.array(q_r), dt=ts, loop=True)

    # Assemble data for objective
    r = dict(q_r=np.array(q_r), dq_r=np.array(dq_r), ddq_r=np.array(ddq_r))
    const = dict(Ts=ts, Tsim=Tsim, time=time, n_DoFs=n_DoFs,
                 r=r, Robot=Robot, Robot_friction=friction, q_0=q_0, toll_qerr=10*np.pi/180, masses=randomized_masses)

    outs = []
    targets = []

    if bo:
        # Define bounds for Bayesian optimization
        bounds = {}
        for i in range(1, 8):
            bounds[f"Kp{i}"] = (0, 400)
            bounds[f"Ki{i}"] = (0, 500)
            bounds[f"Kd{i}"] = (0, 50)

        def black_box_function(**kwargs):
            return -np.log(obj_PID_panda(kwargs, const))  # Minimize cost

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=bounds,
            verbose=2,
            random_state=np.random.randint(10000000),
        )

        optimizer.maximize(
            init_points=10,
            n_iter=1490,
            acquisition_function=UtilityFunction(kind='ei')
        )

        print("Best Parameters:", optimizer.max['params'])
        iters = optimizer.res
        best_params = optimizer.max['params']

        for it in iters:
            outs.append(list(it['params'].values()))
            targets.append(it['target'])

    else:
        bounds = [
            {'name': 'Kd1', 'type': 'continuous', 'domain': (0, 50)},
            {'name': 'Kd2', 'type': 'continuous', 'domain': (0, 50)},
            {'name': 'Kd3', 'type': 'continuous', 'domain': (0, 50)},
            {'name': 'Kd4', 'type': 'continuous', 'domain': (0, 50)},
            {'name': 'Kd5', 'type': 'continuous', 'domain': (0, 50)},
            {'name': 'Kd6', 'type': 'continuous', 'domain': (0, 50)},
            {'name': 'Kd7', 'type': 'continuous', 'domain': (0, 50)},
            {'name': 'Ki1', 'type': 'continuous', 'domain': (0, 500)},
            {'name': 'Ki2', 'type': 'continuous', 'domain': (0, 500)},
            {'name': 'Ki3', 'type': 'continuous', 'domain': (0, 500)},
            {'name': 'Ki4', 'type': 'continuous', 'domain': (0, 500)},
            {'name': 'Ki5', 'type': 'continuous', 'domain': (0, 500)},
            {'name': 'Ki6', 'type': 'continuous', 'domain': (0, 500)},
            {'name': 'Ki7', 'type': 'continuous', 'domain': (0, 500)},
            {'name': 'Kp1', 'type': 'continuous', 'domain': (0, 400)},
            {'name': 'Kp2', 'type': 'continuous', 'domain': (0, 400)},
            {'name': 'Kp3', 'type': 'continuous', 'domain': (0, 400)},
            {'name': 'Kp4', 'type': 'continuous', 'domain': (0, 400)},
            {'name': 'Kp5', 'type': 'continuous', 'domain': (0, 400)},
            {'name': 'Kp6', 'type': 'continuous', 'domain': (0, 400)},
            {'name': 'Kp7', 'type': 'continuous', 'domain': (0, 400)},
        ]

        # Wrap objective
        def black_box_function(param_array):
            param_dict = {}
            groups = ['Kd', 'Ki', 'Kp']
            idx = 0
            for group in groups:
                for i in range(1, 8):  # from 1 to 7
                    key = f"{group}{i}"
                    param_dict[key] = param_array[idx]
                    idx += 1
            return np.log(obj_PID_panda(param_dict, const))  # Minimize cost


        # IDWGOPT initialization
        nvars = len(bounds)
        lb = np.zeros((nvars, 1)).flatten("c")
        ub = lb.copy()
        for i in range(0, nvars):
            lb[i] = bounds[i]['domain'][0]
            ub[i] = bounds[i]['domain'][1]

        prob = GLIS(bounds=(lb, ub), n_initial_random=10)  # initialize GLIS object
        xopt, fopt = prob.solve(black_box_function, 2000)  # solve optimization problem

        f_iters = prob.F
        X_iters = prob.X

        idx_opt = np.argmin(f_iters, axis=0)

        param_dict = {}
        groups = ['Kd', 'Ki', 'Kp']
        idx = 0
        for group in groups:
            for i in range(1, 8):  # from 1 to 7
                key = f"{group}{i}"
                param_dict[key] = xopt[idx]
                idx += 1
        best_params = param_dict
        print(f"f_best_val: {fopt.item():.3f}")

        for o, t in zip(X_iters, f_iters):
            outs.append(o)
            targets.append(t)

    runs_outputs.append(outs)
    runs_targets.append(targets)

    # Run one simulation with best parameters and get joint trajectories
    cost, q_msr = obj_PID_panda(best_params, const, return_trace=True)
    q_r = r['q_r']

    q_r = q_r[:len(time)]
    q_msr = q_msr[:len(time)]

    t = const['time']
    q_err = q_r - q_msr

    q_measured.append(q_msr)

    # Plot joint tracking errors (rad)
    # plt.figure(figsize=(10, 5))
    # for i in range(n_DoFs):
    #     plt.plot(t, q_err[:, i], label=f'q{i+1}')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Tracking error [rad]")
    # plt.title("Joint Position Tracking Errors")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('tracking_errors.png', dpi=300)
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # for i in range(n_DoFs):
    #     plt.plot(t, q_r[:, i], linestyle='--', label=f'q_r{i+1}')
    #     plt.plot(t, q_msr[:, i], linestyle='-', label=f'q_msr{i+1}')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Joint Angle [rad]")
    # plt.title("Desired vs Measured Joint Angles")
    # plt.legend(ncol=2)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # fig, axs = plt.subplots(n_DoFs, 1, figsize=(10, 2.5 * n_DoFs), sharex=True)
    #
    # for i in range(n_DoFs):
    #     axs[i].plot(t, q_r[:, i], linestyle='--', label=f'q_r{i+1}')
    #     axs[i].plot(t, q_msr[:, i], linestyle='-', label=f'q_msr{i+1}')
    #     axs[i].set_ylabel("Angle [rad]")
    #     axs[i].set_title(f"Joint {i+1}")
    #     axs[i].legend()
    #     axs[i].grid(True)
    #
    # axs[-1].set_xlabel("Time [s]")
    # plt.tight_layout()
    # plt.savefig('joint_pos.png', dpi=300)
    # plt.show()

    # Optionally downsample for faster plotting
    # q_msr_sampled = q_msr[::20]
    #
    # # Create GIF (or animation in window)
    # # anim = Robot.plot(q_msr_sampled, dt=ts, block=False)  # capture animation object
    # # anim.save('robot_trajectory.gif', writer='pillow', fps=int(1/ts))
    # Robot.plot(q_msr_sampled, dt=ts)

q_r = np.array(q_r)
q_measured = np.array(q_measured)
runs_outputs = np.array(runs_outputs)
runs_targets = np.array(runs_targets)

# np.save('q_r_qerr_mallrand.npy', q_r)
# np.save('q_measured_qerr_mallrand.npy', q_measured)
# np.save('runs_outputs_qerr_mallrand.npy', runs_outputs)
# np.save('runs_targets_qerr_mallrand.npy', runs_targets)
