import numpy as np
from frankapy import FrankaArm
import time

def obj_PID_panda_real(x_var, const, return_trace=False):
    Kp = np.array([x_var[f"Kp{i+1}"] for i in range(7)])
    Ki = np.array([x_var[f"Ki{i+1}"] for i in range(7)])
    Kd = np.array([x_var[f"Kd{i+1}"] for i in range(7)])

    Ts = const['Ts']
    time_vec = const['time']
    n_steps = len(time_vec)
    n_DoFs = const['n_DoFs']
    q_ref = const['r']['q_r']
    dq_ref = const['r']['dq_r']

    q_err_threshold = 2 * np.pi / 180  # 2 degrees in radians

    fa = FrankaArm()
    print("Resetting joints...")
    fa.reset_joints()
    time.sleep(1)

    print("Starting torque control...")
    fa.run_torque_control()

    ierr = np.zeros(n_DoFs)
    q_measured = []
    dq_measured = []

    start_time = time.time()
    safe_exit = True

    for jj in range(n_steps):
        t_loop = time.time() - start_time

        q_msr = fa.get_joints()
        dq_msr = fa.get_joint_velocities()

        q_measured.append(q_msr.copy())
        dq_measured.append(dq_msr.copy())

        q_err = q_ref[jj] - q_msr
        dq_err = dq_ref[jj] - dq_msr
        ierr += q_err * Ts

        if np.max(np.abs(q_err)) > q_err_threshold:
            print(f"[SAFETY] Joint deviation exceeded threshold at t={t_loop:.3f}s, step {jj}")
            safe_exit = False
            break

        tau_cmd = Kp * q_err + Ki * ierr - Kd * dq_msr

        # SAFE: only send torque to joint 1
        tau_cmd_safe = np.zeros(n_DoFs)
        tau_cmd_safe[0] = tau_cmd[0]

        tau_cmd_safe = np.clip(tau_cmd_safe, -20, 20)

        fa.set_joint_torques(tau_cmd_safe)

        while time.time() - start_time < jj * Ts:
            time.sleep(0.0005)

    fa.stop_torque_control()
    print("Torque control stopped.")

    q_measured = np.array(q_measured)
    dq_measured = np.array(dq_measured)

    if not safe_exit:
        print("Returning high cost due to tracking failure.")
        return 1e6, q_measured if return_trace else 1e6

    q_ref = np.array(q_ref[:len(q_measured)])
    dq_ref = np.array(dq_ref[:len(dq_measured)])

    q_err_total = q_ref - q_measured
    dq_err_total = dq_ref - dq_measured
    cost = (
        np.sqrt(np.mean(q_err_total**2)) +
        np.sqrt(np.mean(dq_err_total**2)) +
        np.max(np.abs(q_err_total))
    )

    if return_trace:
        return cost, q_measured
    return cost


def obj_PID_panda_hw(x_var, const, return_trace=False):
    """
    Hardware PID tracker for Franka Panda using FrankaPy.
    Mirrors the sim objective: computes J and optionally returns measured q trace.

    Required fields in `const` (same as your sim dict layout):
      - 'Ts': sample time [s]
      - 'time': 1D array of time stamps
      - 'n_DoFs': 7
      - 'r': {'q_r', 'dq_r', 'ddq_r'}  (reference trajectories, same shapes as sim)
      - 'toll_qerr': per-joint or scalar tolerance for early abort
      - 'Robot_friction': scalar or length-7 viscous coeff (optional)
    And pass gains in x_var as: Kp1..Kp7, Ki1..Ki7, Kd1..Kd7
    """
    sim_var = const.copy()
    sim_var.update(x_var)

    # Gains
    Kp = np.diag([sim_var[f"Kp{i+1}"] for i in range(7)])
    Ki = np.diag([sim_var[f"Ki{i+1}"] for i in range(7)])
    Kd = np.diag([sim_var[f"Kd{i+1}"] for i in range(7)])

    # Timing / refs
    Ts   = float(sim_var['Ts'])
    t    = np.asarray(sim_var['time'])
    q_r  = np.asarray(sim_var['r']['q_r'])
    dq_r = np.asarray(sim_var['r']['dq_r'])
    ddq_r = np.asarray(sim_var['r']['ddq_r'])
    n_DoFs = int(sim_var.get('n_DoFs', 7))
    assert n_DoFs == 7, "This function assumes a 7-DoF Panda."

    # Optional viscous friction feedforward (same semantics as your sim)
    f = sim_var.get('Robot_friction', 0.0)
    if np.isscalar(f):
        f = np.full(7, float(f))
    else:
        f = np.asarray(f).astype(float)
        assert f.shape == (7,)

    toll_qerr = sim_var['toll_qerr']
    if np.isscalar(toll_qerr):
        toll_qerr = np.full(7, float(toll_qerr))
    else:
        toll_qerr = np.asarray(toll_qerr).astype(float)
        assert toll_qerr.shape == (7,)

    # Safety limits (tunable)
    tau_limit = np.asarray(sim_var.get('tau_limit', np.array([60, 60, 60, 60, 20, 20, 20], dtype=float)))  # Nm
    tau_rate_limit = np.asarray(sim_var.get('tau_rate_limit', np.full(7, 1000.0)))  # Nm/s
    torque_thresholds = sim_var.get('torque_thresholds', None)  # passed to franka as extra safety

    # Init Franka
    fa = FrankaArm()
    fa.wait_for_franka_interface()  # blocks until ready

    # Initialize integrator & logs
    ierr = np.zeros(7)
    q_log   = np.zeros_like(q_r)
    dq_log  = np.zeros_like(dq_r)
    qerr    = np.zeros_like(q_r)
    dqerr   = np.zeros_like(dq_r)
    penalty = np.zeros(7)

    # Seed initial state & sync to t[0]
    q = np.array(fa.get_joints())
    dq = np.array(fa.get_joint_velocities())
    q_log[0]  = q
    dq_log[0] = dq

    # For torque rate limiting
    tau_prev = np.zeros(7)

    exit_flag = False
    start = time.monotonic()
    for jj in range(1, len(t)):
        # Wait until next sample time
        next_tick = start + t[jj]
        now = time.monotonic()
        sleep_dt = next_tick - now
        if sleep_dt > 0:
            time.sleep(sleep_dt)

        # Read sensors
        q = np.array(fa.get_joints())
        dq = np.array(fa.get_joint_velocities())

        # Errors
        e   = q_r[jj]  - q
        de  = dq_r[jj] - dq
        ierr += e * Ts

        # PID (joint-space)
        tau_pid = (Kp @ e) + (Ki @ ierr) - (Kd @ dq)

        # Optional viscous feedforward (same sign convention as your sim)
        tau_ff = f * dq

        # Final torque before gravity removal (gravity is handled by Franka if remove_gravity=1)
        tau = tau_pid + tau_ff

        # Torque rate limit
        max_step = tau_rate_limit * Ts
        tau = np.clip(tau, tau_prev - max_step, tau_prev + max_step)

        # Torque saturation
        tau = np.clip(tau, -tau_limit, tau_limit)

        # Stream torques for one sample interval
        fa.execute_joint_torques(
            joint_torques=tau.tolist(),
            selection=[1]*7,
            remove_gravity=[1]*7,          # robot compensates gravity internally
            duration=Ts,
            dynamic=True,
            torque_thresholds=torque_thresholds,
            block=True
        )

        tau_prev = tau

        # Log
        q_log[jj]  = q
        dq_log[jj] = dq
        qerr[jj]   = q_r[jj]  - q
        dqerr[jj]  = dq_r[jj] - dq

        # Early stop if diverging/unsafe
        if np.any(np.isnan(q)) or np.any(np.abs(qerr[jj]) > toll_qerr):
            penalty = 1e5 * np.exp(-t[jj])
            exit_flag = True
            break

    # Stop / zero-out torques safely
    try:
        # one short gravity-only window to settle
        fa.execute_joint_torques(
            joint_torques=[0.0]*7,
            selection=[1]*7,
            remove_gravity=[1]*7,
            duration=0.1,
            dynamic=False,
            block=True
        )
    finally:
        # Ensure any running skill is stopped
        try:
            fa.stop_skill()
        except Exception:
            pass

    # Cost J (mirrors your sim metric)
    if not exit_flag:
        J = 0.0
        J += np.sum(np.sqrt(np.mean(qerr**2, axis=0)))
        J += np.sum(np.sqrt(np.mean(dqerr**2, axis=0)))
        J += np.sum(np.max(np.abs(qerr), axis=0))
        J += np.sum(np.max(np.abs(dqerr), axis=0))
    else:
        J = np.sum(penalty)

    return (J, q_log) if return_trace else J