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

    fa = FrankaArm(franka_address='YOUR.ROBOT.IP.HERE')
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
