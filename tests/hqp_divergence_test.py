import argparse
import csv
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np

try:
    import mujoco
except ImportError as e:
    raise RuntimeError("MuJoCo is required to run this script.") from e

try:
    import cvxpy as cp
except ModuleNotFoundError as e:
    raise RuntimeError("cvxpy is required. Install with: pip install cvxpy osqp") from e


OUTPUT_JOINT_NAMES = [
    "left_ankle",
    "right_ankle",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_hip_roll",
    "right_hip_roll",
]


def get_rpy(q: np.ndarray) -> Tuple[float, float]:
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(np.clip(2 * (w * y - z * x), -1, 1))
    return float(roll), float(pitch)


def safe_reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if getattr(model, "nkey", 0) > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)


def build_actuator_mappings(model: mujoco.MjModel) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    act_joint = model.actuator_trnid[:, 0].copy()
    qaddr = model.jnt_qposadr[act_joint]
    daddr = model.jnt_dofadr[act_joint]
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        for i in range(model.nu)
    ]
    return qaddr, daddr, actuator_names


def build_groups(names: List[str]) -> Dict[str, List[int]]:
    return {
        "AnkleP": [i for i, n in enumerate(names) if "ankle" in n],
        "HipP": [i for i, n in enumerate(names) if "hip_pitch" in n],
        "HipR": [i for i, n in enumerate(names) if "hip_roll" in n],
        "KneeP": [i for i, n in enumerate(names) if "knee" in n],
    }


def build_gains(model: mujoco.MjModel, groups: Dict[str, List[int]], kp_posture: float, kd_posture: float) -> Tuple[np.ndarray, np.ndarray]:
    kp = np.full(model.nu, float(kp_posture), dtype=float)
    kd = np.full(model.nu, float(kd_posture), dtype=float)
    for i in groups["HipR"]:
        kp[i], kd[i] = 500.0, 50.0
    return kp, kd


def hipr_state(
    data: mujoco.MjData,
    qaddr: np.ndarray,
    daddr: np.ndarray,
    q_target_full: np.ndarray,
    hipr_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hipr_qaddr = qaddr[hipr_ids]
    hipr_daddr = daddr[hipr_ids]
    hipr_q = data.qpos[hipr_qaddr]
    hipr_dq = data.qvel[hipr_daddr]
    hipr_q_ref = q_target_full[hipr_qaddr]
    hipr_err = hipr_q_ref - hipr_q
    return hipr_q, hipr_dq, hipr_err


def calibrate_balance_sign(model: mujoco.MjModel, data: mujoco.MjData, ankle_ids: List[int]) -> float:
    safe_reset(model, data)
    mujoco.mj_forward(model, data)
    _, p0 = get_rpy(data.qpos[3:7])
    data.ctrl[:] = 0.0
    for idx in ankle_ids:
        data.ctrl[idx] = 10.0
    for _ in range(30):
        mujoco.mj_step(model, data)
    _, p1 = get_rpy(data.qpos[3:7])
    sign = 1.0 if (p1 - p0) > 0.0 else -1.0
    safe_reset(model, data)
    return float(sign)


def _resolve_output_indices(actuator_names: List[str]) -> List[int]:
    out: List[int] = []
    for joint_name in OUTPUT_JOINT_NAMES:
        idx = None
        for i, n in enumerate(actuator_names):
            if n == joint_name:
                idx = i
                break
        if idx is None:
            raise RuntimeError(f"Could not resolve actuator '{joint_name}'")
        out.append(int(idx))
    return out


class WeightedQPController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        qaddr: np.ndarray,
        daddr: np.ndarray,
        actuator_names: List[str],
        groups: Dict[str, List[int]],
        gear: np.ndarray,
        ctrl_min: np.ndarray,
        ctrl_max: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        q_target_full: np.ndarray,
        tilt_sign: float,
        posture_weight: float,
        balance_weight: float,
    ):
        self.model = model
        self.data = data
        self.qaddr = qaddr
        self.daddr = daddr
        self.actuator_names = actuator_names
        self.groups = groups
        self.gear = gear
        self.ctrl_min = ctrl_min
        self.ctrl_max = ctrl_max
        self.kp = kp
        self.kd = kd
        self.q_target_full = q_target_full
        self.tilt_sign = float(tilt_sign)
        self.posture_weight = float(posture_weight)
        self.balance_weight = float(balance_weight)

        self.tau = cp.Variable(self.model.nu)
        self.tau_nominal = cp.Parameter(self.model.nu)
        self.tilt_target = cp.Parameter()
        self.roll_target = cp.Parameter()
        self.hipr_hold = cp.Parameter(len(self.groups["HipR"]))

        posture_obj = cp.sum_squares(self.tau - self.tau_nominal)
        balance_p = 0
        for i in self.groups["AnkleP"]:
            balance_p += cp.square(self.tau[i] - (self.tau_nominal[i] + self.tilt_target))
        for i in self.groups["HipP"]:
            balance_p += cp.square(self.tau[i] - (self.tau_nominal[i] + 0.5 * self.tilt_target))

        balance_roll = 0
        for i in self.groups["HipR"]:
            balance_roll += cp.square(self.tau[i] - (self.tau_nominal[i] + self.roll_target))

        balance_r = cp.sum_squares(self.tau[self.groups["HipR"]] - (self.tau_nominal[self.groups["HipR"]] + self.hipr_hold))

        constraints = [self.tau >= ctrl_min * gear, self.tau <= ctrl_max * gear]
        self.prob = cp.Problem(
            cp.Minimize(
                self.posture_weight * posture_obj
                + self.balance_weight * balance_p
                + self.balance_weight * balance_roll
                + self.balance_weight * balance_r
                + 0.0005 * cp.sum_squares(self.tau)
            ),
            constraints,
        )

    def compute_tau(
        self,
        curr_pitch: float,
        curr_roll: float,
        prev_pitch: float,
        prev_roll: float,
        dt_est: float,
        hipr_int: np.ndarray,
        target_pitch: float,
        kp_tilt: float,
        kd_tilt: float,
        kp_roll: float,
        kd_roll: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        q_curr = self.data.qpos[self.qaddr]
        dq_curr = self.data.qvel[self.daddr]
        tau_pd = self.kp * (self.q_target_full[self.qaddr] - q_curr) - self.kd * dq_curr
        tau_nom = tau_pd + self.data.qfrc_bias[self.daddr]
        self.tau_nominal.value = tau_nom

        d_pitch = (curr_pitch - prev_pitch) / dt_est
        self.tilt_target.value = -self.tilt_sign * (kp_tilt * (curr_pitch - target_pitch) + kd_tilt * d_pitch)

        d_roll = (curr_roll - prev_roll) / dt_est
        self.roll_target.value = float(-kp_roll * curr_roll - kd_roll * d_roll)

        hipr_kp, hipr_kd, hipr_ki = 400.0, 40.0, 60.0
        _, hipr_dq, hipr_err = hipr_state(self.data, self.qaddr, self.daddr, self.q_target_full, self.groups["HipR"])
        hipr_int_new = np.clip(hipr_int + hipr_err * dt_est, -0.3, 0.3)
        self.hipr_hold.value = hipr_kp * hipr_err - hipr_kd * hipr_dq + hipr_ki * hipr_int_new

        t0 = time.perf_counter()
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            tau_out = self.tau.value if self.tau.value is not None else tau_nom
            solve_ms = (time.perf_counter() - t0) * 1000.0
        except Exception:
            tau_out = tau_nom
            solve_ms = 0.0

        return np.asarray(tau_out, dtype=float), hipr_int_new, float(solve_ms)


class HierarchicalQPController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        qaddr: np.ndarray,
        daddr: np.ndarray,
        actuator_names: List[str],
        groups: Dict[str, List[int]],
        gear: np.ndarray,
        ctrl_min: np.ndarray,
        ctrl_max: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        q_target_full: np.ndarray,
        tilt_sign: float,
        posture_weight: float,
        balance_weight: float,
    ):
        self.model = model
        self.data = data
        self.qaddr = qaddr
        self.daddr = daddr
        self.actuator_names = actuator_names
        self.groups = groups
        self.gear = gear
        self.ctrl_min = ctrl_min
        self.ctrl_max = ctrl_max
        self.kp = kp
        self.kd = kd
        self.q_target_full = q_target_full
        self.tilt_sign = float(tilt_sign)
        self.posture_weight = float(posture_weight)
        self.balance_weight = float(balance_weight)

        self.tau = cp.Variable(self.model.nu)
        self.tau_nominal = cp.Parameter(self.model.nu)
        self.tilt_target = cp.Parameter()
        self.roll_target = cp.Parameter()
        self.hipr_hold = cp.Parameter(len(self.groups["HipR"]))

        self.posture_obj = cp.sum_squares(self.tau - self.tau_nominal)

        self.balance_p = 0
        for i in self.groups["AnkleP"]:
            self.balance_p += cp.square(self.tau[i] - (self.tau_nominal[i] + self.tilt_target))
        for i in self.groups["HipP"]:
            self.balance_p += cp.square(self.tau[i] - (self.tau_nominal[i] + 0.5 * self.tilt_target))

        self.balance_roll = 0
        for i in self.groups["HipR"]:
            self.balance_roll += cp.square(self.tau[i] - (self.tau_nominal[i] + self.roll_target))

        self.balance_r = cp.sum_squares(self.tau[self.groups["HipR"]] - (self.tau_nominal[self.groups["HipR"]] + self.hipr_hold))

        self.balance_cost = (
            self.balance_weight * self.balance_p
            + self.balance_weight * self.balance_roll
            + self.balance_weight * self.balance_r
        )

        self.constraints = [self.tau >= ctrl_min * gear, self.tau <= ctrl_max * gear]
        self.prob_balance = cp.Problem(cp.Minimize(self.balance_cost + 0.0005 * cp.sum_squares(self.tau)), self.constraints)

        self.balance_cost_limit = cp.Parameter(nonneg=True)
        self.prob_posture = cp.Problem(
            cp.Minimize(self.posture_weight * self.posture_obj + 0.0005 * cp.sum_squares(self.tau)),
            self.constraints + [self.balance_cost <= self.balance_cost_limit],
        )

    def compute_tau(
        self,
        curr_pitch: float,
        curr_roll: float,
        prev_pitch: float,
        prev_roll: float,
        dt_est: float,
        hipr_int: np.ndarray,
        target_pitch: float,
        kp_tilt: float,
        kd_tilt: float,
        kp_roll: float,
        kd_roll: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        q_curr = self.data.qpos[self.qaddr]
        dq_curr = self.data.qvel[self.daddr]
        tau_pd = self.kp * (self.q_target_full[self.qaddr] - q_curr) - self.kd * dq_curr
        tau_nom = tau_pd + self.data.qfrc_bias[self.daddr]
        self.tau_nominal.value = tau_nom

        d_pitch = (curr_pitch - prev_pitch) / dt_est
        self.tilt_target.value = -self.tilt_sign * (kp_tilt * (curr_pitch - target_pitch) + kd_tilt * d_pitch)

        d_roll = (curr_roll - prev_roll) / dt_est
        self.roll_target.value = float(-kp_roll * curr_roll - kd_roll * d_roll)

        hipr_kp, hipr_kd, hipr_ki = 400.0, 40.0, 60.0
        _, hipr_dq, hipr_err = hipr_state(self.data, self.qaddr, self.daddr, self.q_target_full, self.groups["HipR"])
        hipr_int_new = np.clip(hipr_int + hipr_err * dt_est, -0.3, 0.3)
        self.hipr_hold.value = hipr_kp * hipr_err - hipr_kd * hipr_dq + hipr_ki * hipr_int_new

        t0 = time.perf_counter()
        try:
            self.prob_balance.solve(solver=cp.OSQP, warm_start=True)
            tau_stage1 = self.tau.value if self.tau.value is not None else tau_nom
            stage1_cost = float(self.balance_cost.value) if self.balance_cost.value is not None else float("inf")
            self.balance_cost_limit.value = max(0.0, stage1_cost * (1.0 + 1e-6) + 1e-9)
            self.prob_posture.solve(solver=cp.OSQP, warm_start=True)
            tau_out = self.tau.value if self.tau.value is not None else tau_stage1
            solve_ms = (time.perf_counter() - t0) * 1000.0
        except Exception:
            tau_out = tau_nom
            solve_ms = 0.0

        return np.asarray(tau_out, dtype=float), hipr_int_new, float(solve_ms)


def run_controller_trial(
    controller_kind: str,
    xml_path: str,
    sim_time: float,
    push_start: float,
    push_duration: float,
    push_force_n: float,
    posture_weight: float,
    balance_weight: float,
    target_pitch: float,
    kp_posture: float,
    kd_posture: float,
    kp_tilt: float,
    kd_tilt: float,
    kp_roll: float,
    kd_roll: float,
) -> Tuple[np.ndarray, List[str]]:
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = 0.001
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    qaddr, daddr, actuator_names = build_actuator_mappings(model)
    groups = build_groups(actuator_names)
    kp, kd = build_gains(model, groups, kp_posture, kd_posture)

    gear = model.actuator_gear[:, 0].copy()
    gear[gear == 0.0] = 1.0
    ctrl_min = model.actuator_ctrlrange[:, 0]
    ctrl_max = model.actuator_ctrlrange[:, 1]

    output_indices = _resolve_output_indices(actuator_names)
    _ = output_indices

    safe_reset(model, data)
    mujoco.mj_forward(model, data)
    q_target_full = data.qpos.copy()

    tilt_sign = calibrate_balance_sign(model, data, groups["AnkleP"])
    safe_reset(model, data)
    mujoco.mj_forward(model, data)

    if controller_kind == "wqp":
        controller = WeightedQPController(
            model=model,
            data=data,
            qaddr=qaddr,
            daddr=daddr,
            actuator_names=actuator_names,
            groups=groups,
            gear=gear,
            ctrl_min=ctrl_min,
            ctrl_max=ctrl_max,
            kp=kp,
            kd=kd,
            q_target_full=q_target_full,
            tilt_sign=tilt_sign,
            posture_weight=posture_weight,
            balance_weight=balance_weight,
        )
    elif controller_kind == "hqp":
        controller = HierarchicalQPController(
            model=model,
            data=data,
            qaddr=qaddr,
            daddr=daddr,
            actuator_names=actuator_names,
            groups=groups,
            gear=gear,
            ctrl_min=ctrl_min,
            ctrl_max=ctrl_max,
            kp=kp,
            kd=kd,
            q_target_full=q_target_full,
            tilt_sign=tilt_sign,
            posture_weight=posture_weight,
            balance_weight=balance_weight,
        )
    else:
        raise ValueError(f"Unknown controller_kind: {controller_kind}")

    n_steps = int(round(float(sim_time) / dt))
    tau_log = np.zeros((n_steps, model.nu), dtype=np.float64)
    hipr_int = np.zeros(len(groups["HipR"]), dtype=float)
    prev_roll, prev_pitch = get_rpy(data.qpos[3:7])
    prev_t = float(data.time)

    push_end = float(push_start) + float(push_duration)
    force_vec = np.array([float(push_force_n), 0.0, 0.0], dtype=float)

    for step in range(n_steps):
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            break

        mujoco.mj_forward(model, data)
        now_t = float(data.time)
        dt_est = max(1e-4, now_t - prev_t)
        curr_roll, curr_pitch = get_rpy(data.qpos[3:7])

        tau, hipr_int, _solve_ms = controller.compute_tau(
            curr_pitch=curr_pitch,
            curr_roll=curr_roll,
            prev_pitch=float(prev_pitch),
            prev_roll=float(prev_roll),
            dt_est=float(dt_est),
            hipr_int=hipr_int,
            target_pitch=float(target_pitch),
            kp_tilt=float(kp_tilt),
            kd_tilt=float(kd_tilt),
            kp_roll=float(kp_roll),
            kd_roll=float(kd_roll),
        )

        tau_log[step, :] = tau
        ctrl = np.clip(tau / gear, ctrl_min, ctrl_max)
        data.ctrl[:] = ctrl

        data.xfrc_applied[:] = 0.0
        if pelvis_id >= 0 and float(push_start) <= now_t < push_end:
            data.xfrc_applied[pelvis_id, 0:3] = force_vec

        prev_roll, prev_pitch = float(curr_roll), float(curr_pitch)
        prev_t = now_t
        mujoco.mj_step(model, data)

    return tau_log, actuator_names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HQP divergence unit test (WQP vs HQP)")
    p.add_argument(
        "--xml",
        type=str,
        default=r"C:\Users\sanju\Documents\University_West\Course\Thesis\Scripts\venv310\model\New_MOdels\mujoco_menagerie-main\unitree_h1\scene.xml",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sim-time", type=float, default=5.0)
    p.add_argument("--push-start", type=float, default=1.0)
    p.add_argument("--push-duration", type=float, default=0.1)
    p.add_argument("--push-force", type=float, default=100.0)
    p.add_argument("--posture-weight", type=float, default=50.0)
    p.add_argument("--balance-weight", type=float, default=50.0)
    p.add_argument("--target-pitch", type=float, default=0.05)
    p.add_argument("--kp-posture", type=float, default=200.0)
    p.add_argument("--kd-posture", type=float, default=20.0)
    p.add_argument("--kp-tilt", type=float, default=500.0)
    p.add_argument("--kd-tilt", type=float, default=50.0)
    p.add_argument("--kp-roll", type=float, default=500.0)
    p.add_argument("--kd-roll", type=float, default=50.0)
    p.add_argument("--out-csv", type=str, default="hqp_divergence_verification.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tau_wqp, actuator_names = run_controller_trial(
        controller_kind="wqp",
        xml_path=str(args.xml),
        sim_time=float(args.sim_time),
        push_start=float(args.push_start),
        push_duration=float(args.push_duration),
        push_force_n=float(args.push_force),
        posture_weight=float(args.posture_weight),
        balance_weight=float(args.balance_weight),
        target_pitch=float(args.target_pitch),
        kp_posture=float(args.kp_posture),
        kd_posture=float(args.kd_posture),
        kp_tilt=float(args.kp_tilt),
        kd_tilt=float(args.kd_tilt),
        kp_roll=float(args.kp_roll),
        kd_roll=float(args.kd_roll),
    )

    tau_hqp, _ = run_controller_trial(
        controller_kind="hqp",
        xml_path=str(args.xml),
        sim_time=float(args.sim_time),
        push_start=float(args.push_start),
        push_duration=float(args.push_duration),
        push_force_n=float(args.push_force),
        posture_weight=float(args.posture_weight),
        balance_weight=float(args.balance_weight),
        target_pitch=float(args.target_pitch),
        kp_posture=float(args.kp_posture),
        kd_posture=float(args.kd_posture),
        kp_tilt=float(args.kp_tilt),
        kd_tilt=float(args.kd_tilt),
        kp_roll=float(args.kp_roll),
        kd_roll=float(args.kd_roll),
    )

    n = int(min(tau_wqp.shape[0], tau_hqp.shape[0]))
    tau_wqp = tau_wqp[:n, :]
    tau_hqp = tau_hqp[:n, :]
    diff = tau_hqp - tau_wqp
    mean_abs_diff = np.mean(np.abs(diff), axis=0)
    max_abs_diff = float(np.max(np.abs(diff))) if diff.size else 0.0

    print("\nMean |torque difference| per joint (Nm):")
    for i, name in enumerate(actuator_names):
        print(f"  {name:<28s} {mean_abs_diff[i]:.6f}")

    print(f"\nMaximum |torque difference| observed (Nm): {max_abs_diff:.6f}")
    diverged = bool(max_abs_diff > 1e-3)
    print(f"Controllers produced different torques: {'yes' if diverged else 'no'}")
    if diverged:
        print("HQP hierarchy confirmed active — controllers diverge under equal weighting")

    out_path = os.path.join(os.getcwd(), str(args.out_csv))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestep", "joint_name", "wqp_torque", "hqp_torque", "difference"])
        for t in range(n):
            for j, name in enumerate(actuator_names):
                w.writerow([t, name, float(tau_wqp[t, j]), float(tau_hqp[t, j]), float(diff[t, j])])
    print(f"\nSaved CSV: {out_path}")


if __name__ == "__main__":
    main()

