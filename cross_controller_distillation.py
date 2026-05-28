import argparse
import csv
import datetime
import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ModuleNotFoundError:
    cp = None
    CVXPY_AVAILABLE = False

try:
    from disturbance import BODY_WEIGHTS, DisturbanceProfile, Push

    DISTURBANCE_AVAILABLE = True
except ModuleNotFoundError:
    DISTURBANCE_AVAILABLE = False
    BODY_WEIGHTS = {}
    DisturbanceProfile = None
    Push = None


SINGLE_FRAME_DIM = 52
HISTORY_LENGTH = 5
INPUT_DIM = SINGLE_FRAME_DIM * HISTORY_LENGTH
OUTPUT_DIM = 6
HIDDEN_DIM = 128
TAU_BOUND = 20.0

OUTPUT_JOINT_NAMES = [
    "left_ankle",
    "right_ankle",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_hip_roll",
    "right_hip_roll",
]

CSV_FIELDNAMES = [
    "session_id",
    "controller_name",
    "condition_id",
    "trial_in_condition",
    "body_label",
    "body_name",
    "direction",
    "force_x",
    "force_y",
    "force_z",
    "force_magnitude",
    "max_com_disp",
    "max_com_disp_x",
    "max_com_disp_y",
    "recovery_time",
    "fall_time",
    "is_stable",
    "tau_rms_total",
    "tau_rms_ankle",
    "tau_rms_knee",
    "tau_rms_hip",
    "max_torso_pitch_rad",
    "max_torso_roll_rad",
    "qp_solve_time_mean_ms",
    "qp_solve_time_max_ms",
    "qp_solver_failures",
    "disturbance_seed",
    "n_pushes_scheduled",
    "n_pushes_delivered",
    "total_impulse_Ns",
    "max_instantaneous_force_N",
    "mean_push_magnitude_N",
    "push_log_json",
    "survival_time_s",
    "rms_com_disp_m",
    "integrated_torque_Nms",
]


def get_rpy(q: np.ndarray) -> Tuple[float, float]:
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(np.clip(2 * (w * y - z * x), -1, 1))
    return float(roll), float(pitch)


def safe_reset(model: "mujoco.MjModel", data: "mujoco.MjData") -> None:
    if getattr(model, "nkey", 0) > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)


def compute_com_position(model: "mujoco.MjModel", data: "mujoco.MjData") -> np.ndarray:
    masses = model.body_mass
    total_mass = float(np.sum(masses))
    return (masses[:, None] * data.xipos).sum(axis=0) / total_mass


def build_actuator_mappings(model: "mujoco.MjModel") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    act_joint = model.actuator_trnid[:, 0].copy()
    qaddr = model.jnt_qposadr[act_joint]
    daddr = model.jnt_dofadr[act_joint]
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        for i in range(model.nu)
    ]
    return qaddr, daddr, actuator_names


def build_groups(actuator_names: List[str]) -> Dict[str, List[int]]:
    return {
        "AnkleP": [i for i, n in enumerate(actuator_names) if "ankle" in n],
        "HipP": [i for i, n in enumerate(actuator_names) if "hip_pitch" in n],
        "KneeP": [i for i, n in enumerate(actuator_names) if "knee" in n],
        "HipR": [i for i, n in enumerate(actuator_names) if "hip_roll" in n],
    }


def build_gains(
    model: "mujoco.MjModel",
    groups: Dict[str, List[int]],
    kp_posture: float,
    kd_posture: float,
) -> Tuple[np.ndarray, np.ndarray]:
    kp = np.full(model.nu, float(kp_posture), dtype=float)
    kd = np.full(model.nu, float(kd_posture), dtype=float)
    for i in groups["HipR"]:
        kp[i], kd[i] = 500.0, 50.0
    return kp, kd


def hipr_state(
    data: "mujoco.MjData",
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


def calibrate_balance_sign(
    model: "mujoco.MjModel", data: "mujoco.MjData", ankle_ids: List[int]
) -> float:
    safe_reset(model, data)
    mujoco.mj_forward(model, data)
    _, pitch0 = get_rpy(data.qpos[3:7])
    data.ctrl[:] = 0.0
    for idx in ankle_ids:
        data.ctrl[idx] = 10.0
    for _ in range(30):
        mujoco.mj_step(model, data)
    _, pitch1 = get_rpy(data.qpos[3:7])
    sign = 1.0 if (pitch1 - pitch0) > 0.0 else -1.0
    safe_reset(model, data)
    return float(sign)


def build_qp(
    model: "mujoco.MjModel",
    ctrl_min: np.ndarray,
    ctrl_max: np.ndarray,
    gear: np.ndarray,
    groups: Dict[str, List[int]],
):
    if cp is None:
        raise RuntimeError("cvxpy is required. Install with: pip install cvxpy osqp")
    tau = cp.Variable(model.nu)
    tau_nominal = cp.Parameter(model.nu)
    tilt_target = cp.Parameter()
    roll_target = cp.Parameter()
    hipr_hold = cp.Parameter(len(groups["HipR"]))

    posture_obj = cp.sum_squares(tau - tau_nominal)

    balance_p = 0
    for i in groups["AnkleP"]:
        balance_p += cp.square(tau[i] - (tau_nominal[i] + tilt_target))
    for i in groups["HipP"]:
        balance_p += cp.square(tau[i] - (tau_nominal[i] + 0.5 * tilt_target))

    balance_roll = 0
    for i in groups["HipR"]:
        balance_roll += cp.square(tau[i] - (tau_nominal[i] + roll_target))

    balance_r = cp.sum_squares(
        tau[groups["HipR"]] - (tau_nominal[groups["HipR"]] + hipr_hold)
    )

    hipr_weight = 300.0
    total_obj = cp.Minimize(
        1.0 * posture_obj
        + 2000.0 * balance_p
        + hipr_weight * balance_r
        + 300.0 * balance_roll
        + 0.0005 * cp.sum_squares(tau)
    )

    constraints = [tau >= ctrl_min * gear, tau <= ctrl_max * gear]
    prob = cp.Problem(total_obj, constraints)
    return prob, tau, tau_nominal, tilt_target, roll_target, hipr_hold


class DistillationMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        tau_bound: float = TAU_BOUND,
    ):
        super().__init__()
        self.tau_bound = float(tau_bound)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.tanh(self.net(x)) * self.tau_bound


@dataclass(frozen=True)
class DatasetBundle:
    states: np.ndarray
    tau_passivity_6: np.ndarray
    tau_wqp_6: np.ndarray
    targets: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray


class DistillationEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.model = mujoco.MjModel.from_xml_path(args.xml)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        self.qaddr, self.daddr, self.actuator_names = build_actuator_mappings(self.model)
        self.groups = build_groups(self.actuator_names)

        self.knee_ids = self.groups["KneeP"]
        self.hip_ids = [
            i
            for i, n in enumerate(self.actuator_names)
            if "hip_pitch" in n or "hip_roll" in n or "hip_yaw" in n
        ]

        self.gear = self.model.actuator_gear[:, 0].copy()
        self.gear[self.gear == 0.0] = 1.0
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1]

        self.tilt_sign = calibrate_balance_sign(self.model, self.data, self.groups["AnkleP"])
        mujoco.mj_forward(self.model, self.data)
        self.q_target_full = self.data.qpos.copy()

        self.kp_wqp, self.kd_wqp = build_gains(
            self.model, self.groups, float(args.kp_passivity), float(args.kd_passivity)
        )

        self.kp_pass = np.full(self.model.nu, float(args.kp_passivity), dtype=float)
        self.kd_pass = np.full(self.model.nu, float(args.kd_passivity), dtype=float)

        self.prob, self.tau_var, self.tau_nominal, self.tilt_target, self.roll_target, self.hipr_hold = build_qp(
            self.model, self.ctrl_min, self.ctrl_max, self.gear, self.groups
        )

        self.mlp_output_indices: List[int] = []
        for joint_name in OUTPUT_JOINT_NAMES:
            for i, aname in enumerate(self.actuator_names):
                if aname == joint_name:
                    self.mlp_output_indices.append(i)
                    break
        if len(self.mlp_output_indices) != OUTPUT_DIM:
            raise RuntimeError(
                f"Could not resolve all {OUTPUT_DIM} output joints. "
                f"Found: {self.mlp_output_indices} for {OUTPUT_JOINT_NAMES}"
            )

        import collections

        self.obs_history: Deque[np.ndarray] = collections.deque(maxlen=HISTORY_LENGTH)
        for _ in range(HISTORY_LENGTH):
            self.obs_history.append(np.zeros(SINGLE_FRAME_DIM, dtype=np.float32))

    def _reset_history(self) -> None:
        self.obs_history.clear()
        for _ in range(HISTORY_LENGTH):
            self.obs_history.append(np.zeros(SINGLE_FRAME_DIM, dtype=np.float32))

    def _build_state_vec(
        self,
        curr_pitch: float,
        curr_roll: float,
        prev_pitch: float,
        prev_roll: float,
        dt_est: float,
        com_error_x: float,
        com_error_y: float,
        com_vel_x: float,
        com_vel_y: float,
        tau_base_6: np.ndarray,
    ) -> np.ndarray:
        d_pitch = (curr_pitch - prev_pitch) / dt_est
        d_roll = (curr_roll - prev_roll) / dt_est
        q = self.data.qpos[self.qaddr]
        dq = self.data.qvel[self.daddr]
        tau_base_6 = np.asarray(tau_base_6, dtype=np.float32).reshape(-1)
        if tau_base_6.shape[0] != OUTPUT_DIM:
            raise ValueError(f"tau_base_6 must have shape ({OUTPUT_DIM},), got {tau_base_6.shape}")
        frame = np.concatenate(
            [
                [curr_pitch, curr_roll, d_pitch, d_roll],
                q,
                dq,
                [com_error_x, com_error_y, com_vel_x, com_vel_y],
                tau_base_6,
            ],
            axis=0,
        ).astype(np.float32)
        if frame.shape[0] != SINGLE_FRAME_DIM:
            raise ValueError(f"Expected SINGLE_FRAME_DIM={SINGLE_FRAME_DIM}, got {frame.shape[0]}")
        self.obs_history.append(frame)
        stacked = np.concatenate(list(self.obs_history), axis=0).astype(np.float32)
        if stacked.shape[0] != INPUT_DIM:
            raise ValueError(f"Expected INPUT_DIM={INPUT_DIM}, got {stacked.shape[0]}")
        return stacked

    def _compute_passivity_torque(
        self,
        curr_pitch: float,
        curr_roll: float,
        prev_pitch: float,
        prev_roll: float,
        dt_est: float,
        hipr_int: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        tau_g = self.data.qfrc_bias[self.daddr].copy()

        q_curr = self.data.qpos[self.qaddr]
        tau_pd = self.kp_pass * (self.q_target_full[self.qaddr] - q_curr)

        dq_curr = self.data.qvel[self.daddr]
        tau_d = -float(self.args.damping_gain) * dq_curr

        tau_e = np.zeros(self.model.nu, dtype=float)

        d_pitch = (curr_pitch - prev_pitch) / dt_est
        pitch_error = curr_pitch - float(self.args.target_pitch)
        tilt_corr = -self.tilt_sign * (
            float(self.args.kp_tilt) * pitch_error + float(self.args.kd_tilt) * d_pitch
        )
        tilt_corr = float(np.clip(tilt_corr, -150.0, 150.0))
        for idx in self.groups["AnkleP"]:
            tau_e[idx] += tilt_corr
        for idx in self.groups["HipP"]:
            tau_e[idx] += 0.5 * tilt_corr

        d_roll = (curr_roll - prev_roll) / dt_est
        roll_corr = -(float(self.args.kp_roll) * curr_roll + float(self.args.kd_roll) * d_roll)

        hipr_kp, hipr_kd, hipr_ki = 400.0, 40.0, 60.0
        hipr_q = self.data.qpos[self.qaddr][self.groups["HipR"]]
        hipr_dq = self.data.qvel[self.daddr][self.groups["HipR"]]
        hipr_q_ref = self.q_target_full[self.qaddr][self.groups["HipR"]]
        hipr_err = hipr_q_ref - hipr_q
        hipr_int_new = np.clip(hipr_int + hipr_err * dt_est, -0.3, 0.3)
        hipr_hold = hipr_kp * hipr_err - hipr_kd * hipr_dq + hipr_ki * hipr_int_new

        for i, idx in enumerate(self.groups["HipR"]):
            tau_e[idx] += roll_corr + hipr_hold[i]

        tau_wbc = tau_g + tau_pd + tau_d + tau_e
        return tau_wbc, hipr_int_new

    def _compute_wqp_torque(
        self,
        curr_pitch: float,
        curr_roll: float,
        prev_pitch: float,
        prev_roll: float,
        dt_est: float,
        hipr_int: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        q_curr = self.data.qpos[self.qaddr]
        dq_curr = self.data.qvel[self.daddr]
        tau_pd = self.kp_wqp * (self.q_target_full[self.qaddr] - q_curr) - self.kd_wqp * dq_curr
        tau_nom = tau_pd + self.data.qfrc_bias[self.daddr]
        self.tau_nominal.value = tau_nom

        d_pitch = (curr_pitch - prev_pitch) / dt_est
        self.tilt_target.value = -self.tilt_sign * (
            float(self.args.kp_tilt) * (curr_pitch - float(self.args.target_pitch))
            + float(self.args.kd_tilt) * d_pitch
        )

        d_roll = (curr_roll - prev_roll) / dt_est
        self.roll_target.value = float(-float(self.args.kp_roll) * curr_roll - float(self.args.kd_roll) * d_roll)

        hipr_kp, hipr_kd, hipr_ki = 400.0, 40.0, 60.0
        _, hipr_dq, hipr_err = hipr_state(self.data, self.qaddr, self.daddr, self.q_target_full, self.groups["HipR"])
        hipr_int_new = np.clip(hipr_int + hipr_err * dt_est, -0.3, 0.3)
        self.hipr_hold.value = hipr_kp * hipr_err - hipr_kd * hipr_dq + hipr_ki * hipr_int_new

        if self.tau_nominal.value is None or not np.all(np.isfinite(self.tau_nominal.value)):
            self.tau_nominal.value = np.zeros(self.model.nu)
        if self.tilt_target.value is None or not np.isfinite(float(self.tilt_target.value)):
            self.tilt_target.value = 0.0
        if self.roll_target.value is None or not np.isfinite(float(self.roll_target.value)):
            self.roll_target.value = 0.0
        if self.hipr_hold.value is None or not np.all(np.isfinite(self.hipr_hold.value)):
            self.hipr_hold.value = np.zeros(len(self.groups["HipR"]))

        t0 = time.perf_counter()
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            tau_wbc = self.tau_var.value if self.tau_var.value is not None else tau_nom
            solve_ms = (time.perf_counter() - t0) * 1000.0
        except Exception:
            tau_wbc = tau_nom
            solve_ms = 0.0

        return np.asarray(tau_wbc, dtype=float), hipr_int_new, float(solve_ms), int(1 if solve_ms <= 0.0 else 0)

    def run_trial_collect(
        self,
        profile: "DisturbanceProfile",
        collect_all: bool,
    ) -> Tuple[
        Dict[str, Any],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        float,
        int,
        float,
    ]:
        safe_reset(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._reset_history()

        sim_end = float(self.args.sim_time)

        body_id_for: Dict[str, int] = {}
        for p in profile.pushes:
            if p.body_name not in body_id_for:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, p.body_name)
                body_id_for[p.body_name] = int(bid)

        first_push_start = profile.pushes[0].start_time_s if profile.pushes else 1.0
        last_push_end = profile.pushes[-1].end_time_s if profile.pushes else first_push_start

        pre_push_window_samples = max(1, int(round(float(self.args.pre_push_window) / self.dt)))
        recovery_window_samples = max(1, int(round(float(self.args.recovery_window) / self.dt)))

        pre_push_com_xy: List[np.ndarray] = []
        com_xy_ref: Optional[np.ndarray] = None

        max_com_xy_disp = 0.0
        max_com_x_disp = 0.0
        max_com_y_disp = 0.0
        max_torso_pitch = 0.0
        max_torso_roll = 0.0
        recovery_time = float("nan")
        recovery_buffer: List[int] = []
        fall_time = float("nan")

        com_sq_integral = 0.0
        com_integral_duration = 0.0
        total_tau_sq_integral = 0.0

        tau_sq_sum_total = 0.0
        tau_sq_sum_ankle = 0.0
        tau_sq_sum_knee = 0.0
        tau_sq_sum_hip = 0.0
        tau_sq_duration = 0.0

        qp_solve_times: List[float] = []

        _r0, _p0 = get_rpy(self.data.qpos[3:7])
        prev_pitch = float(_p0)
        prev_roll = float(_r0)
        prev_t = float(self.data.time)
        prev_com_xy = compute_com_position(self.model, self.data)[0:2].copy()

        hipr_int_wqp = np.zeros(len(self.groups["HipR"]), dtype=float)
        hipr_int_pass = np.zeros(len(self.groups["HipR"]), dtype=float)

        states: List[np.ndarray] = []
        tau_pass_6: List[np.ndarray] = []
        tau_wqp_6: List[np.ndarray] = []
        targets: List[np.ndarray] = []

        delta_abs_all: List[float] = []

        while self.data.time < sim_end:
            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                mujoco.mj_step(self.model, self.data)
                continue

            mujoco.mj_forward(self.model, self.data)
            now_t = float(self.data.time)
            dt_est = max(1e-4, now_t - prev_t)

            curr_roll, curr_pitch = get_rpy(self.data.qpos[3:7])

            if now_t >= first_push_start:
                max_torso_pitch = max(max_torso_pitch, abs(curr_pitch))
                max_torso_roll = max(max_torso_roll, abs(curr_roll))

            pelvis_height = float(self.data.xipos[self.pelvis_id, 2]) if self.pelvis_id >= 0 else float("inf")
            pitch_fallen = abs(curr_pitch) > float(self.args.fall_pitch_thresh)
            height_fallen = pelvis_height < float(self.args.fall_height_thresh)
            if (pitch_fallen or height_fallen) and not np.isfinite(fall_time):
                fall_time = now_t

            com = compute_com_position(self.model, self.data)
            com_xy = com[0:2]

            if now_t < first_push_start:
                pre_push_com_xy.append(com_xy.copy())
                if len(pre_push_com_xy) > pre_push_window_samples:
                    del pre_push_com_xy[: len(pre_push_com_xy) - pre_push_window_samples]
                com_xy_ref = np.mean(np.array(pre_push_com_xy), axis=0)

            if com_xy_ref is not None:
                com_err = com_xy - com_xy_ref
            else:
                com_err = np.zeros(2, dtype=float)

            com_vel = (com_xy - prev_com_xy) / dt_est

            tau_wqp, hipr_int_wqp, solve_ms, _fail = self._compute_wqp_torque(
                curr_pitch, curr_roll, prev_pitch, prev_roll, dt_est, hipr_int_wqp
            )
            qp_solve_times.append(float(solve_ms))

            tau_pass, hipr_int_pass = self._compute_passivity_torque(
                curr_pitch, curr_roll, prev_pitch, prev_roll, dt_est, hipr_int_pass
            )

            tau_wqp_base6 = tau_wqp[self.mlp_output_indices].copy().astype(np.float32)
            state_vec = self._build_state_vec(
                curr_pitch,
                curr_roll,
                prev_pitch,
                prev_roll,
                dt_est,
                float(com_err[0]),
                float(com_err[1]),
                float(com_vel[0]),
                float(com_vel[1]),
                tau_base_6=tau_wqp_base6,
            )

            tau_applied = tau_wqp

            if now_t >= first_push_start:
                ankle_ids = self.groups["AnkleP"]
                knee_ids = self.knee_ids
                hip_ids = self.hip_ids
                tau_sq_sum_total += float(np.sum(tau_applied ** 2)) * self.dt
                tau_sq_sum_ankle += float(np.sum(tau_applied[ankle_ids] ** 2)) * self.dt
                tau_sq_sum_knee += float(np.sum(tau_applied[knee_ids] ** 2)) * self.dt
                tau_sq_sum_hip += float(np.sum(tau_applied[hip_ids] ** 2)) * self.dt
                tau_sq_duration += self.dt

            total_tau_sq_integral += float(np.sum(tau_applied ** 2)) * self.dt

            if now_t >= first_push_start:
                com_xy_disp = float(np.linalg.norm(com_err))
                max_com_xy_disp = max(max_com_xy_disp, com_xy_disp)
                max_com_x_disp = max(max_com_x_disp, abs(float(com_err[0])))
                max_com_y_disp = max(max_com_y_disp, abs(float(com_err[1])))
                com_sq_integral += (com_xy_disp ** 2) * self.dt
                com_integral_duration += self.dt

                if now_t >= last_push_end and not np.isfinite(recovery_time):
                    recovery_buffer.append(1 if com_xy_disp <= float(self.args.recovery_thresh) else 0)
                    if len(recovery_buffer) > recovery_window_samples:
                        recovery_buffer.pop(0)
                    if len(recovery_buffer) == recovery_window_samples and sum(recovery_buffer) >= int(
                        0.8 * recovery_window_samples
                    ):
                        recovery_time = now_t - last_push_end

            if now_t >= first_push_start:
                tau_pass6 = tau_pass[self.mlp_output_indices].copy().astype(np.float32)
                delta = (tau_pass6 - tau_wqp_base6).astype(np.float32)
                delta_abs_all.append(float(np.mean(np.abs(delta))))
                states.append(state_vec)
                tau_pass_6.append(tau_pass6)
                tau_wqp_6.append(tau_wqp_base6)
                targets.append(delta)

            ctrl = np.clip(tau_applied / self.gear, self.ctrl_min, self.ctrl_max)
            self.data.ctrl[:] = ctrl

            self.data.xfrc_applied[:] = 0.0
            active = profile.active_push_at(now_t)
            if active is not None:
                bid = body_id_for.get(active.body_name, -1)
                if bid >= 0:
                    self.data.xfrc_applied[bid, 0:3] = active.force_vec

            prev_com_xy = com_xy.copy()
            prev_pitch = float(curr_pitch)
            prev_roll = float(curr_roll)
            prev_t = now_t

            mujoco.mj_step(self.model, self.data)
            if np.isfinite(fall_time):
                break

        recovery_time_out = float(recovery_time) if np.isfinite(recovery_time) else -1.0
        fall_time_out = float(fall_time) if np.isfinite(fall_time) else -1.0
        is_stable = 1 if (recovery_time_out >= 0.0 and fall_time_out < 0.0) else 0
        survival_time_s = float(fall_time_out) if fall_time_out > 0 else float(sim_end)

        com_safe_dur = max(float(com_integral_duration), 1e-6)
        rms_com_disp_m = float(np.sqrt(com_sq_integral / com_safe_dur))

        safe_dur = max(float(tau_sq_duration), 1e-6)
        tau_rms_total = float(np.sqrt(float(tau_sq_sum_total) / safe_dur))
        tau_rms_ankle = float(np.sqrt(float(tau_sq_sum_ankle) / safe_dur))
        tau_rms_knee = float(np.sqrt(float(tau_sq_sum_knee) / safe_dur))
        tau_rms_hip = float(np.sqrt(float(tau_sq_sum_hip) / safe_dur))

        qp_solve_time_mean_ms = float(np.mean(qp_solve_times)) if qp_solve_times else -1.0
        qp_solve_time_max_ms = float(np.max(qp_solve_times)) if qp_solve_times else -1.0
        qp_solver_failures = int(sum(1 for t in qp_solve_times if float(t) <= 0.0))

        n_pushes_delivered = profile.pushes_delivered_by(survival_time_s)

        if profile.pushes:
            fp = profile.pushes[0]
            mag_rep = float(profile.max_instantaneous_force_n)
            body_label_rep = fp.body_name
            force_vec_rep = fp.force_vec
        else:
            mag_rep = 0.0
            body_label_rep = "none"
            force_vec_rep = (0.0, 0.0, 0.0)

        result = {
            "session_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "controller_name": "weighted_qp",
            "condition_id": int(profile.seed),
            "trial_in_condition": 0,
            "body_label": body_label_rep,
            "body_name": body_label_rep,
            "direction": "random",
            "force_x": float(force_vec_rep[0]),
            "force_y": float(force_vec_rep[1]),
            "force_z": float(force_vec_rep[2]),
            "force_magnitude": float(mag_rep),
            "max_com_disp": float(max_com_xy_disp),
            "max_com_disp_x": float(max_com_x_disp),
            "max_com_disp_y": float(max_com_y_disp),
            "recovery_time": recovery_time_out,
            "fall_time": fall_time_out,
            "is_stable": int(is_stable),
            "tau_rms_total": float(tau_rms_total),
            "tau_rms_ankle": float(tau_rms_ankle),
            "tau_rms_knee": float(tau_rms_knee),
            "tau_rms_hip": float(tau_rms_hip),
            "max_torso_pitch_rad": float(max_torso_pitch),
            "max_torso_roll_rad": float(max_torso_roll),
            "qp_solve_time_mean_ms": float(qp_solve_time_mean_ms),
            "qp_solve_time_max_ms": float(qp_solve_time_max_ms),
            "qp_solver_failures": int(qp_solver_failures),
            "disturbance_seed": int(profile.seed),
            "n_pushes_scheduled": int(profile.n_pushes_scheduled),
            "n_pushes_delivered": int(n_pushes_delivered),
            "total_impulse_Ns": float(profile.total_impulse_ns),
            "max_instantaneous_force_N": float(profile.max_instantaneous_force_n),
            "mean_push_magnitude_N": float(profile.mean_push_magnitude_n),
            "push_log_json": profile.to_json(),
            "survival_time_s": float(survival_time_s),
            "rms_com_disp_m": float(rms_com_disp_m),
            "integrated_torque_Nms": float(total_tau_sq_integral),
        }

        trial_delta_sum = float(np.sum(delta_abs_all)) if delta_abs_all else 0.0
        trial_delta_count = int(len(delta_abs_all))
        trial_delta_max = float(np.max(delta_abs_all)) if delta_abs_all else 0.0

        if (not collect_all) and int(is_stable) == 0:
            states = []
            tau_pass_6 = []
            tau_wqp_6 = []
            targets = []

        return (
            result,
            states,
            tau_pass_6,
            tau_wqp_6,
            targets,
            trial_delta_sum,
            trial_delta_count,
            trial_delta_max,
        )

    def run_trial_evaluate(
        self,
        profile: "DisturbanceProfile",
        mlp: Optional["nn.Module"],
        device: Optional[str],
    ) -> Dict[str, Any]:
        safe_reset(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._reset_history()

        sim_end = float(self.args.sim_time)

        body_id_for: Dict[str, int] = {}
        for p in profile.pushes:
            if p.body_name not in body_id_for:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, p.body_name)
                body_id_for[p.body_name] = int(bid)

        first_push_start = profile.pushes[0].start_time_s if profile.pushes else 1.0
        last_push_end = profile.pushes[-1].end_time_s if profile.pushes else first_push_start

        pre_push_window_samples = max(1, int(round(float(self.args.pre_push_window) / self.dt)))
        recovery_window_samples = max(1, int(round(float(self.args.recovery_window) / self.dt)))

        pre_push_com_xy: List[np.ndarray] = []
        com_xy_ref: Optional[np.ndarray] = None

        max_com_xy_disp = 0.0
        max_com_x_disp = 0.0
        max_com_y_disp = 0.0
        max_torso_pitch = 0.0
        max_torso_roll = 0.0
        recovery_time = float("nan")
        recovery_buffer: List[int] = []
        fall_time = float("nan")

        com_sq_integral = 0.0
        com_integral_duration = 0.0
        total_tau_sq_integral = 0.0

        tau_sq_sum_total = 0.0
        tau_sq_sum_ankle = 0.0
        tau_sq_sum_knee = 0.0
        tau_sq_sum_hip = 0.0
        tau_sq_duration = 0.0

        qp_solve_times: List[float] = []

        _r0, _p0 = get_rpy(self.data.qpos[3:7])
        prev_pitch = float(_p0)
        prev_roll = float(_r0)
        prev_t = float(self.data.time)
        prev_com_xy = compute_com_position(self.model, self.data)[0:2].copy()

        hipr_int_wqp = np.zeros(len(self.groups["HipR"]), dtype=float)

        while self.data.time < sim_end:
            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                mujoco.mj_step(self.model, self.data)
                continue

            mujoco.mj_forward(self.model, self.data)
            now_t = float(self.data.time)
            dt_est = max(1e-4, now_t - prev_t)

            curr_roll, curr_pitch = get_rpy(self.data.qpos[3:7])

            if now_t >= first_push_start:
                max_torso_pitch = max(max_torso_pitch, abs(curr_pitch))
                max_torso_roll = max(max_torso_roll, abs(curr_roll))

            pelvis_height = float(self.data.xipos[self.pelvis_id, 2]) if self.pelvis_id >= 0 else float("inf")
            pitch_fallen = abs(curr_pitch) > float(self.args.fall_pitch_thresh)
            height_fallen = pelvis_height < float(self.args.fall_height_thresh)
            if (pitch_fallen or height_fallen) and not np.isfinite(fall_time):
                fall_time = now_t

            com = compute_com_position(self.model, self.data)
            com_xy = com[0:2]

            if now_t < first_push_start:
                pre_push_com_xy.append(com_xy.copy())
                if len(pre_push_com_xy) > pre_push_window_samples:
                    del pre_push_com_xy[: len(pre_push_com_xy) - pre_push_window_samples]
                com_xy_ref = np.mean(np.array(pre_push_com_xy), axis=0)

            if com_xy_ref is not None:
                com_err = com_xy - com_xy_ref
            else:
                com_err = np.zeros(2, dtype=float)

            com_vel = (com_xy - prev_com_xy) / dt_est

            tau_wqp, hipr_int_wqp, solve_ms, _fail = self._compute_wqp_torque(
                curr_pitch, curr_roll, prev_pitch, prev_roll, dt_est, hipr_int_wqp
            )
            qp_solve_times.append(float(solve_ms))

            tau_wqp_base6 = tau_wqp[self.mlp_output_indices].copy().astype(np.float32)
            state_vec = self._build_state_vec(
                curr_pitch,
                curr_roll,
                prev_pitch,
                prev_roll,
                dt_est,
                float(com_err[0]),
                float(com_err[1]),
                float(com_vel[0]),
                float(com_vel[1]),
                tau_base_6=tau_wqp_base6,
            )

            tau_applied = tau_wqp.copy()
            if mlp is not None and now_t >= first_push_start:
                with torch.no_grad():
                    x_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                    if device:
                        x_t = x_t.to(device)
                    delta = mlp(x_t).squeeze(0).cpu().numpy().astype(float)
                for k, idx in enumerate(self.mlp_output_indices):
                    tau_applied[idx] += float(delta[k])

            if now_t >= first_push_start:
                ankle_ids = self.groups["AnkleP"]
                knee_ids = self.knee_ids
                hip_ids = self.hip_ids
                tau_sq_sum_total += float(np.sum(tau_applied ** 2)) * self.dt
                tau_sq_sum_ankle += float(np.sum(tau_applied[ankle_ids] ** 2)) * self.dt
                tau_sq_sum_knee += float(np.sum(tau_applied[knee_ids] ** 2)) * self.dt
                tau_sq_sum_hip += float(np.sum(tau_applied[hip_ids] ** 2)) * self.dt
                tau_sq_duration += self.dt

            total_tau_sq_integral += float(np.sum(tau_applied ** 2)) * self.dt

            if now_t >= first_push_start:
                com_xy_disp = float(np.linalg.norm(com_err))
                max_com_xy_disp = max(max_com_xy_disp, com_xy_disp)
                max_com_x_disp = max(max_com_x_disp, abs(float(com_err[0])))
                max_com_y_disp = max(max_com_y_disp, abs(float(com_err[1])))
                com_sq_integral += (com_xy_disp ** 2) * self.dt
                com_integral_duration += self.dt

                if now_t >= last_push_end and not np.isfinite(recovery_time):
                    recovery_buffer.append(1 if com_xy_disp <= float(self.args.recovery_thresh) else 0)
                    if len(recovery_buffer) > recovery_window_samples:
                        recovery_buffer.pop(0)
                    if len(recovery_buffer) == recovery_window_samples and sum(recovery_buffer) >= int(
                        0.8 * recovery_window_samples
                    ):
                        recovery_time = now_t - last_push_end

            ctrl = np.clip(tau_applied / self.gear, self.ctrl_min, self.ctrl_max)
            self.data.ctrl[:] = ctrl

            self.data.xfrc_applied[:] = 0.0
            active = profile.active_push_at(now_t)
            if active is not None:
                bid = body_id_for.get(active.body_name, -1)
                if bid >= 0:
                    self.data.xfrc_applied[bid, 0:3] = active.force_vec

            prev_com_xy = com_xy.copy()
            prev_pitch = float(curr_pitch)
            prev_roll = float(curr_roll)
            prev_t = now_t
            mujoco.mj_step(self.model, self.data)
            if np.isfinite(fall_time):
                break

        recovery_time_out = float(recovery_time) if np.isfinite(recovery_time) else -1.0
        fall_time_out = float(fall_time) if np.isfinite(fall_time) else -1.0
        is_stable = 1 if (recovery_time_out >= 0.0 and fall_time_out < 0.0) else 0
        survival_time_s = float(fall_time_out) if fall_time_out > 0 else float(sim_end)

        com_safe_dur = max(float(com_integral_duration), 1e-6)
        rms_com_disp_m = float(np.sqrt(com_sq_integral / com_safe_dur))

        safe_dur = max(float(tau_sq_duration), 1e-6)
        tau_rms_total = float(np.sqrt(float(tau_sq_sum_total) / safe_dur))
        tau_rms_ankle = float(np.sqrt(float(tau_sq_sum_ankle) / safe_dur))
        tau_rms_knee = float(np.sqrt(float(tau_sq_sum_knee) / safe_dur))
        tau_rms_hip = float(np.sqrt(float(tau_sq_sum_hip) / safe_dur))

        qp_solve_time_mean_ms = float(np.mean(qp_solve_times)) if qp_solve_times else -1.0
        qp_solve_time_max_ms = float(np.max(qp_solve_times)) if qp_solve_times else -1.0
        qp_solver_failures = int(sum(1 for t in qp_solve_times if float(t) <= 0.0))

        n_pushes_delivered = profile.pushes_delivered_by(survival_time_s)

        if profile.pushes:
            fp = profile.pushes[0]
            mag_rep = float(profile.max_instantaneous_force_n)
            body_label_rep = fp.body_name
            force_vec_rep = fp.force_vec
        else:
            mag_rep = 0.0
            body_label_rep = "none"
            force_vec_rep = (0.0, 0.0, 0.0)

        return {
            "session_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "controller_name": "wqp_distilled",
            "condition_id": int(profile.seed),
            "trial_in_condition": 0,
            "body_label": body_label_rep,
            "body_name": body_label_rep,
            "direction": "random",
            "force_x": float(force_vec_rep[0]),
            "force_y": float(force_vec_rep[1]),
            "force_z": float(force_vec_rep[2]),
            "force_magnitude": float(mag_rep),
            "max_com_disp": float(max_com_xy_disp),
            "max_com_disp_x": float(max_com_x_disp),
            "max_com_disp_y": float(max_com_y_disp),
            "recovery_time": recovery_time_out,
            "fall_time": fall_time_out,
            "is_stable": int(is_stable),
            "tau_rms_total": float(tau_rms_total),
            "tau_rms_ankle": float(tau_rms_ankle),
            "tau_rms_knee": float(tau_rms_knee),
            "tau_rms_hip": float(tau_rms_hip),
            "max_torso_pitch_rad": float(max_torso_pitch),
            "max_torso_roll_rad": float(max_torso_roll),
            "qp_solve_time_mean_ms": float(qp_solve_time_mean_ms),
            "qp_solve_time_max_ms": float(qp_solve_time_max_ms),
            "qp_solver_failures": int(qp_solver_failures),
            "disturbance_seed": int(profile.seed),
            "n_pushes_scheduled": int(profile.n_pushes_scheduled),
            "n_pushes_delivered": int(n_pushes_delivered),
            "total_impulse_Ns": float(profile.total_impulse_ns),
            "max_instantaneous_force_N": float(profile.max_instantaneous_force_n),
            "mean_push_magnitude_N": float(profile.mean_push_magnitude_n),
            "push_log_json": profile.to_json(),
            "survival_time_s": float(survival_time_s),
            "rms_com_disp_m": float(rms_com_disp_m),
            "integrated_torque_Nms": float(total_tau_sq_integral),
        }


def run_collect(args: argparse.Namespace) -> None:
    if not MUJOCO_AVAILABLE:
        raise RuntimeError("MuJoCo not available. Cannot run collect mode.")
    if not DISTURBANCE_AVAILABLE:
        raise RuntimeError("disturbance.py not found. Place it in the same directory.")
    if not CVXPY_AVAILABLE:
        raise RuntimeError("cvxpy not available. Install with: pip install cvxpy osqp")

    engine = DistillationEngine(args)

    available_bodies = [
        name
        for name in BODY_WEIGHTS
        if mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    ]

    states_all: List[np.ndarray] = []
    tau_pass_all: List[np.ndarray] = []
    tau_wqp_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []

    delta_sum_all = 0.0
    delta_count_all = 0
    max_delta_seen = 0.0
    n_stable = 0

    for seed in range(int(args.n_collect_seeds)):
        profile = DisturbanceProfile.generate(
            seed=seed,
            sim_duration_s=float(args.sim_time),
            available_bodies=available_bodies,
        )

        res, s_list, tp_list, tw_list, y_list, d_sum, d_count, d_max = engine.run_trial_collect(
            profile, collect_all=bool(args.collect_all)
        )
        d_mean = (float(d_sum) / float(d_count)) if int(d_count) > 0 else 0.0

        stable_str = "STABLE" if int(res["is_stable"]) == 1 else "fell  "
        print(
            f"  [{stable_str}] seed={seed:4d} | "
            f"max_F={res['max_instantaneous_force_N']:5.1f}N | "
            f"survival={res['survival_time_s']:.2f}s | "
            f"delta_tau_mean={d_mean:.2f}Nm"
        )

        if int(res["is_stable"]) == 1:
            n_stable += 1

        states_all.extend(s_list)
        tau_pass_all.extend(tp_list)
        tau_wqp_all.extend(tw_list)
        targets_all.extend(y_list)
        if len(s_list) > 0:
            delta_sum_all += float(d_sum)
            delta_count_all += int(d_count)
        max_delta_seen = max(max_delta_seen, float(d_max))

    if not states_all:
        raise RuntimeError("No timesteps collected.")

    states_arr = np.array(states_all, dtype=np.float32)
    tau_pass_arr = np.array(tau_pass_all, dtype=np.float32)
    tau_wqp_arr = np.array(tau_wqp_all, dtype=np.float32)
    targets_arr = np.array(targets_all, dtype=np.float32)

    state_mean = states_arr.mean(axis=0)
    state_std = states_arr.std(axis=0) + 1e-8

    os.makedirs(str(args.output_dir), exist_ok=True)
    out_path = os.path.join(str(args.output_dir), str(args.dataset_file))
    np.savez_compressed(
        out_path,
        states=states_arr,
        tau_passivity_6=tau_pass_arr,
        tau_wqp_6=tau_wqp_arr,
        targets=targets_arr,
        state_mean=state_mean,
        state_std=state_std,
        input_dim=np.array(INPUT_DIM),
        output_dim=np.array(OUTPUT_DIM),
    )

    mean_delta_all = (float(delta_sum_all) / float(delta_count_all)) if delta_count_all > 0 else 0.0

    print("\nCOLLECT COMPLETE")
    print(f"  Timesteps: {len(states_arr):,}")
    print(f"  Stable trials: {n_stable}/{int(args.n_collect_seeds)}")
    print(f"  Mean delta_tau across all timesteps: {mean_delta_all:.2f} Nm")
    print(f"  Max delta_tau seen: {max_delta_seen:.2f} Nm")
    print(f"  State shape: {states_arr.shape}")
    print(f"  Target shape: {targets_arr.shape}")
    print(f"  Saved to: {out_path}")


def run_train(args: argparse.Namespace) -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")

    dataset_path = os.path.join(str(args.output_dir), str(args.dataset_file))
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}\nRun --mode collect first.")

    data_np = np.load(dataset_path)
    states_all = data_np["states"]
    targets_all = data_np["targets"]
    targets_all = np.clip(targets_all, -30.0, 30.0)
    state_mean = data_np["state_mean"]
    state_std = data_np["state_std"]

    if states_all.shape[1] != INPUT_DIM:
        raise ValueError(f"Dataset has {states_all.shape[1]} features but INPUT_DIM={INPUT_DIM}.")
    if targets_all.shape[1] != OUTPUT_DIM:
        raise ValueError(f"Dataset has {targets_all.shape[1]} outputs but OUTPUT_DIM={OUTPUT_DIM}.")

    states_norm = (states_all - state_mean) / state_std

    n_total = len(states_norm)
    n_train = int(n_total * 0.8)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_train = torch.tensor(states_norm[train_idx], dtype=torch.float32)
    y_train = torch.tensor(targets_all[train_idx], dtype=torch.float32)
    X_val = torch.tensor(states_norm[val_idx], dtype=torch.float32)
    y_val = torch.tensor(targets_all[val_idx], dtype=torch.float32)

    train_dl = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
    )
    val_dl = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistillationMLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, tau_bound=TAU_BOUND).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)
    reg_lambda = float(args.reg_lambda)

    best_val_loss = float("inf")
    patience_count = 0
    best_state_dict = None

    print("\nTraining...")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}")
    print("-" * 50)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            
            
            
            huber_loss = nn.functional.huber_loss(pred, y_b, delta=5.0)
            reg_loss = reg_lambda * (pred ** 2).mean()
            loss = huber_loss + reg_loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(X_b)
        train_loss = train_loss_sum / len(train_dl.dataset)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                
                           
                huber_loss = nn.functional.huber_loss(pred, y_b, delta=5.0)
                reg_loss = reg_lambda * (pred ** 2).mean()
                loss = huber_loss + reg_loss
                
                val_loss_sum += loss.item() * len(X_b)
        val_loss = val_loss_sum / len(val_dl.dataset)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {current_lr:>10.2e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= int(args.early_stop_patience):
                print(f"\nEarly stopping at epoch {epoch} (patience={int(args.early_stop_patience)})")
                break

    if best_state_dict is None:
        best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    os.makedirs(str(args.output_dir), exist_ok=True)
    weights_path = os.path.join(str(args.output_dir), str(args.weights_file))
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "state_mean": state_mean,
            "state_std": state_std,
            "input_dim": int(INPUT_DIM),
            "output_dim": int(OUTPUT_DIM),
            "hidden_dim": int(HIDDEN_DIM),
            "tau_bound": float(TAU_BOUND),
            "output_joints": OUTPUT_JOINT_NAMES,
            "best_val_loss": float(best_val_loss),
        },
        weights_path,
    )

    print("\nTRAIN COMPLETE")
    print(f"  Best val loss : {best_val_loss:.6f}")
    print(f"  Saved to      : {weights_path}")


def run_evaluate(args: argparse.Namespace) -> None:
    if not MUJOCO_AVAILABLE:
        raise RuntimeError("MuJoCo not available. Cannot run evaluate mode.")
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")
    if not DISTURBANCE_AVAILABLE:
        raise RuntimeError("disturbance.py not found. Place it in the same directory.")
    if not CVXPY_AVAILABLE:
        raise RuntimeError("cvxpy not available. Install with: pip install cvxpy osqp")

    weights_path = os.path.join(str(args.output_dir), str(args.weights_file))
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}\nRun --mode train first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if int(checkpoint["input_dim"]) != INPUT_DIM:
        raise ValueError(f"Weights input_dim={int(checkpoint['input_dim'])} but expected INPUT_DIM={INPUT_DIM}.")

    base_model = DistillationMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        tau_bound=float(checkpoint["tau_bound"]),
    ).to(device)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()

    state_mean = checkpoint["state_mean"]
    state_std = checkpoint["state_std"]

    class NormalisedMLP(nn.Module):
        def __init__(self, base: nn.Module, mean: np.ndarray, std: np.ndarray):
            super().__init__()
            self.base = base
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if x.device != self.mean.device:
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)
            return self.base((x - self.mean) / self.std)

    mlp = NormalisedMLP(base_model, state_mean, state_std).to(device)
    mlp.eval()

    engine = DistillationEngine(args)

    available_bodies = [
        name
        for name in BODY_WEIGHTS
        if mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    ]

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(str(args.output_dir), exist_ok=True)
    csv_path = os.path.join(str(args.output_dir), f"wqp_distilled_random_{session_id}.csv")

    all_results: List[Dict[str, Any]] = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for seed in range(int(args.n_eval_seeds)):
            profile = DisturbanceProfile.generate(
                seed=seed,
                sim_duration_s=float(args.sim_time),
                available_bodies=available_bodies,
            )
            res = engine.run_trial_evaluate(profile, mlp=mlp, device=device)
            all_results.append(res)
            stable_str = "STABLE" if int(res["is_stable"]) == 1 else "FELL  "
            print(
                f"  [{stable_str}] seed={seed:4d} | "
                f"max_F={res['max_instantaneous_force_N']:5.1f}N | "
                f"survival={res['survival_time_s']:.2f}s | "
                f"rms_com={res['rms_com_disp_m']:.4f}m"
            )
            writer.writerow({k: res.get(k, "") for k in CSV_FIELDNAMES})
            f.flush()

    n = len(all_results)
    n_stable = sum(int(r["is_stable"]) for r in all_results)
    fall_count = sum(1 for r in all_results if float(r["fall_time"]) > 0.0)
    neither = n - n_stable - fall_count
    mean_surv = float(np.mean([float(r["survival_time_s"]) for r in all_results]))

    print("\nEVALUATE COMPLETE")
    print(f"  Stable: {n_stable}/{n} ({100.0 * n_stable / max(1, n):.1f}%)")
    print("  WQP baseline: 103/200 (51.5%)")
    print("  Passivity baseline: 110/200 (55.0%)")
    print(f"  Fell: {fall_count}/{n}")
    print(f"  Neither: {neither}/{n}")
    print(f"  Mean survival: {mean_surv:.2f} s")
    print(f"  CSV saved: {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-controller distillation: WQP learns passivity corrections")
    p.add_argument("--mode", type=str, required=True, choices=["collect", "train", "evaluate"])
    p.add_argument(
        "--xml",
        type=str,
        default=r"C:\Users\sanju\Documents\University_West\Course\Thesis\Scripts\venv310\model\New_MOdels\mujoco_menagerie-main\unitree_h1\scene.xml",
    )
    p.add_argument("--output-dir", type=str, default="experiment_results")
    p.add_argument("--dataset-file", type=str, default="distillation_dataset.npz")
    p.add_argument("--weights-file", type=str, default="distillation_weights.pt")
    p.add_argument("--n-collect-seeds", type=int, default=200)
    p.add_argument("--n-eval-seeds", type=int, default=200)

    p.add_argument("--sim-time", type=float, default=10.0)
    p.add_argument("--kp-passivity", type=float, default=200.0)
    p.add_argument("--kd-passivity", type=float, default=20.0)
    p.add_argument("--kp-tilt", type=float, default=500.0)
    p.add_argument("--kd-tilt", type=float, default=50.0)
    p.add_argument("--kp-roll", type=float, default=500.0)
    p.add_argument("--kd-roll", type=float, default=50.0)
    p.add_argument("--damping-gain", type=float, default=20.0)
    p.add_argument("--target-pitch", type=float, default=0.05)

    p.add_argument("--fall-pitch-thresh", type=float, default=1.2)
    p.add_argument("--fall-height-thresh", type=float, default=0.65)
    p.add_argument("--recovery-thresh", type=float, default=0.02)
    p.add_argument("--recovery-window", type=float, default=0.5)
    p.add_argument("--pre-push-window", type=float, default=1.0)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg-lambda", type=float, default=0.0001)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--collect-all", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "collect":
        run_collect(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
