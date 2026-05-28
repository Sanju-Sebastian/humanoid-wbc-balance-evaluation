"""
Automated humanoid disturbance experiments for Passivity-based controller.

Modes:
1) Batch mode (--mode batch): Part 1a step-push grid.
2) Random mode (--mode random): Part 1b seeded multi-push stochastic trials.
3) Debug mode (--mode debug): single visualized condition.
"""

import argparse
import csv
import datetime
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

# Part 1b random multi-push disturbance generator.
try:
    from disturbance import BODY_WEIGHTS, DisturbanceProfile, Push
except ModuleNotFoundError:
    DisturbanceProfile = None
    Push = None
    BODY_WEIGHTS = {}


PUSH_LOCATION_PRESETS = {
    "torso":  ["torso_link"],
    "pelvis": ["pelvis"],
}

DIRECTION_PRESETS = {
    "forward":  np.array([ 1.0, 0.0, 0.0], dtype=float),
    "backward": np.array([-1.0, 0.0, 0.0], dtype=float),
    "lateral":  np.array([ 0.0, 1.0, 0.0], dtype=float),
}

def get_rpy(q: np.ndarray) -> Tuple[float, float]:
    w, x, y, z = q
    roll  = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(np.clip(2 * (w * y - z * x), -1, 1))
    return float(roll), float(pitch)

def safe_reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if getattr(model, "nkey", 0) > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

def compute_com_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    masses = model.body_mass
    total_mass = float(np.sum(masses))
    return (masses[:, None] * data.xipos).sum(axis=0) / total_mass

def build_actuator_mappings(model: mujoco.MjModel) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
        "AnkleP": [i for i, n in enumerate(actuator_names) if "ankle"     in n],
        "HipP":   [i for i, n in enumerate(actuator_names) if "hip_pitch" in n],
        "KneeP":  [i for i, n in enumerate(actuator_names) if "knee"      in n],
        "HipR":   [i for i, n in enumerate(actuator_names) if "hip_roll"  in n],
    }

def build_gains(
    model: mujoco.MjModel,
    groups: Dict[str, List[int]],
    kp_posture: float,
    kd_posture: float,
) -> Tuple[np.ndarray, np.ndarray]:
    kp = np.full(model.nu, float(kp_posture), dtype=float)
    kd = np.full(model.nu, float(kd_posture), dtype=float)
    return kp, kd

def select_body(model: mujoco.MjModel, candidates: List[str]) -> Tuple[str, int]:
    for name in candidates:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return str(name), int(body_id)
    return "", -1

def calibrate_balance_sign(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ankle_ids: List[int],
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

def parse_csv_list(text: str) -> List[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]

def parse_csv_floats(text: str) -> List[float]:
    return [float(s) for s in parse_csv_list(text)]


class HumanoidExperimentEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.model = mujoco.MjModel.from_xml_path(args.xml)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

        self.qaddr, self.daddr, self.actuator_names = build_actuator_mappings(self.model)
        self.groups = build_groups(self.actuator_names)

        self.knee_ids = self.groups["KneeP"]
        self.hip_ids  = [i for i, n in enumerate(self.actuator_names)
                         if "hip_pitch" in n or "hip_roll" in n or "hip_yaw" in n]

        self.kp, self.kd = build_gains(
            self.model, self.groups, self.args.kp_posture, self.args.kd_posture
        )

        self.gear = self.model.actuator_gear[:, 0].copy()
        self.gear[self.gear == 0.0] = 1.0
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1]

        self.tilt_sign = calibrate_balance_sign(
            self.model, self.data, self.groups["AnkleP"]
        )
        mujoco.mj_forward(self.model, self.data)
        self.q_target_full = self.data.qpos.copy()

        self.output_dir = str(self.args.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        controller_label = getattr(self.args, "controller_name", "controller")
        self.csv_path = os.path.join(
            self.output_dir, f"{controller_label}_{self.session_id}.csv"
        )

        self.all_results: List[Dict[str, Any]] = []
        self.rng = np.random.default_rng(int(self.args.seed))

    def _apply_small_init_noise(self) -> None:
        noise_std = float(self.args.init_noise)
        if noise_std <= 0.0:
            return
        self.data.qpos[self.qaddr] += self.rng.normal(
            0.0, noise_std, size=self.data.qpos[self.qaddr].shape
        )
        self.data.qpos[3:7] += self.rng.normal(0.0, noise_std * 0.1, size=4)
        q = self.data.qpos[3:7]
        self.data.qpos[3:7] = q / np.linalg.norm(q)
        mujoco.mj_forward(self.model, self.data)

    def _compute_passivity_torque(
        self,
        curr_pitch: float,
        curr_roll: float,
        prev_pitch: float,
        prev_roll: float,
        dt_est: float,
        hipr_int: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the 4-layer passivity-based control torque.

        Returns:
            tau_wbc: full actuator torque vector (model.nu,)
            hipr_int_new: updated hip-roll integrator state
        """
        # Layer 1: Gravity compensation
        tau_g = self.data.qfrc_bias[self.daddr].copy()

        # Layer 2: Posture PD (zeroed for balance joints via tau_e below)
        q_curr = self.data.qpos[self.qaddr]
        tau_pd = self.kp * (self.q_target_full[self.qaddr] - q_curr)

        # Layer 3: Damping injection (Passivity condition — kinetic energy dissipation)
        dq_curr = self.data.qvel[self.daddr]
        tau_d = -float(self.args.damping_gain) * dq_curr

        # Layer 4: Energy shaping (Balance)
        tau_e = np.zeros(self.model.nu, dtype=float)

        # Pitch stabiliser
        d_pitch = (curr_pitch - prev_pitch) / dt_est
        pitch_error = curr_pitch - float(self.args.target_pitch)
        tilt_corr = -self.tilt_sign * (
            float(self.args.kp_tilt) * pitch_error
            + float(self.args.kd_tilt) * d_pitch
        )
        tilt_corr = float(np.clip(tilt_corr, -150.0, 150.0))
        for idx in self.groups["AnkleP"]:
            tau_e[idx] += tilt_corr
        for idx in self.groups["HipP"]:
            tau_e[idx] += 0.5 * tilt_corr

        # Roll stabiliser
        d_roll = (curr_roll - prev_roll) / dt_est
        roll_corr = -(
            float(self.args.kp_roll) * curr_roll
            + float(self.args.kd_roll) * d_roll
        )

        # HipR PID hold
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

    def run_single_trial(
        self, config: Dict[str, Any], trial_in_condition: int, visualize: bool
    ) -> Dict[str, Any]:
        safe_reset(self.model, self.data)
        self._apply_small_init_noise()
        mujoco.mj_forward(self.model, self.data)

        push_body_id   = int(config["body_id"])
        push_body_name = str(config["body_name"])
        force_vec      = np.array(config["force_vec"], dtype=float)
        direction      = str(config["dir_name"])
        magnitude      = float(config["mag"])

        push_start    = float(self.args.push_start)
        push_duration = float(self.args.push_duration)
        push_end      = push_start + push_duration
        sim_end       = float(self.args.sim_time)

        pre_push_window_samples = max(1, int(round(float(self.args.pre_push_window) / self.dt)))
        recovery_window_samples = max(1, int(round(float(self.args.recovery_window) / self.dt)))

        pre_push_com_xy: List[np.ndarray] = []
        com_xy_ref = None
        max_com_xy_disp = 0.0
        max_com_x_disp = 0.0
        max_com_y_disp = 0.0
        max_torso_pitch = 0.0
        max_torso_roll = 0.0
        recovery_time = float("nan")
        recovery_buffer: List[int] = []
        fall_time = float("nan")

        tau_sq_sum_total = 0.0
        tau_sq_sum_ankle = 0.0
        tau_sq_sum_knee = 0.0
        tau_sq_sum_hip = 0.0
        tau_sq_duration = 0.0

        _r0, _p0 = get_rpy(self.data.qpos[3:7])
        prev_pitch = float(_p0)
        prev_roll = float(_r0)
        prev_t = float(self.data.time)

        hipr_int = np.zeros(len(self.groups["HipR"]), dtype=float)

        def step_once() -> None:
            nonlocal com_xy_ref, fall_time, max_com_x_disp, max_com_xy_disp, \
                max_com_y_disp, max_torso_pitch, max_torso_roll, prev_pitch, \
                prev_roll, prev_t, recovery_buffer, recovery_time, tau_sq_duration, \
                tau_sq_sum_ankle, tau_sq_sum_hip, tau_sq_sum_knee, tau_sq_sum_total, \
                hipr_int

            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                mujoco.mj_step(self.model, self.data)
                return

            mujoco.mj_forward(self.model, self.data)
            now_t = float(self.data.time)
            dt_est = max(1e-4, now_t - prev_t)

            curr_roll, curr_pitch = get_rpy(self.data.qpos[3:7])

            if now_t >= push_start:
                max_torso_pitch = max(max_torso_pitch, abs(curr_pitch))
                max_torso_roll = max(max_torso_roll, abs(curr_roll))

            pelvis_height = (
                float(self.data.xipos[self.pelvis_id, 2])
                if self.pelvis_id >= 0 else float("inf")
            )
            if (abs(curr_pitch) > float(self.args.fall_pitch_thresh)
                    or pelvis_height < float(self.args.fall_height_thresh)) \
                    and not np.isfinite(fall_time):
                fall_time = now_t

            tau_wbc, hipr_int = self._compute_passivity_torque(
                curr_pitch, curr_roll, prev_pitch, prev_roll, dt_est, hipr_int
            )

            ctrl = np.clip(tau_wbc / self.gear, self.ctrl_min, self.ctrl_max)
            self.data.ctrl[:] = ctrl

            if now_t >= push_start:
                ankle_ids = self.groups["AnkleP"]
                tau_sq_sum_total += float(np.sum(tau_wbc ** 2)) * self.dt
                tau_sq_sum_ankle += float(np.sum(tau_wbc[ankle_ids] ** 2)) * self.dt
                tau_sq_sum_knee  += float(np.sum(tau_wbc[self.knee_ids] ** 2)) * self.dt
                tau_sq_sum_hip   += float(np.sum(tau_wbc[self.hip_ids]  ** 2)) * self.dt
                tau_sq_duration  += self.dt

            self.data.xfrc_applied[:] = 0.0
            if push_body_id >= 0 and push_start <= now_t < push_end:
                self.data.xfrc_applied[push_body_id, 0:3] = force_vec

            com = compute_com_position(self.model, self.data)
            if now_t < push_start:
                pre_push_com_xy.append(com[0:2].copy())
                if len(pre_push_com_xy) > pre_push_window_samples:
                    del pre_push_com_xy[: len(pre_push_com_xy) - pre_push_window_samples]
                com_xy_ref = np.mean(np.array(pre_push_com_xy), axis=0)

            if com_xy_ref is not None:
                com_xy_disp = float(np.linalg.norm(com[0:2] - com_xy_ref))
                if now_t >= push_start:
                    max_com_xy_disp = max(max_com_xy_disp, com_xy_disp)
                    max_com_x_disp = max(max_com_x_disp, abs(float(com[0] - com_xy_ref[0])))
                    max_com_y_disp = max(max_com_y_disp, abs(float(com[1] - com_xy_ref[1])))

                if now_t >= push_end and not np.isfinite(recovery_time):
                    recovery_buffer.append(
                        1 if com_xy_disp <= float(self.args.recovery_thresh) else 0
                    )
                    if len(recovery_buffer) > recovery_window_samples:
                        recovery_buffer.pop(0)
                    if (len(recovery_buffer) == recovery_window_samples
                            and sum(recovery_buffer) >= int(0.8 * recovery_window_samples)):
                        recovery_time = now_t - push_end

            prev_pitch = curr_pitch
            prev_roll = curr_roll
            prev_t = now_t
            mujoco.mj_step(self.model, self.data)

        if visualize:
            try:
                with mujoco.viewer.launch_passive(self.model, self.data) as v:
                    while v.is_running() and self.data.time < sim_end:
                        step_once()
                        v.sync()
                        if int(self.args.viz_realtime) == 1:
                            time.sleep(self.dt)
                        if np.isfinite(fall_time):
                            break
                    if int(self.args.viz_hold) == 1:
                        print("Close viewer to continue...")
                        while v.is_running():
                            v.sync()
                            time.sleep(0.01)
            except Exception as e:
                print(f"WARNING: Visualization failed ({type(e).__name__}). Headless fallback.")
                while self.data.time < sim_end:
                    step_once()
                    if np.isfinite(fall_time):
                        break
        else:
            while self.data.time < sim_end:
                step_once()
                if np.isfinite(fall_time):
                    break

        safe_dur = max(float(tau_sq_duration), 1e-6)
        tau_rms_total = float(np.sqrt(float(tau_sq_sum_total) / safe_dur))
        tau_rms_ankle = float(np.sqrt(float(tau_sq_sum_ankle) / safe_dur))
        tau_rms_knee = float(np.sqrt(float(tau_sq_sum_knee) / safe_dur))
        tau_rms_hip = float(np.sqrt(float(tau_sq_sum_hip) / safe_dur))

        recovery_time_out = float(recovery_time) if np.isfinite(recovery_time) else -1.0
        fall_time_out = float(fall_time) if np.isfinite(fall_time) else -1.0
        is_stable = 1 if (recovery_time_out >= 0.0 and fall_time_out < 0.0) else 0

        return {
            "session_id": self.session_id,
            "controller_name": str(self.args.controller_name),
            "condition_id": int(config["condition_id"]),
            "trial_in_condition": int(trial_in_condition),
            "body_label": str(config["body_label"]),
            "body_name": push_body_name,
            "direction": direction,
            "force_x": float(force_vec[0]),
            "force_y": float(force_vec[1]),
            "force_z": float(force_vec[2]),
            "force_magnitude": float(magnitude),
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
            "qp_solve_time_mean_ms": 0.0,
            "qp_solve_time_max_ms": 0.0,
            "qp_solver_failures": 0,
            # Part 1b columns — sentinels for Part 1a trials.
            "disturbance_seed": -1,
            "n_pushes_scheduled": 1,
            "n_pushes_delivered": 1,
            "total_impulse_Ns": float(magnitude * push_duration),
            "max_instantaneous_force_N": float(magnitude),
            "mean_push_magnitude_N": float(magnitude),
            "push_log_json": "",
            "survival_time_s": float(fall_time_out if fall_time_out > 0 else sim_end),
            "rms_com_disp_m": -1.0,
            "integrated_torque_Nms": -1.0,
        }

    # ==================================================================
    # Part 1b: Randomised multi-push trials (stochastic disturbances)
    # ==================================================================

    def _resolve_available_push_bodies(self) -> List[str]:
        """Check which of the six canonical push bodies exist in the MJCF."""
        candidates = list(BODY_WEIGHTS.keys())
        available: List[str] = []
        for name in candidates:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                available.append(name)
        return available

    def run_random_trial(self, profile: "DisturbanceProfile", visualize: bool = False) -> Dict[str, Any]:
        """One Part 1b trial under a seeded random disturbance profile."""
        safe_reset(self.model, self.data)
        self._apply_small_init_noise()
        mujoco.mj_forward(self.model, self.data)

        sim_end = float(self.args.sim_time)

        # Pre-resolve body ids so step_once doesn't do name lookups every step.
        body_id_for: Dict[str, int] = {}
        for p in profile.pushes:
            if p.body_name not in body_id_for:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, p.body_name)
                body_id_for[p.body_name] = int(bid)

        first_push_start = profile.pushes[0].start_time_s if profile.pushes else 1.0
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

        tau_sq_sum_total = 0.0
        tau_sq_sum_ankle = 0.0
        tau_sq_sum_knee = 0.0
        tau_sq_sum_hip = 0.0
        tau_sq_duration = 0.0

        # Part 1b metrics.
        com_sq_integral = 0.0
        com_integral_duration = 0.0
        total_tau_sq_integral = 0.0

        _r0, _p0 = get_rpy(self.data.qpos[3:7])
        prev_pitch = float(_p0)
        prev_roll = float(_r0)
        prev_t = float(self.data.time)

        hipr_int = np.zeros(len(self.groups["HipR"]), dtype=float)

        def step_once() -> None:
            nonlocal com_xy_ref, fall_time, max_com_x_disp, max_com_xy_disp, \
                max_com_y_disp, max_torso_pitch, max_torso_roll, prev_pitch, \
                prev_roll, prev_t, recovery_buffer, recovery_time, tau_sq_duration, \
                tau_sq_sum_ankle, tau_sq_sum_hip, tau_sq_sum_knee, tau_sq_sum_total, \
                hipr_int, com_sq_integral, com_integral_duration, total_tau_sq_integral

            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                mujoco.mj_step(self.model, self.data)
                return

            mujoco.mj_forward(self.model, self.data)
            now_t = float(self.data.time)
            dt_est = max(1e-4, now_t - prev_t)

            curr_roll, curr_pitch = get_rpy(self.data.qpos[3:7])

            if now_t >= first_push_start:
                max_torso_pitch = max(max_torso_pitch, abs(curr_pitch))
                max_torso_roll = max(max_torso_roll, abs(curr_roll))

            pelvis_height = (
                float(self.data.xipos[self.pelvis_id, 2])
                if self.pelvis_id >= 0 else float("inf")
            )
            if (abs(curr_pitch) > float(self.args.fall_pitch_thresh)
                    or pelvis_height < float(self.args.fall_height_thresh)) \
                    and not np.isfinite(fall_time):
                fall_time = now_t

            tau_wbc, hipr_int = self._compute_passivity_torque(
                curr_pitch, curr_roll, prev_pitch, prev_roll, dt_est, hipr_int
            )

            ctrl = np.clip(tau_wbc / self.gear, self.ctrl_min, self.ctrl_max)
            self.data.ctrl[:] = ctrl

            if now_t >= first_push_start:
                ankle_ids = self.groups["AnkleP"]
                tau_sq_sum_total += float(np.sum(tau_wbc ** 2)) * self.dt
                tau_sq_sum_ankle += float(np.sum(tau_wbc[ankle_ids] ** 2)) * self.dt
                tau_sq_sum_knee  += float(np.sum(tau_wbc[self.knee_ids] ** 2)) * self.dt
                tau_sq_sum_hip   += float(np.sum(tau_wbc[self.hip_ids]  ** 2)) * self.dt
                tau_sq_duration  += self.dt

            total_tau_sq_integral += float(np.sum(tau_wbc ** 2)) * self.dt

            # Apply scheduled push(es).
            self.data.xfrc_applied[:] = 0.0
            active = profile.active_push_at(now_t)
            if active is not None:
                bid = body_id_for.get(active.body_name, -1)
                if bid >= 0:
                    self.data.xfrc_applied[bid, 0:3] = active.force_vec

            com = compute_com_position(self.model, self.data)
            if now_t < first_push_start:
                pre_push_com_xy.append(com[0:2].copy())
                if len(pre_push_com_xy) > pre_push_window_samples:
                    del pre_push_com_xy[: len(pre_push_com_xy) - pre_push_window_samples]
                com_xy_ref = np.mean(np.array(pre_push_com_xy), axis=0)

            if com_xy_ref is not None:
                com_xy_disp = float(np.linalg.norm(com[0:2] - com_xy_ref))
                if now_t >= first_push_start:
                    max_com_xy_disp = max(max_com_xy_disp, com_xy_disp)
                    max_com_x_disp = max(max_com_x_disp, abs(float(com[0] - com_xy_ref[0])))
                    max_com_y_disp = max(max_com_y_disp, abs(float(com[1] - com_xy_ref[1])))
                    com_sq_integral += (com_xy_disp ** 2) * self.dt
                    com_integral_duration += self.dt

                last_push_end = profile.pushes[-1].end_time_s if profile.pushes else first_push_start
                if now_t >= last_push_end and not np.isfinite(recovery_time):
                    recovery_buffer.append(1 if com_xy_disp <= float(self.args.recovery_thresh) else 0)
                    if len(recovery_buffer) > recovery_window_samples:
                        recovery_buffer.pop(0)
                    if len(recovery_buffer) == recovery_window_samples and sum(recovery_buffer) >= int(0.8 * recovery_window_samples):
                        recovery_time = now_t - last_push_end

            prev_pitch = curr_pitch
            prev_roll = curr_roll
            prev_t = now_t
            mujoco.mj_step(self.model, self.data)

        if visualize:
            try:
                with mujoco.viewer.launch_passive(self.model, self.data) as v:
                    while v.is_running() and self.data.time < sim_end:
                        step_once()
                        v.sync()
                        if int(self.args.viz_realtime) == 1:
                            time.sleep(self.dt)
                        if np.isfinite(fall_time):
                            break
                    if int(self.args.viz_hold) == 1:
                        print("  Trial ended. Close the viewer window to continue...")
                        while v.is_running():
                            v.sync()
                            time.sleep(0.01)
            except Exception as e:
                print(f"  [WARN] Viewer failed ({type(e).__name__}: {e}). Headless fallback.")
                while self.data.time < sim_end:
                    step_once()
                    if np.isfinite(fall_time):
                        break
        else:
            while self.data.time < sim_end:
                step_once()
                if np.isfinite(fall_time):
                    break

        safe_dur = max(float(tau_sq_duration), 1e-6)
        tau_rms_total = float(np.sqrt(float(tau_sq_sum_total) / safe_dur))
        tau_rms_ankle = float(np.sqrt(float(tau_sq_sum_ankle) / safe_dur))
        tau_rms_knee = float(np.sqrt(float(tau_sq_sum_knee) / safe_dur))
        tau_rms_hip = float(np.sqrt(float(tau_sq_sum_hip) / safe_dur))

        recovery_time_out = float(recovery_time) if np.isfinite(recovery_time) else -1.0
        fall_time_out = float(fall_time) if np.isfinite(fall_time) else -1.0
        is_stable = 1 if (recovery_time_out >= 0.0 and fall_time_out < 0.0) else 0

        survival_time_s = float(fall_time_out) if fall_time_out > 0 else float(sim_end)
        com_safe_dur = max(float(com_integral_duration), 1e-6)
        rms_com_disp_m = float(np.sqrt(com_sq_integral / com_safe_dur))
        n_pushes_delivered = profile.pushes_delivered_by(survival_time_s)

        if profile.pushes:
            first_push = profile.pushes[0]
            force_x_rep = float(first_push.force_vec[0])
            force_y_rep = float(first_push.force_vec[1])
            force_z_rep = float(first_push.force_vec[2])
            body_label_rep = first_push.body_name
            body_name_rep = first_push.body_name
            direction_rep = "random"
            mag_rep = float(profile.max_instantaneous_force_n)
        else:
            force_x_rep = 0.0
            force_y_rep = 0.0
            force_z_rep = 0.0
            body_label_rep = "none"
            body_name_rep = "none"
            direction_rep = "none"
            mag_rep = 0.0

        result = {
            "session_id": self.session_id,
            "controller_name": str(self.args.controller_name),
            "condition_id": int(profile.seed),
            "trial_in_condition": 0,
            "body_label": body_label_rep,
            "body_name": body_name_rep,
            "direction": direction_rep,
            "force_x": force_x_rep,
            "force_y": force_y_rep,
            "force_z": force_z_rep,
            "force_magnitude": mag_rep,
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
            "qp_solve_time_mean_ms": 0.0,
            "qp_solve_time_max_ms": 0.0,
            "qp_solver_failures": 0,
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
        return result

    def run_random_batch(self) -> None:
        if DisturbanceProfile is None:
            raise RuntimeError("disturbance.py is missing. Place it in the same directory as this script.")

        seed_start = int(self.args.seed_start)
        n_trials = int(self.args.n_random_trials)
        sim_duration = float(self.args.sim_time)
        visualize = bool(int(getattr(self.args, "visualize_random", 0)))

        available_bodies = self._resolve_available_push_bodies()
        if len(available_bodies) < len(BODY_WEIGHTS):
            print(
                f"[WARN] Only {len(available_bodies)}/{len(BODY_WEIGHTS)} canonical push bodies "
                f"found in MJCF. Missing bodies will be excluded. Present: {available_bodies}"
            )

        controller_label = getattr(self.args, "controller_name", "controller")
        self.csv_path = os.path.join(self.output_dir, f"{controller_label}_random_{self.session_id}.csv")

        print(f"\nPart 1b random mode")
        print(f"- Seeds: [{seed_start}, {seed_start + n_trials - 1}] ({n_trials} trials)")
        print(f"- Sim duration: {sim_duration} s per trial")
        print(f"- Available push bodies: {available_bodies}")
        print(f"- CSV output: {self.csv_path}")
        if visualize:
            print(f"- Visualization: ON (each trial opens viewer window)")

        header_written = False
        with open(self.csv_path, "w", newline="") as f:
            writer = None
            for i in range(n_trials):
                seed = seed_start + i
                profile = DisturbanceProfile.generate(
                    seed=seed,
                    sim_duration_s=sim_duration,
                    available_bodies=available_bodies,
                )
                res = self.run_random_trial(profile, visualize=visualize)
                self.all_results.append(res)

                stable_str = "STABLE" if int(res["is_stable"]) == 1 else "FELL  "
                print(
                    f"  [{stable_str}] seed={seed:5d} | "
                    f"n_pushes={int(res['n_pushes_scheduled'])}/{int(res['n_pushes_delivered'])} | "
                    f"max_F={float(res['max_instantaneous_force_N']):5.1f}N | "
                    f"survival={float(res['survival_time_s']):.2f}s | "
                    f"max_com={float(res['max_com_disp']):.3f}m | "
                    f"rms_com={float(res['rms_com_disp_m']):.4f}m"
                )

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(res.keys()))
                if not header_written:
                    writer.writeheader()
                    header_written = True
                writer.writerow(res)
                f.flush()

        print("\n==============================")
        print("RANDOM BATCH COMPLETE")
        print(f"Saved: {self.csv_path}")
        self._print_random_summary()

    def _print_random_summary(self) -> None:
        if not self.all_results:
            return
        rs = self.all_results
        n = len(rs)
        n_stable = sum(int(r["is_stable"]) for r in rs)
        fall_count = sum(1 for r in rs if float(r["fall_time"]) > 0)
        mean_max_force = float(np.mean([float(r["max_instantaneous_force_N"]) for r in rs]))
        mean_survival = float(np.mean([float(r["survival_time_s"]) for r in rs]))

        print(f"\nQuick Summary (n={n})")
        print(f"  Stable trials:      {n_stable}/{n} ({100.0*n_stable/n:.1f}%)")
        print(f"  Trials with fall:   {fall_count}/{n}")
        print(f"  Mean max force:     {mean_max_force:.1f} N")
        print(f"  Mean survival time: {mean_survival:.2f} s")

    def _should_visualize_trial(
        self, config: Dict[str, Any], trial_in_condition: int, viz_count_so_far: int
    ) -> bool:
        policy = str(self.args.viz_policy)
        if policy == "none":
            return False
        if viz_count_so_far >= int(self.args.viz_max):
            return False
        if policy == "first_trial_each_condition":
            return trial_in_condition == 0
        if policy == "only_force":
            return (trial_in_condition == 0
                    and float(config["mag"]) == float(self.args.viz_force))
        if policy == "every_nth_condition":
            step = max(1, int(self.args.viz_step))
            return (trial_in_condition == 0
                    and int(config["condition_id"]) % step == 0)
        return False

    def _print_trial_result(
        self, cfg: Dict[str, Any], trial_num: int, total_trials: int, res: Dict[str, Any]
    ) -> None:
        stable_str = "STABLE" if int(res["is_stable"]) == 1 else "FELL  "
        rt = float(res["recovery_time"])
        rt_str = f"{rt:.2f}s" if rt > 0 else "none"
        print(
            f"  [{stable_str}] "
            f"Cond {int(cfg['condition_id']):03d} | "
            f"Trial {trial_num + 1}/{total_trials} | "
            f"{str(cfg['body_label']):<6} {str(cfg['dir_name']):<8} {float(cfg['mag']):>5.0f}N | "
            f"CoM={float(res['max_com_disp']):.3f}m | "
            f"RT={rt_str} | "
            f"tau_rms={float(res['tau_rms_total']):.1f}Nm"
        )

    def run_batch_experiments(self) -> None:
        magnitudes = parse_csv_floats(self.args.magnitudes)
        target_labels = parse_csv_list(self.args.targets)
        direction_names = parse_csv_list(self.args.directions)
        num_trials = int(self.args.num_trials)

        configs: List[Dict[str, Any]] = []
        condition_id = 0
        for label in target_labels:
            if label not in PUSH_LOCATION_PRESETS:
                continue
            body_name, body_id = select_body(self.model, PUSH_LOCATION_PRESETS[label])
            if body_id < 0:
                continue
            for mag in magnitudes:
                for dname in direction_names:
                    if dname not in DIRECTION_PRESETS:
                        continue
                    force_vec = (DIRECTION_PRESETS[dname] * float(mag)).tolist()
                    configs.append({
                        "condition_id": condition_id,
                        "body_label": label,
                        "body_name": body_name,
                        "body_id": body_id,
                        "dir_name": dname,
                        "mag": float(mag),
                        "force_vec": force_vec,
                    })
                    condition_id += 1

        print(f"Batch mode: {len(configs)} conditions, {num_trials} trials each")
        print(f"Total simulations: {len(configs) * num_trials}")
        print(f"CSV output: {self.csv_path}")

        header_written = False
        viz_count = 0

        with open(self.csv_path, "w", newline="") as f:
            writer = None
            for cfg in configs:
                print(
                    f"Condition {cfg['condition_id']:04d} | "
                    f"{cfg['body_label']} | {cfg['dir_name']} | {cfg['mag']}N"
                )
                for t in range(num_trials):
                    visualize = self._should_visualize_trial(cfg, t, viz_count)
                    if visualize:
                        viz_count += 1
                    res = self.run_single_trial(cfg, t, visualize=visualize)
                    self.all_results.append(res)
                    self._print_trial_result(cfg, t, num_trials, res)
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(res.keys()))
                    if not header_written:
                        writer.writeheader()
                        header_written = True
                    writer.writerow(res)
                    f.flush()

        print("\n==============================")
        print("BATCH COMPLETE")
        print(f"Saved: {self.csv_path}")
        self.print_summary_stats()

    def run_debug_visualization(self) -> None:
        target_label = str(self.args.debug_target)
        direction = str(self.args.debug_direction)
        magnitude = float(self.args.debug_force)

        if target_label in PUSH_LOCATION_PRESETS:
            body_name, body_id = select_body(self.model, PUSH_LOCATION_PRESETS[target_label])
            body_label = target_label
        else:
            body_label = "custom"
            body_name, body_id = select_body(self.model, [target_label])

        if body_id < 0:
            raise RuntimeError(f"Debug target body not found: {target_label}")
        if direction not in DIRECTION_PRESETS:
            raise RuntimeError(f"Unknown direction: {direction}")

        force_vec = (DIRECTION_PRESETS[direction] * magnitude).tolist()
        cfg = {
            "condition_id": 0,
            "body_label": body_label,
            "body_name": body_name,
            "body_id": body_id,
            "dir_name": direction,
            "mag": magnitude,
            "force_vec": force_vec,
        }

        print("Debug mode")
        print(f"- body:       {cfg['body_name']}")
        print(f"- direction:  {cfg['dir_name']}")
        print(f"- force:      {cfg['mag']} N")
        print(f"- sim_time:   {self.args.sim_time} s")
        print(f"- tilt_sign:  {self.tilt_sign}")
        print(f"- damping:    {self.args.damping_gain}")
        if int(self.args.debug_visualize) == 1:
            print("Viewer window will open.")
        else:
            print("Running headless.")

        prev_noise = float(self.args.init_noise)
        self.args.init_noise = 0.0
        res = self.run_single_trial(
            cfg, trial_in_condition=0,
            visualize=bool(int(self.args.debug_visualize))
        )
        self.args.init_noise = prev_noise

        print("\n==== Debug Result ====")
        print(f"max_com_disp:   {res['max_com_disp']:.4f} m")
        print(f"recovery_time:  {res['recovery_time']:.4f} s (-1 = no recovery)")
        print(f"fall_time:      {res['fall_time']:.4f} s (-1 = no fall)")

    def print_summary_stats(self) -> None:
        if not self.all_results:
            return
        summary: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
        for r in self.all_results:
            key = (str(r["body_label"]), str(r["direction"]), float(r["force_magnitude"]))
            if key not in summary:
                summary[key] = {"stable": 0, "total": 0, "com_disp": []}
            summary[key]["total"] += 1
            summary[key]["stable"] += int(r["is_stable"])
            summary[key]["com_disp"].append(float(r["max_com_disp"]))

        print("\nQuick Summary")
        print(f"{'Location':<12} | {'Dir':<8} | {'Force(N)':<8} | {'Stable%':<8} | {'MeanCoM(m)':<10}")
        print("-" * 60)
        for (loc, d, force), data in sorted(summary.items()):
            rate = (data["stable"] / data["total"]) * 100.0
            mean_disp = float(np.mean(data["com_disp"])) if data["com_disp"] else float("nan")
            print(f"{loc:<12} | {d:<8} | {force:<8.0f} | {rate:<8.1f} | {mean_disp:<10.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Passivity-Based WBC experiments on Unitree H1")

    p.add_argument("--xml", type=str,
        default=r"C:\Users\sanju\Documents\University_West\Course\Thesis\Scripts\venv310\model\New_MOdels\mujoco_menagerie-main\unitree_h1\scene.xml")
    p.add_argument("--mode", type=str, default="batch", choices=["batch", "debug", "random"])
    p.add_argument("--output-dir", type=str, default="experiment_results")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--sim-time",      type=float, default=10.0)
    p.add_argument("--push-start",    type=float, default=3.0)
    p.add_argument("--push-duration", type=float, default=0.1)

    p.add_argument("--pre-push-window",    type=float, default=1.0)
    p.add_argument("--recovery-thresh",    type=float, default=0.02)
    p.add_argument("--recovery-window",    type=float, default=0.5)
    p.add_argument("--fall-pitch-thresh",  type=float, default=1.2)
    p.add_argument("--fall-height-thresh", type=float, default=0.65)

    p.add_argument("--target-pitch", type=float, default=0.05)
    p.add_argument("--kp-posture",   type=float, default=200.0)
    p.add_argument("--kd-posture",   type=float, default=20.0)
    p.add_argument("--kp-tilt",      type=float, default=500.0)
    p.add_argument("--kd-tilt",      type=float, default=50.0)
    p.add_argument("--kp-roll",      type=float, default=500.0)
    p.add_argument("--kd-roll",      type=float, default=50.0)
    p.add_argument("--damping-gain", type=float, default=20.0)

    p.add_argument("--init-noise",      type=float, default=0.01)
    p.add_argument("--controller-name", type=str,   default="passivity_based")

    p.add_argument("--magnitudes", type=str, default="50,100,150,200,250,300")
    p.add_argument("--targets",    type=str, default="torso,pelvis")
    p.add_argument("--directions", type=str, default="forward,backward,lateral")
    p.add_argument("--num-trials", type=int, default=5)

    # Part 1b random mode arguments.
    p.add_argument("--seed-start", type=int, default=0,
                   help="Starting seed for --mode random.")
    p.add_argument("--n-random-trials", type=int, default=200,
                   help="Number of seeded random trials for --mode random.")
    p.add_argument("--visualize-random", type=int, default=0,
                   help="1 = open MuJoCo viewer for each random trial. 0 = headless.")

    p.add_argument("--viz-policy", type=str, default="only_force",
        choices=["none", "first_trial_each_condition", "only_force", "every_nth_condition"])
    p.add_argument("--viz-force",    type=float, default=100.0)
    p.add_argument("--viz-step",     type=int,   default=10)
    p.add_argument("--viz-max",      type=int,   default=20)
    p.add_argument("--viz-realtime", type=int,   default=1)
    p.add_argument("--viz-hold",     type=int,   default=0)

    p.add_argument("--debug-target",    type=str,   default="pelvis")
    p.add_argument("--debug-direction", type=str,   default="forward",
        choices=list(DIRECTION_PRESETS.keys()))
    p.add_argument("--debug-force",     type=float, default=100.0)
    p.add_argument("--debug-visualize", type=int,   default=1)

    return p.parse_args()

def main() -> None:
    args = parse_args()
    engine = HumanoidExperimentEngine(args)
    if args.mode == "debug":
        engine.run_debug_visualization()
        return
    if args.mode == "random":
        engine.run_random_batch()
        return
    engine.run_batch_experiments()

if __name__ == "__main__":
    main()
