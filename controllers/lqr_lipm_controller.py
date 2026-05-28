"""
Automated humanoid disturbance experiments for LQR-on-LIPM controller.

This is the reduced-order-model baseline controller for the Part 1a/1b
comparison. Unlike WQP/HQP/Passivity (which all use full-body dynamics),
LQR-on-LIPM models the robot as a simple 3D linear inverted pendulum and
computes optimal CoP feedback via a single discrete algebraic Riccati
equation solved once at startup.

Theoretical basis:
- Kajita 1991, 2001: 3D LIPM — CoM moves at constant height, foot massless,
  CoM dynamics CoM_ddot = (g/z_c) * (CoM - CoP)
- Stephens 2007 "Integral Control of Humanoid Balance" — LQR on LIPM with
  CoP constraint, shows expected failure mode (LQR saturates at foot edge).
- Kajita 2010 "Biped Walking Stabilization Based on Linear Inverted Pendulum
  Tracking" — the reference for LIPM-based stabilization on HRP-4C.

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
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

# Part 1b random multi-push disturbance generator.
try:
    from disturbance import BODY_WEIGHTS, DisturbanceProfile, Push
except ModuleNotFoundError:
    DisturbanceProfile = None
    Push = None
    BODY_WEIGHTS = {}


PUSH_LOCATION_PRESETS = {
    "torso": ["torso_link"],
    "pelvis": ["pelvis"],
}

DIRECTION_PRESETS = {
    "forward": np.array([1.0, 0.0, 0.0], dtype=float),
    "backward": np.array([-1.0, 0.0, 0.0], dtype=float),
    "lateral": np.array([0.0, 1.0, 0.0], dtype=float),
}

# H1 double-stance support polygon approximation (relative to pelvis projection).
# Not used for hard clipping by default, but LQR's commanded CoP will often
# exceed this — flag for Chapter 4 discussion.
SUPPORT_POLYGON_X = (-0.12, 0.12)   # fore-aft (m)
SUPPORT_POLYGON_Y = (-0.25, 0.25)   # lateral (m)


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


def compute_com_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    masses = model.body_mass
    total_mass = float(np.sum(masses))
    return (masses[:, None] * data.xipos).sum(axis=0) / total_mass


def compute_com_velocity(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Approximate CoM velocity using body mass-weighted cvel (angular+linear)."""
    masses = model.body_mass
    total_mass = float(np.sum(masses))
    # cvel is [angular (3); linear (3)] per body, expressed at body COM in world frame.
    lin_vel = data.cvel[:, 3:6]
    return (masses[:, None] * lin_vel).sum(axis=0) / total_mass


def build_actuator_mappings(model: mujoco.MjModel) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    act_joint = model.actuator_trnid[:, 0].copy()
    qaddr = model.jnt_qposadr[act_joint]
    daddr = model.jnt_dofadr[act_joint]
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(model.nu)]
    return qaddr, daddr, actuator_names


def build_groups(names: List[str]) -> Dict[str, List[int]]:
    return {
        "AnkleP": [i for i, n in enumerate(names) if "ankle" in n],
        "HipP": [i for i, n in enumerate(names) if "hip_pitch" in n],
        "KneeP": [i for i, n in enumerate(names) if "knee" in n],
        "HipR": [i for i, n in enumerate(names) if "hip_roll" in n],
    }


def build_gains(model: mujoco.MjModel, groups: Dict[str, List[int]],
                kp_posture: float, kd_posture: float) -> Tuple[np.ndarray, np.ndarray]:
    kp = np.full(model.nu, float(kp_posture), dtype=float)
    kd = np.full(model.nu, float(kd_posture), dtype=float)
    for i in groups["HipR"]:
        kp[i], kd[i] = 500.0, 50.0
    return kp, kd


def select_body(model: mujoco.MjModel, candidates: List[str]) -> Tuple[str, int]:
    for name in candidates:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return str(name), int(body_id)
    return "", -1


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


def design_lqr_lipm(z_c: float, dt: float, q_pos: float, q_vel: float,
                    r_control: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Design LQR gain for 1D LIPM dynamics. Returns (K, A_d, B_d, P).

    Continuous dynamics:  x_ddot = (g/z_c) * (x - u)
                          s = [x; x_dot],  u = CoP position
                          A_c = [[0, 1], [g/z_c, 0]]
                          B_c = [[0], [-g/z_c]]

    Since the 3D LIPM is decoupled between sagittal (x) and lateral (y) planes
    under the constant-height assumption, we design one 1D LQR and use the
    same gain for both axes (Kajita 2001, Stephens 2007).
    """
    g = 9.81
    omega_sq = g / float(z_c)
    A_c = np.array([[0.0, 1.0], [omega_sq, 0.0]])
    B_c = np.array([[0.0], [-omega_sq]])
    C = np.eye(2)
    D = np.zeros((2, 1))
    # Zero-order hold discretisation (more accurate than Euler at 1ms).
    (A_d, B_d, _, _, _) = cont2discrete((A_c, B_c, C, D), float(dt), method="zoh")
    Q = np.diag([float(q_pos), float(q_vel)])
    R = np.array([[float(r_control)]])
    P = solve_discrete_are(A_d, B_d, Q, R)
    K = np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
    return K, A_d, B_d, P


def parse_csv_list(text: str) -> List[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_csv_floats(text: str) -> List[float]:
    out = []
    for s in parse_csv_list(text):
        out.append(float(s))
    return out


class HumanoidExperimentEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = mujoco.MjModel.from_xml_path(args.xml)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        safe_reset(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

        self.qaddr, self.daddr, self.actuator_names = build_actuator_mappings(self.model)
        self.groups = build_groups(self.actuator_names)
        self.knee_ids = self.groups["KneeP"]
        self.hip_ids = [i for i, n in enumerate(self.actuator_names)
                        if ("hip_pitch" in n) or ("hip_roll" in n) or ("hip_yaw" in n)]

        self.kp, self.kd = build_gains(self.model, self.groups,
                                       self.args.kp_posture, self.args.kd_posture)

        self.gear = self.model.actuator_gear[:, 0].copy()
        self.gear[self.gear == 0.0] = 1.0
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1]

        self.q_target_full = self.data.qpos.copy()

        # Measure CoM reference at home pose (BEFORE any disturbance).
        # This is the CoM we want to regulate TO. Stored once at init.
        com_home = compute_com_position(self.model, self.data)
        self.com_ref_xy = com_home[0:2].copy()
        self.z_c = float(com_home[2])  # CoM height ~ 0.98m for H1

        self.tilt_sign = calibrate_balance_sign(self.model, self.data, self.groups["AnkleP"])
        safe_reset(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Design the LQR gain ONCE (offline, static).
        # This is the "reduced-order baseline" architectural choice —
        # no online re-computation, unlike MPC.
        self.K_lqr, self.A_d, self.B_d, self.P_dare = design_lqr_lipm(
            z_c=self.z_c,
            dt=self.dt,
            q_pos=float(self.args.lqr_q_pos),
            q_vel=float(self.args.lqr_q_vel),
            r_control=float(self.args.lqr_r),
        )

        # Report LQR design at startup.
        eig_cl = np.linalg.eigvals(self.A_d - self.B_d @ self.K_lqr)
        print(f"[LQR-LIPM] z_c={self.z_c:.4f}m, dt={self.dt:.4f}s")
        print(f"[LQR-LIPM] Q=diag({self.args.lqr_q_pos},{self.args.lqr_q_vel}), R={self.args.lqr_r}")
        print(f"[LQR-LIPM] K = [{self.K_lqr[0,0]:+.3f}, {self.K_lqr[0,1]:+.3f}]  "
              f"(position gain, velocity gain)")
        print(f"[LQR-LIPM] closed-loop |eig| = {[float(abs(e)) for e in eig_cl]}  "
              f"(stable if <1)")

        self.output_dir = str(self.args.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        controller_label = getattr(self.args, "controller_name", "controller")
        self.csv_path = os.path.join(self.output_dir, f"{controller_label}_{self.session_id}.csv")

        self.all_results: List[Dict[str, Any]] = []
        self.rng = np.random.default_rng(int(self.args.seed))
        self.hipr_int = np.zeros(len(self.groups["HipR"]), dtype=float)

    def _apply_small_init_noise(self) -> None:
        noise_std = float(self.args.init_noise)
        if noise_std <= 0.0:
            return
        self.data.qpos[self.qaddr] += self.rng.normal(0.0, noise_std,
                                                     size=self.data.qpos[self.qaddr].shape)
        self.data.qpos[3:7] += self.rng.normal(0.0, noise_std * 0.1, size=4)
        q = self.data.qpos[3:7]
        self.data.qpos[3:7] = q / np.linalg.norm(q)
        mujoco.mj_forward(self.model, self.data)

    def _compute_lqr_torque(
        self,
        com_ref_xy: np.ndarray,
        curr_pitch: float,
        prev_pitch: float,
        curr_roll: float,
        prev_roll: float,
        dt_est: float,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        tau_g = self.data.qfrc_bias[self.daddr].copy()

        q_curr = self.data.qpos[self.qaddr]
        dq_curr = self.data.qvel[self.daddr]
        tau_pd = self.kp * (self.q_target_full[self.qaddr] - q_curr)
        tau_d = -float(self.args.damping_gain) * dq_curr

        com_pos = compute_com_position(self.model, self.data)
        com_vel = compute_com_velocity(self.model, self.data)

        s_x = np.array([com_pos[0] - float(com_ref_xy[0]), com_vel[0]], dtype=float)
        s_y = np.array([com_pos[1] - float(com_ref_xy[1]), com_vel[1]], dtype=float)

        cop_x_rel = float((-self.K_lqr @ s_x)[0])
        cop_y_rel = float((-self.K_lqr @ s_y)[0])

        cop_saturated = 0
        cop_x = float(np.clip(cop_x_rel, SUPPORT_POLYGON_X[0], SUPPORT_POLYGON_X[1]))
        cop_y = float(np.clip(cop_y_rel, SUPPORT_POLYGON_Y[0], SUPPORT_POLYGON_Y[1]))
        if cop_x != cop_x_rel or cop_y != cop_y_rel:
            cop_saturated = 1
        cop_cmd = np.array([cop_x, cop_y], dtype=float)

        tau_lqr = np.zeros(self.model.nu, dtype=float)
        fz_total = float(np.sum(self.model.body_mass)) * 9.81

        pitch_moment = 0.5 * fz_total * cop_x * float(self.args.cop_to_ankle_gain)
        for idx in self.groups["AnkleP"]:
            tau_lqr[idx] += -self.tilt_sign * pitch_moment
        for idx in self.groups["HipP"]:
            tau_lqr[idx] += -self.tilt_sign * 0.5 * pitch_moment

        hipr_cmd = cop_y * float(self.args.cop_to_hiproll_gain)
        for idx in self.groups["HipR"]:
            tau_lqr[idx] += hipr_cmd

        d_pitch = (float(curr_pitch) - float(prev_pitch)) / max(1e-4, float(dt_est))
        d_roll = (float(curr_roll) - float(prev_roll)) / max(1e-4, float(dt_est))

        tilt_corr = -self.tilt_sign * (
            float(self.args.kp_tilt) * (float(curr_pitch) - float(self.args.target_pitch))
            + float(self.args.kd_tilt) * d_pitch
        )
        tilt_corr = float(np.clip(tilt_corr, -150.0, 150.0))
        for idx in self.groups["AnkleP"]:
            tau_lqr[idx] += tilt_corr
        for idx in self.groups["HipP"]:
            tau_lqr[idx] += 0.5 * tilt_corr

        roll_corr = -(
            float(self.args.kp_roll) * float(curr_roll) + float(self.args.kd_roll) * d_roll
        )

        hipr_ids = self.groups["HipR"]
        if hipr_ids:
            hipr_qaddr = self.qaddr[hipr_ids]
            hipr_daddr = self.daddr[hipr_ids]
            hipr_q = self.data.qpos[hipr_qaddr]
            hipr_dq = self.data.qvel[hipr_daddr]
            hipr_q_ref = self.q_target_full[hipr_qaddr]
            hipr_err = hipr_q_ref - hipr_q
            self.hipr_int = np.clip(self.hipr_int + hipr_err * float(dt_est), -0.3, 0.3)
            hipr_hold = 400.0 * hipr_err - 40.0 * hipr_dq + 60.0 * self.hipr_int
            for i, idx in enumerate(hipr_ids):
                tau_lqr[idx] += roll_corr + float(hipr_hold[i])

        tau_wbc = tau_g + tau_pd + tau_d + tau_lqr
        return tau_wbc, cop_cmd, cop_saturated
    # ==================================================================
    # run_single_trial (Part 1a: step-push)
    # ==================================================================
    def run_single_trial(self, config: Dict[str, Any], trial_in_condition: int,
                          visualize: bool) -> Dict[str, Any]:
        safe_reset(self.model, self.data)
        self._apply_small_init_noise()
        mujoco.mj_forward(self.model, self.data)
        self.hipr_int = np.zeros(len(self.groups["HipR"]), dtype=float)

        push_body_id = int(config["body_id"])
        push_body_name = str(config["body_name"])
        force_vec = np.array(config["force_vec"], dtype=float)
        direction = str(config["dir_name"])
        magnitude = float(config["mag"])

        push_start = float(self.args.push_start)
        push_duration = float(self.args.push_duration)
        push_end = push_start + push_duration
        sim_end = float(self.args.sim_time)

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

        cop_saturation_count = 0
        total_steps_after_push = 0

        _r0, _p0 = get_rpy(self.data.qpos[3:7])
        prev_pitch = float(_p0)
        prev_roll = float(_r0)
        prev_t = float(self.data.time)

        def step_once() -> None:
            nonlocal com_xy_ref, fall_time, max_com_x_disp, max_com_xy_disp, \
                max_com_y_disp, max_torso_pitch, max_torso_roll, prev_pitch, \
                prev_roll, prev_t, recovery_buffer, recovery_time, tau_sq_duration, \
                tau_sq_sum_ankle, tau_sq_sum_hip, tau_sq_sum_knee, tau_sq_sum_total, \
                cop_saturation_count, total_steps_after_push

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

            pelvis_height = (float(self.data.xipos[self.pelvis_id, 2])
                             if self.pelvis_id >= 0 else float("inf"))
            if ((abs(curr_pitch) > float(self.args.fall_pitch_thresh))
                    or (pelvis_height < float(self.args.fall_height_thresh))) \
                    and not np.isfinite(fall_time):
                fall_time = now_t

            com = compute_com_position(self.model, self.data)
            if now_t < push_start:
                pre_push_com_xy.append(com[0:2].copy())
                if len(pre_push_com_xy) > pre_push_window_samples:
                    del pre_push_com_xy[: len(pre_push_com_xy) - pre_push_window_samples]
                com_xy_ref = np.mean(np.array(pre_push_com_xy), axis=0)

            com_ref = com_xy_ref if com_xy_ref is not None else self.com_ref_xy
            tau_wbc, _cop, cop_sat = self._compute_lqr_torque(
                com_ref, curr_pitch, prev_pitch, curr_roll, prev_roll, dt_est
            )
            
            ctrl = np.clip(tau_wbc / self.gear, self.ctrl_min, self.ctrl_max)
            self.data.ctrl[:] = ctrl

            if now_t >= push_start:
                ankle_ids = self.groups["AnkleP"]
                tau_sq_sum_total += float(np.sum(tau_wbc ** 2)) * self.dt
                tau_sq_sum_ankle += float(np.sum(tau_wbc[ankle_ids] ** 2)) * self.dt
                tau_sq_sum_knee += float(np.sum(tau_wbc[self.knee_ids] ** 2)) * self.dt
                tau_sq_sum_hip += float(np.sum(tau_wbc[self.hip_ids] ** 2)) * self.dt
                tau_sq_duration += self.dt
                cop_saturation_count += cop_sat
                total_steps_after_push += 1

            self.data.xfrc_applied[:] = 0.0
            if push_body_id >= 0 and push_start <= now_t < push_end:
                self.data.xfrc_applied[push_body_id, 0:3] = force_vec

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
            "qp_solve_time_mean_ms": 0.0,   # LQR has no QP
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
    # run_random_trial (Part 1b: stochastic disturbances)
    # ==================================================================
    def _resolve_available_push_bodies(self) -> List[str]:
        candidates = list(BODY_WEIGHTS.keys())
        available: List[str] = []
        for name in candidates:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                available.append(name)
        return available

    def run_random_trial(self, profile: "DisturbanceProfile",
                          visualize: bool = False) -> Dict[str, Any]:
        safe_reset(self.model, self.data)
        self._apply_small_init_noise()
        mujoco.mj_forward(self.model, self.data)
        self.hipr_int = np.zeros(len(self.groups["HipR"]), dtype=float)

        sim_end = float(self.args.sim_time)

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

        com_sq_integral = 0.0
        com_integral_duration = 0.0
        total_tau_sq_integral = 0.0

        _r0, _p0 = get_rpy(self.data.qpos[3:7])
        prev_pitch = float(_p0)
        prev_roll = float(_r0)
        prev_t = float(self.data.time)

        def step_once() -> None:
            nonlocal com_xy_ref, fall_time, max_com_x_disp, max_com_xy_disp, \
                max_com_y_disp, max_torso_pitch, max_torso_roll, prev_pitch, \
                prev_roll, prev_t, recovery_buffer, recovery_time, tau_sq_duration, \
                tau_sq_sum_ankle, tau_sq_sum_hip, tau_sq_sum_knee, tau_sq_sum_total, \
                com_sq_integral, com_integral_duration, total_tau_sq_integral

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

            pelvis_height = (float(self.data.xipos[self.pelvis_id, 2])
                             if self.pelvis_id >= 0 else float("inf"))
            if ((abs(curr_pitch) > float(self.args.fall_pitch_thresh))
                    or (pelvis_height < float(self.args.fall_height_thresh))) \
                    and not np.isfinite(fall_time):
                fall_time = now_t

            com = compute_com_position(self.model, self.data)
            if now_t < first_push_start:
                pre_push_com_xy.append(com[0:2].copy())
                if len(pre_push_com_xy) > pre_push_window_samples:
                    del pre_push_com_xy[: len(pre_push_com_xy) - pre_push_window_samples]
                com_xy_ref = np.mean(np.array(pre_push_com_xy), axis=0)

            com_ref = com_xy_ref if com_xy_ref is not None else self.com_ref_xy
            tau_wbc, _cop, _sat = self._compute_lqr_torque(
                com_ref, curr_pitch, prev_pitch, curr_roll, prev_roll, dt_est
            )
            
            ctrl = np.clip(tau_wbc / self.gear, self.ctrl_min, self.ctrl_max)
            self.data.ctrl[:] = ctrl

            if now_t >= first_push_start:
                ankle_ids = self.groups["AnkleP"]
                tau_sq_sum_total += float(np.sum(tau_wbc ** 2)) * self.dt
                tau_sq_sum_ankle += float(np.sum(tau_wbc[ankle_ids] ** 2)) * self.dt
                tau_sq_sum_knee += float(np.sum(tau_wbc[self.knee_ids] ** 2)) * self.dt
                tau_sq_sum_hip += float(np.sum(tau_wbc[self.hip_ids] ** 2)) * self.dt
                tau_sq_duration += self.dt

            total_tau_sq_integral += float(np.sum(tau_wbc ** 2)) * self.dt

            self.data.xfrc_applied[:] = 0.0
            active = profile.active_push_at(now_t)
            if active is not None:
                bid = body_id_for.get(active.body_name, -1)
                if bid >= 0:
                    self.data.xfrc_applied[bid, 0:3] = active.force_vec

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
                    recovery_buffer.append(
                        1 if com_xy_disp <= float(self.args.recovery_thresh) else 0
                    )
                    if len(recovery_buffer) > recovery_window_samples:
                        recovery_buffer.pop(0)
                    if (len(recovery_buffer) == recovery_window_samples
                            and sum(recovery_buffer) >= int(0.8 * recovery_window_samples)):
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
            force_x_rep = force_y_rep = force_z_rep = 0.0
            body_label_rep = body_name_rep = "none"
            direction_rep = "none"
            mag_rep = 0.0

        return {
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

    def run_random_batch(self) -> None:
        if DisturbanceProfile is None:
            raise RuntimeError("disturbance.py is missing.")

        seed_start = int(self.args.seed_start)
        n_trials = int(self.args.n_random_trials)
        sim_duration = float(self.args.sim_time)
        visualize = bool(int(getattr(self.args, "visualize_random", 0)))

        available_bodies = self._resolve_available_push_bodies()
        if len(available_bodies) < len(BODY_WEIGHTS):
            print(f"[WARN] Only {len(available_bodies)}/{len(BODY_WEIGHTS)} push bodies found. "
                  f"Present: {available_bodies}")

        controller_label = getattr(self.args, "controller_name", "controller")
        self.csv_path = os.path.join(self.output_dir, f"{controller_label}_random_{self.session_id}.csv")

        print(f"\nPart 1b random mode")
        print(f"- Seeds: [{seed_start}, {seed_start + n_trials - 1}] ({n_trials} trials)")
        print(f"- Sim duration: {sim_duration} s per trial")
        print(f"- Available push bodies: {available_bodies}")
        print(f"- CSV output: {self.csv_path}")

        header_written = False
        with open(self.csv_path, "w", newline="") as f:
            writer = None
            for i in range(n_trials):
                seed = seed_start + i
                profile = DisturbanceProfile.generate(
                    seed=seed, sim_duration_s=sim_duration,
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

    def _should_visualize_trial(self, config: Dict[str, Any], trial_in_condition: int,
                                 viz_count_so_far: int) -> bool:
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

    def _print_trial_result(self, cfg: Dict[str, Any], trial_num: int,
                             total_trials: int, res: Dict[str, Any]) -> None:
        stable_str = "STABLE" if int(res["is_stable"]) == 1 else "FELL  "
        rt = float(res["recovery_time"])
        rt_str = f"{rt:.2f}s" if rt > 0 else "none"
        print(
            f"  [{stable_str}] "
            f"Cond {int(cfg['condition_id']):03d} | "
            f"Trial {trial_num+1}/{total_trials} | "
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
                print(f"Condition {cfg['condition_id']:04d} | {cfg['body_label']} | "
                      f"{cfg['dir_name']} | {cfg['mag']}N")
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
            "condition_id": 0, "body_label": body_label, "body_name": body_name,
            "body_id": body_id, "dir_name": direction, "mag": magnitude,
            "force_vec": force_vec,
        }
        print("Debug mode")
        print(f"- body:       {cfg['body_name']}")
        print(f"- direction:  {cfg['dir_name']}")
        print(f"- force:      {cfg['mag']} N")
        print(f"- sim_time:   {self.args.sim_time} s")
        prev_noise = float(self.args.init_noise)
        self.args.init_noise = 0.0
        res = self.run_single_trial(cfg, trial_in_condition=0,
                                     visualize=bool(int(self.args.debug_visualize)))
        self.args.init_noise = prev_noise
        print("\n==== Debug Result ====")
        print(f"max_com_disp:   {res['max_com_disp']:.4f} m")
        print(f"recovery_time:  {res['recovery_time']:.4f} s (-1 = no recovery)")
        print(f"fall_time:      {res['fall_time']:.4f} s (-1 = no fall)")
        print(f"tau_rms_total:  {res['tau_rms_total']:.3f} Nm")

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
    p = argparse.ArgumentParser(description="LQR-on-LIPM baseline controller for Unitree H1")

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

    # Posture / torso control (Layer 2/3)
    p.add_argument("--target-pitch", type=float, default=0.05)
    p.add_argument("--kp-posture",   type=float, default=200.0)
    p.add_argument("--kd-posture",   type=float, default=20.0)
    p.add_argument("--kp-tilt",      type=float, default=500.0)
    p.add_argument("--kd-tilt",      type=float, default=50.0)
    p.add_argument("--kp-roll",      type=float, default=500.0)
    p.add_argument("--kd-roll",      type=float, default=50.0)
    p.add_argument("--damping-gain", type=float, default=20.0)

    # LQR-on-LIPM design parameters (Layer 4)
    p.add_argument("--lqr-q-pos", type=float, default=100.0,
                   help="LQR Q weight on CoM position error.")
    p.add_argument("--lqr-q-vel", type=float, default=10.0,
                   help="LQR Q weight on CoM velocity.")
    p.add_argument("--lqr-r",     type=float, default=1.0,
                   help="LQR R weight (cost on CoP command). Higher R = softer control.")
    p.add_argument("--cop-to-ankle-gain", type=float, default=0.1,
                   help="Scaling from commanded CoP to ankle pitch torque. "
                        "Raw physical value is 1.0 (Fz*CoP), but empirically 0.1 "
                        "works better because the full physical mapping would saturate ankles.")
    p.add_argument("--cop-to-hiproll-gain", type=float, default=30.0,
                   help="Scaling from commanded lateral CoP to hip roll torque. "
                        "H1 has no ankle roll, so lateral CoP control goes through hip roll.")

    p.add_argument("--init-noise",      type=float, default=0.01)
    p.add_argument("--controller-name", type=str,   default="lqr_lipm")

    p.add_argument("--magnitudes", type=str, default="50,100,150,200,250,300")
    p.add_argument("--targets",    type=str, default="torso,pelvis")
    p.add_argument("--directions", type=str, default="forward,backward,lateral")
    p.add_argument("--num-trials", type=int, default=5)

    # Part 1b random mode
    p.add_argument("--seed-start",        type=int, default=0)
    p.add_argument("--n-random-trials",   type=int, default=200)
    p.add_argument("--visualize-random",  type=int, default=0)

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
