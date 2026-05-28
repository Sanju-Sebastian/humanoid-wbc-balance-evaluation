"""
Residual MLP for passivity-based WBC correction.  v2 — CoM-augmented input.

KEY CHANGE vs v1:
  INPUT_DIM 42 → 46.  Four new features appended to state vector:
    [43] com_error_x   — horizontal CoM displacement from reference (m), sagittal
    [44] com_error_y   — horizontal CoM displacement from reference (m), lateral
    [45] com_vel_x     — CoM velocity x  (m/s)
    [46] com_vel_y     — CoM velocity y  (m/s)

Motivation: the single-timestep 42-dim snapshot could not distinguish "being pushed"
from "already recovered."  The CoM error gives the MLP the disturbance signature
it needs (Danaa / Thulsiram 2026 uses ΔX_CoM in his input vector; FAST (2025)
also adds CoM-related observations explicitly for balance improvement).

No change to architecture depth, output joints, TAU_BOUND, or training procedure.
collect mode must be re-run to regenerate the dataset with 46-dim states.

Three modes:
  --mode collect   Re-runs passivity controller on seeds 0-199, logs per-timestep
                   (state_46dim, tau_base) to residual_mlp_dataset.npz
  --mode train     Loads dataset, trains 46->64->64->64->6 network, saves weights
  --mode evaluate  Runs passivity+MLP controller on seeds 0-199, writes CSV

Usage examples:
  python residual_mlp.py --mode collect
  python residual_mlp.py --mode train
  python residual_mlp.py --mode evaluate
"""

import argparse
import csv
import datetime
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── MuJoCo (only needed for collect / evaluate) ───────────────────────────────
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

# ── PyTorch (only needed for train / evaluate) ────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Disturbance generator ─────────────────────────────────────────────────────
try:
    from disturbance import BODY_WEIGHTS, DisturbanceProfile, Push
    DISTURBANCE_AVAILABLE = True
except ModuleNotFoundError:
    DISTURBANCE_AVAILABLE = False
    BODY_WEIGHTS = {}

# =============================================================================
# CONSTANTS
# =============================================================================

SINGLE_FRAME_DIM = 52
HISTORY_LENGTH = 5
INPUT_DIM = SINGLE_FRAME_DIM * HISTORY_LENGTH
OUTPUT_DIM = 6
HIDDEN_DIM = 128
TAU_BOUND = 20.0

FEATURE_NAMES = (
    ["pitch", "roll", "d_pitch", "d_roll"]
    + [f"q_{i}" for i in range(19)]
    + [f"dq_{i}" for i in range(19)]
    + ["com_err_x", "com_err_y", "com_vel_x", "com_vel_y"]
    + [f"tau_base_{i}" for i in range(OUTPUT_DIM)]
)
assert len(FEATURE_NAMES) == SINGLE_FRAME_DIM

# Output joint names in order (must match OUTPUT_DIM = 6)
OUTPUT_JOINT_NAMES = [
    "left_ankle",
    "right_ankle",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_hip_roll",
    "right_hip_roll",
]

# Train / val seed split
TRAIN_SEEDS = list(range(0, 160))   # seeds 0-159
VAL_SEEDS   = list(range(160, 200)) # seeds 160-199

# =============================================================================
# SHARED UTILITIES
# =============================================================================

def get_rpy(q: np.ndarray) -> Tuple[float, float]:
    """Returns (roll, pitch) from quaternion [w, x, y, z]."""
    w, x, y, z = q
    roll  = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(np.clip(2 * (w * y - z * x), -1, 1))
    return float(roll), float(pitch)

def safe_reset(model, data) -> None:
    if getattr(model, "nkey", 0) > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

def compute_com_position(model, data) -> np.ndarray:
    masses = model.body_mass
    total_mass = float(np.sum(masses))
    return (masses[:, None] * data.xipos).sum(axis=0) / total_mass

def build_actuator_mappings(model):
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

def calibrate_balance_sign(model, data, ankle_ids: List[int]) -> float:
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

# =============================================================================
# MLP ARCHITECTURE
# =============================================================================

class ResidualMLP(nn.Module):
    """MLP: INPUT_DIM → 128 → 128 → 128 → 6."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM,
                 output_dim: int = OUTPUT_DIM, tau_bound: float = TAU_BOUND):
        super().__init__()
        self.tau_bound = tau_bound
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


# =============================================================================
# SIMULATION ENGINE (shared by collect and evaluate)
# =============================================================================

class PassivityEngine:
    """
    Stripped-down passivity controller engine.
    Used by both collect mode (logs data) and evaluate mode (applies MLP).
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.model = mujoco.MjModel.from_xml_path(args.xml)
        self.model.opt.timestep = 0.001
        self.data  = mujoco.MjData(self.model)
        self.dt    = float(self.model.opt.timestep)

        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        self.qaddr, self.daddr, self.actuator_names = build_actuator_mappings(self.model)
        self.groups = build_groups(self.actuator_names)

        self.knee_ids = self.groups["KneeP"]
        self.hip_ids  = [i for i, n in enumerate(self.actuator_names)
                         if "hip_pitch" in n or "hip_roll" in n or "hip_yaw" in n]

        # PD gains
        self.kp = np.full(self.model.nu, float(args.kp_posture), dtype=float)
        self.kd = np.full(self.model.nu, float(args.kd_posture), dtype=float)

        self.gear     = self.model.actuator_gear[:, 0].copy()
        self.gear[self.gear == 0.0] = 1.0
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1]

        self.tilt_sign = calibrate_balance_sign(
            self.model, self.data, self.groups["AnkleP"]
        )
        mujoco.mj_forward(self.model, self.data)
        self.q_target_full = self.data.qpos.copy()

        # Resolve MLP output indices (which actuator indices to patch)
        self.mlp_output_indices: List[int] = []
        for joint_name in OUTPUT_JOINT_NAMES:
            for i, aname in enumerate(self.actuator_names):
                if aname == joint_name:
                    self.mlp_output_indices.append(i)
                    break
        assert len(self.mlp_output_indices) == OUTPUT_DIM, (
            f"Could not resolve all {OUTPUT_DIM} MLP output joints. "
            f"Found: {self.mlp_output_indices} for {OUTPUT_JOINT_NAMES}"
        )

        # Dataset accumulator (collect mode)
        self._collect_states: List[np.ndarray] = []
        self._collect_taus:   List[np.ndarray] = []

        import collections

        self.obs_history = collections.deque(maxlen=HISTORY_LENGTH)
        for _ in range(HISTORY_LENGTH):
            self.obs_history.append(np.zeros(SINGLE_FRAME_DIM, dtype=np.float32))

    # ── Passivity torque (unchanged from v1) ──────────────────────────────────

    def _compute_passivity_torque(
        self,
        curr_pitch: float,
        curr_roll:  float,
        prev_pitch: float,
        prev_roll:  float,
        dt_est:     float,
        hipr_int:   np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        args = self.args

        # Layer 1: gravity compensation
        tau_g = self.data.qfrc_bias[self.daddr].copy()

        # Layer 2: posture PD
        q_curr = self.data.qpos[self.qaddr]
        tau_pd = self.kp * (self.q_target_full[self.qaddr] - q_curr)

        # Layer 3: damping injection
        dq_curr = self.data.qvel[self.daddr]
        tau_d   = -float(args.damping_gain) * dq_curr

        # Layer 4: energy shaping (balance)
        tau_e = np.zeros(self.model.nu, dtype=float)

        d_pitch    = (curr_pitch - prev_pitch) / dt_est
        pitch_error = curr_pitch - float(args.target_pitch)
        tilt_corr  = -self.tilt_sign * (
            float(args.kp_tilt) * pitch_error + float(args.kd_tilt) * d_pitch
        )
        tilt_corr = float(np.clip(tilt_corr, -150.0, 150.0))
        for idx in self.groups["AnkleP"]:
            tau_e[idx] += tilt_corr
        for idx in self.groups["HipP"]:
            tau_e[idx] += 0.5 * tilt_corr

        d_roll    = (curr_roll - prev_roll) / dt_est
        roll_corr = -(float(args.kp_roll) * curr_roll + float(args.kd_roll) * d_roll)

        hipr_kp, hipr_kd, hipr_ki = 400.0, 40.0, 60.0
        hipr_q    = self.data.qpos[self.qaddr][self.groups["HipR"]]
        hipr_dq   = self.data.qvel[self.daddr][self.groups["HipR"]]
        hipr_q_ref = self.q_target_full[self.qaddr][self.groups["HipR"]]
        hipr_err  = hipr_q_ref - hipr_q
        hipr_int_new = np.clip(hipr_int + hipr_err * dt_est, -0.3, 0.3)
        hipr_hold = hipr_kp * hipr_err - hipr_kd * hipr_dq + hipr_ki * hipr_int_new

        for i, idx in enumerate(self.groups["HipR"]):
            tau_e[idx] += roll_corr + hipr_hold[i]

        tau_wbc = tau_g + tau_pd + tau_d + tau_e
        return tau_wbc, hipr_int_new

    # ── State vector builder — v2: 42 → 46 dims ───────────────────────────────

    def _build_state_vec(
        self,
        curr_pitch:   float,
        curr_roll:    float,
        prev_pitch:   float,
        prev_roll:    float,
        dt_est:       float,
        com_error_x:  float,   # NEW — CoM displacement from reference, sagittal (m)
        com_error_y:  float,   # NEW — CoM displacement from reference, lateral (m)
        com_vel_x:    float,   # NEW — CoM velocity x (m/s)
        com_vel_y:    float,   # NEW — CoM velocity y (m/s)
        tau_base_6:   np.ndarray,
    ) -> np.ndarray:
        d_pitch = (curr_pitch - prev_pitch) / dt_est
        d_roll  = (curr_roll  - prev_roll)  / dt_est
        q       = self.data.qpos[self.qaddr]   # shape (19,)
        dq      = self.data.qvel[self.daddr]   # shape (19,)
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
            ]
        ).astype(np.float32)
        if frame.shape[0] != SINGLE_FRAME_DIM:
            raise ValueError(f"Single-frame feature dim mismatch: expected {SINGLE_FRAME_DIM}, got {frame.shape[0]}")
        self.obs_history.append(frame)
        stacked = np.concatenate(list(self.obs_history), axis=0).astype(np.float32)
        return stacked

    # ── Single trial runner ───────────────────────────────────────────────────

    def run_trial(
        self,
        profile: "DisturbanceProfile",
        collect_data: bool = False,
        mlp_model: Optional["ResidualMLP"] = None,
        mlp_device: Optional[str] = None,
        stable_only_collect: bool = False,
    ) -> Dict[str, Any]:
        safe_reset(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if hasattr(self, "obs_history"):
            self.obs_history.clear()
            for _ in range(HISTORY_LENGTH):
                self.obs_history.append(np.zeros(SINGLE_FRAME_DIM, dtype=np.float32))

        sim_end = float(self.args.sim_time)

        body_id_for: Dict[str, int] = {}
        for p in profile.pushes:
            if p.body_name not in body_id_for:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, p.body_name)
                body_id_for[p.body_name] = int(bid)

        first_push_start = profile.pushes[0].start_time_s if profile.pushes else 1.0
        recovery_window_samples = max(1, int(round(float(self.args.recovery_window) / self.dt)))
        pre_push_window_samples = max(1, int(round(float(self.args.pre_push_window) / self.dt)))

        pre_push_com_xy: List[np.ndarray] = []
        com_xy_ref: Optional[np.ndarray]  = None
        max_com_xy_disp  = 0.0
        max_torso_pitch  = 0.0
        max_torso_roll   = 0.0
        recovery_time    = float("nan")
        recovery_buffer: List[int] = []
        fall_time        = float("nan")

        com_sq_integral       = 0.0
        com_integral_duration = 0.0
        total_tau_sq_integral = 0.0
        tau_sq_sum_total      = 0.0
        tau_sq_duration       = 0.0

        # Per-trial data buffer (only flushed if stable)
        trial_states: List[np.ndarray] = []
        trial_taus:   List[np.ndarray] = []

        _r0, _p0 = get_rpy(self.data.qpos[3:7])
        prev_pitch = float(_p0)
        prev_roll  = float(_r0)
        prev_t     = float(self.data.time)

        # ── v2: track previous CoM position for finite-difference velocity ──
        prev_com_xy = compute_com_position(self.model, self.data)[0:2].copy()

        hipr_int = np.zeros(len(self.groups["HipR"]), dtype=float)

        def step_once() -> None:
            nonlocal com_xy_ref, fall_time, max_com_xy_disp, max_torso_pitch, \
                max_torso_roll, prev_pitch, prev_roll, prev_t, recovery_buffer, \
                recovery_time, tau_sq_duration, tau_sq_sum_total, hipr_int, \
                com_sq_integral, com_integral_duration, total_tau_sq_integral, \
                prev_com_xy   # ← v2 addition

            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                mujoco.mj_step(self.model, self.data)
                return

            mujoco.mj_forward(self.model, self.data)
            now_t  = float(self.data.time)
            dt_est = max(1e-4, now_t - prev_t)

            curr_roll, curr_pitch = get_rpy(self.data.qpos[3:7])

            if now_t >= first_push_start:
                max_torso_pitch = max(max_torso_pitch, abs(curr_pitch))
                max_torso_roll  = max(max_torso_roll,  abs(curr_roll))

            pelvis_height = (
                float(self.data.xipos[self.pelvis_id, 2])
                if self.pelvis_id >= 0 else float("inf")
            )
            if (abs(curr_pitch) > float(self.args.fall_pitch_thresh)
                    or pelvis_height < float(self.args.fall_height_thresh)) \
                    and not np.isfinite(fall_time):
                fall_time = now_t

            # ── v2: compute CoM position and derived features ─────────────
            com = compute_com_position(self.model, self.data)
            com_xy = com[0:2]

            # Update reference (pre-push average)
            if now_t < first_push_start:
                pre_push_com_xy.append(com_xy.copy())
                if len(pre_push_com_xy) > pre_push_window_samples:
                    del pre_push_com_xy[: len(pre_push_com_xy) - pre_push_window_samples]
                com_xy_ref = np.mean(np.array(pre_push_com_xy), axis=0)

            # CoM error: displacement from reference
            if com_xy_ref is not None:
                com_err = com_xy - com_xy_ref          # shape (2,)
            else:
                com_err = np.zeros(2, dtype=float)

            # CoM velocity: finite difference from previous step
            com_vel = (com_xy - prev_com_xy) / dt_est  # shape (2,)

            # ── Base passivity torque ─────────────────────────────────────
            tau_wbc, hipr_int = self._compute_passivity_torque(
                curr_pitch, curr_roll, prev_pitch, prev_roll, dt_est, hipr_int
            )

            tau_base_6 = tau_wbc[self.mlp_output_indices].copy().astype(np.float32)
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
                tau_base_6=tau_base_6,
            )

            # ── MLP correction (evaluate mode) ────────────────────────────
            if mlp_model is not None and now_t >= first_push_start:
                with torch.no_grad():
                    x_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                    if mlp_device:
                        x_t = x_t.to(mlp_device)
                    delta_tau = mlp_model(x_t).squeeze(0).cpu().numpy()
                for k, idx in enumerate(self.mlp_output_indices):
                    tau_wbc[idx] += float(delta_tau[k])

            # ── Collect: log (state_46, tau_base) ────────────────────────
            if collect_data and now_t >= first_push_start:
                tau_out = tau_base_6.copy()
                trial_states.append(state_vec)
                trial_taus.append(tau_out.astype(np.float32))

            ctrl = np.clip(tau_wbc / self.gear, self.ctrl_min, self.ctrl_max)
            self.data.ctrl[:] = ctrl

            if now_t >= first_push_start:
                tau_sq_sum_total += float(np.sum(tau_wbc ** 2)) * self.dt
                tau_sq_duration  += self.dt

            total_tau_sq_integral += float(np.sum(tau_wbc ** 2)) * self.dt

            # Apply push
            self.data.xfrc_applied[:] = 0.0
            active = profile.active_push_at(now_t)
            if active is not None:
                bid = body_id_for.get(active.body_name, -1)
                if bid >= 0:
                    self.data.xfrc_applied[bid, 0:3] = active.force_vec

            # Update metrics
            if com_xy_ref is not None:
                com_xy_disp = float(np.linalg.norm(com_err))
                if now_t >= first_push_start:
                    max_com_xy_disp = max(max_com_xy_disp, com_xy_disp)
                    com_sq_integral       += (com_xy_disp ** 2) * self.dt
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

            # ── v2: update previous CoM for next step ─────────────────────
            prev_com_xy = com_xy.copy()

            prev_pitch = curr_pitch
            prev_roll  = curr_roll
            prev_t     = now_t
            mujoco.mj_step(self.model, self.data)

        # ── Run simulation ────────────────────────────────────────────────────
        while self.data.time < sim_end:
            step_once()
            if np.isfinite(fall_time):
                break

        # ── Post-trial bookkeeping ────────────────────────────────────────────
        recovery_time_out = float(recovery_time) if np.isfinite(recovery_time) else -1.0
        fall_time_out     = float(fall_time)     if np.isfinite(fall_time)     else -1.0
        is_stable         = 1 if (recovery_time_out >= 0.0 and fall_time_out < 0.0) else 0
        survival_time_s   = float(fall_time_out) if fall_time_out > 0 else float(sim_end)

        com_safe_dur   = max(float(com_integral_duration), 1e-6)
        rms_com_disp_m = float(np.sqrt(com_sq_integral / com_safe_dur))

        safe_dur      = max(float(tau_sq_duration), 1e-6)
        tau_rms_total = float(np.sqrt(float(tau_sq_sum_total) / safe_dur))

        n_pushes_delivered = profile.pushes_delivered_by(survival_time_s)

        # Flush trial data to global buffer only if stable (collect mode)
        if collect_data:
            if (not stable_only_collect) or (is_stable == 1):
                self._collect_states.extend(trial_states)
                self._collect_taus.extend(trial_taus)

        if profile.pushes:
            fp = profile.pushes[0]
            mag_rep        = float(profile.max_instantaneous_force_n)
            body_label_rep = fp.body_name
        else:
            mag_rep        = 0.0
            body_label_rep = "none"

        return {
            "session_id":               datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "controller_name":          str(self.args.controller_name),
            "condition_id":             int(profile.seed),
            "trial_in_condition":       0,
            "body_label":               body_label_rep,
            "body_name":                body_label_rep,
            "direction":                "random",
            "force_x":                  float(profile.pushes[0].force_vec[0]) if profile.pushes else 0.0,
            "force_y":                  float(profile.pushes[0].force_vec[1]) if profile.pushes else 0.0,
            "force_z":                  float(profile.pushes[0].force_vec[2]) if profile.pushes else 0.0,
            "force_magnitude":          mag_rep,
            "max_com_disp":             float(max_com_xy_disp),
            "max_com_disp_x":           0.0,
            "max_com_disp_y":           0.0,
            "recovery_time":            recovery_time_out,
            "fall_time":                fall_time_out,
            "is_stable":                int(is_stable),
            "tau_rms_total":            float(tau_rms_total),
            "tau_rms_ankle":            0.0,
            "tau_rms_knee":             0.0,
            "tau_rms_hip":              0.0,
            "max_torso_pitch_rad":      float(max_torso_pitch),
            "max_torso_roll_rad":       float(max_torso_roll),
            "qp_solve_time_mean_ms":    0.0,
            "qp_solve_time_max_ms":     0.0,
            "qp_solver_failures":       0,
            "disturbance_seed":         int(profile.seed),
            "n_pushes_scheduled":       int(profile.n_pushes_scheduled),
            "n_pushes_delivered":       int(n_pushes_delivered),
            "total_impulse_Ns":         float(profile.total_impulse_ns),
            "max_instantaneous_force_N": float(profile.max_instantaneous_force_n),
            "mean_push_magnitude_N":    float(profile.mean_push_magnitude_n),
            "push_log_json":            profile.to_json(),
            "survival_time_s":          float(survival_time_s),
            "rms_com_disp_m":           float(rms_com_disp_m),
            "integrated_torque_Nms":    float(total_tau_sq_integral),
        }


# =============================================================================
# MODE: COLLECT
# =============================================================================

def run_collect(args: argparse.Namespace) -> None:
    if not MUJOCO_AVAILABLE:
        raise RuntimeError("MuJoCo not available. Cannot run collect mode.")
    if not DISTURBANCE_AVAILABLE:
        raise RuntimeError("disturbance.py not found. Place it in the same directory.")

    print("=" * 60)
    print("COLLECT MODE  (v3 — history + base-torque input)")
    print(f"Seeds: 0 – {args.n_collect_seeds - 1}  ({args.n_collect_seeds} trials)")
    print("Stable-only logging: False")
    print(f"Input features: {INPUT_DIM}  ({SINGLE_FRAME_DIM} × {HISTORY_LENGTH} history)")
    print("=" * 60)

    engine = PassivityEngine(args)

    available_bodies = [
        name for name in BODY_WEIGHTS
        if mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    ]

    n_stable = 0
    for seed in range(args.n_collect_seeds):
        profile = DisturbanceProfile.generate(
            seed=seed,
            sim_duration_s=float(args.sim_time),
            available_bodies=available_bodies,
        )
        res = engine.run_trial(
            profile,
            collect_data=True,
            stable_only_collect=False,
        )
        stable_str = "STABLE" if res["is_stable"] else "fell  "
        ts_count   = len(engine._collect_states)
        print(
            f"  [{stable_str}] seed={seed:4d} | "
            f"max_F={res['max_instantaneous_force_N']:5.1f}N | "
            f"survival={res['survival_time_s']:.2f}s | "
            f"total_timesteps_so_far={ts_count}"
        )
        if res["is_stable"]:
            n_stable += 1

    n_timesteps = len(engine._collect_states)
    if n_timesteps == 0:
        print("WARNING: No timesteps collected. Check that some trials are stable.")
        return

    states_arr = np.array(engine._collect_states, dtype=np.float32)
    taus_arr   = np.array(engine._collect_taus,   dtype=np.float32)  # (N, 6)

    assert states_arr.shape[1] == INPUT_DIM, \
        f"Expected {INPUT_DIM} features, got {states_arr.shape[1]}"

    state_mean = states_arr.mean(axis=0)
    state_std  = states_arr.std(axis=0) + 1e-8
    tau_mean   = taus_arr.mean(axis=0)
    tau_std    = taus_arr.std(axis=0)  + 1e-8

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.dataset_file)
    np.savez_compressed(
        out_path,
        states=states_arr,
        taus=taus_arr,
        state_mean=state_mean,
        state_std=state_std,
        tau_mean=tau_mean,
        tau_std=tau_std,
        input_dim=np.array(INPUT_DIM),
        train_seeds=np.array(TRAIN_SEEDS),
        val_seeds=np.array(VAL_SEEDS),
        feature_names=np.array(FEATURE_NAMES),
    )

    print("\n" + "=" * 60)
    print("COLLECT COMPLETE")
    print(f"  Timesteps collected : {n_timesteps:,}")
    print(f"  Stable trials       : {n_stable}/{args.n_collect_seeds}")
    print(f"  State shape         : {states_arr.shape}  ← should be (N, {INPUT_DIM})")
    print(f"  Tau shape           : {taus_arr.shape}")
    last = (HISTORY_LENGTH - 1) * SINGLE_FRAME_DIM
    print(f"  CoM error mean x/y  : {state_mean[last + 42]:.4f} / {state_mean[last + 43]:.4f} m")
    print(f"  CoM vel   mean x/y  : {state_mean[last + 44]:.4f} / {state_mean[last + 45]:.4f} m/s")
    print(f"  Saved to            : {out_path}")
    print("=" * 60)


# =============================================================================
# LIPM one-step prediction loss (unchanged from v1)
# =============================================================================

def lipm_prediction_loss(
    states: "torch.Tensor",
    delta_tau: "torch.Tensor",
    dt: float = 0.001,
    g: float = 9.81,
    z_c: float = 0.9504,
    m: float = 47.0,
) -> "torch.Tensor":
    """
    LIPM prediction loss.
    v2: can now also use com_err_x (index 42) directly, but pitch (index 0)
    is kept as the primary variable for backward compatibility.
    """
    pitch   = states[:, 0]      # index 0
    d_pitch = states[:, 2]      # index 2

    # Optionally use com_error_x as a cross-check signal (indices 42-45 now available)
    # For simplicity we keep the same formulation as v1 — the CoM features are
    # the primary improvement via the input representation, not the loss.
    delta_ankle = (delta_tau[:, 0] + delta_tau[:, 1]) * 0.5

    k = float(g / z_c)
    predicted_d_pitch_next = d_pitch + k * pitch * dt - (1.0 / (m * z_c)) * delta_ankle * dt
    target = torch.zeros_like(predicted_d_pitch_next)
    return nn.functional.mse_loss(predicted_d_pitch_next, target)


# =============================================================================
# MODE: TRAIN
# =============================================================================

def run_train(args: argparse.Namespace) -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")

    dataset_path = os.path.join(args.output_dir, args.dataset_file)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}\nRun --mode collect first.")

    print("=" * 60)
    print("TRAIN MODE  (v3 — history + pitch-rate cloning target)")
    print(f"Dataset: {dataset_path}")
    print("=" * 60)

    data_np    = np.load(dataset_path)
    states_all = data_np["states"]
    taus_all   = data_np["taus"]
    state_mean = data_np["state_mean"]
    state_std  = data_np["state_std"]

    # Verify we have the right dataset
    actual_input_dim = states_all.shape[1]
    if actual_input_dim != INPUT_DIM:
        raise ValueError(
            f"Dataset has {actual_input_dim} features but INPUT_DIM={INPUT_DIM}.\n"
            f"You need to re-run --mode collect to regenerate the dataset."
        )

    print(f"  Total timesteps : {len(states_all):,}")
    print(f"  State shape     : {states_all.shape}  ← should be (N, {INPUT_DIM})")
    print(f"  Tau shape       : {taus_all.shape}")
    last = (HISTORY_LENGTH - 1) * SINGLE_FRAME_DIM
    print(f"  CoM err mean    : x={state_mean[last + 42]:.4f}m  y={state_mean[last + 43]:.4f}m")
    print(f"  CoM vel mean    : x={state_mean[last + 44]:.4f}m/s  y={state_mean[last + 45]:.4f}m/s")

    states_norm = (states_all - state_mean) / state_std

    n_total = len(states_norm)
    n_train = int(n_total * 0.8)
    rng  = np.random.default_rng(42)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    X_train = torch.tensor(states_norm[train_idx], dtype=torch.float32)
    X_val = torch.tensor(states_norm[val_idx], dtype=torch.float32)

    d_pitch_idx = (HISTORY_LENGTH - 1) * SINGLE_FRAME_DIM + 2
    d_pitch_all = states_all[:, d_pitch_idx].astype(np.float32)
    alpha = np.clip(np.abs(d_pitch_all) / 1.0, 0.0, 1.0).astype(np.float32)
    sign = (-np.sign(d_pitch_all)).astype(np.float32)
    corr = alpha * sign
    y_all = np.zeros((len(states_all), OUTPUT_DIM), dtype=np.float32)
    y_all[:, 0] = corr * 8.0
    y_all[:, 1] = corr * 8.0
    y_all[:, 2] = corr * 4.0
    y_all[:, 3] = corr * 4.0
    y_all[:, 4] = 0.0
    y_all[:, 5] = 0.0

    y_train = torch.tensor(y_all[train_idx], dtype=torch.float32)
    y_val = torch.tensor(y_all[val_idx], dtype=torch.float32)

    print(f"  Train samples   : {len(X_train):,}")
    print(f"  Val samples     : {len(X_val):,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device          : {device}")

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size,
                          shuffle=True, num_workers=0)
    val_dl   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=args.batch_size,
                          shuffle=False, num_workers=0)

    # Model uses INPUT_DIM=46 automatically
    model     = ResidualMLP(input_dim=INPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

    reg_lambda     = float(args.reg_lambda)
    best_val_loss  = float("inf")
    patience_count = 0
    best_state_dict = None

    print("\nTraining...")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred  = model(X_b)
            mse_loss = nn.functional.mse_loss(pred, y_b)
            reg_loss = reg_lambda * (pred ** 2).mean()
            loss = mse_loss + reg_loss
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
                pred  = model(X_b)
                mse_loss = nn.functional.mse_loss(pred, y_b)
                reg_loss = reg_lambda * (pred ** 2).mean()
                loss = mse_loss + reg_loss
                val_loss_sum += loss.item() * len(X_b)
        val_loss = val_loss_sum / len(val_dl.dataset)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {current_lr:>10.2e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss   = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count  = 0
        else:
            patience_count += 1
            if patience_count >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.early_stop_patience})")
                break

    if best_state_dict is None:
        best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    os.makedirs(args.output_dir, exist_ok=True)
    weights_path = os.path.join(args.output_dir, args.weights_file)
    torch.save({
        "model_state_dict": best_state_dict,
        "state_mean":       state_mean,
        "state_std":        state_std,
        "input_dim":        INPUT_DIM,
        "output_dim":       OUTPUT_DIM,
        "hidden_dim":       HIDDEN_DIM,
        "tau_bound":        TAU_BOUND,
        "output_joints":    OUTPUT_JOINT_NAMES,
        "best_val_loss":    best_val_loss,
        "version":          "v2_com_augmented",
        "feature_names":    FEATURE_NAMES,
    }, weights_path)

    print("\n" + "=" * 60)
    print("TRAIN COMPLETE")
    print(f"  Best val loss : {best_val_loss:.6f}")
    print(f"  Saved to      : {weights_path}")
    print("=" * 60)


# =============================================================================
# MODE: EVALUATE
# =============================================================================

def run_evaluate(args: argparse.Namespace) -> None:
    if not MUJOCO_AVAILABLE:
        raise RuntimeError("MuJoCo not available.")
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")
    if not DISTURBANCE_AVAILABLE:
        raise RuntimeError("disturbance.py not found.")

    weights_path = os.path.join(args.output_dir, args.weights_file)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}\nRun --mode train first.")

    print("=" * 60)
    print("EVALUATE MODE  (v2 — 46-dim input)")
    print(f"Weights : {weights_path}")
    print(f"Seeds   : 0 – {args.n_eval_seeds - 1}  ({args.n_eval_seeds} trials)")
    print("=" * 60)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    saved_input_dim = int(checkpoint["input_dim"])
    if saved_input_dim != INPUT_DIM:
        raise ValueError(
            f"Weights were trained with input_dim={saved_input_dim} "
            f"but current INPUT_DIM={INPUT_DIM}.\n"
            f"Re-run --mode collect and --mode train to regenerate."
        )

    mlp_model = ResidualMLP(
        input_dim  = saved_input_dim,
        hidden_dim = int(checkpoint["hidden_dim"]),
        output_dim = int(checkpoint["output_dim"]),
        tau_bound  = float(checkpoint["tau_bound"]),
    ).to(device)
    mlp_model.load_state_dict(checkpoint["model_state_dict"])
    mlp_model.eval()

    state_mean = checkpoint["state_mean"]
    state_std  = checkpoint["state_std"]
    version    = checkpoint.get("version", "v1")
    print(f"  Model version       : {version}")
    print(f"  Best val loss       : {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"  Input dim           : {saved_input_dim}")

    args.controller_name = "passivity_mlp_v3"

    engine = PassivityEngine(args)

    class NormalisedMLP(nn.Module):
        def __init__(self, base_model, mean, std):
            super().__init__()
            self.base = base_model
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std  = torch.tensor(std,  dtype=torch.float32)

        def forward(self, x):
            if x.device != self.mean.device:
                self.mean = self.mean.to(x.device)
                self.std  = self.std.to(x.device)
            return self.base((x - self.mean) / self.std)

    norm_mlp = NormalisedMLP(mlp_model, state_mean, state_std).to(device)
    norm_mlp.eval()

    available_bodies = [
        name for name in BODY_WEIGHTS
        if mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    ]

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path   = os.path.join(args.output_dir, f"passivity_mlp_v3_random_{session_id}.csv")

    all_results: List[Dict[str, Any]] = []
    writer = None
    header_written = False

    with open(csv_path, "w", newline="") as f:
        for seed in range(args.n_eval_seeds):
            profile = DisturbanceProfile.generate(
                seed=seed,
                sim_duration_s=float(args.sim_time),
                available_bodies=available_bodies,
            )
            res = engine.run_trial(
                profile,
                collect_data=False,
                mlp_model=norm_mlp,
                mlp_device=device,
            )
            all_results.append(res)

            stable_str = "STABLE" if res["is_stable"] else "FELL  "
            print(
                f"  [{stable_str}] seed={seed:4d} | "
                f"max_F={res['max_instantaneous_force_N']:5.1f}N | "
                f"survival={res['survival_time_s']:.2f}s | "
                f"rms_com={res['rms_com_disp_m']:.4f}m"
            )

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(res.keys()))
            if not header_written:
                writer.writeheader()
                header_written = True
            writer.writerow(res)
            f.flush()

    n           = len(all_results)
    n_stable    = sum(r["is_stable"]   for r in all_results)
    fall_count  = sum(1 for r in all_results if float(r["fall_time"]) > 0)
    neither     = n - n_stable - fall_count
    mean_surv   = float(np.mean([r["survival_time_s"] for r in all_results]))
    mean_rms    = float(np.mean([r["rms_com_disp_m"]  for r in all_results]))

    print("\n" + "=" * 60)
    print("EVALUATE COMPLETE  (v2)")
    print(f"  Stable          : {n_stable}/{n}  ({100*n_stable/n:.1f}%)")
    print(f"  Passivity base  : 110/200 (55.0%)  ← compare against this")
    print(f"  Fell            : {fall_count}/{n}")
    print(f"  Neither         : {neither}/{n}")
    print(f"  Mean survival   : {mean_surv:.2f} s")
    print(f"  Mean rms CoM    : {mean_rms:.4f} m")
    print(f"  CSV saved       : {csv_path}")
    print("=" * 60)


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Residual MLP v2 — CoM-augmented input")

    p.add_argument("--mode", type=str, required=True,
                   choices=["collect", "train", "evaluate"])

    p.add_argument("--xml", type=str,
        default=r"C:\Users\sanju\Documents\University_West\Course\Thesis\Scripts\venv310\model\New_MOdels\mujoco_menagerie-main\unitree_h1\scene.xml")
    p.add_argument("--output-dir",    type=str, default="experiment_results")
    p.add_argument("--dataset-file",  type=str, default="residual_mlp_dataset_v3.npz")
    p.add_argument("--weights-file",  type=str, default="residual_mlp_weights_v3.pt")
    p.add_argument("--controller-name", type=str, default="passivity_based")

    # Simulation parameters (identical to passivity script)
    p.add_argument("--sim-time",           type=float, default=10.0)
    p.add_argument("--pre-push-window",    type=float, default=1.0)
    p.add_argument("--recovery-thresh",    type=float, default=0.02)
    p.add_argument("--recovery-window",    type=float, default=0.5)
    p.add_argument("--fall-pitch-thresh",  type=float, default=1.2)
    p.add_argument("--fall-height-thresh", type=float, default=0.65)
    p.add_argument("--target-pitch",       type=float, default=0.05)
    p.add_argument("--kp-posture",         type=float, default=200.0)
    p.add_argument("--kd-posture",         type=float, default=20.0)
    p.add_argument("--kp-tilt",            type=float, default=500.0)
    p.add_argument("--kd-tilt",            type=float, default=50.0)
    p.add_argument("--kp-roll",            type=float, default=500.0)
    p.add_argument("--kd-roll",            type=float, default=50.0)
    p.add_argument("--damping-gain",       type=float, default=20.0)
    p.add_argument("--init-noise",         type=float, default=0.0)

    p.add_argument("--n-collect-seeds", type=int, default=200)
    p.add_argument("--collect-all",     action="store_true")

    p.add_argument("--epochs",              type=int,   default=100)
    p.add_argument("--batch-size",          type=int,   default=256)
    p.add_argument("--lr",                  type=float, default=1e-3)
    p.add_argument("--reg-lambda",          type=float, default=0.0001)
    p.add_argument("--early-stop-patience", type=int,   default=15)

    p.add_argument("--n-eval-seeds", type=int, default=200)

    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

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
