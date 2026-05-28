import argparse
import csv
import datetime
import math
import os
import time
from typing import Any, Dict, List, Tuple

import mujoco
import mujoco.viewer
import numpy as np

try:
    import cvxpy as cp
except ModuleNotFoundError:
    cp = None


PUSH_LOCATION_PRESETS = {
    "torso": ["torso_link"],
    "pelvis": ["pelvis"],
}

DIRECTION_PRESETS = {
    "forward": np.array([1.0, 0.0, 0.0], dtype=float),
    "backward": np.array([-1.0, 0.0, 0.0], dtype=float),
    "lateral": np.array([0.0, 1.0, 0.0], dtype=float),
}


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
        "HipR": [i for i, n in enumerate(names) if "hip_roll" in n],
    }


def build_gains(model: mujoco.MjModel, groups: Dict[str, List[int]], kp_posture: float, kd_posture: float) -> Tuple[np.ndarray, np.ndarray]:
    kp = np.full(model.nu, float(kp_posture), dtype=float)
    kd = np.full(model.nu, float(kd_posture), dtype=float)
    for i in groups["HipR"]:
        kp[i], kd[i] = 500.0, 50.0
    return kp, kd


def hipr_state(data: mujoco.MjData, qaddr: np.ndarray, daddr: np.ndarray, q_target_full: np.ndarray, hipr_ids: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hipr_qaddr = qaddr[hipr_ids]
    hipr_daddr = daddr[hipr_ids]
    hipr_q = data.qpos[hipr_qaddr]
    hipr_dq = data.qvel[hipr_daddr]
    hipr_q_ref = q_target_full[hipr_qaddr]
    hipr_err = hipr_q_ref - hipr_q
    return hipr_q, hipr_dq, hipr_err


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


def build_hqp(model: mujoco.MjModel, ctrl_min: np.ndarray, ctrl_max: np.ndarray, gear: np.ndarray, groups: Dict[str, List[int]]):
    if cp is None:
        raise RuntimeError("cvxpy is required. Install with: pip install cvxpy osqp")
    tau = cp.Variable(model.nu)
    tau_nominal = cp.Parameter(model.nu)
    tilt_target = cp.Parameter()
    roll_target = cp.Parameter()
    hipr_hold = cp.Parameter(len(groups["HipR"]))
    tau_balance_ref = cp.Parameter(model.nu)
    balance_p = 0
    for i in groups["AnkleP"]:
        balance_p += cp.square(tau[i] - (tau_nominal[i] + tilt_target))
    for i in groups["HipP"]:
        balance_p += cp.square(tau[i] - (tau_nominal[i] + 0.5 * tilt_target))
    balance_r = cp.sum_squares(tau[groups["HipR"]] - (tau_nominal[groups["HipR"]] + hipr_hold))
    balance_roll = 0
    for i in groups["HipR"]:
        balance_roll += cp.square(tau[i] - (tau_nominal[i] + roll_target))
    posture_obj = cp.sum_squares(tau - tau_nominal)
    constraints = [tau >= ctrl_min * gear, tau <= ctrl_max * gear]
    prob_balance = cp.Problem(
        cp.Minimize(
            1.0 * posture_obj
            + 2000.0 * balance_p
            + 300.0 * balance_roll
            + 300.0 * balance_r
            + 0.0005 * cp.sum_squares(tau)
        ),
        constraints,
    )
    pri_ids = sorted(set(groups["AnkleP"] + groups["HipP"] + groups["HipR"]))
    eqs = [tau[i] == tau_balance_ref[i] for i in pri_ids]
    prob_posture = cp.Problem(
        cp.Minimize(1.0 * posture_obj + 300.0 * balance_r + 0.0005 * cp.sum_squares(tau)),
        constraints + eqs,
    )
    return prob_balance, prob_posture, tau, tau_nominal, tilt_target, roll_target, hipr_hold, tau_balance_ref


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
        self.knee_ids = [i for i, n in enumerate(self.actuator_names) if "knee" in n]
        self.hip_ids = [i for i, n in enumerate(self.actuator_names) if ("hip_pitch" in n) or ("hip_roll" in n) or ("hip_yaw" in n)]
        self.kp, self.kd = build_gains(self.model, self.groups, self.args.kp_posture, self.args.kd_posture)
        self.gear = self.model.actuator_gear[:, 0].copy()
        self.gear[self.gear == 0.0] = 1.0
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1]
        self.q_target_full = self.data.qpos.copy()
        if float(getattr(self.args, "tilt_sign", 0.0)) != 0.0:
            self.tilt_sign = float(np.sign(float(self.args.tilt_sign)))
        else:
            self.tilt_sign = calibrate_balance_sign(self.model, self.data, self.groups["AnkleP"])
        safe_reset(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.prob_balance, self.prob_posture, self.tau_var, self.tau_nominal, self.tilt_target, self.roll_target, self.hipr_hold, self.tau_balance_ref = build_hqp(
            self.model, self.ctrl_min, self.ctrl_max, self.gear, self.groups
        )
        self.output_dir = str(self.args.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        controller_label = getattr(self.args, "controller_name", "controller")
        self.csv_path = os.path.join(self.output_dir, f"{controller_label}_{self.session_id}.csv")
        self.all_results: List[Dict[str, Any]] = []
        self.rng = np.random.default_rng(int(self.args.seed))

    def _apply_small_init_noise(self) -> None:
        noise_std = float(self.args.init_noise)
        if noise_std <= 0.0:
            return
        self.data.qpos[self.qaddr] += self.rng.normal(0.0, noise_std, size=self.data.qpos[self.qaddr].shape)
        self.data.qpos[3:7] += self.rng.normal(0.0, noise_std * 0.1, size=4)
        q = self.data.qpos[3:7]
        self.data.qpos[3:7] = q / np.linalg.norm(q)
        mujoco.mj_forward(self.model, self.data)

    def run_single_trial(self, config: Dict[str, Any], trial_in_condition: int, visualize: bool) -> Dict[str, Any]:
        safe_reset(self.model, self.data)
        self._apply_small_init_noise()
        mujoco.mj_forward(self.model, self.data)
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
        qp_solve_times: List[float] = []
        tau_sq_sum_total = 0.0
        tau_sq_sum_ankle = 0.0
        tau_sq_sum_knee = 0.0
        tau_sq_sum_hip = 0.0
        tau_sq_duration = 0.0
        hipr_int = np.zeros(len(self.groups["HipR"]), dtype=float)
        hipr_kp, hipr_kd, hipr_ki = 400.0, 40.0, 60.0
        prev_pitch = 0.0
        prev_t = float(self.data.time)
        self._prev_roll = 0.0
        dbg_last_print_t = -1e9
        h_ref = float(self.data.xipos[self.pelvis_id, 2]) if self.pelvis_id >= 0 else 0.9

        def step_once() -> None:
            nonlocal com_xy_ref, fall_time, hipr_int, max_com_x_disp, max_com_xy_disp, max_com_y_disp, max_torso_pitch, max_torso_roll, prev_pitch, prev_t, qp_solve_times, recovery_buffer, recovery_time, tau_sq_duration, tau_sq_sum_ankle, tau_sq_sum_hip, tau_sq_sum_knee, tau_sq_sum_total, dbg_last_print_t
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
            pelvis_height = float(self.data.xipos[self.pelvis_id, 2]) if self.pelvis_id >= 0 else float("inf")
            pitch_fallen = abs(curr_pitch) > float(self.args.fall_pitch_thresh)
            height_fallen = pelvis_height < float(self.args.fall_height_thresh)
            if (pitch_fallen or height_fallen) and not np.isfinite(fall_time):
                fall_time = now_t
            h_err = float(h_ref - pelvis_height)
            q_curr = self.data.qpos[self.qaddr]
            dq_curr = self.data.qvel[self.daddr]
            tau_pd = self.kp * (self.q_target_full[self.qaddr] - q_curr) - self.kd * dq_curr
            tau_nom = tau_pd + self.data.qfrc_bias[self.daddr]
            self.tau_nominal.value = tau_nom
            d_pitch = (curr_pitch - prev_pitch) / dt_est
            self.tilt_target.value = -self.tilt_sign * (self.args.kp_tilt * (curr_pitch - self.args.target_pitch) + self.args.kd_tilt * d_pitch)
            d_roll = (curr_roll - float(self._prev_roll)) / dt_est
            self.roll_target.value = float(-self.args.kp_roll * curr_roll - self.args.kd_roll * d_roll)
            self._prev_roll = float(curr_roll)
            _, hipr_dq, hipr_err = hipr_state(self.data, self.qaddr, self.daddr, self.q_target_full, self.groups["HipR"])
            hipr_int = np.clip(hipr_int + hipr_err * dt_est, -0.3, 0.3)
            self.hipr_hold.value = hipr_kp * hipr_err - hipr_kd * hipr_dq + hipr_ki * hipr_int
            if self.tau_nominal.value is None or not np.all(np.isfinite(self.tau_nominal.value)):
                self.tau_nominal.value = np.zeros(self.model.nu)
            if self.tilt_target.value is None or not np.isfinite(float(self.tilt_target.value)):
                self.tilt_target.value = 0.0
            if self.roll_target.value is None or not np.isfinite(float(self.roll_target.value)):
                self.roll_target.value = 0.0
            if self.hipr_hold.value is None or not np.all(np.isfinite(self.hipr_hold.value)):
                self.hipr_hold.value = np.zeros(len(self.groups["HipR"]))
            _t0 = time.perf_counter()
            tau_balance = tau_nom
            balance_status = "err"
            try:
                self.prob_balance.solve(solver=cp.OSQP, warm_start=True, max_iter=20000)
                balance_status = str(self.prob_balance.status)
                if self.prob_balance.status in ("optimal", "optimal_inaccurate") and self.tau_var.value is not None:
                    if np.all(np.isfinite(self.tau_var.value)):
                        tau_balance = self.tau_var.value.copy()
            except Exception:
                tau_balance = tau_nom
                balance_status = "err"
            tau_wbc = tau_balance
            posture_status = "err"
            try:
                self.tau_balance_ref.value = tau_balance
                self.prob_posture.solve(solver=cp.OSQP, warm_start=True, max_iter=20000)
                posture_status = str(self.prob_posture.status)
                if self.prob_posture.status in ("optimal", "optimal_inaccurate") and self.tau_var.value is not None:
                    if np.all(np.isfinite(self.tau_var.value)):
                        tau_wbc = self.tau_var.value.copy()
            except Exception:
                tau_wbc = tau_balance
                posture_status = "err"
            _solve_ms = (time.perf_counter() - _t0) * 1000.0
            qp_solve_times.append(float(_solve_ms))
            ctrl = np.clip(tau_wbc / self.gear, self.ctrl_min, self.ctrl_max)
            self.data.ctrl[:] = ctrl

            if int(getattr(self.args, "hqp_debug", 0)) == 1 and str(getattr(self.args, "mode", "")) == "debug":
                dbg_dt = float(getattr(self.args, "hqp_debug_dt", 0.05))
                dbg_until = float(getattr(self.args, "hqp_debug_until", 2.0))
                if now_t <= dbg_until and (now_t - dbg_last_print_t) >= dbg_dt:
                    dbg_last_print_t = now_t
                    sat = int(np.sum((ctrl <= (self.ctrl_min + 1e-9)) | (ctrl >= (self.ctrl_max - 1e-9))))
                    ankle_ids = self.groups["AnkleP"]
                    hipp_ids = self.groups["HipP"]
                    hipr_ids = self.groups["HipR"]
                    print(
                        f"[HQP2] t={now_t:.3f} h={pelvis_height:.3f} pitch={curr_pitch:+.3f} roll={curr_roll:+.3f} "
                        f"tilt={float(self.tilt_target.value):+.2f} roll_t={float(self.roll_target.value):+.2f} "
                        f"h_ref={h_ref:.3f} h_err={h_err:+.3f} "
                        f"bal={balance_status} post={posture_status} "
                        f"tau|max nom={float(np.max(np.abs(tau_nom))):.1f} bal={float(np.max(np.abs(tau_balance))):.1f} wbc={float(np.max(np.abs(tau_wbc))):.1f} "
                        f"ctrl|max={float(np.max(np.abs(ctrl))):.3f} sat={sat}/{self.model.nu} "
                        f"ankle_tau={[round(float(tau_wbc[i]),1) for i in ankle_ids]} "
                        f"hipp_tau={[round(float(tau_wbc[i]),1) for i in hipp_ids]} "
                        f"hipr_tau={[round(float(tau_wbc[i]),1) for i in hipr_ids]}"
                    )
            if now_t >= push_start:
                ankle_ids = self.groups["AnkleP"]
                knee_ids = self.knee_ids
                hip_ids = self.hip_ids
                hip_ids = self.hip_ids
                tau_sq_sum_total += float(np.sum(tau_wbc**2)) * self.dt
                tau_sq_sum_ankle += float(np.sum(tau_wbc[ankle_ids] ** 2)) * self.dt
                tau_sq_sum_knee += float(np.sum(tau_wbc[knee_ids] ** 2)) * self.dt
                tau_sq_sum_hip += float(np.sum(tau_wbc[hip_ids] ** 2)) * self.dt
                tau_sq_duration += self.dt
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
                    recovery_buffer.append(1 if com_xy_disp <= float(self.args.recovery_thresh) else 0)
                    if len(recovery_buffer) > recovery_window_samples:
                        recovery_buffer.pop(0)
                    if len(recovery_buffer) == recovery_window_samples and sum(recovery_buffer) >= int(0.8 * recovery_window_samples):
                        recovery_time = now_t - push_end
            prev_pitch = curr_pitch
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
                        while v.is_running():
                            v.sync()
                            time.sleep(0.01)
            except Exception:
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
        qp_solve_time_mean_ms = float(np.mean(qp_solve_times)) if qp_solve_times else -1.0
        qp_solve_time_max_ms = float(np.max(qp_solve_times)) if qp_solve_times else -1.0
        qp_solver_failures = int(sum(1 for t in qp_solve_times if float(t) <= 0.0))
        recovery_time_out = float(recovery_time) if np.isfinite(recovery_time) else -1.0
        fall_time_out = float(fall_time) if np.isfinite(fall_time) else -1.0
        is_stable = 1 if (recovery_time_out >= 0.0 and fall_time_out < 0.0) else 0
        result = {
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
            "qp_solve_time_mean_ms": float(qp_solve_time_mean_ms),
            "qp_solve_time_max_ms": float(qp_solve_time_max_ms),
            "qp_solver_failures": int(qp_solver_failures),
        }
        return result

    def _should_visualize_trial(self, config: Dict[str, Any], trial_in_condition: int, viz_count_so_far: int) -> bool:
        policy = str(self.args.viz_policy)
        if policy == "none":
            return False
        if viz_count_so_far >= int(self.args.viz_max):
            return False
        if policy == "first_trial_each_condition":
            return trial_in_condition == 0
        if policy == "only_force":
            return trial_in_condition == 0 and float(config["mag"]) == float(self.args.viz_force)
        if policy == "every_nth_condition":
            step = max(1, int(self.args.viz_step))
            return trial_in_condition == 0 and (int(config["condition_id"]) % step == 0)
        return False

    def _print_trial_result(self, cfg: Dict[str, Any], trial_num: int, total_trials: int, res: Dict[str, Any]) -> None:
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
                    configs.append(
                        {
                            "condition_id": condition_id,
                            "body_label": label,
                            "body_name": body_name,
                            "body_id": body_id,
                            "dir_name": dname,
                            "mag": float(mag),
                            "force_vec": force_vec,
                        }
                    )
                    condition_id += 1
        print(f"Batch mode: {len(configs)} conditions, {num_trials} trials each")
        print(f"Total simulations: {len(configs) * num_trials}")
        print(f"CSV output: {self.csv_path}")
        print(f"Visualization policy: {self.args.viz_policy} (max {self.args.viz_max} trials)")
        header_written = False
        viz_count = 0
        with open(self.csv_path, "w", newline="") as f:
            writer = None
            for cfg in configs:
                print(f"Condition {cfg['condition_id']:04d} | {cfg['body_label']} | {cfg['dir_name']} | {cfg['mag']}N")
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
            raise RuntimeError(f"Debug target body not found in model: {target_label}")
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
        print("Debug visualization mode")
        print(f"- body_label: {cfg['body_label']}")
        print(f"- body_name:  {cfg['body_name']}")
        print(f"- direction:  {cfg['dir_name']}")
        print(f"- magnitude:  {cfg['mag']} N")
        print(f"- sim_time:   {self.args.sim_time} s")
        if int(self.args.debug_visualize) == 1:
            print("A viewer window will open. Close it to end the run.")
        else:
            print("Debug is running headless (no viewer).")
        if int(getattr(self.args, "hqp_debug", 0)) == 1:
            ankle_ids = self.groups["AnkleP"]
            ankle_names = [self.actuator_names[i] for i in ankle_ids]
            ankle_gears = [float(self.gear[i]) for i in ankle_ids]
            ankle_ranges = [(float(self.ctrl_min[i]), float(self.ctrl_max[i])) for i in ankle_ids]
            print(f"- target_pitch: {float(self.args.target_pitch):+.3f}")
            print(f"- tilt_sign:    {float(self.tilt_sign):+.0f}")
            print(f"- ankle_ids:    {ankle_ids}")
            print(f"- ankle_names:  {ankle_names}")
            print(f"- ankle_gear:   {ankle_gears}")
            print(f"- ankle_ctrl:   {ankle_ranges}")
        prev_noise = float(self.args.init_noise)
        self.args.init_noise = 0.0
        res = self.run_single_trial(cfg, trial_in_condition=0, visualize=bool(int(self.args.debug_visualize)))
        self.args.init_noise = prev_noise
        print("\n==== Debug Result (single run) ====")
        print(f"max_com_disp:   {res['max_com_disp']:.4f} m")
        print(f"recovery_time:  {res['recovery_time']:.4f} s (or -1 means no recovery)")
        print(f"fall_time:      {res['fall_time']:.4f} s (or -1 means no fall detected)")
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
        print("\nQuick Summary (stability rate and mean CoM displacement)")
        print(f"{'Location':<12} | {'Dir':<8} | {'Force(N)':<8} | {'Stable%':<8} | {'MeanCoM(m)':<10}")
        print("-" * 60)
        for (loc, d, force), data in sorted(summary.items()):
            rate = (data["stable"] / data["total"]) * 100.0
            mean_disp = float(np.mean(data["com_disp"])) if data["com_disp"] else float("nan")
            print(f"{loc:<12} | {d:<8} | {force:<8.0f} | {rate:<8.1f} | {mean_disp:<10.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--xml",
        type=str,
        default=r"C:\Users\sanju\Documents\University_West\Course\Thesis\Scripts\venv310\model\New_MOdels\mujoco_menagerie-main\unitree_h1\scene.xml",
    )
    p.add_argument("--mode", type=str, default="batch", choices=["batch", "debug"])
    p.add_argument("--output-dir", type=str, default="experiment_results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-time", type=float, default=10.0)
    p.add_argument("--push-start", type=float, default=3.0)
    p.add_argument("--push-duration", type=float, default=0.1)
    p.add_argument("--pre-push-window", type=float, default=1.0)
    p.add_argument("--recovery-thresh", type=float, default=0.02)
    p.add_argument("--recovery-window", type=float, default=0.5)
    p.add_argument("--fall-pitch-thresh", type=float, default=1.2)
    p.add_argument(
        "--fall-height-thresh",
        type=float,
        default=0.65,
    )
    p.add_argument("--target-pitch", type=float, default=0.05)
    p.add_argument("--kp-posture", type=float, default=200.0)
    p.add_argument("--kd-posture", type=float, default=20.0)
    p.add_argument("--kp-tilt", type=float, default=500.0)
    p.add_argument("--kd-tilt", type=float, default=50.0)
    p.add_argument("--kp-roll", type=float, default=500.0)
    p.add_argument("--kd-roll", type=float, default=50.0)
    p.add_argument("--kp-com", type=float, default=0.0)
    p.add_argument("--kd-com", type=float, default=0.0)
    p.add_argument("--kp-height", type=float, default=0.0)
    p.add_argument("--tilt-sign", type=float, default=0.0)
    p.add_argument("--hqp-debug", type=int, default=1)
    p.add_argument("--hqp-debug-dt", type=float, default=0.05)
    p.add_argument("--hqp-debug-until", type=float, default=2.0)
    p.add_argument(
        "--init-noise",
        type=float,
        default=0.01,
    )
    p.add_argument(
        "--controller-name",
        type=str,
        default="hierarchical_qp",
    )
    p.add_argument("--magnitudes", type=str, default="50,100,150,200,250,300")
    p.add_argument("--targets", type=str, default="torso,pelvis")
    p.add_argument("--directions", type=str, default="forward,backward,lateral")
    p.add_argument("--num-trials", type=int, default=5)
    p.add_argument(
        "--viz-policy",
        type=str,
        default="only_force",
        choices=["none", "first_trial_each_condition", "only_force", "every_nth_condition"],
    )
    p.add_argument("--viz-force", type=float, default=100.0)
    p.add_argument("--viz-step", type=int, default=10)
    p.add_argument("--viz-max", type=int, default=20)
    p.add_argument("--viz-realtime", type=int, default=1)
    p.add_argument("--viz-hold", type=int, default=0)
    p.add_argument("--debug-target", type=str, default="pelvis")
    p.add_argument("--debug-direction", type=str, default="forward", choices=list(DIRECTION_PRESETS.keys()))
    p.add_argument("--debug-force", type=float, default=100.0)
    p.add_argument("--debug-visualize", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engine = HumanoidExperimentEngine(args)
    if args.mode == "debug":
        engine.run_debug_visualization()
        return
    engine.run_batch_experiments()


if __name__ == "__main__":
    main()
