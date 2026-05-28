"""
disturbance.py — Randomised multi-push disturbance generator for Part 1b.

This module defines the DisturbanceProfile class, which generates a sequence
of discrete push disturbances for one simulation trial. Each trial:

- Has N pushes (Poisson-distributed, clamped to [1, 5])
- Each push has independently sampled location, direction, magnitude, timing
- Disturbance is fully determined by a single integer seed (reproducible)
- Can serialise to/from JSON for post-hoc analysis

Design decisions:
- Magnitude: log-uniform [20, 250] N, scaled by body mass ratio
- Locations: 10 body points covering upper body (pelvis, torso, shoulders,
  elbows) AND lower body (hip_pitch, knee), weighted by segment mass
- Directions: uniform azimuth in XY, ±15 deg vertical tilt
- Duration: uniform [0.05, 0.20] s
- Timing: first push in [1.0, 2.5] s, subsequent ≥1.5 s apart, all before 8.5 s

Body weight design (v2 — 10 sites):
  Upper body (heavy trunk):
    pelvis          0.22  scale 1.00  (5.39 kg — direct CoM coupling)
    torso_link      0.22  scale 1.00  (17.79 kg — largest moment arm)
  Upper body (arms):
    left/right_shoulder_roll_link  0.07 each  scale 0.60  (0.79 kg)
    left/right_elbow_link          0.07 each  scale 0.60  (0.67 kg)
  Lower body (new in v2):
    left/right_hip_pitch_link      0.07 each  scale 0.80  (4.15 kg)
    left/right_knee_link           0.04 each  scale 0.60  (1.72 kg)
  Total: 0.22+0.22+0.07+0.07+0.07+0.07+0.07+0.07+0.04+0.04 = 1.00 ✓

Hip_pitch force scale 0.80: mass ratio ~4.15/17.79 ≈ 0.23, but hip_pitch sits
at knee height and directly loads the knee joint, so a moderate scale (0.80)
is appropriate to test joint-loading failure modes at realistic magnitudes.

Knee_link force scale 0.60: same as arm segments by mass ratio.

Literature grounding:
- Kim et al. 2025: up to 150 N for 0.05 s on iCub
- Wiedebach et al. 2025: up to 100 N for 0.2 s
- Castano et al. 2022: upper-body pushes standard in WBC benchmarking;
  lower-body pushes included here as thesis extension.

Run standalone to inspect sample profiles:
    python disturbance.py
    python disturbance.py --seeds 42,43,44 --verbose
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Body locations and their sampling weights (must sum to 1.0).
# v2: added left/right_hip_pitch_link and left/right_knee_link.
# Existing upper-body weights renormalised to accommodate new sites.
BODY_WEIGHTS: Dict[str, float] = {
    # Upper body — trunk
    "pelvis":                       0.20,
    "torso_link":                   0.20,
    # Upper body — arms
    "left_shoulder_roll_link":      0.08,
    "right_shoulder_roll_link":     0.08,
    "left_elbow_link":              0.08,
    "right_elbow_link":             0.08,
    # Lower body — thigh (NEW)
    "left_hip_pitch_link":          0.08,
    "right_hip_pitch_link":         0.08,
    # Lower body — shin (NEW)
    "left_knee_link":               0.06,
    "right_knee_link":              0.06,
}

# Verify weights sum to 1.0 at import time (catches copy-paste errors).
_WEIGHT_SUM = sum(BODY_WEIGHTS.values())
assert abs(_WEIGHT_SUM - 1.0) < 1e-9, (
    f"BODY_WEIGHTS must sum to 1.0, got {_WEIGHT_SUM:.10f}"
)

# Force magnitude scaling by body.
# Reflects the mass ratio of each segment relative to the torso.
# Smaller segments receive lower peak forces so the disturbance magnitude
# distribution remains physically plausible across all sites.
BODY_FORCE_SCALE: Dict[str, float] = {
    # Upper body — trunk (full force range)
    "pelvis":                       1.00,
    "torso_link":                   1.00,
    # Upper body — arms (light segments, ~0.67–0.79 kg)
    "left_shoulder_roll_link":      0.60,
    "right_shoulder_roll_link":     0.60,
    "left_elbow_link":              0.60,
    "right_elbow_link":             0.60,
    # Lower body — thigh (~4.15 kg, moderate force scale)
    "left_hip_pitch_link":          0.80,
    "right_hip_pitch_link":         0.80,
    # Lower body — shin (~1.72 kg, same scale as arm segments)
    "left_knee_link":               0.60,
    "right_knee_link":              0.60,
}

# Magnitude distribution: log-uniform [20, 250] N.
FORCE_MIN_N = 20.0
FORCE_MAX_N = 250.0

# Push duration: uniform [0.05, 0.20] s.
DURATION_MIN_S = 0.05
DURATION_MAX_S = 0.20

# Direction: random azimuth in XY plane, small vertical tilt allowed.
VERTICAL_TILT_MAX_RAD = math.radians(15.0)

# Push count: Poisson-distributed, clamped.
POISSON_LAMBDA = 2.5
MIN_PUSHES = 1
MAX_PUSHES = 5

# Timing constraints.
FIRST_PUSH_MIN_S = 1.0
FIRST_PUSH_MAX_S = 2.5
MIN_GAP_BETWEEN_PUSHES_S = 1.5
LAST_PUSH_BEFORE_S = 8.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Push:
    """One discrete push event applied to one body for a fixed duration."""

    body_name: str
    start_time_s: float
    duration_s: float
    force_vec: Tuple[float, float, float]  # world-frame Fx, Fy, Fz in Newtons

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.duration_s

    @property
    def magnitude_n(self) -> float:
        return float(np.linalg.norm(self.force_vec))

    @property
    def impulse_ns(self) -> float:
        """Force integral over duration — scalar measure of push severity."""
        return self.magnitude_n * self.duration_s

    def is_active_at(self, t: float) -> bool:
        return self.start_time_s <= t < self.end_time_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "body_name": self.body_name,
            "start_time_s": round(self.start_time_s, 4),
            "duration_s": round(self.duration_s, 4),
            "force_vec": [round(float(f), 3) for f in self.force_vec],
            "magnitude_n": round(self.magnitude_n, 3),
        }


@dataclass
class DisturbanceProfile:
    """A complete disturbance schedule for one trial, generated from a seed."""

    seed: int
    pushes: List[Push] = field(default_factory=list)
    sim_duration_s: float = 10.0

    @classmethod
    def generate(
        cls,
        seed: int,
        sim_duration_s: float = 10.0,
        available_bodies: Optional[List[str]] = None,
    ) -> "DisturbanceProfile":
        """Create a reproducible DisturbanceProfile from the given seed.

        Args:
            seed: Integer seed. Same seed → same profile.
            sim_duration_s: Total trial duration in seconds.
            available_bodies: Optional list of body names actually present in
                the MuJoCo model. If provided, BODY_WEIGHTS is filtered to
                only include these bodies (and renormalised). If None, all
                ten canonical bodies are used.

        Returns:
            A fully populated DisturbanceProfile.
        """
        rng = np.random.default_rng(seed)

        # Filter body weights to only include bodies present in the model.
        if available_bodies is None:
            body_weights = dict(BODY_WEIGHTS)
        else:
            body_weights = {
                b: w for b, w in BODY_WEIGHTS.items() if b in available_bodies
            }
            if not body_weights:
                raise ValueError(
                    "No bodies from BODY_WEIGHTS are present in available_bodies. "
                    f"Wanted one of {list(BODY_WEIGHTS.keys())}, "
                    f"got {available_bodies}."
                )

        # Renormalise weights after filtering.
        total_weight = sum(body_weights.values())
        body_names = list(body_weights.keys())
        body_probs = np.array(
            [body_weights[b] / total_weight for b in body_names]
        )

        # 1) Sample number of pushes.
        n_pushes = int(
            np.clip(rng.poisson(POISSON_LAMBDA), MIN_PUSHES, MAX_PUSHES)
        )

        # 2) Sample push start times with required gap constraints.
        last_allowed = min(
            LAST_PUSH_BEFORE_S, sim_duration_s - DURATION_MAX_S - 0.1
        )
        start_times = cls._sample_start_times(rng, n_pushes, last_allowed)
        n_pushes = len(start_times)  # may be reduced if timing didn't fit

        # 3) Sample per-push attributes independently.
        pushes: List[Push] = []
        for t_start in start_times:
            body_name = str(rng.choice(body_names, p=body_probs))

            # Magnitude: log-uniform, then scaled by body.
            log_f = rng.uniform(math.log(FORCE_MIN_N), math.log(FORCE_MAX_N))
            f_raw = math.exp(log_f)
            f_scaled = f_raw * BODY_FORCE_SCALE[body_name]

            # Direction: uniform azimuth + small vertical tilt.
            azimuth = rng.uniform(0.0, 2.0 * math.pi)
            elevation = rng.uniform(
                -VERTICAL_TILT_MAX_RAD, VERTICAL_TILT_MAX_RAD
            )
            fx = f_scaled * math.cos(elevation) * math.cos(azimuth)
            fy = f_scaled * math.cos(elevation) * math.sin(azimuth)
            fz = f_scaled * math.sin(elevation)

            duration = float(rng.uniform(DURATION_MIN_S, DURATION_MAX_S))

            pushes.append(
                Push(
                    body_name=body_name,
                    start_time_s=float(t_start),
                    duration_s=duration,
                    force_vec=(float(fx), float(fy), float(fz)),
                )
            )

        return cls(seed=seed, pushes=pushes, sim_duration_s=sim_duration_s)

    @staticmethod
    def _sample_start_times(
        rng: np.random.Generator, n_pushes: int, last_allowed: float
    ) -> List[float]:
        """Sample n_pushes start times with the required gap constraint."""
        times: List[float] = []
        first = rng.uniform(FIRST_PUSH_MIN_S, FIRST_PUSH_MAX_S)
        if first > last_allowed:
            return times
        times.append(float(first))

        for _ in range(n_pushes - 1):
            min_next = times[-1] + MIN_GAP_BETWEEN_PUSHES_S
            if min_next > last_allowed:
                break
            next_t = rng.uniform(min_next, last_allowed)
            times.append(float(next_t))

        return times

    # ---- Runtime API ----

    def active_push_at(self, t: float) -> Optional[Push]:
        """Return the push currently active at time t, or None."""
        for p in self.pushes:
            if p.is_active_at(t):
                return p
        return None

    # ---- Summary metrics ----

    @property
    def n_pushes_scheduled(self) -> int:
        return len(self.pushes)

    @property
    def total_impulse_ns(self) -> float:
        return float(sum(p.impulse_ns for p in self.pushes))

    @property
    def max_instantaneous_force_n(self) -> float:
        if not self.pushes:
            return 0.0
        return float(max(p.magnitude_n for p in self.pushes))

    @property
    def mean_push_magnitude_n(self) -> float:
        if not self.pushes:
            return 0.0
        return float(np.mean([p.magnitude_n for p in self.pushes]))

    def pushes_delivered_by(self, cutoff_time_s: float) -> int:
        return sum(1 for p in self.pushes if p.start_time_s < cutoff_time_s)

    # ---- Serialisation ----

    def to_json(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "sim_duration_s": self.sim_duration_s,
                "pushes": [p.to_dict() for p in self.pushes],
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, s: str) -> "DisturbanceProfile":
        d = json.loads(s)
        pushes = [
            Push(
                body_name=p["body_name"],
                start_time_s=float(p["start_time_s"]),
                duration_s=float(p["duration_s"]),
                force_vec=tuple(float(x) for x in p["force_vec"]),
            )
            for p in d["pushes"]
        ]
        return cls(
            seed=int(d["seed"]),
            pushes=pushes,
            sim_duration_s=float(d["sim_duration_s"]),
        )


# ---------------------------------------------------------------------------
# Standalone inspection
# ---------------------------------------------------------------------------


def _print_profile_summary(
    profile: DisturbanceProfile, verbose: bool = False
) -> None:
    print(f"\n=== Profile seed={profile.seed} ===")
    print(
        f"  n_pushes: {profile.n_pushes_scheduled}, "
        f"total_impulse: {profile.total_impulse_ns:.2f} Ns, "
        f"max_force: {profile.max_instantaneous_force_n:.1f} N, "
        f"mean_force: {profile.mean_push_magnitude_n:.1f} N"
    )
    if verbose:
        for i, p in enumerate(profile.pushes):
            fx, fy, fz = p.force_vec
            print(
                f"  Push {i+1}: t=[{p.start_time_s:5.2f} → {p.end_time_s:5.2f}] s "
                f"body={p.body_name:<30s} "
                f"F=({fx:+7.1f}, {fy:+7.1f}, {fz:+7.1f}) N "
                f"|F|={p.magnitude_n:6.1f} N"
            )


def _run_statistical_summary(n_seeds: int = 1000) -> None:
    """Verify distribution across many seeds."""
    print(f"\n=== Statistical summary over {n_seeds} seeds ===")

    n_pushes_all: List[int] = []
    magnitudes_all: List[float] = []
    durations_all: List[float] = []
    body_counts: Dict[str, int] = {b: 0 for b in BODY_WEIGHTS.keys()}

    for seed in range(n_seeds):
        prof = DisturbanceProfile.generate(seed)
        n_pushes_all.append(prof.n_pushes_scheduled)
        for p in prof.pushes:
            magnitudes_all.append(p.magnitude_n)
            durations_all.append(p.duration_s)
            body_counts[p.body_name] = body_counts.get(p.body_name, 0) + 1

    n_p = np.array(n_pushes_all)
    mags = np.array(magnitudes_all)
    durs = np.array(durations_all)
    total_pushes = int(n_p.sum())

    print(
        f"  n_pushes per trial:   mean={n_p.mean():.2f}, "
        f"std={n_p.std():.2f}, "
        f"range=[{n_p.min()}, {n_p.max()}]"
    )
    print(
        f"  magnitude [N]:        mean={mags.mean():.1f}, "
        f"median={np.median(mags):.1f}, "
        f"range=[{mags.min():.1f}, {mags.max():.1f}]"
    )
    print(
        f"  duration [s]:         mean={durs.mean():.3f}, "
        f"range=[{durs.min():.3f}, {durs.max():.3f}]"
    )
    print("  body distribution (actual vs expected):")

    # Group by region for readability
    upper_trunk = ["pelvis", "torso_link"]
    upper_arms  = ["left_shoulder_roll_link", "right_shoulder_roll_link",
                   "left_elbow_link", "right_elbow_link"]
    lower_thigh = ["left_hip_pitch_link", "right_hip_pitch_link"]
    lower_shin  = ["left_knee_link", "right_knee_link"]

    for group_name, group in [
        ("Upper trunk", upper_trunk),
        ("Upper arms ", upper_arms),
        ("Lower thigh", lower_thigh),
        ("Lower shin ", lower_shin),
    ]:
        print(f"  [{group_name}]")
        for body in group:
            count = body_counts.get(body, 0)
            actual = count / total_pushes if total_pushes > 0 else 0.0
            expected = BODY_WEIGHTS[body]
            print(
                f"    {body:<32s}  actual={actual:.3f}  "
                f"expected={expected:.3f}"
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,42",
        help="Comma-separated integer seeds to inspect. Default: 0,1,2,42",
    )
    p.add_argument(
        "--sim-duration",
        type=float,
        default=10.0,
        help="Trial duration in seconds. Default: 10.0",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print every push in each profile.",
    )
    p.add_argument(
        "--stats",
        type=int,
        default=1000,
        help=(
            "Run statistical summary over this many seeds. "
            "Set to 0 to skip. Default: 1000"
        ),
    )
    return p.parse_args()


def _main() -> None:
    args = _parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(
        f"Inspecting {len(seeds)} profile(s) with "
        f"sim_duration={args.sim_duration} s"
    )
    for seed in seeds:
        profile = DisturbanceProfile.generate(
            seed, sim_duration_s=args.sim_duration
        )
        _print_profile_summary(profile, verbose=args.verbose)

    if args.stats > 0:
        _run_statistical_summary(n_seeds=args.stats)
        import matplotlib.pyplot as plt

        bodies = ["pelvis", "torso", "L shoulder", "R shoulder", "L elbow", "R elbow", "L hip", "R hip", "L knee", "R knee"]
        weights = [0.20, 0.20, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.06, 0.06]
        colors = ["#2196F3", "#2196F3",
                  "#FF9800", "#FF9800", "#FF9800", "#FF9800",
                  "#4CAF50", "#4CAF50", "#4CAF50", "#4CAF50"]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(bodies, weights, color=colors)
        ax.set_ylabel("Sampling Probability")
        ax.set_xlabel("Body Site")
        ax.set_title("Part 1b Push Body Site Distribution (1000 seeds)")
        ax.set_ylim(0, 0.25)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("figure3_2_right_panel.png", dpi=300, bbox_inches="tight")
        plt.show()
        print("Saved figure3_2_right_panel.png")

    # Round-trip serialisation test
    test_profile = DisturbanceProfile.generate(
        seeds[0], sim_duration_s=args.sim_duration
    )
    json_str = test_profile.to_json()
    recovered = DisturbanceProfile.from_json(json_str)
    assert recovered.seed == test_profile.seed
    assert len(recovered.pushes) == len(test_profile.pushes)
    print(
        f"\n=== JSON round-trip OK "
        f"(seed={seeds[0]}, {len(json_str)} chars) ==="
    )


if __name__ == "__main__":
    _main()
