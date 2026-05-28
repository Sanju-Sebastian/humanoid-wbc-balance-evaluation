import argparse
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ── CONTROLLER REGISTRY ───────────────────────────────────────────────────────
# Order determines plot order throughout all figures.
# Base controllers first, then MLP-augmented pairs, then distillation.

CTRL_ORDER = [
    "weighted_qp",
    "hierarchical_qp",
    "passivity_based",
    "lqr_lipm",
    "passivity_mlp",       # v1/v2 null result (historical)
    "passivity_mlp_v3",    # v3 POSITIVE — passivity base
    "wqp_mlp_v3",          # v3 POSITIVE — WQP base   (NEW)
    "hqp_mlp_v3",          # v3 POSITIVE — HQP base   (NEW)
    "lqr_mlp_v3",          # v3 POSITIVE — LQR base   (NEW)
    "wqp_distilled",       # cross-controller distillation
]

# These four are the base controllers used for all Chapter 4 figures
CTRL_ORDER_MAIN = ["weighted_qp", "hierarchical_qp", "passivity_based", "lqr_lipm"]

# Paired layout used in MLP cross-controller figures:
#   each tuple is (base_ctrl, mlp_ctrl, short_label_for_pair)
MLP_PAIRS = [
    ("passivity_based", "passivity_mlp_v3", "Passivity"),
    ("weighted_qp",     "wqp_mlp_v3",       "WQP"),
    ("hierarchical_qp", "hqp_mlp_v3",       "HQP"),
    ("lqr_lipm",        "lqr_mlp_v3",       "LQR"),
]

CTRL_LABELS = {
    "weighted_qp":      "Weighted QP",
    "hierarchical_qp":  "Hierarchical QP",
    "passivity_based":  "Passivity-Based",
    "lqr_lipm":         "LQR-on-LIPM",
    "passivity_mlp":    "Passivity+MLP (v1/v2)",
    "passivity_mlp_v3": "Passivity+MLP v3",
    "wqp_mlp_v3":       "WQP+MLP v3",
    "hqp_mlp_v3":       "HQP+MLP v3",
    "lqr_mlp_v3":       "LQR+MLP v3",
    "wqp_distilled":    "WQP+Distillation",
}

CTRL_COLORS = {
    "weighted_qp":      "#2166ac",
    "hierarchical_qp":  "#d6604d",
    "passivity_based":  "#1a9850",
    "lqr_lipm":         "#762a83",
    "passivity_mlp":    "#f46d43",
    "passivity_mlp_v3": "#74c476",   # lighter green — paired with passivity
    "wqp_mlp_v3":       "#6baed6",   # lighter blue  — paired with WQP
    "hqp_mlp_v3":       "#fc8d59",   # lighter red   — paired with HQP
    "lqr_mlp_v3":       "#b2abd2",   # lighter purple — paired with LQR
    "wqp_distilled":    "#4d4d4d",
}

CTRL_MARKERS = {
    "weighted_qp":      "o",
    "hierarchical_qp":  "s",
    "passivity_based":  "^",
    "lqr_lipm":         "D",
    "passivity_mlp":    "P",
    "passivity_mlp_v3": "*",
    "wqp_mlp_v3":       "*",
    "hqp_mlp_v3":       "*",
    "lqr_mlp_v3":       "*",
    "wqp_distilled":    "X",
}

OUTCOME_ORDER  = ["Stable", "Fell", "Neither"]
OUTCOME_COLORS = {"Stable": "#1a9850", "Fell": "#d73027", "Neither": "#999999"}

FORCE_BINS = [
    (-np.inf,  75.0,  "<75N",      37.5),
    ( 75.0,   125.0,  "75–125N",  100.0),
    (125.0,   175.0,  "125–175N", 150.0),
    (175.0,   250.0,  "175–250N", 212.5),
]
FORCE_BIN_LABELS = [b[2] for b in FORCE_BINS]
FORCE_BIN_MIDS   = [b[3] for b in FORCE_BINS]


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class InputCSVs:
    weighted_qp:      str
    hierarchical_qp:  str
    passivity_based:  str
    lqr_lipm:         str
    passivity_mlp:    str = ""
    passivity_mlp_v3: str = ""
    wqp_mlp_v3:       str = ""   # NEW
    hqp_mlp_v3:       str = ""   # NEW
    lqr_mlp_v3:       str = ""   # NEW
    wqp_distilled:    str = ""


# ── STYLE & UTILITIES ─────────────────────────────────────────────────────────

def _percent(x, pos):
    return f"{100.0 * x:.0f}%"

def set_thesis_style():
    sns.set_theme(style="whitegrid", font_scale=1.15)
    matplotlib.rcParams.update({
        "font.family":       "serif",
        "font.size":         12,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

def ensure_output_dir(script_dir):
    out_dir = os.path.join(script_dir, "analysis_outputs_1b")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_fig(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_part1b_csv(path, controller_override=None):
    cols = [
        "controller_name", "disturbance_seed", "is_stable", "fall_time",
        "max_com_disp", "rms_com_disp_m", "tau_rms_total",
        "qp_solve_time_mean_ms", "survival_time_s",
        "max_instantaneous_force_N", "n_pushes_scheduled",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    if controller_override:
        df["controller_name"] = str(controller_override)
    df["controller_name"] = df["controller_name"].astype(str)
    df = df[df["controller_name"].isin(CTRL_ORDER)].copy()
    df = df[df["disturbance_seed"].astype(float) >= 0].copy()
    for col, dtype in [
        ("disturbance_seed",          int),
        ("is_stable",                 int),
        ("fall_time",                 float),
        ("max_instantaneous_force_N", float),
        ("max_com_disp",              float),
        ("rms_com_disp_m",            float),
        ("tau_rms_total",             float),
        ("qp_solve_time_mean_ms",     float),
        ("survival_time_s",           float),
        ("n_pushes_scheduled",        int),
    ]:
        df[col] = df[col].astype(dtype)
    return df

def add_outcome_column(df):
    out = df.copy()
    out["outcome"] = "Neither"
    out.loc[out["is_stable"] == 1, "outcome"] = "Stable"
    out.loc[(out["is_stable"] == 0) & (out["fall_time"] > 0.0), "outcome"] = "Fell"
    out["outcome"] = pd.Categorical(out["outcome"], categories=OUTCOME_ORDER, ordered=True)
    return out

def add_force_bins(df):
    out = df.copy()
    force = out["max_instantaneous_force_N"].to_numpy(dtype=float)
    bin_label = np.full(len(out), None, dtype=object)
    bin_mid   = np.full(len(out), np.nan, dtype=float)
    for low, high, lab, mid in FORCE_BINS:
        mask = (force >= float(low)) & (force < float(high))
        bin_label[mask] = lab
        bin_mid[mask]   = float(mid)
    out["force_bin"]     = bin_label
    out["force_bin_mid"] = bin_mid
    out = out[out["force_bin"].notna()].copy()
    out["force_bin"] = pd.Categorical(out["force_bin"], categories=FORCE_BIN_LABELS, ordered=True)
    return out

def load_all_controllers(csvs: InputCSVs):
    dfs = [
        load_part1b_csv(csvs.weighted_qp,     "weighted_qp"),
        load_part1b_csv(csvs.hierarchical_qp, "hierarchical_qp"),
        load_part1b_csv(csvs.passivity_based,  "passivity_based"),
        load_part1b_csv(csvs.lqr_lipm,         "lqr_lipm"),
    ]
    # Optional CSVs — load each if a path was provided
    optional = [
        (csvs.passivity_mlp,    "passivity_mlp"),
        (csvs.passivity_mlp_v3, "passivity_mlp_v3"),
        (csvs.wqp_mlp_v3,       "wqp_mlp_v3"),
        (csvs.hqp_mlp_v3,       "hqp_mlp_v3"),
        (csvs.lqr_mlp_v3,       "lqr_mlp_v3"),
        (csvs.wqp_distilled,    "wqp_distilled"),
    ]
    for path, ctrl in optional:
        if path:
            try:
                tmp = pd.read_csv(path)
                tmp["controller_name"] = ctrl
                dfs.append(tmp)
                print(f"[INFO] Loaded {ctrl} from {path}")
            except Exception as e:
                print(f"[WARN] Could not load {ctrl}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    df = add_outcome_column(df)
    df = add_force_bins(df)
    df["controller_name"] = pd.Categorical(
        df["controller_name"], categories=CTRL_ORDER, ordered=True
    )
    return df


# ── STATISTICS HELPERS ────────────────────────────────────────────────────────

def _wilson_ci_manual(k, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom  = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half   = (z / denom) * np.sqrt((phat * (1 - phat) / n) + (z * z) / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))

def wilson_ci(k, n):
    try:
        from statsmodels.stats.proportion import proportion_confint
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        return (float(lo), float(hi))
    except:
        return _wilson_ci_manual(k, n)

def compute_bin_stability(df, ctrl_list=None):
    if ctrl_list is None:
        ctrl_list = CTRL_ORDER
    rows = []
    for ctrl in ctrl_list:
        d0 = df[df["controller_name"] == ctrl]
        for b in FORCE_BIN_LABELS:
            sub = d0[d0["force_bin"] == b]
            n   = int(len(sub))
            k   = int(sub["is_stable"].sum()) if n else 0
            rate = float(k / n) if n else np.nan
            lo, hi = wilson_ci(k, n)
            rows.append({
                "controller_name":    ctrl,
                "force_bin":          b,
                "n_trials":           n,
                "n_stable":           k,
                "stability_rate":     rate,
                "wilson_ci_low":      lo,
                "wilson_ci_high":     hi,
                "mean_max_com_disp_m": float(sub["max_com_disp"].mean()) if n else np.nan,
                "mean_rms_com_disp_m": float(sub["rms_com_disp_m"].mean()) if n else np.nan,
            })
    out = pd.DataFrame(rows)
    out["controller_name"] = pd.Categorical(
        out["controller_name"], categories=CTRL_ORDER, ordered=True
    )
    out["force_bin"] = pd.Categorical(
        out["force_bin"], categories=FORCE_BIN_LABELS, ordered=True
    )
    return out.sort_values(["force_bin", "controller_name"]).reset_index(drop=True)

def overall_stability(df, ctrl):
    """Returns (n_stable, n_total, rate) for a single controller."""
    sub = df[df["controller_name"] == ctrl]
    n = int(len(sub))
    k = int(sub["is_stable"].sum()) if n else 0
    return k, n, float(k / n) if n else np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 4 FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_stability_rate_by_force_bin(df, out_dir):
    stats = compute_bin_stability(df, CTRL_ORDER_MAIN)
    fig, ax = plt.subplots(figsize=(10, 6))
    x     = np.arange(len(FORCE_BIN_LABELS), dtype=float)
    width = 0.20
    for i, ctrl in enumerate(CTRL_ORDER_MAIN):
        sub = stats[stats["controller_name"] == ctrl].set_index("force_bin").reindex(FORCE_BIN_LABELS)
        y   = sub["stability_rate"].to_numpy(dtype=float)
        lo  = sub["wilson_ci_low"].to_numpy(dtype=float)
        hi  = sub["wilson_ci_high"].to_numpy(dtype=float)
        ax.bar(
            x + (i - 1.5) * width, y, width=width,
            color=CTRL_COLORS[ctrl], label=CTRL_LABELS[ctrl],
            yerr=np.vstack([y - lo, hi - y]), capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(FORCE_BIN_LABELS)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_title("Stability Rate by Force Bin (Part 1b)")
    ax.set_xlabel("Max Instantaneous Force Bin")
    ax.set_ylabel("Stability Rate (Wilson 95% CI)")
    ax.legend(ncols=2, frameon=True)
    save_fig(fig, out_dir, "fig1_stability_rate_by_force_bin")


def _logistic_decreasing(x, k, x0):
    return 1.0 / (1.0 + np.exp(k * (np.asarray(x, dtype=float) - x0)))

def _fit_logistic(df_ctrl):
    x = df_ctrl["max_instantaneous_force_N"].to_numpy(dtype=float)
    y = df_ctrl["is_stable"].to_numpy(dtype=float)
    params, _ = curve_fit(
        _logistic_decreasing, x, y,
        p0=[0.05, np.median(x)],
        bounds=([1e-4, 0.0], [2.0, 250.0]),
        maxfev=20000,
    )
    return float(params[0]), float(params[1])

def fig2_stability_sigmoid_curves(df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_stats = compute_bin_stability(df, CTRL_ORDER_MAIN)
    xs = np.linspace(0, 250, 400)
    for ctrl in CTRL_ORDER_MAIN:
        d0 = df[df["controller_name"] == ctrl]
        k, x0 = _fit_logistic(d0)
        ax.plot(xs, _logistic_decreasing(xs, k, x0),
                color=CTRL_COLORS[ctrl], linewidth=2, label=CTRL_LABELS[ctrl])
        bsub = (bin_stats[bin_stats["controller_name"] == ctrl]
                .set_index("force_bin").reindex(FORCE_BIN_LABELS))
        ax.scatter(FORCE_BIN_MIDS, bsub["stability_rate"].to_numpy(dtype=float),
                   color=CTRL_COLORS[ctrl], marker=CTRL_MARKERS[ctrl], s=55, zorder=5)
        ax.axvline(x0, color=CTRL_COLORS[ctrl], linestyle="--", alpha=0.35, linewidth=1)
        ax.text(x0, 0.52, f"{CTRL_LABELS[ctrl]} 50% @ {x0:.1f}N",
                rotation=90, ha="left", va="bottom", fontsize=9, color=CTRL_COLORS[ctrl])
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_title("Stability Sigmoid Curves (Logistic Fit) — Part 1b")
    ax.set_xlabel("Max Instantaneous Force (N)")
    ax.set_ylabel("Stability Probability")
    ax.legend(frameon=True)
    save_fig(fig, out_dir, "fig2_stability_sigmoid_curves")


def _seed_merge(df, ctrl_a, ctrl_b):
    """Merge two controllers on disturbance_seed for per-seed comparison."""
    a = (df[df["controller_name"] == ctrl_a]
         [["disturbance_seed", "max_com_disp", "is_stable", "fall_time"]].copy()
         .rename(columns={"max_com_disp": f"{ctrl_a}_com",
                          "is_stable":   f"{ctrl_a}_stable",
                          "fall_time":   f"{ctrl_a}_fall"}))
    b = (df[df["controller_name"] == ctrl_b]
         [["disturbance_seed", "max_com_disp", "is_stable", "fall_time"]].copy()
         .rename(columns={"max_com_disp": f"{ctrl_b}_com",
                          "is_stable":   f"{ctrl_b}_stable",
                          "fall_time":   f"{ctrl_b}_fall"}))
    m = pd.merge(a, b, on="disturbance_seed", how="inner").sort_values("disturbance_seed").reset_index(drop=True)
    m["abs_diff_m"] = (m[f"{ctrl_a}_com"] - m[f"{ctrl_b}_com"]).abs()
    m["outcome_match"] = (m[f"{ctrl_a}_stable"].astype(int) == m[f"{ctrl_b}_stable"].astype(int)).astype(int)
    return m

def fig3_wqp_vs_hqp_per_seed_equivalence_scatter(df, out_dir):
    m = _seed_merge(df, "weighted_qp", "hierarchical_qp")

    def classify(row):
        ws = int(row["weighted_qp_stable"]) == 1
        hs = int(row["hierarchical_qp_stable"]) == 1
        wf = (int(row["weighted_qp_stable"]) == 0) and (float(row["weighted_qp_fall"]) > 0)
        hf = (int(row["hierarchical_qp_stable"]) == 0) and (float(row["hierarchical_qp_fall"]) > 0)
        if ws and hs:   return "both_stable"
        if wf and hf:   return "both_fell"
        return "neither"

    m["pair_class"] = m.apply(classify, axis=1)
    color_map = {"both_stable": "#1a9850", "both_fell": "#d73027", "neither": "#666666"}
    fig, ax = plt.subplots(figsize=(7, 7))
    for key, label in [("both_stable", "both stable"), ("both_fell", "both fell"), ("neither", "neither")]:
        sub = m[m["pair_class"] == key]
        ax.scatter(sub["weighted_qp_com"], sub["hierarchical_qp_com"],
                   s=28, alpha=0.75, color=color_map[key], label=label)
    lim = float(max(m["weighted_qp_com"].max(), m["hierarchical_qp_com"].max()) * 1.05)
    ax.plot([0, lim], [0, lim], linestyle="--", color="black", linewidth=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.text(0.02, 0.98,
            f"mean CoM diff = {float(m['abs_diff_m'].mean() * 1000):.2f}mm\n"
            f"max diff = {float(m['abs_diff_m'].max() * 1000):.2f}mm\n"
            f"outcome match = {int(m['outcome_match'].sum())}/{len(m)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    ax.set_title("WQP vs HQP Per-Seed Equivalence (Part 1b)")
    ax.set_xlabel("WQP max_com_disp (m)")
    ax.set_ylabel("HQP max_com_disp (m)")
    ax.legend(frameon=True, loc="lower right")
    save_fig(fig, out_dir, "fig3_wqp_vs_hqp_per_seed_equivalence_scatter")
    return m


def fig4_three_category_outcome_stacked_bar(df, out_dir):
    plot_order = [c for c in CTRL_ORDER if c in df["controller_name"].unique()]
    counts = (df.groupby(["controller_name", "outcome"]).size()
               .reindex(pd.MultiIndex.from_product([plot_order, OUTCOME_ORDER],
                                                    names=["controller_name", "outcome"]))
               .fillna(0).reset_index(name="n"))
    totals = counts.groupby("controller_name")["n"].sum().to_dict()
    pivot = counts.pivot(index="controller_name", columns="outcome", values="n").reindex(plot_order).fillna(0)
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.7 * len(plot_order))))
    left = np.zeros(len(plot_order), dtype=float)
    for outcome in OUTCOME_ORDER:
        vals = pivot[outcome].to_numpy(dtype=float)
        frac = np.array([vals[i] / max(1.0, float(totals[plot_order[i]]))
                         for i in range(len(plot_order))], dtype=float)
        ax.barh(np.arange(len(plot_order)), frac, left=left,
                color=OUTCOME_COLORS[outcome], label=outcome, height=0.6)
        for i in range(len(plot_order)):
            if frac[i] > 0.06:
                ax.text(left[i] + frac[i] / 2, i, f"{100 * frac[i]:.0f}%",
                        ha="center", va="center", color="white", fontsize=10)
        left += frac
    ax.set_yticks(np.arange(len(plot_order)))
    ax.set_yticklabels([CTRL_LABELS[c] for c in plot_order])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_title("Three-Category Outcomes Across Controllers (Part 1b)")
    ax.set_xlabel("Percentage of Trials")
    ax.legend(frameon=True, ncols=3, loc="lower right")
    save_fig(fig, out_dir, "fig4_outcome_stacked_bar")


def fig5_com_displacement_boxplots_stable_only(df, out_dir):
    plot_order = [c for c in CTRL_ORDER_MAIN if c in df["controller_name"].unique()]
    stable = df[(df["is_stable"] == 1) & (df["controller_name"].isin(plot_order))].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=stable, x="controller_name", y="max_com_disp",
                order=plot_order, palette=CTRL_COLORS, ax=ax, showfliers=False)
    sns.stripplot(data=stable, x="controller_name", y="max_com_disp",
                  order=plot_order, palette=CTRL_COLORS, ax=ax, jitter=True, alpha=0.55, size=4)
    ax.set_xticklabels([CTRL_LABELS[c] for c in plot_order], rotation=15, ha="right")
    ax.axhline(0.02, linestyle="--", color="black", linewidth=1)
    ax.text(0.98, 0.02, "2cm recovery threshold", ha="right", va="bottom",
            transform=ax.get_yaxis_transform())
    means = stable.groupby("controller_name")["max_com_disp"].mean().to_dict()
    try:
        pas = float(means.get("passivity_based", np.nan))
        ref = float(np.nanmean([means.get("weighted_qp", np.nan), means.get("hierarchical_qp", np.nan)]))
        if np.isfinite(pas) and np.isfinite(ref) and ref > 0:
            ax.text(0.02, 0.95,
                    f"Passivity mean = {pas:.4f}m vs WQP/HQP mean = {ref:.4f}m ({100 * (ref - pas) / ref:.0f}% lower)",
                    transform=ax.transAxes, ha="left", va="top", fontsize=10)
    except:
        pass
    ax.set_title("CoM Displacement Boxplots (Stable Trials Only) — Part 1b")
    ax.set_xlabel("")
    ax.set_ylabel("max_com_disp (m)")
    save_fig(fig, out_dir, "fig5_com_disp_boxplots_stable_only")


def fig6_rms_com_displacement_by_force_bin(df, out_dir):
    agg = (df[df["controller_name"].isin(CTRL_ORDER_MAIN)]
           .groupby(["controller_name", "force_bin"])["rms_com_disp_m"]
           .agg(["mean", "std", "count"]).reset_index())
    agg["sem"] = agg["std"] / np.sqrt(np.maximum(1, agg["count"]))
    fig, ax = plt.subplots(figsize=(10, 6))
    for ctrl in CTRL_ORDER_MAIN:
        sub = agg[agg["controller_name"] == ctrl].set_index("force_bin").reindex(FORCE_BIN_LABELS)
        y    = sub["mean"].to_numpy(dtype=float)
        ysem = sub["sem"].to_numpy(dtype=float)
        ax.plot(FORCE_BIN_MIDS, y, marker=CTRL_MARKERS[ctrl],
                color=CTRL_COLORS[ctrl], linewidth=2, label=CTRL_LABELS[ctrl])
        ax.fill_between(FORCE_BIN_MIDS, y - ysem, y + ysem, color=CTRL_COLORS[ctrl], alpha=0.15)
    ax.set_title("RMS CoM Displacement by Force Bin (Part 1b)")
    ax.set_xlabel("Force Bin"); ax.set_ylabel("rms_com_disp_m (m)")
    ax.set_xticks(FORCE_BIN_MIDS); ax.set_xticklabels(FORCE_BIN_LABELS)
    ax.legend(frameon=True, ncols=2)
    save_fig(fig, out_dir, "fig6_rms_com_disp_by_force_bin")


def fig7_torque_cost_comparison(df, out_dir):
    plot_order = CTRL_ORDER_MAIN
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ts = (df[df["controller_name"].isin(plot_order)]
          .groupby("controller_name")["tau_rms_total"]
          .agg(["mean", "std"]).reindex(plot_order))
    x = np.arange(len(plot_order), dtype=float)
    axes[0].bar(x, ts["mean"].to_numpy(dtype=float),
                yerr=ts["std"].to_numpy(dtype=float), capsize=3,
                color=[CTRL_COLORS[c] for c in plot_order])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([CTRL_LABELS[c] for c in plot_order], rotation=15, ha="right")
    axes[0].set_title("Mean tau_rms_total (All Trials)")
    axes[0].set_ylabel("tau_rms_total (Nm)")

    ss = (df[df["controller_name"].isin(plot_order)]
          .groupby("controller_name")["qp_solve_time_mean_ms"]
          .agg(["mean", "std"]).reindex(plot_order))
    ms = ss["mean"].to_numpy(dtype=float).copy()
    for i, c in enumerate(plot_order):
        if c in ("passivity_based", "lqr_lipm"):
            ms[i] = 0.0
    axes[1].bar(x, ms, yerr=ss["std"].to_numpy(dtype=float), capsize=3,
                color=[CTRL_COLORS[c] for c in plot_order])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([CTRL_LABELS[c] for c in plot_order], rotation=15, ha="right")
    axes[1].set_title("Mean QP Solve Time per Step")
    axes[1].set_ylabel("qp_solve_time_mean_ms (ms)")
    for i, c in enumerate(plot_order):
        if c in ("passivity_based", "lqr_lipm"):
            axes[1].text(i, 0.02, "No QP solver", ha="center", va="bottom", fontsize=10)
    try:
        wqp = float(ss.loc["weighted_qp", "mean"])
        hqp = float(ss.loc["hierarchical_qp", "mean"])
        ratio = hqp / wqp if (np.isfinite(wqp) and wqp > 0 and np.isfinite(hqp)) else np.nan
        if np.isfinite(ratio):
            y0 = float(max(ms[0], ms[1]) * 1.10 + 0.01)
            axes[1].plot([0, 0, 1, 1], [y0 - 0.01, y0, y0, y0 - 0.01], color="black", linewidth=1)
            axes[1].text(0.5, y0 + 0.01, f"HQP = {ratio:.2f}× WQP", ha="center", va="bottom", fontsize=10)
    except:
        pass
    fig.suptitle("Torque Cost vs Compute Cost (Part 1b)")
    save_fig(fig, out_dir, "fig7_torque_cost_comparison")


def fig8_survival_time_distribution(df, out_dir):
    plot_order = CTRL_ORDER_MAIN
    d = df[df["controller_name"].isin(plot_order)].copy()
    d["fell_flag"] = np.where(d["fall_time"] > 0.0, "Fell", "Did not fall")
    d["fell_flag"] = pd.Categorical(d["fell_flag"],
                                    categories=["Did not fall", "Fell"], ordered=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        sns.violinplot(data=d, x="controller_name", y="survival_time_s",
                       hue="fell_flag", order=plot_order,
                       hue_order=["Did not fall", "Fell"],
                       palette={"Did not fall": "#4d4d4d", "Fell": "#d73027"},
                       split=True, inner="quartile", cut=0, ax=ax)
    except:
        sns.violinplot(data=d, x="controller_name", y="survival_time_s",
                       hue="fell_flag", order=plot_order,
                       hue_order=["Did not fall", "Fell"],
                       palette={"Did not fall": "#4d4d4d", "Fell": "#d73027"},
                       dodge=True, inner="quartile", cut=0, ax=ax)
    means = d.groupby("controller_name")["survival_time_s"].mean().to_dict()
    try:
        ax.text(0.02, 0.98,
                "Means (s): " + ", ".join(
                    [f"{CTRL_LABELS[c]} {float(means.get(c, np.nan)):.2f}" for c in plot_order]),
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    except:
        pass
    ax.set_xticklabels([CTRL_LABELS[c] for c in plot_order], rotation=15, ha="right")
    ax.set_title("Survival Time Distribution (Part 1b)")
    ax.set_xlabel(""); ax.set_ylabel("survival_time_s (s)")
    ax.legend(title="", frameon=True)
    save_fig(fig, out_dir, "fig8_survival_time_distribution")


def fig9_neither_category_deep_dive(df, out_dir):
    ndf = df[(df["outcome"] == "Neither") & (df["controller_name"].isin(CTRL_ORDER_MAIN))].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for ctrl in CTRL_ORDER_MAIN:
        sub = ndf[ndf["controller_name"] == ctrl]
        ax.scatter(sub["max_instantaneous_force_N"], sub["max_com_disp"],
                   color=CTRL_COLORS[ctrl], marker=CTRL_MARKERS[ctrl],
                   s=45, alpha=0.8, label=CTRL_LABELS[ctrl])
    ax.axhline(0.02, linestyle="--", color="black", linewidth=1)
    ax.axhline(0.10, linestyle="--", color="black", linewidth=1)
    counts = ndf.groupby("controller_name").size().to_dict()
    w = int(counts.get("weighted_qp", 0))
    h = int(counts.get("hierarchical_qp", 0))
    p = int(counts.get("passivity_based", 0))
    l = int(counts.get("lqr_lipm", 0))
    ax.text(0.02, 0.98,
            f'WQP/HQP = {w + h} "Neither" seeds, PAS/LQR = {p + l} "Neither" seeds',
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    ax.set_xlim(0, 250)
    ax.set_title('"Neither" Category Deep Dive (Part 1b)')
    ax.set_xlabel("max_instantaneous_force_N (N)")
    ax.set_ylabel("max_com_disp (m)")
    ax.legend(frameon=True, ncols=2)
    save_fig(fig, out_dir, "fig9_neither_deep_dive")


def fig10_push_count_vs_stability(df, out_dir):
    d = df[df["controller_name"].isin(CTRL_ORDER_MAIN)].copy()
    d = d[(d["n_pushes_scheduled"] >= 1) & (d["n_pushes_scheduled"] <= 5)].copy()
    counts = d.groupby("n_pushes_scheduled").size()
    valid_push_counts = sorted([int(k) for k, n in counts.items() if int(n) >= 5]) or [1, 2, 3, 4, 5]
    rows = []
    for pc in valid_push_counts:
        for ctrl in CTRL_ORDER_MAIN:
            sub = d[(d["n_pushes_scheduled"] == pc) & (d["controller_name"] == ctrl)]
            n = int(len(sub))
            if n < 5:
                continue
            rows.append({
                "n_pushes_scheduled": pc,
                "controller_name":    ctrl,
                "stability_rate":     float(int(sub["is_stable"].sum()) / n),
                "n":                  n,
            })
    s = pd.DataFrame(rows)
    if len(s) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    x     = np.arange(len(valid_push_counts), dtype=float)
    width = 0.20
    for i, ctrl in enumerate(CTRL_ORDER_MAIN):
        ys = []
        for pc in valid_push_counts:
            row = s[(s["controller_name"] == ctrl) & (s["n_pushes_scheduled"] == pc)]
            ys.append(float(row["stability_rate"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (i - 1.5) * width, ys, width=width,
               color=CTRL_COLORS[ctrl], label=CTRL_LABELS[ctrl])
    ax.set_xticks(x)
    ax.set_xticklabels([str(pc) for pc in valid_push_counts])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_title("Push Count vs Stability (Part 1b)")
    ax.set_xlabel("Number of Pushes Scheduled")
    ax.set_ylabel("Stability Rate")
    ax.legend(frameon=True, ncols=2)
    save_fig(fig, out_dir, "fig10_push_count_vs_stability")


# ── TABLES ────────────────────────────────────────────────────────────────────

def table1_stats_part1b_summary(df, out_dir):
    ctrl_present = [c for c in CTRL_ORDER if c in df["controller_name"].unique()]
    rows = []
    for ctrl in ctrl_present:
        sub     = df[df["controller_name"] == ctrl]
        n_total = int(len(sub))
        n_stable  = int((sub["is_stable"] == 1).sum())
        n_fell    = int(((sub["is_stable"] == 0) & (sub["fall_time"] > 0.0)).sum())
        n_neither = n_total - n_stable - n_fell
        rows.append({
            "controller":          CTRL_LABELS[ctrl],
            "n_stable":            n_stable,
            "n_fell":              n_fell,
            "n_neither":           n_neither,
            "stability_pct":       float(100.0 * n_stable / n_total) if n_total else np.nan,
            "mean_max_com_disp_m": float(sub["max_com_disp"].mean()) if n_total else np.nan,
            "std_max_com_disp_m":  float(sub["max_com_disp"].std()) if n_total else np.nan,
            "mean_rms_com_disp_m": float(sub["rms_com_disp_m"].mean()) if n_total else np.nan,
            "mean_tau_rms_Nm":     float(sub["tau_rms_total"].mean()) if n_total else np.nan,
            "mean_survival_s":     float(sub["survival_time_s"].mean()) if n_total else np.nan,
            "mean_qp_solve_ms":    float(sub["qp_solve_time_mean_ms"].mean()) if n_total else np.nan,
        })
    base = pd.DataFrame(rows)
    base.to_csv(os.path.join(out_dir, "stats_part1b_summary.csv"), index=False)
    return base

def table2_stats_part1b_force_bins(df, out_dir):
    ctrl_present = [c for c in CTRL_ORDER_MAIN if c in df["controller_name"].unique()]
    stats = compute_bin_stability(df, ctrl_present)
    out = stats.rename(columns={"controller_name": "controller"})[
        ["controller", "force_bin", "n_trials", "n_stable",
         "stability_rate", "wilson_ci_low", "wilson_ci_high",
         "mean_max_com_disp_m", "mean_rms_com_disp_m"]
    ].copy()
    out["controller"] = out["controller"].map(lambda c: CTRL_LABELS.get(str(c), str(c)))
    out.to_csv(os.path.join(out_dir, "stats_part1b_force_bins.csv"), index=False)
    return out

def table3_stats_part1b_wqp_hqp_equivalence(df, out_dir):
    m = _seed_merge(df, "weighted_qp", "hierarchical_qp")
    out = pd.DataFrame({
        "seed":               m["disturbance_seed"].astype(int),
        "wqp_max_com_disp":   m["weighted_qp_com"].astype(float),
        "hqp_max_com_disp":   m["hierarchical_qp_com"].astype(float),
        "abs_diff_m":         m["abs_diff_m"].astype(float),
        "wqp_is_stable":      m["weighted_qp_stable"].astype(int),
        "hqp_is_stable":      m["hierarchical_qp_stable"].astype(int),
        "outcome_match":      m["outcome_match"].astype(int),
    })
    out.to_csv(os.path.join(out_dir, "stats_part1b_wqp_hqp_equivalence.csv"), index=False)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 5 FIGURES — MLP v3 CROSS-CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

def fig_mlp_progression_table(df, out_dir):
    """
    Updated progression table: 7 rows.
    v1 (null), v2 (null), then v3 on all 4 controllers, then distillation.
    """
    # Hard-coded from confirmed experimental results (v16 memory)
    versions = [
        ("v1",          "Passivity", "42",  "Zero (LIPM loss)", "No",          "109/200 (54.5%)", "-0.5%", "#d73027"),
        ("v2",          "Passivity", "46",  "Zero (LIPM loss)", "No",          "109/200 (54.5%)", "-0.5%", "#d73027"),
        ("v3",          "Passivity", "260", "Pitch-rate BC",    "Yes (5 fr.)", "125/200 (62.5%)", "+7.5pp","#1a9850"),
        ("v3",          "WQP",       "260", "Pitch-rate BC",    "Yes (5 fr.)", "114/200 (57.0%)", "+5.5pp","#1a9850"),
        ("v3",          "HQP",       "260", "Pitch-rate BC",    "Yes (5 fr.)", "114/200 (57.0%)", "+5.5pp","#1a9850"),
        ("v3",          "LQR",       "260", "Pitch-rate BC",    "Yes (5 fr.)", "120/200 (60.0%)", "+2.5pp","#1a9850"),
        ("Distillation","WQP→Pass.", "260", "Passivity torques","Yes (5 fr.)", "125/200 (62.5%)", "+11.0pp","#2166ac"),
    ]
    col_labels = ["Version", "Base", "Input dim", "Target", "History", "Stable/200", "vs Baseline"]
    fig, ax = plt.subplots(figsize=(16, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=[[v[0], v[1], v[2], v[3], v[4], v[5], v[6]] for v in versions],
        colLabels=col_labels,
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#333333")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i, v in enumerate(versions):
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(matplotlib.colors.to_rgba(v[7], alpha=0.18))
    ax.set_title("MLP Residual Learning — Complete Version Progression (All Controllers)",
                 fontsize=13, pad=12)
    save_fig(fig, out_dir, "fig_mlp_progression_table")


def fig_mlp_all_controllers_stability_bar(df, out_dir):
    """
    Figure A1: 8-bar grouped chart.
    For each of the 4 controller pairs (Passivity, WQP, HQP, LQR):
    one bar for baseline, one for +MLP v3.
    Groups are visually separated on the x-axis.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    gap    = 0.3   # extra space between pairs
    width  = 0.38
    x_pos  = []
    x_tick_pos = []
    x_tick_lab = []
    current_x  = 0.0
    bar_meta   = []   # (x, rate, lo, hi, color, label, n_stable, n_total)

    for base_ctrl, mlp_ctrl, pair_label in MLP_PAIRS:
        available = [c for c in [base_ctrl, mlp_ctrl] if c in df["controller_name"].unique()]
        pair_center = current_x + width / 2
        x_tick_pos.append(pair_center)
        x_tick_lab.append(pair_label)
        for j, ctrl in enumerate([base_ctrl, mlp_ctrl]):
            if ctrl not in df["controller_name"].unique():
                current_x += width
                continue
            k, n, rate = overall_stability(df, ctrl)
            lo, hi = wilson_ci(k, n)
            bar_meta.append((current_x, rate, lo, hi, CTRL_COLORS[ctrl],
                             CTRL_LABELS[ctrl], k, n))
            current_x += width
        current_x += gap

    for (x, rate, lo, hi, color, label, k, n) in bar_meta:
        if np.isnan(rate):
            continue
        ax.bar(x, rate, width=width * 0.92, color=color, label=label,
               yerr=np.array([[rate - lo], [hi - rate]]), capsize=5, alpha=0.9)
        ax.text(x + width * 0.46, rate + 0.025,
                f"{k}/{n}\n({100 * rate:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Draw improvement arrows for each pair
    pair_gap = 2 * width + gap
    for i, (base_ctrl, mlp_ctrl, _) in enumerate(MLP_PAIRS):
        if base_ctrl not in df["controller_name"].unique():
            continue
        if mlp_ctrl not in df["controller_name"].unique():
            continue
        kb, nb, rb = overall_stability(df, base_ctrl)
        km, nm, rm = overall_stability(df, mlp_ctrl)
        if not (np.isfinite(rb) and np.isfinite(rm)):
            continue
        delta = rm - rb
        x_base = i * pair_gap
        x_mid  = x_base + width / 2 + width / 2
        y_top  = max(rb, rm) + 0.08
        col    = "#1a9850" if delta >= 0 else "#d73027"
        ax.annotate("", xy=(x_base + width, y_top - 0.02),
                    xytext=(x_base, y_top - 0.02),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
        ax.text(x_mid, y_top,
                f"{'+' if delta >= 0 else ''}{100 * delta:.1f}pp",
                ha="center", va="bottom", fontsize=11, color=col, fontweight="bold")

    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels([f"{lab}\nBaseline / +MLP v3" for lab in x_tick_lab], fontsize=11)
    ax.set_ylim(0, 0.90)
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_title("MLP v3 Residual Compensation: All Four Base Controllers\n"
                 "Grouped by controller pair — baseline vs +MLP v3 (200 seeds each)")
    ax.set_ylabel("Stability Rate (Wilson 95% CI)")
    # De-duplicate legend
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True, ncols=2, loc="upper right")
    save_fig(fig, out_dir, "fig_mlp_all_controllers_stability_bar")


def fig_mlp_improvement_summary(df, out_dir):
    """
    Figure A2: Clean delta bar chart.
    4 bars showing the percentage-point improvement for each controller.
    Sorted by improvement (descending).
    Key visual for the 'scales with error structure' argument.
    """
    pairs_data = []
    for base_ctrl, mlp_ctrl, pair_label in MLP_PAIRS:
        if base_ctrl not in df["controller_name"].unique():
            continue
        if mlp_ctrl not in df["controller_name"].unique():
            continue
        kb, nb, rb = overall_stability(df, base_ctrl)
        km, nm, rm = overall_stability(df, mlp_ctrl)
        if not (np.isfinite(rb) and np.isfinite(rm)):
            continue
        delta = rm - rb
        pairs_data.append((pair_label, delta, CTRL_COLORS[base_ctrl],
                           kb, nb, km, nm))

    if not pairs_data:
        print("[fig_mlp_improvement_summary] No MLP v3 data available — skipping.")
        return

    # Sort descending
    pairs_data.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, delta, color, kb, nb, km, nm) in enumerate(pairs_data):
        bar_color = color
        ax.bar(i, delta * 100, color=bar_color, width=0.55, alpha=0.9,
               edgecolor="white", linewidth=0.5)
        ax.text(i, delta * 100 + 0.2,
                f"{'+' if delta >= 0 else ''}{100 * delta:.1f}pp\n"
                f"({kb}/{nb}→{km}/{nm})",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(pairs_data)))
    ax.set_xticklabels([p[0] for p in pairs_data], fontsize=12)
    ax.set_ylabel("Stability Rate Improvement (percentage points)")
    ax.set_title("MLP v3 Improvement by Base Controller\n"
                 "Improvement scales with correctable error structure, not controller complexity")
    y_max = max(p[1] for p in pairs_data) * 100 + 3.5
    ax.set_ylim(-1, y_max)

    # Annotation box explaining the pattern
    ax.text(0.98, 0.03,
            "Passivity > WQP = HQP > LQR\n"
            "LQR overlaps with BC target → less correctable residual\n"
            "WQP/HQP identical (Finding 1 holds under MLP augmentation)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
    save_fig(fig, out_dir, "fig_mlp_improvement_summary")


def fig_mlp_force_bin_all_controllers(df, out_dir):
    """
    Figure A3: Force bin stability for all 4 controller pairs.
    2×2 subplot grid — one subplot per controller pair.
    Each subplot shows baseline vs +MLP v3 per force bin.
    Replaces the old fig_mlp_force_bin_comparison (passivity-only version).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    axes_flat = axes.flatten()
    x = np.arange(len(FORCE_BIN_LABELS), dtype=float)
    width = 0.35

    for idx, (base_ctrl, mlp_ctrl, pair_label) in enumerate(MLP_PAIRS):
        ax = axes_flat[idx]
        if base_ctrl not in df["controller_name"].unique():
            ax.set_title(f"{pair_label} — no data")
            continue

        ctls_avail = [c for c in [base_ctrl, mlp_ctrl] if c in df["controller_name"].unique()]
        stats = compute_bin_stability(df, ctls_avail)

        for j, ctrl in enumerate(ctls_avail):
            sub = stats[stats["controller_name"] == ctrl].set_index("force_bin").reindex(FORCE_BIN_LABELS)
            y  = sub["stability_rate"].to_numpy(dtype=float)
            lo = sub["wilson_ci_low"].to_numpy(dtype=float)
            hi = sub["wilson_ci_high"].to_numpy(dtype=float)
            offset = (j - 0.5) * width
            bars = ax.bar(x + offset, y, width=width,
                          color=CTRL_COLORS[ctrl], label=CTRL_LABELS[ctrl],
                          yerr=np.vstack([y - lo, hi - y]), capsize=3, alpha=0.88)
            for b_idx, (bar, bl) in enumerate(zip(bars, FORCE_BIN_LABELS)):
                if bl in sub.index and np.isfinite(y[b_idx]):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            float(y[b_idx]) + 0.04,
                            f"{int(sub.loc[bl, 'n_stable'])}/{int(sub.loc[bl, 'n_trials'])}",
                            ha="center", va="bottom", fontsize=8)

        # Delta annotations
        if len(ctls_avail) == 2:
            bs  = stats[stats["controller_name"] == base_ctrl].set_index("force_bin").reindex(FORCE_BIN_LABELS)
            ms_ = stats[stats["controller_name"] == mlp_ctrl].set_index("force_bin").reindex(FORCE_BIN_LABELS)
            for b_idx, bl in enumerate(FORCE_BIN_LABELS):
                br = float(bs.loc[bl, "stability_rate"]) if bl in bs.index else np.nan
                mr = float(ms_.loc[bl, "stability_rate"]) if bl in ms_.index else np.nan
                if np.isfinite(br) and np.isfinite(mr):
                    delta = mr - br
                    col   = "#1a9850" if delta >= 0 else "#d73027"
                    ax.text(x[b_idx], max(br, mr) + 0.10,
                            f"{'+' if delta >= 0 else ''}{100 * delta:.0f}pp",
                            ha="center", va="bottom", fontsize=9,
                            color=col, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(FORCE_BIN_LABELS, fontsize=9)
        ax.set_ylim(0, 1.10)
        ax.yaxis.set_major_formatter(FuncFormatter(_percent))
        ax.set_title(f"{pair_label}: Baseline vs +MLP v3 by Force Bin")
        ax.set_xlabel("Max Instantaneous Force Bin")
        ax.set_ylabel("Stability Rate (Wilson 95% CI)")
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle("MLP v3 Force-Bin Breakdown — All Four Base Controllers",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, out_dir, "fig_mlp_force_bin_all_controllers")


def fig_mlp_survival_all_controllers(df, out_dir):
    """
    Figure A4: Survival time comparison across all 4 controller pairs.
    2×2 subplot grid — overlapping histograms per pair.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for idx, (base_ctrl, mlp_ctrl, pair_label) in enumerate(MLP_PAIRS):
        ax = axes_flat[idx]
        for ctrl, ls in [(base_ctrl, "solid"), (mlp_ctrl, "dashed")]:
            if ctrl not in df["controller_name"].unique():
                continue
            sub  = df[df["controller_name"] == ctrl]["survival_time_s"]
            mean = float(sub.mean())
            ax.hist(sub, bins=20, alpha=0.55, color=CTRL_COLORS[ctrl],
                    linestyle=ls, edgecolor="white", linewidth=0.5,
                    label=f"{CTRL_LABELS[ctrl]} (mean {mean:.2f}s)")
        ax.set_title(f"{pair_label}: Survival Time")
        ax.set_xlabel("Survival Time (s)")
        ax.set_ylabel("Trials")
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle("Survival Time Distribution: Baseline vs +MLP v3 (All Controllers)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, out_dir, "fig_mlp_survival_all_controllers")


def fig_mlp_neither_under_mlp(df, out_dir):
    """
    Figure A5: Neither category count — baseline vs +MLP v3.
    Shows whether MLP increases or decreases lingering oscillation.
    """
    rows = []
    for base_ctrl, mlp_ctrl, pair_label in MLP_PAIRS:
        for ctrl in [base_ctrl, mlp_ctrl]:
            if ctrl not in df["controller_name"].unique():
                continue
            sub = df[df["controller_name"] == ctrl]
            n_total   = int(len(sub))
            n_neither = int(((sub["is_stable"] == 0) & (sub["fall_time"] <= 0.0)).sum())
            rows.append({
                "controller": ctrl,
                "pair_label": pair_label,
                "n_neither":  n_neither,
                "pct_neither": float(100.0 * n_neither / n_total) if n_total else 0.0,
                "label":      CTRL_LABELS[ctrl],
                "is_mlp":     (ctrl != base_ctrl),
            })
    if not rows:
        return
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(MLP_PAIRS), dtype=float)
    width  = 0.35
    for j, is_mlp in enumerate([False, True]):
        sub   = d[d["is_mlp"] == is_mlp]
        vals  = []
        labs  = []
        for _, pair_label, _ in MLP_PAIRS:
            row = sub[sub["pair_label"] == pair_label]
            vals.append(float(row["pct_neither"].iloc[0]) if len(row) else 0.0)
            labs.append(row["label"].iloc[0] if len(row) else "")
        pattern = "///" if is_mlp else ""
        ax.bar(x + (j - 0.5) * width, vals, width=width,
               label="Baseline" if not is_mlp else "+MLP v3",
               color=["#cccccc" if not is_mlp else "#4d9e4d"] * len(MLP_PAIRS),
               hatch=pattern, edgecolor="white", alpha=0.85)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(x[i] + (j - 0.5) * width, v + 0.1, f"{v:.1f}%",
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in MLP_PAIRS], fontsize=12)
    ax.set_ylabel('% of Trials Classified as "Neither"')
    ax.set_title('"Neither" Category Under MLP v3 Augmentation\n'
                 'MLP should not increase lingering oscillation')
    ax.legend(frameon=True)
    save_fig(fig, out_dir, "fig_mlp_neither_under_mlp")


def fig_wqp_hqp_mlp_equivalence_scatter(df, out_dir):
    """
    Figure C1: WQP+MLP vs HQP+MLP per-seed equivalence scatter.
    Strongest confirmation of Finding 16 — equivalence holds under augmentation.
    """
    if "wqp_mlp_v3" not in df["controller_name"].unique():
        print("[fig_wqp_hqp_mlp_equivalence_scatter] wqp_mlp_v3 not in data — skipping.")
        return
    if "hqp_mlp_v3" not in df["controller_name"].unique():
        print("[fig_wqp_hqp_mlp_equivalence_scatter] hqp_mlp_v3 not in data — skipping.")
        return

    m = _seed_merge(df, "wqp_mlp_v3", "hqp_mlp_v3")

    def classify(row):
        ws = int(row["wqp_mlp_v3_stable"]) == 1
        hs = int(row["hqp_mlp_v3_stable"]) == 1
        wf = (int(row["wqp_mlp_v3_stable"]) == 0) and (float(row["wqp_mlp_v3_fall"]) > 0)
        hf = (int(row["hqp_mlp_v3_stable"]) == 0) and (float(row["hqp_mlp_v3_fall"]) > 0)
        if ws and hs:   return "both_stable"
        if wf and hf:   return "both_fell"
        return "neither"

    m["pair_class"] = m.apply(classify, axis=1)
    color_map = {"both_stable": "#1a9850", "both_fell": "#d73027", "neither": "#666666"}
    fig, ax = plt.subplots(figsize=(7, 7))
    for key, label in [("both_stable", "both stable"), ("both_fell", "both fell"), ("neither", "neither")]:
        sub = m[m["pair_class"] == key]
        ax.scatter(sub["wqp_mlp_v3_com"], sub["hqp_mlp_v3_com"],
                   s=28, alpha=0.75, color=color_map[key], label=label)
    lim = float(max(m["wqp_mlp_v3_com"].max(), m["hqp_mlp_v3_com"].max()) * 1.05)
    ax.plot([0, lim], [0, lim], linestyle="--", color="black", linewidth=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.text(0.02, 0.98,
            f"mean CoM diff = {float(m['abs_diff_m'].mean() * 1000):.2f}mm\n"
            f"max diff = {float(m['abs_diff_m'].max() * 1000):.2f}mm\n"
            f"outcome match = {int(m['outcome_match'].sum())}/{len(m)}\n\n"
            "WQP ≡ HQP holds even under MLP augmentation\n(Finding 16)",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    ax.set_title("WQP+MLP v3 vs HQP+MLP v3 Per-Seed Equivalence\n"
                 "Confirms Finding 16: strict hierarchy irrelevant at any augmentation level")
    ax.set_xlabel("WQP+MLP v3 max_com_disp (m)")
    ax.set_ylabel("HQP+MLP v3 max_com_disp (m)")
    ax.legend(frameon=True, loc="lower right")
    save_fig(fig, out_dir, "fig_wqp_hqp_mlp_equivalence_scatter")


# ── Legacy passivity-only figures (kept for backwards compatibility) ───────────

def fig_mlp_stability_comparison(df, out_dir):
    """Passivity baseline vs Passivity+MLP v3 — two-bar chart. Kept for Chapter 5 Section 5.6."""
    ctls = [c for c in ["passivity_based", "passivity_mlp_v3"]
            if c in df["controller_name"].unique()]
    if len(ctls) < 2:
        print("[fig_mlp_stability_comparison] passivity_mlp_v3 not in data — skipping.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, ctrl in enumerate(ctls):
        k, n, rate = overall_stability(df, ctrl)
        lo, hi = wilson_ci(k, n)
        ax.bar(i, rate, color=CTRL_COLORS[ctrl],
               yerr=np.array([[rate - lo], [hi - rate]]), capsize=6, width=0.55)
        ax.text(i, rate + 0.025,
                f"{k}/{n}\n({100 * rate:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(ctls)))
    ax.set_xticklabels([CTRL_LABELS[c] for c in ctls], rotation=10, ha="right")
    ax.set_ylim(0, 0.90)
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_title("Residual MLP v3 vs Passivity Baseline: Stability Rate\n"
                 "McNemar p < 0.001 (exact p = 3.1×10⁻⁵); b=0 (no destabilisation)")
    ax.set_ylabel("Stability Rate (Wilson 95% CI)")
    save_fig(fig, out_dir, "fig_mlp_stability_comparison")


def fig_mlp_force_bin_comparison(df, out_dir):
    """Passivity-only force bin comparison. Kept for thesis Section 5.6."""
    ctls = [c for c in ["passivity_based", "passivity_mlp_v3"]
            if c in df["controller_name"].unique()]
    if len(ctls) < 2:
        return
    stats = compute_bin_stability(df, ctls)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(FORCE_BIN_LABELS), dtype=float)
    width = 0.35
    for i, ctrl in enumerate(ctls):
        sub = stats[stats["controller_name"] == ctrl].set_index("force_bin").reindex(FORCE_BIN_LABELS)
        y  = sub["stability_rate"].to_numpy(dtype=float)
        lo = sub["wilson_ci_low"].to_numpy(dtype=float)
        hi = sub["wilson_ci_high"].to_numpy(dtype=float)
        bars = ax.bar(x + (i - 0.5) * width, y, width=width,
                      color=CTRL_COLORS[ctrl], label=CTRL_LABELS[ctrl],
                      yerr=np.vstack([y - lo, hi - y]), capsize=4, alpha=0.88)
        for j, (bar, b) in enumerate(zip(bars, FORCE_BIN_LABELS)):
            if b in sub.index:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        float(y[j]) + 0.03,
                        f"{int(sub.loc[b, 'n_stable'])}/{int(sub.loc[b, 'n_trials'])}",
                        ha="center", va="bottom", fontsize=9)
    try:
        ps = stats[stats["controller_name"] == "passivity_based"].set_index("force_bin").reindex(FORCE_BIN_LABELS)
        vs = stats[stats["controller_name"] == "passivity_mlp_v3"].set_index("force_bin").reindex(FORCE_BIN_LABELS)
        for j, b in enumerate(FORCE_BIN_LABELS):
            pr = float(ps.loc[b, "stability_rate"]) if b in ps.index else np.nan
            vr = float(vs.loc[b, "stability_rate"]) if b in vs.index else np.nan
            if np.isfinite(pr) and np.isfinite(vr):
                delta = vr - pr
                col   = "#1a9850" if delta >= 0 else "#d73027"
                ax.text(x[j], max(pr, vr) + 0.08,
                        f"{'+' if delta >= 0 else ''}{100 * delta:.0f}pp",
                        ha="center", va="bottom", fontsize=10, color=col, fontweight="bold")
    except:
        pass
    ax.set_xticks(x); ax.set_xticklabels(FORCE_BIN_LABELS)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_title("MLP v3 vs Passivity Baseline: Stability Rate by Force Bin\n"
                 "Largest improvement in 125–175N range (+13pp) — base controller near feasibility boundary")
    ax.set_xlabel("Max Instantaneous Force Bin")
    ax.set_ylabel("Stability Rate (Wilson 95% CI)")
    ax.legend(frameon=True)
    save_fig(fig, out_dir, "fig_mlp_force_bin_comparison")


def fig_mlp_com_comparison(df, out_dir):
    """CoM displacement on stable trials: passivity baseline vs +MLP v3."""
    ctls = [c for c in ["passivity_based", "passivity_mlp_v3"]
            if c in df["controller_name"].unique()]
    if len(ctls) < 2:
        return
    stable = df[(df["controller_name"].isin(ctls)) & (df["is_stable"] == 1)].copy()
    if stable.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {c: CTRL_COLORS.get(c, "#aaaaaa") for c in ctls}
    sns.boxplot(data=stable, x="controller_name", y="max_com_disp",
                order=ctls, palette=palette, width=0.5, ax=ax,
                flierprops=dict(marker=".", markersize=4, alpha=0.5))
    sns.stripplot(data=stable, x="controller_name", y="max_com_disp",
                  order=ctls, palette=palette, ax=ax, jitter=True, alpha=0.4, size=3)
    means = stable.groupby("controller_name")["max_com_disp"].mean().to_dict()
    for i, c in enumerate(ctls):
        m = means.get(c, np.nan)
        if np.isfinite(m):
            ax.text(i, m + 0.003, f"mean={m:.4f}m",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticklabels([CTRL_LABELS.get(c, c) for c in ctls])
    ax.set_title("Max CoM Displacement on Stable Trials\nPassivity Baseline vs Passivity+MLP v3")
    ax.set_xlabel(""); ax.set_ylabel("Max Horizontal CoM Displacement (m)")
    save_fig(fig, out_dir, "fig_mlp_com_comparison")


def fig_mlp_survival_comparison(df, out_dir):
    """Survival time histogram: passivity baseline vs +MLP v3. Kept for Section 5.6."""
    ctls = [c for c in ["passivity_based", "passivity_mlp_v3"]
            if c in df["controller_name"].unique()]
    if len(ctls) < 2:
        return
    d = df[df["controller_name"].isin(ctls)].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for ctrl in ctls:
        sub  = d[d["controller_name"] == ctrl]["survival_time_s"]
        mean = float(sub.mean())
        ax.hist(sub, bins=20, alpha=0.55, color=CTRL_COLORS[ctrl],
                label=f"{CTRL_LABELS[ctrl]} (mean {mean:.2f}s)",
                edgecolor="white", linewidth=0.5)
    ax.set_title("Survival Time Distribution: Passivity Baseline vs MLP v3 (All Trials)")
    ax.set_xlabel("Survival Time (s)")
    ax.set_ylabel("Number of Trials")
    ax.legend(frameon=True)
    save_fig(fig, out_dir, "fig_mlp_survival_comparison")


# ── DISTILLATION FIGURES ──────────────────────────────────────────────────────

def fig_distillation_comparison(df, out_dir):
    """Four-bar: WQP baseline / WQP+Distillation / Passivity baseline / Passivity+MLP v3."""
    ctls_wanted = ["weighted_qp", "wqp_distilled", "passivity_based", "passivity_mlp_v3"]
    ctls = [c for c in ctls_wanted if c in df["controller_name"].unique()]
    if "wqp_distilled" not in ctls:
        print("[fig_distillation_comparison] wqp_distilled not in data — skipping.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, ctrl in enumerate(ctls):
        k, n, rate = overall_stability(df, ctrl)
        lo, hi = wilson_ci(k, n)
        ax.bar(i, rate, color=CTRL_COLORS[ctrl], label=CTRL_LABELS[ctrl],
               yerr=np.array([[rate - lo], [hi - rate]]), capsize=5, width=0.55, alpha=0.9)
        ax.text(i, rate + 0.025,
                f"{k}/{n}\n({100 * rate:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(ctls)))
    ax.set_xticklabels([CTRL_LABELS[c] for c in ctls], rotation=10, ha="right")
    ax.set_ylim(0, 0.90)
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_title("Cross-Controller Knowledge Transfer\n"
                 "WQP+Distillation closes most of the gap to Passivity baseline")
    ax.set_ylabel("Stability Rate (Wilson 95% CI)")
    ax.legend(frameon=True, ncols=2)
    save_fig(fig, out_dir, "fig_distillation_comparison")


def fig_distillation_force_bin(df, out_dir):
    """Force bin: WQP baseline vs WQP+Distillation vs Passivity baseline."""
    ctls = [c for c in ["weighted_qp", "passivity_based", "wqp_distilled"]
            if c in df["controller_name"].unique()]
    if "wqp_distilled" not in ctls:
        print("[fig_distillation_force_bin] wqp_distilled not in data — skipping.")
        return
    stats = compute_bin_stability(df, ctls)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(FORCE_BIN_LABELS), dtype=float)
    width = 0.25
    for i, ctrl in enumerate(ctls):
        sub = stats[stats["controller_name"] == ctrl].set_index("force_bin").reindex(FORCE_BIN_LABELS)
        y  = sub["stability_rate"].to_numpy(dtype=float)
        lo = sub["wilson_ci_low"].to_numpy(dtype=float)
        hi = sub["wilson_ci_high"].to_numpy(dtype=float)
        bars = ax.bar(x + (i - 1) * width, y, width=width,
                      color=CTRL_COLORS[ctrl], label=CTRL_LABELS[ctrl],
                      yerr=np.vstack([y - lo, hi - y]), capsize=3, alpha=0.88)
        for j, (bar, b) in enumerate(zip(bars, FORCE_BIN_LABELS)):
            if b in sub.index:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        float(y[j]) + 0.03,
                        f"{int(sub.loc[b, 'n_stable'])}/{int(sub.loc[b, 'n_trials'])}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(FORCE_BIN_LABELS)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_title("WQP+Distillation vs WQP Baseline vs Passivity: Stability Rate by Force Bin")
    ax.set_xlabel("Max Instantaneous Force Bin")
    ax.set_ylabel("Stability Rate (Wilson 95% CI)")
    ax.legend(frameon=True)
    save_fig(fig, out_dir, "fig_distillation_force_bin")


# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Part 1b analysis — 4 WBC controllers + MLP v3 (all controllers) + distillation"
    )
    # ── Base controllers (required — defaults point to primary April 22 dataset) ──
    p.add_argument("--weighted-qp-csv",
        type=str, default=r"experiment_results\Old data\2\weighted_qp_random_20260422_131731.csv")
    p.add_argument("--hierarchical-qp-csv",
        type=str, default=r"experiment_results\Old data\2\hierarchical_qp_random_20260422_135759.csv")
    p.add_argument("--passivity-csv",
        type=str, default=r"experiment_results\Old data\2\passivity_based_random_20260422_145210.csv")
    p.add_argument("--lqr-csv",
        type=str, default=r"experiment_results\Old data\2\lqr_lipm_random_20260422_145642.csv")
    # ── MLP CSVs (optional) ───────────────────────────────────────────────────
    p.add_argument("--mlp-csv",
        type=str, default="", help="v1/v2 null result CSV (optional, historical)")
    p.add_argument("--mlp-v3-csv",
        type=str, default="", help="Passivity+MLP v3 positive result CSV")
    p.add_argument("--wqp-mlp-v3-csv",
        type=str, default="", help="WQP+MLP v3 positive result CSV")       # NEW
    p.add_argument("--hqp-mlp-v3-csv",
        type=str, default="", help="HQP+MLP v3 positive result CSV")       # NEW
    p.add_argument("--lqr-mlp-v3-csv",
        type=str, default="", help="LQR+MLP v3 positive result CSV")       # NEW
    # ── Distillation (optional) ───────────────────────────────────────────────
    p.add_argument("--distillation-csv",
        type=str, default="", help="WQP→Passivity distillation result CSV")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    set_thesis_style()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = ensure_output_dir(script_dir)
    args       = parse_args()

    csvs = InputCSVs(
        weighted_qp      = args.weighted_qp_csv,
        hierarchical_qp  = args.hierarchical_qp_csv,
        passivity_based  = args.passivity_csv,
        lqr_lipm         = args.lqr_csv,
        passivity_mlp    = args.mlp_csv,
        passivity_mlp_v3 = args.mlp_v3_csv,
        wqp_mlp_v3       = args.wqp_mlp_v3_csv,
        hqp_mlp_v3       = args.hqp_mlp_v3_csv,
        lqr_mlp_v3       = args.lqr_mlp_v3_csv,
        wqp_distilled    = args.distillation_csv,
    )

    df = load_all_controllers(csvs)

    # ── CHAPTER 4 ─────────────────────────────────────────────────────────────
    print("\n--- Chapter 4 figures ---")
    fig1_stability_rate_by_force_bin(df, out_dir)
    fig2_stability_sigmoid_curves(df, out_dir)
    fig3_wqp_vs_hqp_per_seed_equivalence_scatter(df, out_dir)
    fig4_three_category_outcome_stacked_bar(df, out_dir)
    fig5_com_displacement_boxplots_stable_only(df, out_dir)
    fig6_rms_com_displacement_by_force_bin(df, out_dir)
    fig7_torque_cost_comparison(df, out_dir)
    fig8_survival_time_distribution(df, out_dir)
    fig9_neither_category_deep_dive(df, out_dir)
    fig10_push_count_vs_stability(df, out_dir)
    print("[Chapter 4 done — fig1 through fig10]")

    # ── SUMMARY TABLES ────────────────────────────────────────────────────────
    print("\n--- Summary tables ---")
    table1_stats_part1b_summary(df, out_dir)
    table2_stats_part1b_force_bins(df, out_dir)
    table3_stats_part1b_wqp_hqp_equivalence(df, out_dir)
    print("[Tables done]")

    # ── CHAPTER 5 — MLP figures ───────────────────────────────────────────────
    print("\n--- Chapter 5 MLP figures ---")

    # D1 — always generated (uses hard-coded confirmed results)
    fig_mlp_progression_table(df, out_dir)
    print("[fig_mlp_progression_table saved]")

    has_passivity_mlp_v3 = bool(args.mlp_v3_csv)
    has_wqp_mlp_v3       = bool(args.wqp_mlp_v3_csv)
    has_hqp_mlp_v3       = bool(args.hqp_mlp_v3_csv)
    has_lqr_mlp_v3       = bool(args.lqr_mlp_v3_csv)
    any_mlp_v3           = any([has_passivity_mlp_v3, has_wqp_mlp_v3,
                                 has_hqp_mlp_v3, has_lqr_mlp_v3])

    if has_passivity_mlp_v3:
        # Legacy passivity-only figures for Section 5.6
        fig_mlp_stability_comparison(df, out_dir)
        fig_mlp_force_bin_comparison(df, out_dir)
        fig_mlp_com_comparison(df, out_dir)
        fig_mlp_survival_comparison(df, out_dir)
        print("[Passivity-only MLP v3 figures saved]")

    if any_mlp_v3:
        # Cross-controller figures — A1, A2, A3, A4, A5
        fig_mlp_all_controllers_stability_bar(df, out_dir)   # A1
        fig_mlp_improvement_summary(df, out_dir)             # A2
        fig_mlp_force_bin_all_controllers(df, out_dir)       # A3 (2×2 grid)
        fig_mlp_survival_all_controllers(df, out_dir)        # A4
        fig_mlp_neither_under_mlp(df, out_dir)               # A5
        print("[Cross-controller MLP v3 figures saved (A1–A5)]")
    else:
        print("[No MLP v3 CSVs provided — cross-controller figures skipped]")
        print("  Add: --mlp-v3-csv, --wqp-mlp-v3-csv, --hqp-mlp-v3-csv, --lqr-mlp-v3-csv")

    if has_wqp_mlp_v3 and has_hqp_mlp_v3:
        fig_wqp_hqp_mlp_equivalence_scatter(df, out_dir)     # C1
        print("[WQP+MLP vs HQP+MLP equivalence scatter saved (C1)]")

    # ── CHAPTER 5 — Distillation figures ─────────────────────────────────────
    print("\n--- Chapter 5 Distillation figures ---")
    if args.distillation_csv:
        fig_distillation_comparison(df, out_dir)
        fig_distillation_force_bin(df, out_dir)
        print("[Distillation figures saved]")
    else:
        print("[No --distillation-csv provided — skipping distillation figures]")
        print(r"  Add: --distillation-csv experiment_results\wqp_distilled_random_20260506_155337.csv")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\nAll figures saved to: {out_dir}")
    print()
    print("Chapter 4 figures:  fig1 through fig10 + 3 CSV tables")
    print("Chapter 5 figures:")
    print("  fig_mlp_progression_table          ← updated 7-row table (always)")
    print("  fig_mlp_stability_comparison        ← passivity pair (Section 5.6)")
    print("  fig_mlp_force_bin_comparison        ← passivity pair (Section 5.6)")
    print("  fig_mlp_com_comparison              ← passivity pair (Section 5.6)")
    print("  fig_mlp_survival_comparison         ← passivity pair (Section 5.6)")
    print("  fig_mlp_all_controllers_stability_bar  ← A1: 8-bar grouped chart")
    print("  fig_mlp_improvement_summary            ← A2: delta bar chart")
    print("  fig_mlp_force_bin_all_controllers      ← A3: 2×2 force bin grid")
    print("  fig_mlp_survival_all_controllers       ← A4: 2×2 survival histograms")
    print("  fig_mlp_neither_under_mlp              ← A5: neither category under MLP")
    print("  fig_wqp_hqp_mlp_equivalence_scatter    ← C1: Finding 16 confirmation")
    print("  fig_distillation_comparison")
    print("  fig_distillation_force_bin")


if __name__ == "__main__":
    main()