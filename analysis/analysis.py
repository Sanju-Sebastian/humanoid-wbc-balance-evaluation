import glob
import os
import warnings
from itertools import combinations

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "experiment_results")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "analysis_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CTRL_ORDER = ["weighted_qp", "hierarchical_qp", "passivity_based"]

CTRL_LABELS = {
    "weighted_qp": "Weighted QP",
    "hierarchical_qp": "Hierarchical QP",
    "passivity_based": "Passivity-Based",
}

CTRL_COLORS = {
    "weighted_qp": "#2166ac",
    "hierarchical_qp": "#d6604d",
    "passivity_based": "#1a9850",
}

CTRL_MARKERS = {
    "weighted_qp": "o",
    "hierarchical_qp": "s",
    "passivity_based": "^",
}


def set_thesis_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.2)
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    for ext in ["pdf", "png"]:
        path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def load_all_csvs(results_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {results_dir}")
    dfs: list[pd.DataFrame] = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Skipped: {os.path.basename(f)}  ({type(e).__name__})")
            continue
        if len(df) == 0:
            print(f"Skipped: {os.path.basename(f)}  (0 rows)")
            continue
        dfs.append(df)
        ctrl_val = df["controller_name"].iloc[0] if "controller_name" in df.columns and len(df) else "unknown"
        print(f"Loaded: {os.path.basename(f)}  ({len(df)} rows, controller={ctrl_val})")
    if not dfs:
        raise RuntimeError(f"No usable CSV files in: {results_dir}")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(combined)}")
    if "controller_name" in combined.columns:
        controllers = combined["controller_name"].unique().tolist()
        print(f"Controllers: {controllers}")
        combined = combined[combined["controller_name"].isin(CTRL_ORDER)].copy()
        extra = [c for c in controllers if c not in CTRL_ORDER]
        if extra:
            print(f"Note: Ignoring unknown controllers: {extra}")
    return combined


def _percent_formatter(x: float, pos: int) -> str:
    return f"{100.0 * x:.0f}%"


def _filter_force_leq(df: pd.DataFrame, max_force: float) -> pd.DataFrame:
    return df[df["force_magnitude"] <= float(max_force)].copy()


def _add_saturation_region(ax: plt.Axes, start_force: float = 200.0) -> None:
    xmin, xmax = ax.get_xlim()
    span_start = max(float(start_force), float(xmin))
    if span_start < float(xmax):
        ax.axvspan(span_start, xmax, color="grey", alpha=0.12)
        ax.text(
            (span_start + xmax) / 2.0,
            0.05,
            "Ankle saturation regime",
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
        )


def plot_fig1_stability(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    stability = (
        df.groupby(["controller_name", "force_magnitude"])["is_stable"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    for ctrl in CTRL_ORDER:
        sub = stability[stability["controller_name"] == ctrl].sort_values("force_magnitude")
        x = sub["force_magnitude"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        ysem = sub["sem"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker=CTRL_MARKERS[ctrl],
            color=CTRL_COLORS[ctrl],
            label=CTRL_LABELS[ctrl],
            linewidth=2,
        )
        ax.fill_between(x, y - ysem, y + ysem, color=CTRL_COLORS[ctrl], alpha=0.15)

    ax.axhline(0.5, linestyle="--", color="black", linewidth=1)
    ax.text(
        0.99,
        0.5,
        "50% threshold",
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )

    ax.set_title("Stability Rate vs Disturbance Magnitude")
    ax.set_xlabel("Applied Force (N)")
    ax.set_ylabel("Stability Rate")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(FuncFormatter(_percent_formatter))

    _add_saturation_region(ax, start_force=200.0)

    try:
        anno_x = 100.0
        anno_y = float(
            stability[
                (stability["controller_name"] == "weighted_qp")
                & (stability["force_magnitude"] == anno_x)
            ]["mean"].iloc[0]
        )
        ax.annotate(
            "WQP and HQP curves overlap exactly",
            xy=(anno_x, anno_y),
            xytext=(anno_x + 30.0, min(0.95, anno_y + 0.25)),
            arrowprops=dict(arrowstyle="->", lw=1),
            fontsize=10,
        )
    except Exception:
        pass

    try:
        x150 = 150.0
        ypass = float(
            stability[
                (stability["controller_name"] == "passivity_based")
                & (stability["force_magnitude"] == x150)
            ]["mean"].iloc[0]
        )
        ax.scatter([x150], [ypass], color=CTRL_COLORS["passivity_based"], s=60, zorder=5)
        ax.annotate(
            f"Passivity best at 150N ({100.0*ypass:.1f}%)",
            xy=(x150, ypass),
            xytext=(x150 + 20.0, max(0.1, ypass - 0.25)),
            arrowprops=dict(arrowstyle="->", lw=1),
            fontsize=10,
        )
    except Exception:
        pass

    ax.legend()
    save_fig(fig, "fig1_stability_vs_force")


def plot_fig2_com_displacement(df: pd.DataFrame) -> None:
    df2 = _filter_force_leq(df, 200.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    agg = (
        df2.groupby(["controller_name", "force_magnitude"])["max_com_disp"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    for ctrl in CTRL_ORDER:
        sub = agg[agg["controller_name"] == ctrl].sort_values("force_magnitude")
        x = sub["force_magnitude"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        ysem = sub["sem"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker=CTRL_MARKERS[ctrl],
            color=CTRL_COLORS[ctrl],
            label=CTRL_LABELS[ctrl],
            linewidth=2,
        )
        ax.fill_between(x, y - ysem, y + ysem, color=CTRL_COLORS[ctrl], alpha=0.15)

    ax.set_title("Peak CoM Displacement vs Disturbance Magnitude")
    ax.set_xlabel("Applied Force (N)")
    ax.set_ylabel("Peak CoM Displacement (m)")

    try:
        ax.set_xlim(float(df["force_magnitude"].min()), float(df["force_magnitude"].max()))
    except Exception:
        pass
    _add_saturation_region(ax, start_force=200.0)

    ax.text(
        0.02,
        0.95,
        "All controllers fall at ≥200N",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    try:
        anno_x = 100.0
        anno_y = float(
            agg[
                (agg["controller_name"] == "weighted_qp")
                & (agg["force_magnitude"] == anno_x)
            ]["mean"].iloc[0]
        )
        ax.annotate(
            "WQP and HQP overlap exactly",
            xy=(anno_x, anno_y),
            xytext=(anno_x + 25.0, anno_y + 0.01),
            arrowprops=dict(arrowstyle="->", lw=1),
            fontsize=10,
        )
    except Exception:
        pass

    ax.legend()
    save_fig(fig, "fig2_com_displacement")


def plot_fig3_recovery_time(df: pd.DataFrame) -> None:
    stable = df[(df["is_stable"] == 1) & (df["recovery_time"] > 0)].copy()
    force_levels = [50.0, 100.0, 150.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, force in zip(axes, force_levels, strict=True):
        sub = stable[stable["force_magnitude"] == float(force)]
        sns.boxplot(
            data=sub,
            x="controller_name",
            y="recovery_time",
            hue="controller_name",
            order=CTRL_ORDER,
            hue_order=CTRL_ORDER,
            palette=CTRL_COLORS,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="controller_name",
            y="recovery_time",
            hue="controller_name",
            order=CTRL_ORDER,
            hue_order=CTRL_ORDER,
            dodge=True,
            palette=CTRL_COLORS,
            alpha=0.5,
            size=4,
            jitter=True,
            ax=ax,
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_title(f"Force = {int(force)}N")
        ax.set_xlabel("")
        ax.set_xticklabels([CTRL_LABELS[c] for c in CTRL_ORDER], rotation=15, ha="right")
        ax.set_ylabel("Recovery Time (s)" if ax is axes[0] else "")

    fig.suptitle("Recovery Time Distribution (Stable Trials Only)")
    handles = [mpatches.Patch(color=CTRL_COLORS[c], label=CTRL_LABELS[c]) for c in CTRL_ORDER]
    fig.legend(handles=handles, loc="upper right", frameon=True)
    save_fig(fig, "fig3_recovery_time")


def plot_fig4_torque_rms(df: pd.DataFrame) -> None:
    cols = ["tau_rms_total", "tau_rms_ankle", "tau_rms_knee", "tau_rms_hip"]
    means = df.groupby("controller_name")[cols].mean().reindex(CTRL_ORDER)
    stds = df.groupby("controller_name")[cols].std().reindex(CTRL_ORDER)

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["Total", "Ankle", "Knee", "Hip"]
    x = np.arange(len(labels), dtype=float)
    width = 0.25

    for i, ctrl in enumerate(CTRL_ORDER):
        vals = means.loc[ctrl].to_numpy(dtype=float)
        errs = stds.loc[ctrl].to_numpy(dtype=float)
        positions = x + (i - 1) * width
        bars = ax.bar(
            positions,
            vals,
            width=width,
            color=CTRL_COLORS[ctrl],
            label=CTRL_LABELS[ctrl],
            yerr=errs,
            capsize=3,
        )
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height(),
                f"{b.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Mean Joint Torque RMS by Controller")
    ax.set_xlabel("Joint Group")
    ax.set_ylabel("RMS Torque (Nm)")
    ax.legend()
    save_fig(fig, "fig4_torque_rms")


def plot_fig5_torso_deviation(df: pd.DataFrame) -> None:
    df2 = _filter_force_leq(df, 200.0).copy()
    rad2deg = 180.0 / np.pi
    df2["max_torso_pitch_deg"] = df2["max_torso_pitch_rad"] * rad2deg
    df2["max_torso_roll_deg"] = df2["max_torso_roll_rad"] * rad2deg

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, metric, title in [
        (axes[0], "max_torso_pitch_deg", "Max Torso Pitch Deviation"),
        (axes[1], "max_torso_roll_deg", "Max Torso Roll Deviation"),
    ]:
        agg = (
            df2.groupby(["controller_name", "force_magnitude"])[metric]
            .agg(["mean", "sem"])
            .reset_index()
        )
        for ctrl in CTRL_ORDER:
            sub = agg[agg["controller_name"] == ctrl].sort_values("force_magnitude")
            x = sub["force_magnitude"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            ysem = sub["sem"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker=CTRL_MARKERS[ctrl],
                color=CTRL_COLORS[ctrl],
                label=CTRL_LABELS[ctrl],
                linewidth=2,
            )
            ax.fill_between(x, y - ysem, y + ysem, color=CTRL_COLORS[ctrl], alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel("Applied Force (N)")
        ax.set_ylabel("Max Deviation (degrees)")
        _add_saturation_region(ax, start_force=200.0)

        try:
            anno_x = 100.0
            anno_y = float(
                agg[
                    (agg["controller_name"] == "weighted_qp")
                    & (agg["force_magnitude"] == anno_x)
                ]["mean"].iloc[0]
            )
            ax.annotate(
                "WQP and HQP overlap",
                xy=(anno_x, anno_y),
                xytext=(anno_x + 25.0, anno_y + 5.0),
                arrowprops=dict(arrowstyle="->", lw=1),
                fontsize=10,
            )
        except Exception:
            pass

    axes[0].legend(loc="best")
    fig.suptitle("Torso Angular Deviation vs Disturbance Magnitude")
    save_fig(fig, "fig5_torso_deviation")


def plot_fig6_qp_solve_time(df: pd.DataFrame) -> None:
    solve_times = (
        df.groupby("controller_name")[["qp_solve_time_mean_ms", "qp_solve_time_max_ms"]]
        .mean()
        .reindex(CTRL_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["Mean solve time", "Max solve time"]
    x = np.arange(len(metrics), dtype=float)
    width = 0.25

    mean_vals = solve_times["qp_solve_time_mean_ms"].to_numpy(dtype=float)
    max_vals = solve_times["qp_solve_time_max_ms"].to_numpy(dtype=float)
    data_by_ctrl = {
        "weighted_qp": np.array([mean_vals[0], max_vals[0]], dtype=float),
        "hierarchical_qp": np.array([mean_vals[1], max_vals[1]], dtype=float),
        "passivity_based": np.array([mean_vals[2], max_vals[2]], dtype=float),
    }

    for i, ctrl in enumerate(CTRL_ORDER):
        vals = data_by_ctrl[ctrl]
        positions = x + (i - 1) * width
        bars = ax.bar(positions, vals, width=width, color=CTRL_COLORS[ctrl], label=CTRL_LABELS[ctrl])
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height(),
                f"{b.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("QP Solve Time per Controller")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Time (ms)")
    ax.legend()

    try:
        passivity_mean_pos = x[0] + (2 - 1) * width
        ax.text(
            passivity_mean_pos,
            max(0.02, float(data_by_ctrl["passivity_based"][0]) + 0.05),
            "No QP (0ms)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    except Exception:
        pass

    try:
        wqp_mean = float(data_by_ctrl["weighted_qp"][0])
        hqp_mean = float(data_by_ctrl["hierarchical_qp"][0])
        ax.text(
            x[0],
            max(wqp_mean, hqp_mean) + 0.15,
            "HQP = 1.83× WQP",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    except Exception:
        pass

    save_fig(fig, "fig6_qp_solve_time")


def plot_fig7_stability_heatmap(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    cmap = plt.get_cmap("RdYlGn")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    for ax, ctrl in zip(axes, CTRL_ORDER, strict=True):
        sub = df[df["controller_name"] == ctrl]
        pivot = (
            sub.groupby(["direction", "force_magnitude"])["is_stable"]
            .mean()
            .unstack()
        )
        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            cbar=False,
        )
        ax.set_title(CTRL_LABELS[ctrl])
        ax.set_xlabel("Force (N)")
        ax.set_ylabel("Push Direction" if ax is axes[0] else "")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Stability Rate")
    fig.suptitle("Stability Rate Heatmap (Direction × Force)")
    save_fig(fig, "fig7_stability_heatmap")


def plot_fig8_stability_heatmap_body(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    cmap = plt.get_cmap("RdYlGn")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    for ax, ctrl in zip(axes, CTRL_ORDER, strict=True):
        sub = df[df["controller_name"] == ctrl]
        pivot = (
            sub.groupby(["body_label", "force_magnitude"])["is_stable"]
            .mean()
            .unstack()
        )
        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            cbar=False,
        )
        ax.set_title(CTRL_LABELS[ctrl])
        ax.set_xlabel("Force (N)")
        ax.set_ylabel("Push Location" if ax is axes[0] else "")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Stability Rate")
    fig.suptitle("Stability Rate Heatmap (Body Location × Force)")
    save_fig(fig, "fig8_stability_heatmap_body")


def plot_fig9_fall_time(df: pd.DataFrame) -> None:
    fallen = df[df["fall_time"] > 0].copy()
    fallen = fallen[fallen["force_magnitude"].isin([100.0, 150.0])].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=fallen,
        x="force_magnitude",
        y="fall_time",
        hue="controller_name",
        hue_order=CTRL_ORDER,
        palette=CTRL_COLORS,
        cut=0,
        inner="quartile",
        ax=ax,
    )

    ax.set_title("Fall Time Distribution (Failed Trials Only)")
    ax.set_xlabel("Applied Force (N)")
    ax.set_ylabel("Time Until Fall (s)")
    ax.legend(title="", labels=[CTRL_LABELS[c] for c in CTRL_ORDER])

    try:
        f100 = fallen[fallen["force_magnitude"] == 100.0]
        means = f100.groupby("controller_name")["fall_time"].mean()
        if "passivity_based" in means and "weighted_qp" in means and "hierarchical_qp" in means:
            p = float(means["passivity_based"])
            w = float(means["weighted_qp"])
            h = float(means["hierarchical_qp"])
            ax.text(
                0.02,
                0.95,
                f"Passivity delays falls at 100N: {p:.2f}s vs {w:.2f}s (WQP), {h:.2f}s (HQP)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )
    except Exception:
        pass

    save_fig(fig, "fig9_fall_time")


def plot_fig10_com_xy_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for ctrl in CTRL_ORDER:
        sub = df[df["controller_name"] == ctrl]
        ax.scatter(
            sub["max_com_disp_x"],
            sub["max_com_disp_y"],
            c=CTRL_COLORS[ctrl],
            label=CTRL_LABELS[ctrl],
            alpha=0.4,
            s=20,
            marker=CTRL_MARKERS[ctrl],
        )

    circle = mpatches.Circle((0.0, 0.0), radius=0.02, fill=False, linestyle="--", linewidth=1)
    ax.add_patch(circle)
    ax.text(0.02, 0.0, "2cm threshold", ha="left", va="bottom")

    ax.set_title("CoM Displacement: Anterior-Posterior vs Mediolateral")
    ax.set_xlabel("AP Displacement (m)")
    ax.set_ylabel("ML Displacement (m)")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    save_fig(fig, "fig10_com_xy_scatter")


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def stats_table_kruskal_wallis(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["max_com_disp", "recovery_time", "tau_rms_total", "max_torso_pitch_rad"]
    results: list[dict[str, object]] = []

    for force in sorted(df["force_magnitude"].unique()):
        for metric in metrics:
            if metric == "recovery_time":
                dsub = df[(df["force_magnitude"] == force) & (df["is_stable"] == 1) & (df["recovery_time"] > 0)]
            else:
                dsub = df[df["force_magnitude"] == force]

            groups = [
                dsub[dsub["controller_name"] == ctrl][metric].dropna().to_numpy(dtype=float)
                for ctrl in CTRL_ORDER
            ]

            if all(len(g) > 1 for g in groups):
                H, p = stats.kruskal(*groups)
                results.append(
                    {
                        "force_N": float(force),
                        "metric": metric,
                        "H": round(float(H), 3),
                        "p": round(float(p), 4),
                        "sig": _sig_label(float(p)),
                    }
                )

    out = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "stats_kruskal_wallis.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    if len(out):
        print(out.to_string(index=False))
    return out


def stats_table_pairwise_150N(df: pd.DataFrame) -> pd.DataFrame:
    force = 150.0
    pairs = list(combinations(CTRL_ORDER, 2))
    results: list[dict[str, object]] = []

    for metric in ["max_com_disp", "max_torso_pitch_rad", "tau_rms_total"]:
        for a, b in pairs:
            ga = df[(df["controller_name"] == a) & (df["force_magnitude"] == force)][metric].dropna()
            gb = df[(df["controller_name"] == b) & (df["force_magnitude"] == force)][metric].dropna()
            if len(ga) < 2 or len(gb) < 2:
                continue
            U, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
            r = 1.0 - (2.0 * float(U)) / (float(len(ga)) * float(len(gb)))

            results.append(
                {
                    "metric": metric,
                    "ctrl_A": a,
                    "ctrl_B": b,
                    "U": round(float(U), 1),
                    "p": round(float(p), 4),
                    "effect_r": round(float(r), 3),
                    "sig": _sig_label(float(p)),
                }
            )

    out = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "stats_pairwise_150N.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    if len(out):
        print(out.to_string(index=False))
    return out


def stats_table_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ctrl in CTRL_ORDER:
        for force in sorted(df["force_magnitude"].unique()):
            sub = df[(df["controller_name"] == ctrl) & (df["force_magnitude"] == force)]
            fallen = sub[sub["fall_time"] > 0]
            stable = sub[(sub["is_stable"] == 1) & (sub["recovery_time"] > 0)]
            rows.append(
                {
                    "controller": CTRL_LABELS[ctrl],
                    "force_N": int(float(force)),
                    "n_trials": int(len(sub)),
                    "n_stable": int(sub["is_stable"].sum()),
                    "stability_rate": round(float(sub["is_stable"].mean()), 3) if len(sub) else np.nan,
                    "mean_com_disp_m": round(float(sub["max_com_disp"].mean()), 4) if len(sub) else np.nan,
                    "std_com_disp_m": round(float(sub["max_com_disp"].std()), 4) if len(sub) else np.nan,
                    "mean_recovery_s": round(float(stable["recovery_time"].mean()), 3) if len(stable) else np.nan,
                    "std_recovery_s": round(float(stable["recovery_time"].std()), 3) if len(stable) else np.nan,
                    "mean_tau_rms_Nm": round(float(sub["tau_rms_total"].mean()), 2) if len(sub) else np.nan,
                    "mean_fall_time_s": round(float(fallen["fall_time"].mean()), 3) if len(fallen) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "stats_summary_table.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    if len(out):
        print(out.head(12).to_string(index=False))
    return out


def run_statistical_analysis(df: pd.DataFrame) -> None:
    stats_table_kruskal_wallis(df)
    stats_table_pairwise_150N(df)
    stats_table_summary(df)


def main() -> None:
    set_thesis_style()
    df = load_all_csvs(RESULTS_DIR)

    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)

    plot_fig1_stability(df)
    plot_fig2_com_displacement(df)
    plot_fig3_recovery_time(df)
    plot_fig4_torque_rms(df)
    plot_fig5_torso_deviation(df)
    plot_fig6_qp_solve_time(df)
    plot_fig7_stability_heatmap(df)
    plot_fig8_stability_heatmap_body(df)
    plot_fig9_fall_time(df)
    plot_fig10_com_xy_scatter(df)

    print("\n" + "=" * 60)
    print("Running statistical analysis...")
    print("=" * 60)

    run_statistical_analysis(df)

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
