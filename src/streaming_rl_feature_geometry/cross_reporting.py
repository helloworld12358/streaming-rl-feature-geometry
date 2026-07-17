"""Matplotlib-only figures derived from saved cross-environment metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "observation_only": "#7f7f7f",
    "oracle": "#d62728",
    "raw": "#1f77b4",
    "rms_raw": "#8c564b",
    "standardized": "#e377c2",
    "decorrelated": "#17becf",
    "whitened": "#ff7f0e",
    "gaussian_moment": "#9467bd",
    "unit_sphere": "#bcbd22",
    "sparse": "#2ca02c",
    "bounded": "#aec7e8",
    "matched": "#111111",
}
ENVIRONMENTS = (
    "tmaze",
    "ringworld",
    "two_loop",
    "hidden_velocity",
    "hidden_velocity_informative",
)


def _environment_axes(figsize: tuple[float, float] = (15, 8)):
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=False)
    axes = axes.ravel()
    for ax in axes[len(ENVIRONMENTS) :]:
        ax.set_visible(False)
    return fig, axes


def _finish(
    fig: plt.Figure,
    path: Path,
    *,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _mean_sem(frame: pd.DataFrame, by: list[str], value: str) -> pd.DataFrame:
    grouped = frame.groupby(by, sort=False)[value].agg(["mean", "sem", "count"]).reset_index()
    grouped["sem"] = grouped["sem"].fillna(0.0)
    return grouped


def _faceted_learning_curves(steps: pd.DataFrame, figures: Path) -> None:
    fig, axes = _environment_axes()
    for ax, environment in zip(axes, ENVIRONMENTS):
        subset = steps[steps.environment == environment]
        grouped = _mean_sem(subset, ["condition", "t"], "moving_performance")
        for condition, values in grouped.groupby("condition", sort=False):
            ax.plot(values["t"], values["mean"], label=condition, color=PALETTE.get(condition))
            ax.fill_between(
                values["t"],
                values["mean"] - values["sem"],
                values["mean"] + values["sem"],
                color=PALETTE.get(condition),
                alpha=0.12,
            )
        ax.set_title(environment)
        ax.set_xlabel("Interaction")
        ax.set_ylabel("Moving reward" if environment.startswith("hidden_velocity") else "Moving accuracy")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=min(6, max(1, len(labels))),
    )
    fig.suptitle("Cross-environment control learning curves (mean ± SEM)", y=1.0)
    _finish(
        fig,
        figures / "control_learning_curves_by_environment.png",
        rect=(0.0, 0.13, 1.0, 0.96),
    )


def _faceted_bars(summary: pd.DataFrame, figures: Path, conditions: set[str] | None, name: str) -> None:
    fig, axes = _environment_axes()
    for ax, environment in zip(axes, ENVIRONMENTS):
        subset = summary[summary.environment == environment]
        if conditions is not None:
            subset = subset[subset.condition.isin(conditions)]
        grouped = _mean_sem(subset, ["condition"], "final_performance")
        x = np.arange(len(grouped))
        ax.bar(
            x,
            grouped["mean"],
            yerr=grouped["sem"],
            capsize=3,
            color=[PALETTE.get(value, "#777777") for value in grouped.condition],
            edgecolor="#333333",
            linewidth=0.5,
        )
        ax.set_xticks(x, grouped.condition, rotation=40, ha="right")
        ax.set_title(f"{environment} (n={int(grouped['count'].max()) if len(grouped) else 0} seeds)")
        ax.set_ylabel("Final reward" if environment.startswith("hidden_velocity") else "Final accuracy")
        ax.grid(axis="y", alpha=0.25)
    title = "Task-matched prior versus generic priors" if conditions else "Final performance by representation and environment"
    fig.suptitle(title, y=1.02)
    _finish(fig, figures / name)


def _scatter_by_metric(joined: pd.DataFrame, metric: str, path: Path, xlabel: str) -> None:
    fig, axes = _environment_axes()
    for ax, environment in zip(axes, ENVIRONMENTS):
        subset = joined[joined.environment == environment]
        for condition, values in subset.groupby("condition", sort=False):
            ax.scatter(
                values[metric],
                values.final_performance,
                color=PALETTE.get(condition, "#777777"),
                label=condition,
                alpha=0.8,
            )
        ax.set_title(environment)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Final performance")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=min(6, max(1, len(labels))),
    )
    _finish(fig, path, rect=(0.0, 0.12, 1.0, 1.0))


def _moment_relationship(joined: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(
        axes,
        ("skewness_error", "kurtosis_error"),
        ("Skewness error", "Kurtosis error"),
    ):
        for condition, values in joined.groupby("condition", sort=False):
            ax.scatter(
                values[metric], values.final_performance,
                color=PALETTE.get(condition, "#777777"), label=condition, alpha=0.75,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Final performance")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05),
        ncol=min(6, len(labels)),
    )
    fig.suptitle("Moment error versus control performance", y=1.0)
    _finish(
        fig,
        figures / "moment_error_vs_control_performance.png",
        rect=(0.0, 0.13, 1.0, 0.96),
    )


def _update_stability(updates: pd.DataFrame, figures: Path) -> None:
    grouped = _mean_sem(updates, ["environment", "condition"], "control_update_norm")
    fig, axes = _environment_axes()
    for ax, environment in zip(axes, ENVIRONMENTS):
        values = grouped[grouped.environment == environment]
        x = np.arange(len(values))
        ax.bar(x, values["mean"], yerr=values["sem"], capsize=3, color="#4c78a8")
        ax.set_xticks(x, values.condition, rotation=40, ha="right")
        ax.set_title(environment)
        ax.set_ylabel("Control update norm")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Online update-norm stability (mean ± SEM)", y=1.02)
    _finish(fig, figures / "update_norm_stability.png")


def _eigen_spectra(representation: pd.DataFrame, figures: Path) -> None:
    columns = sorted(column for column in representation if column.startswith("eigenvalue_"))
    fig, axes = _environment_axes()
    for ax, environment in zip(axes, ENVIRONMENTS):
        subset = representation[representation.environment == environment]
        for condition, values in subset.groupby("condition", sort=False):
            spectrum = values[columns].mean().dropna().to_numpy(dtype=float)
            if len(spectrum):
                ax.plot(
                    np.arange(1, len(spectrum) + 1),
                    np.maximum(spectrum, 1e-12),
                    marker="o", ms=3, label=condition, color=PALETTE.get(condition),
                )
        ax.set_yscale("log")
        ax.set_title(environment)
        ax.set_xlabel("Eigenvalue rank")
        ax.set_ylabel("Covariance eigenvalue")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=min(6, max(1, len(labels))),
    )
    fig.suptitle("Representation covariance eigenvalue spectra", y=1.0)
    _finish(
        fig,
        figures / "covariance_eigenvalue_spectra.png",
        rect=(0.0, 0.12, 1.0, 0.96),
    )


def _property_heatmap(representation: pd.DataFrame, figures: Path) -> None:
    pivot = representation.pivot_table(
        index="environment", columns="condition", values="isotropy_error", aggfunc="mean"
    ).reindex(ENVIRONMENTS)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="cividis")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", color="white")
    ax.set_title("Real-stream property fulfillment: isotropy error (lower is closer to I)")
    fig.colorbar(image, ax=ax, label="Isotropy error")
    _finish(fig, figures / "property_fulfillment_summary.png")


def _pca(features: np.ndarray) -> np.ndarray:
    centered = features - features.mean(axis=0)
    if centered.shape[1] == 1:
        return np.column_stack((centered[:, 0], np.zeros(len(centered))))
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return centered @ vectors[:2].T


def _load_matched(root: Path, environment: str) -> dict[str, np.ndarray] | None:
    paths = sorted((root / "runs" / environment / "matched").rglob("diagnostic_samples.npz"))
    if not paths:
        return None
    with np.load(paths[0]) as archive:
        return {key: archive[key] for key in archive.files}


def _environment_geometry(root: Path, figures: Path) -> None:
    payloads = {environment: _load_matched(root, environment) for environment in ENVIRONMENTS}
    fig, axes = _environment_axes((15, 9))
    for ax, environment in zip(axes, ENVIRONMENTS):
        payload = payloads[environment]
        if payload is None or not len(payload["features"]):
            ax.set_visible(False)
            continue
        if environment in {"tmaze", "ringworld", "two_loop"}:
            projection = payload["features"][:, :2]
            x_label, y_label = "Matched coordinate 1", "Matched coordinate 2"
        else:
            projection = _pca(payload["features"])
            x_label, y_label = "Representation PC1", "Representation PC2"
        if environment == "tmaze":
            color = payload["latents"][:, 0]
            cmap = "coolwarm"
            color_label = "True cue (diagnostic only)"
        elif environment == "two_loop":
            color = payload["latents"][:, 0]
            cmap = "coolwarm"
            color_label = "True loop identity (diagnostic only)"
        else:
            color = payload["latents"][:, -1]
            cmap = "twilight"
            color_label = "True phase/velocity (diagnostic only)"
        scatter = ax.scatter(
            projection[:, 0], projection[:, 1], c=color, cmap=cmap, s=8, alpha=0.6
        )
        ax.set_title(environment)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(scatter, ax=ax, fraction=0.046, label=color_label)
    fig.suptitle("Environment-specific latent geometry in matched representations", y=1.01)
    _finish(fig, figures / "environment_specific_latent_geometry.png")

    tmaze = payloads["tmaze"]
    if tmaze is not None:
        projection = tmaze["features"][:, :2]
        fig, ax = plt.subplots(figsize=(7, 5))
        cue = tmaze["latents"][:, 0]
        ax.scatter(projection[cue < 0, 0], projection[cue < 0, 1], s=10, alpha=0.5, label="left")
        ax.scatter(projection[cue > 0, 0], projection[cue > 0, 1], s=10, alpha=0.5, label="right")
        ax.set(
            title="E1 cue clusters from held-out diagnostic samples",
            xlabel="Simplex coordinate 1",
            ylabel="Simplex coordinate 2",
        )
        ax.legend()
        _finish(fig, figures / "e1_cue_clusters.png")

    ring = payloads["ringworld"]
    if ring is not None:
        values = ring["features"][:, :2]
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(values[:, 0], values[:, 1], c=ring["latents"][:, 1], cmap="twilight", s=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set(title="E2 circular phase representation", xlabel="Circular block 1", ylabel="Circular block 2")
        fig.colorbar(scatter, ax=ax, label="True phase (diagnostic only)")
        _finish(fig, figures / "e2_circular_phase.png")

    loops = payloads["two_loop"]
    if loops is not None and loops["features"].shape[1] >= 4:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        identity = loops["latents"][:, 0]
        axes[0].scatter(loops["features"][:, 0], loops["features"][:, 1], c=identity, cmap="coolwarm", s=9)
        axes[0].set(title="Identity simplex block", xlabel="p(loop 0)", ylabel="p(loop 1)")
        scatter = axes[1].scatter(loops["features"][:, 2], loops["features"][:, 3], c=loops["latents"][:, 2], cmap="twilight", s=9)
        axes[1].set(title="Phase circle block", xlabel="phase 1", ylabel="phase 2")
        fig.colorbar(scatter, ax=axes[1], label="True phase (diagnostic only)")
        _finish(fig, figures / "e3_block_representation.png")

    velocity = payloads["hidden_velocity"]
    if velocity is not None and "true_velocity" in velocity:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(velocity["true_velocity"], velocity["predicted_velocity"], s=10, alpha=0.55)
        limits = [
            min(velocity["true_velocity"].min(), velocity["predicted_velocity"].min()),
            max(velocity["true_velocity"].max(), velocity["predicted_velocity"].max()),
        ]
        ax.plot(limits, limits, "--", color="#333333", label="ideal")
        ax.set(title="E4 held-out hidden-velocity prediction", xlabel="True velocity", ylabel="Linear probe prediction")
        ax.legend()
        _finish(fig, figures / "e4_hidden_velocity_prediction.png")


def _nonstationary_adaptation(steps: pd.DataFrame, figures: Path) -> None:
    """Render the preregistered E1 adaptation plot only from an executed change."""

    if "post_change" not in steps or not (steps["post_change"] > 0).any():
        return
    subset = steps[steps.environment == "tmaze"]
    grouped = _mean_sem(subset, ["condition", "t"], "moving_performance")
    change_point = float(subset.loc[subset["post_change"] > 0, "t"].min())
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for condition, values in grouped.groupby("condition", sort=False):
        ax.plot(values["t"], values["mean"], label=condition, color=PALETTE.get(condition))
    ax.axvline(change_point, linestyle="--", color="#333333", label="corridor change")
    ax.set(
        title="E1 adaptation after the remote-only corridor-length change",
        xlabel="Interaction",
        ylabel="Moving accuracy",
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    _finish(fig, figures / "remote_nonstationary_adaptation.png")


def _condition_boxplot(
    frame: pd.DataFrame, metric: str, figures: Path, filename: str, title: str
) -> None:
    usable = frame.dropna(subset=[metric]) if metric in frame else pd.DataFrame()
    if usable.empty:
        return
    labels = list(dict.fromkeys(usable["condition"]))
    values = [usable.loc[usable.condition == label, metric].to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(labels)), 5.5))
    # Keep the declared Matplotlib >=3.7 API surface: tick_labels was added
    # later, while setting ticks separately works across the supported range.
    ax.boxplot(values, showfliers=True)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    for index, data in enumerate(values, start=1):
        jitter = np.linspace(-0.12, 0.12, len(data)) if len(data) else np.empty(0)
        ax.scatter(index + jitter, data, s=18, alpha=0.65, color="#1f77b4")
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " "))
    ax.tick_params(axis="x", rotation=40)
    ax.grid(axis="y", alpha=0.25)
    _finish(fig, figures / filename)


def _hidden_velocity_figures(
    summaries: pd.DataFrame, task: pd.DataFrame, figures: Path
) -> None:
    hidden = summaries[summaries.environment.isin({"hidden_velocity", "hidden_velocity_informative"})]
    if hidden.empty:
        return
    _condition_boxplot(
        hidden, "final_window_reward", figures, "hidden_velocity_final_reward_distribution.png",
        "Hidden velocity final reward distribution",
    )
    _condition_boxplot(
        hidden, "final_window_stabilization_rate", figures,
        "hidden_velocity_stabilization_distribution.png",
        "Hidden velocity final-window stabilization rate",
    )
    _condition_boxplot(
        hidden, "final_window_position_rmse", figures, "hidden_velocity_position_rmse.png",
        "Hidden velocity final-window position RMSE",
    )
    _condition_boxplot(
        hidden, "final_window_velocity_rmse", figures, "hidden_velocity_velocity_rmse.png",
        "Hidden velocity final-window velocity RMSE",
    )
    _condition_boxplot(
        hidden, "final_window_control_effort", figures, "hidden_velocity_control_effort.png",
        "Hidden velocity final-window control effort",
    )
    cost_columns = [
        "final_window_position_cost",
        "final_window_velocity_cost",
        "final_window_action_cost",
    ]
    if set(cost_columns) <= set(hidden):
        cost = hidden.groupby("condition", sort=False)[cost_columns].median()
        fig, ax = plt.subplots(figsize=(12, 5.5))
        cost.plot(kind="bar", ax=ax)
        ax.set(title="Hidden velocity median cost decomposition", ylabel="Cost per step")
        ax.grid(axis="y", alpha=0.25)
        _finish(fig, figures / "hidden_velocity_cost_components.png")
    robust_rows = []
    for condition, frame in hidden.groupby("condition", sort=False):
        values = frame["final_window_reward"].dropna().to_numpy()
        if len(values):
            from .robust_stats import interquartile_mean

            robust_rows.append(
                (condition, np.median(values), interquartile_mean(values), np.quantile(values, 0.1))
            )
    if robust_rows:
        robust = pd.DataFrame(robust_rows, columns=["condition", "median", "IQM", "q10"])
        robust.set_index("condition").plot(kind="bar", figsize=(12, 5.5))
        fig = plt.gcf()
        ax = plt.gca()
        ax.set(title="Robust final-reward summaries", ylabel="Final-window reward")
        ax.grid(axis="y", alpha=0.25)
        _finish(fig, figures / "hidden_velocity_robust_reward_summary.png")
    if "catastrophic_failure" in hidden:
        failure = hidden.groupby("condition", sort=False)["catastrophic_failure"].mean()
        fig, ax = plt.subplots(figsize=(11, 5))
        failure.plot(kind="bar", ax=ax, color="#d62728")
        ax.set(title="Catastrophic failure rate", ylabel="Failure rate", ylim=(0, 1))
        ax.grid(axis="y", alpha=0.25)
        _finish(fig, figures / "hidden_velocity_catastrophic_failure_rate.png")
    baselines = hidden[
        hidden.condition.isin({"observation_only", "oracle", "raw", "whitened", "matched"})
    ]
    _condition_boxplot(
        baselines, "final_window_reward", figures, "hidden_velocity_baseline_comparison.png",
        "Oracle, observation-only, and predictive conditions",
    )
    join_keys = ["environment", "condition", "seed"]
    joined = hidden.merge(task, on=join_keys, suffixes=("", "_task"))
    if "velocity_decoding_r2" in joined:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(joined.velocity_decoding_r2, joined.final_window_reward, alpha=0.7)
        ax.set(xlabel="Velocity decoding R2", ylabel="Final-window reward",
               title="Velocity decoding versus control")
        ax.grid(alpha=0.25)
        _finish(fig, figures / "hidden_velocity_reward_vs_velocity_decoding.png")
    if "mean_control_update_norm" in hidden:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(hidden.mean_control_update_norm, hidden.final_window_reward, alpha=0.7)
        ax.set(xlabel="Mean control update norm", ylabel="Final-window reward",
               title="Control update norm versus reward")
        ax.grid(alpha=0.25)
        _finish(fig, figures / "hidden_velocity_reward_vs_update_norm.png")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for condition, frame in hidden.groupby("condition", sort=False):
        values = np.sort(frame["final_window_reward"].dropna().to_numpy())
        if len(values):
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=condition)
    ax.set(title="Seed-level final reward ECDF", xlabel="Final-window reward", ylabel="ECDF")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    _finish(fig, figures / "hidden_velocity_seed_ecdf.png")


def _decision_probe_figures(root: Path, summaries: pd.DataFrame, task: pd.DataFrame, figures: Path) -> None:
    by_position_path = root / "aggregate_decision_probe_by_position.csv"
    if by_position_path.exists() and by_position_path.stat().st_size > 1:
        try:
            by_position = pd.read_csv(by_position_path)
        except pd.errors.EmptyDataError:
            by_position = pd.DataFrame()
        if len(by_position) and "cue_decodability" in by_position:
            grouped = by_position.groupby(["condition", "corridor_position"], sort=False)[
                "cue_decodability"
            ].mean().reset_index()
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for condition, frame in grouped.groupby("condition", sort=False):
                ax.plot(frame.corridor_position, frame.cue_decodability, marker="o", label=condition)
            ax.set(title="T-maze cue information along the corridor", xlabel="Corridor position",
                   ylabel="Held-out cue decodability")
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            _finish(fig, figures / "tmaze_decodability_by_corridor_position.png")
    specifications = [
        ("tmaze", "cue_decodability", "cue_decodability_at_decision", "tmaze_full_vs_decision_probe.png"),
        ("two_loop", "identity_decodability", "identity_decodability_at_decision", "two_loop_identity_full_vs_decision.png"),
        ("two_loop", "phase_r2", "phase_r2_at_decision", "two_loop_phase_full_vs_decision.png"),
    ]
    for environment, full_metric, decision_metric, filename in specifications:
        if full_metric not in task or decision_metric not in task:
            continue
        frame = task[task.environment == environment].dropna(subset=[full_metric, decision_metric])
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(frame[full_metric], frame[decision_metric], alpha=0.7)
        ax.set(title=f"{environment}: full-stream versus decision-time probe",
               xlabel=full_metric, ylabel=decision_metric)
        ax.grid(alpha=0.25)
        _finish(fig, figures / filename)
    joined = summaries.merge(task, on=["environment", "condition", "seed"], suffixes=("", "_task"))
    metric = np.where(
        joined.environment == "tmaze",
        joined.get("cue_decodability_at_decision", np.nan),
        joined.get("identity_decodability_at_decision", np.nan),
    )
    valid = np.isfinite(metric)
    if valid.any():
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(np.asarray(metric)[valid], joined.loc[valid, "final_performance"], alpha=0.7)
        ax.set(title="Decision-time decodability versus final performance",
               xlabel="Decision-time decodability", ylabel="Final performance")
        ax.grid(alpha=0.25)
        _finish(fig, figures / "decision_decodability_vs_performance.png")
    if "decision_probe_sample_count" in task:
        _condition_boxplot(
            task, "decision_probe_sample_count", figures, "decision_probe_sample_count.png",
            "Decision-conditioned probe sample counts",
        )


def _alpha_tuning_figures(root: Path, summaries: pd.DataFrame, figures: Path) -> None:
    """Plot tuning candidates without pooling distinct controller alphas."""

    selected_path = root / "selected_learning_rates.csv"
    selected = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()
    fig, axes = _environment_axes((16, 9))
    for ax, environment in zip(axes, ENVIRONMENTS):
        subset = summaries[summaries.environment == environment]
        for condition, frame in subset.groupby("condition", sort=False):
            grouped = frame.groupby("candidate_alpha", sort=True)["final_performance"].median()
            ax.plot(grouped.index, grouped.values, marker="o", label=condition)
            if len(selected):
                chosen = selected[
                    (selected.environment == environment) & (selected.condition == condition)
                ]
                if len(chosen):
                    alpha = float(chosen.iloc[0].selected_alpha)
                    ax.axvline(alpha, color=PALETTE.get(condition, "#777777"), alpha=0.12)
        ax.set_xscale("log")
        ax.set(title=environment, xlabel="Candidate controller alpha", ylabel="Median final performance")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, max(1, len(labels))))
    fig.suptitle("Controller-alpha tuning on tuning seeds only")
    _finish(fig, figures / "alpha_tuning_performance.png", rect=(0.0, 0.10, 1.0, 0.97))

    failures = summaries.groupby(
        ["environment", "condition", "candidate_alpha"], sort=False
    )["catastrophic_failure"].mean().reset_index()
    fig, axes = _environment_axes((16, 9))
    for ax, environment in zip(axes, ENVIRONMENTS):
        subset = failures[failures.environment == environment]
        for condition, frame in subset.groupby("condition", sort=False):
            frame = frame.sort_values("candidate_alpha")
            ax.plot(frame.candidate_alpha, frame.catastrophic_failure, marker="o", label=condition)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set(title=environment, xlabel="Candidate controller alpha", ylabel="Failure rate")
        ax.grid(alpha=0.25)
    _finish(fig, figures / "alpha_tuning_failure_rate.png")


def make_cross_figures(
    root: str | Path,
    summaries: pd.DataFrame,
    representation: pd.DataFrame,
    task: pd.DataFrame,
    steps: pd.DataFrame,
    decisions: pd.DataFrame,
    updates: pd.DataFrame,
) -> None:
    root = Path(root)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    if "candidate_alpha" in summaries and not summaries["candidate_alpha"].isna().all():
        _alpha_tuning_figures(root, summaries, figures)
        return
    _faceted_learning_curves(steps, figures)
    _faceted_bars(summaries, figures, None, "final_performance_by_environment.png")
    _faceted_bars(
        summaries, figures, {"raw", "whitened", "gaussian_moment", "matched"},
        "task_matched_vs_generic.png",
    )
    join_keys = ["environment", "condition", "seed"]
    if all(
        "candidate_alpha" in frame and not frame["candidate_alpha"].isna().all()
        for frame in (summaries, representation, task)
    ):
        join_keys.append("candidate_alpha")
    joined = summaries.merge(
        representation,
        on=join_keys,
        suffixes=("", "_representation"),
    ).merge(task, on=join_keys, suffixes=("", "_task"))
    _scatter_by_metric(joined, "effective_rank", figures / "effective_rank_vs_control.png", "Effective rank")
    _scatter_by_metric(joined, "isotropy_error", figures / "isotropy_vs_control.png", "Isotropy error")
    _scatter_by_metric(joined, "task_decodability", figures / "decodability_vs_control.png", "Held-out task decodability")
    _moment_relationship(joined, figures)
    _update_stability(updates, figures)
    _eigen_spectra(representation, figures)
    _property_heatmap(representation, figures)
    _environment_geometry(root, figures)
    _nonstationary_adaptation(steps, figures)
    _hidden_velocity_figures(summaries, task, figures)
    _decision_probe_figures(root, summaries, task, figures)
