"""Aggregate experiment tables into real, source-backed matplotlib figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_bar(frame: pd.DataFrame, column: str, path: Path, ylabel: str) -> None:
    grouped = frame.groupby("condition", sort=False)[column]
    mean = grouped.mean()
    error = grouped.sem().fillna(0.0)
    counts = grouped.count()
    fig, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(mean.index, mean.values, yerr=error.values, capsize=3)
    for bar, count in zip(bars, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f" n={count}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _binned_seed_curve(
    frame: pd.DataFrame, x: str, y: str, bins: int = 120
) -> pd.DataFrame:
    maximum = max(int(frame[x].max()), 1)
    bin_size = max(1, maximum // bins)
    work = frame.copy()
    work["plot_bin"] = (work[x] // bin_size) * bin_size
    per_seed = work.groupby(["condition", "seed", "plot_bin"], as_index=False)[y].mean()
    return per_seed.groupby(["condition", "plot_bin"])[y].agg(["mean", "sem"]).reset_index()


def _save_curve(
    curve: pd.DataFrame,
    path: Path,
    xlabel: str,
    ylabel: str,
    chance_line: float | None = None,
    vertical_line: float | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for condition, part in curve.groupby("condition", sort=False):
        x = part["plot_bin"].to_numpy(dtype=float)
        mean = part["mean"].to_numpy(dtype=float)
        error = part["sem"].fillna(0.0).to_numpy(dtype=float)
        axis.plot(x, mean, label=condition)
        axis.fill_between(x, mean - error, mean + error, alpha=0.15)
    if chance_line is not None:
        axis.axhline(chance_line, color="black", linestyle="--", linewidth=1, alpha=0.6)
    if vertical_line is not None:
        axis.axvline(vertical_line, color="black", linestyle=":", linewidth=1.3)
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.legend(ncol=2, fontsize=7)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_figures(
    root: Path,
    summaries: pd.DataFrame,
    representation: pd.DataFrame,
    hidden: pd.DataFrame,
    trials: pd.DataFrame,
    predictions: pd.DataFrame,
    updates: pd.DataFrame,
) -> None:
    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    _save_bar(
        summaries,
        "final_window_accuracy",
        figure_dir / "final_accuracy.png",
        "Final-window trial accuracy",
    )
    _save_bar(
        summaries,
        "cumulative_reward",
        figure_dir / "cumulative_reward.png",
        "Cumulative reward",
    )
    _save_bar(
        summaries,
        "mean_update_norm",
        figure_dir / "update_norm.png",
        "Mean controller update norm",
    )

    if not trials.empty:
        learning_curve = _binned_seed_curve(trials, "trial", "moving_accuracy")
        _save_curve(
            learning_curve,
            figure_dir / "trial_accuracy_learning_curve.png",
            "Trial",
            "Moving trial accuracy (mean +/- SE)",
            chance_line=0.5,
        )
        if trials["corridor_length"].nunique() > 1:
            adaptation = _binned_seed_curve(trials, "t", "moving_accuracy")
            change_point = float(
                trials.loc[
                    trials["corridor_length"] == trials["corridor_length"].max(), "t"
                ].min()
            )
            _save_curve(
                adaptation,
                figure_dir / "nonstationary_adaptation.png",
                "Interaction",
                "Moving trial accuracy (mean +/- SE)",
                chance_line=0.5,
                vertical_line=change_point,
            )

    if not predictions.empty:
        prediction_curve = _binned_seed_curve(predictions, "t", "gvf_td_mse")
        _save_curve(
            prediction_curve,
            figure_dir / "gvf_prediction_error.png",
            "Interaction",
            "GVF mean squared TD error",
        )
        plt.close("all")

    if not hidden.empty:
        position_order = sorted(
            hidden["position"].unique(),
            key=lambda value: (value == "junction", value),
        )
        fig, axis = plt.subplots(figsize=(9, 4.8))
        for condition, part in hidden.groupby("condition", sort=False):
            position_stats = part.groupby("position")["cue_decodability"].agg(["mean", "sem"])
            means = [position_stats.loc[position, "mean"] for position in position_order]
            errors = [
                0.0
                if pd.isna(position_stats.loc[position, "sem"])
                else position_stats.loc[position, "sem"]
                for position in position_order
            ]
            x = np.arange(len(position_order))
            axis.plot(x, means, marker="o", label=condition)
            axis.fill_between(x, np.asarray(means) - errors, np.asarray(means) + errors, alpha=0.12)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(np.arange(len(position_order)), position_order, rotation=30)
        axis.set(ylabel="Held-out cue balanced accuracy", xlabel="Decision position")
        axis.legend(ncol=2, fontsize=7)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "cue_decodability_by_position.png", dpi=160)
        plt.close(fig)

    if not representation.empty:
        for column, label in (
            ("condition_number", "Regularized covariance condition number"),
            ("effective_rank", "Effective rank"),
            ("isotropy_error", "Covariance isotropy error"),
            ("mean_abs_corr", "Mean absolute off-diagonal correlation"),
            ("skewness_error", "Mean absolute skewness"),
            ("kurtosis_error", "Mean absolute kurtosis error"),
            ("norm_mean", "Mean feature norm"),
        ):
            _save_bar(representation, column, figure_dir / f"{column}.png", label)

        eigen_columns = sorted(
            (column for column in representation.columns if column.startswith("eigenvalue_")),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if eigen_columns:
            fig, axis = plt.subplots(figsize=(9, 4.8))
            for condition, part in representation.groupby("condition", sort=False):
                spectrum = part[eigen_columns].mean().to_numpy(dtype=float)
                axis.plot(
                    np.arange(1, len(spectrum) + 1),
                    np.maximum(spectrum, 1e-12),
                    marker=".",
                    label=condition,
                )
            axis.set(xlabel="Eigenvalue rank", ylabel="Covariance eigenvalue", yscale="log")
            axis.legend(ncol=2, fontsize=7)
            axis.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(figure_dir / "covariance_eigenvalue_spectrum.png", dpi=160)
            plt.close(fig)

        junction = hidden.query("position == 'junction'") if not hidden.empty else pd.DataFrame()
        cue_metric = (
            junction[["condition", "seed", "cue_decodability"]]
            if not junction.empty
            else pd.DataFrame(columns=["condition", "seed", "cue_decodability"])
        )
        merged = summaries.merge(representation, on=["condition", "seed"], how="inner")
        merged = merged.merge(cue_metric, on=["condition", "seed"], how="left")
        for column, label in (
            ("isotropy_error", "Isotropy error"),
            ("effective_rank", "Effective rank"),
            ("cue_decodability", "Junction cue decodability"),
        ):
            if column not in merged or merged[column].isna().all():
                continue
            fig, axis = plt.subplots(figsize=(6.4, 4.8))
            for condition, part in merged.groupby("condition", sort=False):
                axis.scatter(part[column], part["final_window_accuracy"], label=condition)
            axis.set(xlabel=label, ylabel="Final-window trial accuracy")
            axis.legend(fontsize=7)
            axis.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(figure_dir / f"accuracy_vs_{column}.png", dpi=160)
            plt.close(fig)

    _make_cue_projection_and_histograms(root, figure_dir)


def _make_cue_projection_and_histograms(root: Path, figure_dir: Path) -> None:
    sample_paths = sorted((root / "runs").glob("*/seed_*/junction_features.npz"))
    if not sample_paths:
        return
    conditions: list[str] = []
    for path in sample_paths:
        condition = path.parents[1].name
        if condition not in conditions:
            conditions.append(condition)
    columns = min(3, len(conditions))
    rows = int(np.ceil(len(conditions) / columns))
    projection_fig, projection_axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.6 * rows), squeeze=False
    )
    histogram_fig, histogram_axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.6 * rows), squeeze=False
    )
    for projection_axis, histogram_axis, condition in zip(
        projection_axes.flat, histogram_axes.flat, conditions
    ):
        path = next(item for item in sample_paths if item.parents[1].name == condition)
        payload = np.load(path)
        features = payload["features"]
        cues = payload["cues"]
        if len(features) >= 3 and features.shape[1] > 0:
            centered = features - features.mean(axis=0)
            _, _, right = np.linalg.svd(centered, full_matrices=False)
            projection = centered @ right[:2].T
            if projection.shape[1] == 1:
                projection = np.column_stack((projection, np.zeros(len(projection))))
            for cue, label, color in (
                (-1, "left cue", "tab:blue"),
                (1, "right cue", "tab:orange"),
            ):
                mask = cues == cue
                projection_axis.scatter(
                    projection[mask, 0], projection[mask, 1], s=10, alpha=0.45, label=label, color=color
                )
                histogram_axis.hist(
                    projection[mask, 0], bins=20, alpha=0.45, density=True, label=label, color=color
                )
        projection_axis.set_title(condition)
        projection_axis.set(xlabel="PC1", ylabel="PC2")
        projection_axis.grid(alpha=0.2)
        histogram_axis.set_title(condition)
        histogram_axis.set(xlabel="PC1", ylabel="Density")
        histogram_axis.grid(alpha=0.2)
    for axes in (projection_axes, histogram_axes):
        for axis in axes.flat[len(conditions) :]:
            axis.set_visible(False)
        axes.flat[0].legend(fontsize=7)
    projection_fig.tight_layout()
    projection_fig.savefig(figure_dir / "cue_conditioned_feature_projection.png", dpi=160)
    plt.close(projection_fig)
    histogram_fig.tight_layout()
    histogram_fig.savefig(figure_dir / "cue_conditioned_feature_histograms.png", dpi=160)
    plt.close(histogram_fig)
