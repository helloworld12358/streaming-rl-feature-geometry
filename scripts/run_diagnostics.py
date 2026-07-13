"""Generate synthetic property-fulfillment evidence under results/diagnostics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from streaming_rl_feature_geometry.metrics import cue_probe_score
from streaming_rl_feature_geometry.transforms import FeatureTransform, rep_metrics


def run_transform(kind: str, data: np.ndarray) -> tuple[np.ndarray, FeatureTransform]:
    transform = FeatureTransform(
        kind,
        data.shape[1],
        eps=1e-3,
        update_every=25,
        min_samples=96,
        moment_beta=0.005,
        moment_learning_rate=0.0005,
    )
    output = np.vstack([transform.transform(row) for row in data])
    return output, transform


def metric_row(
    method: str,
    stream: str,
    data: np.ndarray,
    output: np.ndarray,
    burn_in: int,
) -> dict[str, object]:
    before = rep_metrics(data[burn_in:])
    after = rep_metrics(output[burn_in:])
    row: dict[str, object] = {"method": method, "stream": stream}
    for name in (
        "mean_error",
        "variance_mean",
        "variance_imbalance",
        "mean_abs_corr",
        "effective_rank",
        "isotropy_error",
        "condition_number",
        "skewness_error",
        "kurtosis_error",
        "norm_mean",
        "norm_var",
        "norm_max",
        "near_zero_fraction",
    ):
        row[f"before_{name}"] = before[name]
        row[f"after_{name}"] = after[name]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name")
    arguments = parser.parse_args()
    run_name = arguments.run_name or datetime.now(timezone.utc).strftime(
        "diagnostics-%Y%m%dT%H%M%SZ"
    )
    root = Path("results") / "diagnostics" / run_name
    root.mkdir(parents=True, exist_ok=False)
    figure_dir = root / "figures"
    figure_dir.mkdir()

    rng = np.random.default_rng(20260713)
    n = 12000
    correlated_base = rng.normal(size=(n, 3))
    correlated = correlated_base @ np.asarray(
        [[3.0, 1.4, 0.8], [0.0, 0.35, 0.2], [0.0, 0.0, 0.08]]
    )
    shifted_scaled = rng.normal(loc=[4.0, -2.0, 1.0], scale=[5.0, 0.3, 2.0], size=(n, 3))
    repeated_base = rng.normal(size=n)
    repeated = np.column_stack((repeated_base, repeated_base, np.ones(n)))
    constant = np.ones((n, 3)) * np.asarray([2.0, -1.0, 0.5])
    gaussian_streams = {
        "normal": rng.normal(size=(n, 1)),
        "positive_skew": (rng.exponential(size=n) - 1.0)[:, None],
        "negative_skew": (1.0 - rng.exponential(size=n))[:, None],
        "heavy_tail": (rng.standard_t(3, size=n) / np.sqrt(3.0))[:, None],
    }
    bimodal_cues = np.repeat([-1, 1], n // 2)
    bimodal = np.concatenate(
        (rng.normal(-2.0, 0.35, n // 2), rng.normal(2.0, 0.35, n // 2))
    )
    permutation = rng.permutation(n)
    bimodal = bimodal[permutation, None]
    bimodal_cues = bimodal_cues[permutation]

    rows: list[dict[str, object]] = []
    cases = (
        ("raw", "correlated", correlated),
        ("rms_raw", "correlated", correlated),
        ("standardized", "shifted_scaled", shifted_scaled),
        ("decorrelated", "correlated", correlated),
        ("whitened", "correlated", correlated),
        ("whitened", "repeated_singular", repeated),
        ("whitened", "constant_singular", constant),
        ("unit_sphere", "correlated", correlated),
    )
    outputs: dict[tuple[str, str], np.ndarray] = {}
    for method, stream, data in cases:
        output, transform = run_transform(method, data)
        if not np.isfinite(output).all():
            raise AssertionError(f"non-finite diagnostic output for {method}/{stream}")
        row = metric_row(method, stream, data, output, burn_in=3000)
        row.update(transform.state_metrics())
        rows.append(row)
        outputs[(method, stream)] = output

    for stream, data in gaussian_streams.items():
        output, transform = run_transform("gaussian_moment", data)
        row = metric_row("gaussian_moment", stream, data, output, burn_in=3000)
        row.update(transform.state_metrics())
        rows.append(row)
        outputs[("gaussian_moment", stream)] = output
    bimodal_output, bimodal_transform = run_transform("gaussian_moment", bimodal)
    bimodal_row = metric_row(
        "gaussian_moment", "bimodal", bimodal, bimodal_output, burn_in=3000
    )
    before_probe = cue_probe_score(
        bimodal[3000:], bimodal_cues[3000:], seed=11
    )
    after_probe = cue_probe_score(
        bimodal_output[3000:], bimodal_cues[3000:], seed=11
    )
    bimodal_row.update(
        {
            "before_cue_decodability": before_probe,
            "after_cue_decodability": after_probe,
            **bimodal_transform.state_metrics(),
        }
    )
    rows.append(bimodal_row)
    outputs[("gaussian_moment", "bimodal")] = bimodal_output

    metrics = pd.DataFrame(rows)
    metrics.to_csv(root / "before_after_metrics.csv", index=False)
    checks = []

    def add_check(method: str, check: str, passed: bool, evidence: str) -> None:
        checks.append(
            {"method": method, "check": check, "passed": bool(passed), "evidence": evidence}
        )

    raw_row = metrics.query("method == 'raw'").iloc[0]
    add_check("raw", "identity", np.allclose(correlated, outputs[("raw", "correlated")]), "output equals input")
    rms = metrics.query("method == 'rms_raw'").iloc[0]
    add_check(
        "rms_raw",
        "unit global RMS without correlation change",
        0.8 <= np.sqrt(rms.after_variance_mean) <= 1.2
        and abs(rms.after_mean_abs_corr - rms.before_mean_abs_corr) < 0.03,
        f"rms={np.sqrt(rms.after_variance_mean):.3f}, corr_delta={rms.after_mean_abs_corr-rms.before_mean_abs_corr:.3f}",
    )
    standardized = metrics.query("method == 'standardized'").iloc[0]
    add_check(
        "standardized",
        "zero mean and unit marginal variance",
        standardized.after_mean_error < 0.25 and 0.8 <= standardized.after_variance_mean <= 1.2,
        f"mean_error={standardized.after_mean_error:.3f}, variance_mean={standardized.after_variance_mean:.3f}",
    )
    decorrelated = metrics.query("method == 'decorrelated'").iloc[0]
    add_check(
        "decorrelated",
        "lower off-diagonal correlation",
        decorrelated.after_mean_abs_corr < 0.5 * decorrelated.before_mean_abs_corr,
        f"before={decorrelated.before_mean_abs_corr:.3f}, after={decorrelated.after_mean_abs_corr:.3f}",
    )
    whitened = metrics.query("method == 'whitened' and stream == 'correlated'").iloc[0]
    add_check(
        "whitened",
        "lower isotropy error",
        whitened.after_isotropy_error < whitened.before_isotropy_error,
        f"before={whitened.before_isotropy_error:.3f}, after={whitened.after_isotropy_error:.3f}",
    )
    singular = metrics.query("method == 'whitened' and stream != 'correlated'")
    diagnostic_columns = [
        column
        for column in singular.columns
        if (column.startswith("before_") or column.startswith("after_"))
        and singular[column].notna().all()
    ]
    add_check(
        "whitened",
        "finite singular/repeated streams",
        np.isfinite(singular[diagnostic_columns].to_numpy(dtype=float)).all(),
        "all saved singular-stream metrics are finite",
    )
    sphere = metrics.query("method == 'unit_sphere'").iloc[0]
    add_check(
        "unit_sphere",
        "unit norm",
        0.98 <= sphere.after_norm_mean <= 1.01 and sphere.after_norm_var < 1e-3,
        f"norm_mean={sphere.after_norm_mean:.4f}, norm_var={sphere.after_norm_var:.6f}",
    )

    gaussian_rows = metrics.query("method == 'gaussian_moment'").set_index("stream")
    add_check(
        "gaussian_moment",
        "positive skew reduced",
        gaussian_rows.loc["positive_skew", "after_skewness_error"]
        < gaussian_rows.loc["positive_skew", "before_skewness_error"],
        "compare absolute skewness before/after",
    )
    add_check(
        "gaussian_moment",
        "negative skew reduced",
        gaussian_rows.loc["negative_skew", "after_skewness_error"]
        < gaussian_rows.loc["negative_skew", "before_skewness_error"],
        "compare absolute skewness before/after",
    )
    add_check(
        "gaussian_moment",
        "heavy-tail kurtosis error reduced",
        gaussian_rows.loc["heavy_tail", "after_kurtosis_error"]
        < gaussian_rows.loc["heavy_tail", "before_kurtosis_error"],
        "compare absolute kurtosis error before/after",
    )
    add_check(
        "gaussian_moment",
        "normal stream not materially damaged",
        gaussian_rows.loc["normal", "after_skewness_error"] < 0.25
        and gaussian_rows.loc["normal", "after_kurtosis_error"] < 0.75,
        "fixed preregistered tolerance",
    )
    add_check(
        "gaussian_moment",
        "bimodal cue separation retained",
        after_probe >= 0.95 and after_probe >= before_probe - 0.05,
        f"before={before_probe:.3f}, after={after_probe:.3f}",
    )
    summary = pd.DataFrame(checks)
    summary.to_csv(root / "pass_fail_summary.csv", index=False)

    plot_metrics(metrics, figure_dir)
    plot_gaussian_histograms(
        gaussian_streams,
        bimodal,
        outputs,
        figure_dir / "gaussian_synthetic_histograms.png",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": 20260713,
        "samples_per_stream": n,
        "burn_in": 3000,
        "all_checks_passed": bool(summary["passed"].all()),
        "checks_passed": int(summary["passed"].sum()),
        "checks_total": len(summary),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"RESULT_DIR={root.resolve()}")
    print(summary.to_string(index=False))
    if not summary["passed"].all():
        raise SystemExit("one or more property diagnostics failed; evidence was retained")


def plot_metrics(metrics: pd.DataFrame, figure_dir: Path) -> None:
    selected = metrics[
        (metrics["stream"].isin(["correlated", "shifted_scaled"]))
        & (~metrics["method"].isin(["raw"]))
    ]
    columns = ["mean_abs_corr", "isotropy_error", "skewness_error", "kurtosis_error"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    x = np.arange(len(selected))
    labels = [f"{method}\n{stream}" for method, stream in zip(selected.method, selected.stream)]
    for axis, column in zip(axes.flat, columns):
        axis.bar(x - 0.18, selected[f"before_{column}"], width=0.36, label="before")
        axis.bar(x + 0.18, selected[f"after_{column}"], width=0.36, label="after")
        axis.set_title(column)
        axis.set_xticks(x, labels, rotation=30, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.2)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "property_before_after.png", dpi=160)
    plt.close(fig)


def plot_gaussian_histograms(
    streams: dict[str, np.ndarray],
    bimodal: np.ndarray,
    outputs: dict[tuple[str, str], np.ndarray],
    path: Path,
) -> None:
    all_streams = {**streams, "bimodal": bimodal}
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for axis, (name, data) in zip(axes.flat, all_streams.items()):
        axis.hist(data[3000:, 0], bins=50, density=True, alpha=0.45, label="before")
        axis.hist(
            outputs[("gaussian_moment", name)][3000:, 0],
            bins=50,
            density=True,
            alpha=0.45,
            label="after",
        )
        axis.set_title(name)
        axis.grid(alpha=0.2)
    axes.flat[-1].set_visible(False)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
