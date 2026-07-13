"""Synthetic and real-stream property checks for the cross-environment extension."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from streaming_rl_feature_geometry.cross_env import ENVIRONMENT_IDS, make_environment
from streaming_rl_feature_geometry.experiment import git_value
from streaming_rl_feature_geometry.predictive import CausalPredictiveBank
from streaming_rl_feature_geometry.priors import MATCHED_PRIOR_FOR_ENV, TaskMatchedTransform
from streaming_rl_feature_geometry.transforms import FeatureTransform, rep_metrics


RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def apply_generic(kind: str, stream: np.ndarray) -> np.ndarray:
    transform = FeatureTransform(
        kind,
        stream.shape[1],
        min_samples=64,
        update_every=25,
        sparse_top_k=max(1, stream.shape[1] // 3),
        bounded_alpha=0.8,
    )
    return np.vstack([transform.transform(row) for row in stream])


def add_check(
    rows: list[dict[str, object]],
    method: str,
    stream: str,
    check: str,
    passed: bool,
    evidence: str,
) -> None:
    rows.append(
        {
            "method": method,
            "stream": stream,
            "check": check,
            "passed": bool(passed),
            "evidence": evidence,
        }
    )


def synthetic_diagnostics(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    n = 8000
    burn = 1500
    base = rng.normal(size=(n, 6))
    correlated = base @ np.asarray(
        [
            [3.0, 1.6, 0.8, 0.4, 0.2, 0.1],
            [0.0, 0.7, 0.3, 0.2, 0.1, 0.0],
            [0.0, 0.0, 0.4, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.2, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.04],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
        ]
    )
    shifted = rng.normal(
        loc=[4.0, -2.0, 1.0, 3.0, -4.0, 0.5],
        scale=[5.0, 0.3, 2.0, 1.5, 0.6, 3.0],
        size=(n, 6),
    )
    positive_skew = rng.exponential(size=(n, 1)) - 1.0
    heavy_tail = rng.standard_t(3, size=(n, 1)) / np.sqrt(3.0)
    cases = {
        "rms_raw": correlated,
        "standardized": shifted,
        "decorrelated": correlated,
        "whitened": correlated,
        "unit_sphere": correlated,
        "sparse": correlated,
        "bounded": shifted,
    }
    metrics_rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    outputs: dict[str, np.ndarray] = {}
    for kind, stream in cases.items():
        output = apply_generic(kind, stream)
        outputs[kind] = output
        before = rep_metrics(stream[burn:])
        after = rep_metrics(output[burn:])
        metrics_rows.append(
            {
                "method": kind,
                "stream": "correlated" if stream is correlated else "shifted_scaled",
                **{f"before_{key}": value for key, value in before.items()},
                **{f"after_{key}": value for key, value in after.items()},
            }
        )
        finite = np.isfinite(output).all()
        add_check(checks, kind, "synthetic", "finite output", finite, f"finite={finite}")

    raw = rep_metrics(correlated[burn:])
    rms = rep_metrics(outputs["rms_raw"][burn:])
    add_check(
        checks,
        "rms_raw",
        "synthetic",
        "global scale without correlation change",
        0.75 <= np.sqrt(rms["variance_mean"]) <= 1.25
        and abs(rms["mean_abs_corr"] - raw["mean_abs_corr"]) < 0.03,
        f"rms={np.sqrt(rms['variance_mean']):.4f}; corr_delta={rms['mean_abs_corr']-raw['mean_abs_corr']:.4f}",
    )
    standardized = rep_metrics(outputs["standardized"][burn:])
    add_check(
        checks,
        "standardized",
        "synthetic",
        "marginal zero mean and unit variance",
        standardized["mean_error"] < 0.3
        and 0.75 <= standardized["variance_mean"] <= 1.25,
        f"mean_error={standardized['mean_error']:.4f}; variance={standardized['variance_mean']:.4f}",
    )
    decorrelated = rep_metrics(outputs["decorrelated"][burn:])
    add_check(
        checks,
        "decorrelated",
        "synthetic",
        "lower off-diagonal correlation",
        decorrelated["mean_abs_corr"] < 0.5 * raw["mean_abs_corr"],
        f"before={raw['mean_abs_corr']:.4f}; after={decorrelated['mean_abs_corr']:.4f}",
    )
    whitened = rep_metrics(outputs["whitened"][burn:])
    add_check(
        checks,
        "whitened",
        "synthetic",
        "lower second-order isotropy error",
        whitened["isotropy_error"] < raw["isotropy_error"],
        f"before={raw['isotropy_error']:.4f}; after={whitened['isotropy_error']:.4f}",
    )
    sphere_norms = np.linalg.norm(outputs["unit_sphere"][burn:], axis=1)
    add_check(
        checks,
        "unit_sphere",
        "synthetic",
        "unit norm",
        np.max(np.abs(sphere_norms - 1.0)) < 0.01,
        f"mean_radius={sphere_norms.mean():.5f}",
    )
    sparse_active = np.mean(np.count_nonzero(outputs["sparse"][burn:], axis=1))
    add_check(
        checks,
        "sparse",
        "synthetic",
        "fixed top-k activity",
        abs(sparse_active - 2.0) < 1e-12,
        f"mean_active_dimensions={sparse_active:.3f}",
    )
    bounded_max = float(np.max(np.abs(outputs["bounded"][burn:])))
    add_check(
        checks,
        "bounded",
        "synthetic",
        "strict boundedness",
        bounded_max < 1.0,
        f"max_abs={bounded_max:.8f}",
    )

    gaussian_skew = apply_generic("gaussian_moment", positive_skew)
    gaussian_tail = apply_generic("gaussian_moment", heavy_tail)
    before_skew = rep_metrics(positive_skew[burn:])["skewness_error"]
    after_skew = rep_metrics(gaussian_skew[burn:])["skewness_error"]
    before_tail = rep_metrics(heavy_tail[burn:])["kurtosis_error"]
    after_tail = rep_metrics(gaussian_tail[burn:])["kurtosis_error"]
    add_check(
        checks,
        "gaussian_moment",
        "positive_skew",
        "skewness error reduced",
        after_skew < before_skew,
        f"before={before_skew:.4f}; after={after_skew:.4f}",
    )
    add_check(
        checks,
        "gaussian_moment",
        "heavy_tail",
        "kurtosis error reduced",
        after_tail < before_tail,
        f"before={before_tail:.4f}; after={after_tail:.4f}",
    )

    identity = rng.integers(0, 2, size=n)
    angles = rng.uniform(-np.pi, np.pi, size=n)
    semantic = np.column_stack(
        (
            np.where(identity == 0, 2.0, -2.0) + rng.normal(0, 0.2, n),
            np.where(identity == 1, 2.0, -2.0) + rng.normal(0, 0.2, n),
            np.cos(angles),
            np.sin(angles),
            rng.normal(size=(n, 2)),
        )
    )
    matched_outputs: dict[str, np.ndarray] = {}
    for kind in ("simplex", "circular", "block", "anisotropic"):
        transform = TaskMatchedTransform(kind, semantic.shape[1])
        matched_outputs[kind] = np.vstack([transform.transform(row) for row in semantic])
    simplex = matched_outputs["simplex"][burn:]
    add_check(
        checks,
        "simplex",
        "synthetic",
        "nonnegative normalized simplex",
        np.min(simplex) >= 0.0 and np.max(np.abs(simplex.sum(axis=1) - 1.0)) < 1e-12,
        f"min={simplex.min():.6f}; max_sum_error={np.max(np.abs(simplex.sum(axis=1)-1.0)):.3g}",
    )
    circle = matched_outputs["circular"][burn:]
    add_check(
        checks,
        "circular",
        "synthetic",
        "unit circular block",
        np.max(np.abs(np.linalg.norm(circle, axis=1) - 1.0)) < 1e-10,
        f"mean_radius={np.mean(np.linalg.norm(circle, axis=1)):.6f}",
    )
    phase_input = semantic.copy()
    phase_input[:, 0] = np.cos(angles)
    phase_input[:, 1] = np.sin(angles)
    phase_transform = TaskMatchedTransform("circular", semantic.shape[1])
    phase_output = np.vstack([phase_transform.transform(row) for row in phase_input])[burn:]
    predicted_angles = np.arctan2(phase_output[:, 1], phase_output[:, 0])
    phase_alignment = float(
        np.abs(np.mean(np.exp(1j * (predicted_angles - angles[burn:]))))
    )
    add_check(
        checks,
        "circular",
        "synthetic",
        "phase order preserved",
        phase_alignment > 0.95,
        f"circular_phase_alignment={phase_alignment:.6f}",
    )
    block = matched_outputs["block"][burn:]
    add_check(
        checks,
        "block",
        "synthetic",
        "simplex by circular block semantics",
        np.min(block[:, :2]) >= 0.0
        and np.max(np.abs(block[:, :2].sum(axis=1) - 1.0)) < 1e-12
        and np.max(np.abs(np.linalg.norm(block[:, 2:4], axis=1) - 1.0)) < 1e-10,
        "simplex and circular block constraints checked independently",
    )
    anisotropic = matched_outputs["anisotropic"][burn:]
    variances = np.diag(np.cov(anisotropic, rowvar=False))
    ratio = float(variances[0] / variances[1])
    add_check(
        checks,
        "anisotropic",
        "synthetic",
        "predeclared covariance ratio",
        7.0 <= ratio <= 11.0,
        f"variance_ratio={ratio:.4f}; target=9.0",
    )
    return pd.DataFrame(metrics_rows), pd.DataFrame(checks), matched_outputs


def real_stream_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    samples: dict[str, np.ndarray] = {}
    for env_index, environment in enumerate(ENVIRONMENT_IDS):
        env = make_environment(environment, seed=100 + env_index)
        bank = CausalPredictiveBank(
            environment,
            env.obs_dim,
            env.event_names,
            seed=200 + env_index,
            bank="mixed",
        )
        prior_name = MATCHED_PRIOR_FOR_ENV[environment]
        transform = TaskMatchedTransform(prior_name, bank.d)
        prediction = bank.features(env.observation)
        output_rows = []
        latent_rows = []
        for step in range(6000):
            output_rows.append(transform.transform(prediction))
            latent_rows.append(env.latent_state.copy())
            next_observation, _, info = env.step(step % env.n_actions)
            bank.update(next_observation, info.events)
            prediction = bank.features(next_observation)
        output = np.asarray(output_rows[1000:])
        latents = np.asarray(latent_rows[1000:])
        samples[environment] = output
        metrics = rep_metrics(output)
        rows.append(
            {
                "environment": environment,
                "matched_prior": prior_name,
                **metrics,
            }
        )
        finite = bool(np.isfinite(output).all())
        add_check(checks, prior_name, environment, "real-stream finite output", finite, f"finite={finite}")
        if prior_name == "simplex":
            passed = np.min(output) >= 0 and np.max(np.abs(output.sum(axis=1) - 1.0)) < 1e-12
            evidence = f"min={output.min():.5f}; sum_error={np.max(np.abs(output.sum(axis=1)-1.0)):.3g}"
        elif prior_name == "circular":
            radii = np.linalg.norm(output, axis=1)
            passed = np.max(np.abs(radii - 1.0)) < 1e-10
            evidence = f"mean_radius={radii.mean():.6f}"
        elif prior_name == "block":
            passed = (
                np.min(output[:, :2]) >= 0
                and np.max(np.abs(output[:, :2].sum(axis=1) - 1.0)) < 1e-12
                and np.max(np.abs(np.linalg.norm(output[:, 2:4], axis=1) - 1.0)) < 1e-10
            )
            evidence = "simplex and circular blocks checked"
        else:
            variances = np.diag(np.cov(output, rowvar=False))
            ratio = float(variances[0] / max(variances[1], 1e-12))
            passed = 4.0 <= ratio <= 18.0
            evidence = f"real_stream_variance_ratio={ratio:.4f}; target=9.0"
        add_check(checks, prior_name, environment, "real-stream named property", passed, evidence)
        if prior_name == "circular":
            predicted_angles = np.arctan2(output[:, 1], output[:, 0])
            true_angles = latents[:, 1]
            forward = np.abs(np.mean(np.exp(1j * (predicted_angles - true_angles))))
            reverse = np.abs(np.mean(np.exp(1j * (predicted_angles + true_angles))))
            alignment = float(max(forward, reverse))
            shuffled = np.random.default_rng(20260713).permutation(predicted_angles)
            shuffled_alignment = float(
                max(
                    np.abs(np.mean(np.exp(1j * (shuffled - true_angles)))),
                    np.abs(np.mean(np.exp(1j * (shuffled + true_angles)))),
                )
            )
            add_check(
                checks,
                prior_name,
                environment,
                "real-stream phase order",
                alignment > 0.3 and alignment > shuffled_alignment + 0.2,
                f"alignment={alignment:.4f}; shuffled={shuffled_alignment:.4f}",
            )
    return pd.DataFrame(rows), pd.DataFrame(checks), samples


def make_figures(
    root: Path,
    checks: pd.DataFrame,
    real_metrics: pd.DataFrame,
    samples: dict[str, np.ndarray],
) -> None:
    figures = root / "figures"
    figures.mkdir()
    labels = [f"{row.method}\n{row.stream}" for row in checks.itertuples()]
    values = checks["passed"].astype(int).to_numpy()
    fig, axis = plt.subplots(figsize=(max(9, 0.38 * len(labels)), 4.8))
    axis.bar(np.arange(len(values)), values, color=np.where(values, "#2ca02c", "#d62728"))
    axis.set_xticks(np.arange(len(values)), labels, rotation=75, ha="right", fontsize=7)
    axis.set_yticks([0, 1], ["fail", "pass"])
    axis.set_title("Cross-environment property-fulfillment checks")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "property_pass_fail.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, environment in zip(axes.ravel(), ENVIRONMENT_IDS):
        output = samples[environment]
        projection = output[:, : min(2, output.shape[1])]
        if projection.shape[1] == 1:
            projection = np.column_stack((projection[:, 0], np.zeros(len(projection))))
        axis.scatter(projection[::5, 0], projection[::5, 1], s=7, alpha=0.45)
        axis.set_title(environment)
        axis.set_xlabel("matched coordinate 1")
        axis.set_ylabel("matched coordinate 2")
        axis.grid(alpha=0.2)
    fig.suptitle("Real-stream matched-prior outputs", y=1.01)
    fig.tight_layout()
    fig.savefig(figures / "real_stream_matched_geometry.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(real_metrics["environment"], real_metrics["effective_rank"], color="#4c78a8")
    axis.set_ylabel("Effective rank")
    axis.set_title("Real-stream matched-representation effective rank")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "real_stream_effective_rank.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        parser.error("--run-name may contain only letters, digits, dot, underscore, and hyphen")
    root = Path("results") / "diagnostics" / "cross_environment" / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_name": args.run_name,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "exact_command": [Path(sys.executable).name, *sys.argv],
        "seed": 20260713,
        "exit_status": "running",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        synthetic_metrics, synthetic_checks, _ = synthetic_diagnostics(
            np.random.default_rng(20260713)
        )
        real_metrics, real_checks, real_samples = real_stream_diagnostics()
        checks = pd.concat((synthetic_checks, real_checks), ignore_index=True)
        synthetic_metrics.to_csv(root / "synthetic_before_after_metrics.csv", index=False)
        real_metrics.to_csv(root / "real_stream_metrics.csv", index=False)
        checks.to_csv(root / "pass_fail_summary.csv", index=False)
        make_figures(root, checks, real_metrics, real_samples)
        passed = bool(checks["passed"].all())
        manifest.update(
            end_time=datetime.now(timezone.utc).isoformat(),
            exit_status="ok" if passed else "failed",
            checks_passed=int(checks["passed"].sum()),
            checks_total=len(checks),
            all_checks_passed=passed,
        )
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"RESULT_DIR={root.resolve()}")
        print(checks.to_string(index=False))
        if not passed:
            raise SystemExit("one or more cross-environment property checks failed")
    except BaseException as error:
        if manifest["exit_status"] == "running":
            manifest.update(
                end_time=datetime.now(timezone.utc).isoformat(),
                exit_status="failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            (root / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        raise


if __name__ == "__main__":
    main()
