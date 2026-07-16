"""Condition-specific controller-alpha selection with disjoint seed sets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .robust_stats import interquartile_mean


DEFAULT_SEED_SETS = {
    "pilot": list(range(200, 205)),
    "tuning": list(range(100, 105)),
    "evaluation": list(range(20)),
    "smoke": [9000, 9001],
}


def validate_seed_sets(seed_sets: Mapping[str, list[int]]) -> None:
    """Reject overlap because evaluation outcomes must never influence design."""

    required = set(DEFAULT_SEED_SETS)
    missing = required - set(seed_sets)
    if missing:
        raise ValueError(f"seed_sets is missing {sorted(missing)}")
    normalized = {name: set(map(int, seed_sets[name])) for name in required}
    if any(not values for values in normalized.values()):
        raise ValueError("every seed set must be non-empty")
    names = sorted(normalized)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = normalized[left] & normalized[right]
            if overlap:
                raise ValueError(f"seed sets {left} and {right} overlap: {sorted(overlap)}")


def candidate_alphas(base_alpha: float, multipliers: list[float]) -> list[tuple[float, float]]:
    if base_alpha <= 0 or not multipliers or any(float(value) <= 0 for value in multipliers):
        raise ValueError("alpha base and multipliers must be positive")
    return [(float(base_alpha) * float(multiplier), float(multiplier)) for multiplier in multipliers]


def select_learning_rates(
    summaries: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Select one alpha per environment/condition using tuning data only."""

    required = {
        "environment",
        "condition",
        "seed",
        "candidate_alpha",
        "base_alpha_multiplier",
        "final_performance",
        "catastrophic_failure",
    }
    missing = required - set(summaries)
    if missing:
        raise ValueError(f"tuning summaries are missing {sorted(missing)}")
    seed_sets = config.get("seed_sets", DEFAULT_SEED_SETS)
    validate_seed_sets(seed_sets)
    allowed = set(map(int, seed_sets["tuning"]))
    observed = set(map(int, summaries["seed"].unique()))
    if not observed <= allowed:
        raise ValueError(f"alpha selection received non-tuning seeds: {sorted(observed - allowed)}")
    tuning = config.get("alpha_tuning", {})
    criterion = str(tuning.get("selection_criterion", "robust_score"))
    if criterion not in {"mean", "median", "iqm", "robust_score"}:
        raise ValueError("selection_criterion must be mean, median, iqm, or robust_score")
    penalty = float(tuning.get("failure_penalty", 1.0))
    tie_rule = str(
        tuning.get(
            "tie_breaking_rule",
            "score desc; catastrophic failure rate asc; IQM desc; alpha asc",
        )
    )
    rows: list[dict[str, Any]] = []
    for (environment, condition, alpha, multiplier), frame in summaries.groupby(
        ["environment", "condition", "candidate_alpha", "base_alpha_multiplier"],
        sort=False,
    ):
        values = frame["final_performance"].to_numpy(dtype=float)
        failure_rate = float(frame["catastrophic_failure"].mean())
        mean = float(np.mean(values))
        median = float(np.median(values))
        iqm = interquartile_mean(values)
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
        robust_score = iqm - penalty * failure_rate
        score = {"mean": mean, "median": median, "iqm": iqm, "robust_score": robust_score}[
            criterion
        ]
        rows.append(
            {
                "environment": environment,
                "condition": condition,
                "candidate_alpha": float(alpha),
                "base_alpha_multiplier": float(multiplier),
                "tuning_seed_set": ",".join(map(str, sorted(frame["seed"].astype(int).unique()))),
                "tuning_mean": mean,
                "tuning_median": median,
                "tuning_IQM": iqm,
                "tuning_SEM": sem,
                "tuning_catastrophic_failure_rate": failure_rate,
                "robust_score": robust_score,
                "selection_score": score,
                "selection_criterion": criterion,
                "failure_penalty": penalty,
                "tie_breaking_rule": tie_rule,
            }
        )
    candidates = pd.DataFrame(rows)
    selected: list[dict[str, Any]] = []
    for _, frame in candidates.groupby(["environment", "condition"], sort=False):
        winner = frame.sort_values(
            [
                "selection_score",
                "tuning_catastrophic_failure_rate",
                "tuning_IQM",
                "candidate_alpha",
            ],
            ascending=[False, True, False, True],
            kind="stable",
        ).iloc[0].to_dict()
        winner["selected_alpha"] = winner["candidate_alpha"]
        selected.append(winner)
    return pd.DataFrame(selected)


def load_selected_learning_rates(
    path: str | Path, expected_pairs: set[tuple[str, str]] | None = None
) -> dict[tuple[str, str], float]:
    """Parse and validate the machine-readable tuning handoff."""

    frame = pd.read_csv(path)
    required = {"environment", "condition", "selected_alpha"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"selected learning-rate file is missing {sorted(missing)}")
    if frame.duplicated(["environment", "condition"]).any():
        raise ValueError("selected learning-rate file has duplicate environment/condition rows")
    if not np.isfinite(frame["selected_alpha"]).all() or (frame["selected_alpha"] <= 0).any():
        raise ValueError("selected alphas must be finite and positive")
    result = {
        (str(row.environment), str(row.condition)): float(row.selected_alpha)
        for row in frame.itertuples(index=False)
    }
    if expected_pairs is not None and not expected_pairs <= set(result):
        missing_pairs = sorted(expected_pairs - set(result))
        raise ValueError(
            f"selected learning-rate pairs mismatch; missing={missing_pairs}"
        )
    return result if expected_pairs is None else {key: result[key] for key in expected_pairs}
