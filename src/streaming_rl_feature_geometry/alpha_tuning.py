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
    summaries: pd.DataFrame,
    config: Mapping[str, Any],
    invalid_candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select one fully valid alpha per environment/condition using tuning data only."""

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
    invalid = invalid_candidates if invalid_candidates is not None else pd.DataFrame()
    if not invalid.empty:
        invalid_required = {"environment", "condition", "seed", "candidate_alpha"}
        invalid_missing = invalid_required - set(invalid)
        if invalid_missing:
            raise ValueError(
                f"invalid tuning candidates are missing {sorted(invalid_missing)}"
            )
        invalid_seeds = set(map(int, invalid["seed"].unique()))
        if not invalid_seeds <= allowed:
            raise ValueError(
                "invalid alpha selection evidence contains non-tuning seeds: "
                f"{sorted(invalid_seeds - allowed)}"
            )
    invalid_alpha_keys = {
        (str(row.environment), str(row.condition), float(row.candidate_alpha))
        for row in invalid.itertuples(index=False)
    }
    eligible = summaries.loc[
        [
            (str(row.environment), str(row.condition), float(row.candidate_alpha))
            not in invalid_alpha_keys
            for row in summaries.itertuples(index=False)
        ]
    ].copy()
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
    for (environment, condition, alpha, multiplier), frame in eligible.groupby(
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
    all_pairs = {
        (str(row.environment), str(row.condition))
        for row in summaries.itertuples(index=False)
    } | {
        (str(row.environment), str(row.condition))
        for row in invalid.itertuples(index=False)
    }
    eligible_pairs = (
        set(zip(candidates["environment"], candidates["condition"], strict=False))
        if not candidates.empty
        else set()
    )
    missing_pairs = sorted(all_pairs - eligible_pairs)
    if missing_pairs:
        raise ValueError(
            "no fully valid learning-rate candidate for environment/condition pairs: "
            f"{missing_pairs}"
        )
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
        pair = (str(winner["environment"]), str(winner["condition"]))
        valid_attempt_count = int(
            (
                (summaries["environment"].astype(str) == pair[0])
                & (summaries["condition"].astype(str) == pair[1])
            ).sum()
        )
        invalid_attempt_count = int(
            (
                (invalid["environment"].astype(str) == pair[0])
                & (invalid["condition"].astype(str) == pair[1])
            ).sum()
        ) if not invalid.empty else 0
        configured_seeds = list(config.get("seeds", seed_sets["tuning"]))
        multipliers = list(config.get("alpha_tuning", {}).get("multipliers", []))
        expected_identity_count = (
            len(configured_seeds) * len(multipliers)
            if multipliers
            else valid_attempt_count + invalid_attempt_count
        )
        eligible_alpha_count = int(
            frame["candidate_alpha"].astype(float).nunique()
        )
        invalid_alpha_count = len(
            {
                float(row.candidate_alpha)
                for row in invalid.itertuples(index=False)
                if str(row.environment) == pair[0] and str(row.condition) == pair[1]
            }
        )
        winner.update(
            valid_candidate_count=eligible_alpha_count,
            invalid_candidate_count=invalid_alpha_count,
            attempted_candidate_count=eligible_alpha_count + invalid_alpha_count,
            expected_candidate_count=(
                len(multipliers)
                if multipliers
                else eligible_alpha_count + invalid_alpha_count
            ),
            valid_attempt_count=valid_attempt_count,
            invalid_attempt_count=invalid_attempt_count,
            attempted_identity_count=valid_attempt_count + invalid_attempt_count,
            expected_identity_count=expected_identity_count,
            all_candidates_attempted=(
                valid_attempt_count + invalid_attempt_count == expected_identity_count
            ),
            eligible_candidate_alpha_count=eligible_alpha_count,
            invalid_candidate_alpha_count=invalid_alpha_count,
        )
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
