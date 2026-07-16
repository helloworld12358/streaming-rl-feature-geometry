import pandas as pd
import pytest

from streaming_rl_feature_geometry.alpha_tuning import (
    DEFAULT_SEED_SETS,
    load_selected_learning_rates,
    select_learning_rates,
    validate_seed_sets,
)
from streaming_rl_feature_geometry.cross_experiment import (
    build_cross_tasks,
    expected_cross_run_count,
    load_cross_config,
)


def test_default_seed_sets_are_pairwise_disjoint():
    validate_seed_sets(DEFAULT_SEED_SETS)
    bad = {key: list(value) for key, value in DEFAULT_SEED_SETS.items()}
    bad["evaluation"] = [100]
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_sets(bad)


def test_selection_uses_tuning_only_and_tie_breaks_to_smaller_alpha(tmp_path):
    rows = []
    for alpha in (0.01, 0.02):
        for seed in DEFAULT_SEED_SETS["tuning"]:
            rows.append(
                {
                    "environment": "tmaze",
                    "condition": "raw",
                    "seed": seed,
                    "candidate_alpha": alpha,
                    "base_alpha_multiplier": alpha / 0.01,
                    "final_performance": 0.8,
                    "catastrophic_failure": 0,
                }
            )
    selected = select_learning_rates(
        pd.DataFrame(rows),
        {"seed_sets": DEFAULT_SEED_SETS, "alpha_tuning": {"failure_penalty": 1.0}},
    )
    assert selected.iloc[0].selected_alpha == 0.01
    path = tmp_path / "selected_learning_rates.csv"
    selected.to_csv(path, index=False)
    assert load_selected_learning_rates(path, {("tmaze", "raw")}) == {("tmaze", "raw"): 0.01}


def test_evaluation_tasks_only_consume_preselected_alpha(tmp_path):
    config = {
        "experiment_stage": "lr_eval",
        "seeds": [0],
        "control_alpha": 0.5,
        "environments": {"tmaze": {"conditions": ["raw"]}},
    }
    tasks = build_cross_tasks(
        config, tmp_path, selected_learning_rates={("tmaze", "raw"): 0.0125}
    )
    assert len(tasks) == 1
    assert tasks[0][5]["controller_alpha"] == 0.0125


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("configs/cross_extension_fixed_full.json", 1000),
        ("configs/cross_extension_lr_tune_full.json", 1500),
        ("configs/cross_extension_lr_eval_full.json", 1000),
        ("configs/cross_extension_norm_scaled_full.json", 700),
    ],
)
def test_formal_extension_profiles_parse_and_have_exact_run_counts(path, expected):
    config = load_cross_config(path)
    assert config["remote_full"] is True
    assert expected_cross_run_count(config) == expected
