"""Guard the frozen budget arithmetic and dataset separation assumptions."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_budget_schedule_matches_acquisition_design() -> None:
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    data = config["data"]
    expected_maximum = (
        data["initial_target_labels"]
        + data["acquisition_batch_size"] * data["acquisition_rounds"]
    )
    assert expected_maximum == 60
    assert config["evaluation"]["budgets"] == [0, 12, 24, 36, 48, 60]


def test_candidate_and_test_sets_are_distinct_in_configuration() -> None:
    data = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())["data"]
    assert data["target_candidate_size"] == 160
    assert data["target_test_size"] == 80
