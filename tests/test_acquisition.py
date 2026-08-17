"""Acquisition selection, bookkeeping and leakage-boundary tests."""

import numpy as np

from value_sced.acquisition import (
    PoolState, decision_sensitive_scores, ensemble_disagreement,
    initial_target_indices, minmax_normalise, operational_criticality,
    select_diversity, select_highest_scores, select_random,
    source_standardisation,
)
from value_sced.experiment import OracleLabelStore, run_acquisition_rounds
from value_sced.grid import build_target_grid


def test_initial_set_is_reproducible_and_shared() -> None:
    first = initial_target_indices(160, 12, 11)
    assert first == initial_target_indices(160, 12, 11)
    assert len(first) == len(set(first)) == 12


def test_random_two_round_smoke_test_reveals_only_selected_labels() -> None:
    initial = initial_target_indices(40, 4, 42)
    state = PoolState.initialise(40, initial)
    store = OracleLabelStore({i: f"label-{i}" for i in range(40)})
    initial_labels = store.reveal(initial)
    assert set(initial_labels) == set(initial)
    rng = np.random.default_rng(42)
    history = run_acquisition_rounds(
        state, store, lambda pool, size: select_random(pool.remaining, size, rng),
        batch_size=4, rounds=2,
    )
    acquired = set(history[0] + history[1])
    assert len(acquired) == 8
    assert store.revealed == set(initial) | acquired
    assert len(state.labelled) == 12
    assert set(state.labelled).isdisjoint(state.remaining)


def test_diversity_is_farthest_first_and_adds_batch_centres() -> None:
    source = np.asarray([[0.0], [1.0]])
    candidates = np.asarray([[0.0], [2.0], [5.0], [9.0]])
    mean, scale = source_standardisation(source)
    selected = select_diversity(
        candidates, remaining=[1, 2, 3], labelled_indices=[0],
        labelled_external_features=np.empty((0, 1)), batch_size=2,
        source_mean=mean, source_scale=scale, rng=np.random.default_rng(7),
    )
    assert selected == (3, 2)


def test_pool_rejects_duplicate_or_previously_labelled_selection() -> None:
    state = PoolState.initialise(5, [0])
    for invalid in ([1, 1], [0]):
        try:
            state.acquire(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid batch was accepted")


def test_ensemble_disagreement_matches_protocol_formula() -> None:
    members = np.asarray([[[0.0, 2.0]], [[2.0, 4.0]]])
    assert np.allclose(ensemble_disagreement(members), [2.0])


def test_operational_criticality_covers_base_and_contingencies() -> None:
    grid = build_target_grid()
    demands = np.asarray([[0.0, 40.0, 35.0, 45.0, 40.0, 20.0]])
    generation = np.asarray([[100.0, 40.0, 20.0, 20.0]])
    base = operational_criticality(grid, demands, generation, [])
    secure = operational_criticality(grid, demands, generation, [3, 4])
    assert base.shape == secure.shape == (1,)
    assert secure[0] >= base[0] >= 0.0


def test_decision_score_normalisation_and_zero_range() -> None:
    assert np.array_equal(minmax_normalise(np.ones(3)), np.zeros(3))
    combined = decision_sensitive_scores(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([3.0, 2.0, 1.0])
    )
    assert np.allclose(combined, [0.0, 0.75, 1.0])


def test_score_selection_has_no_duplicates_and_seeded_ties() -> None:
    scores = np.asarray([2.0, 2.0, 1.0, 0.0])
    first = select_highest_scores([10, 11, 12, 13], scores, 3, np.random.default_rng(9))
    second = select_highest_scores([10, 11, 12, 13], scores, 3, np.random.default_rng(9))
    assert first == second
    assert set(first[:2]) == {10, 11}
    assert len(first) == len(set(first))
