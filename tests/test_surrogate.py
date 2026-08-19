"""Tests for deterministic source splitting and ensemble shapes."""

import numpy as np

from value_sced.surrogate import source_train_validation_indices, train_ensemble


def test_source_split_is_disjoint_and_reproducible() -> None:
    first = source_train_validation_indices(220, 0.2, 42)
    second = source_train_validation_indices(220, 0.2, 42)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert len(first[0]) == 176 and len(first[1]) == 44
    assert set(first[0]).isdisjoint(first[1])


def test_small_ensemble_prediction_shape() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(20, 3))
    targets = features[:, :2]
    ensemble = train_ensemble(
        features[:16], targets[:16], features[16:], targets[16:], ensemble_size=2,
        hidden_sizes=(8,), learning_rate=0.01, max_epochs=5, patience=3, seed=7,
    )
    assert ensemble.predict(features[16:]).shape == (4, 2)
    assert ensemble.predict_members(features[16:]).shape == (2, 4, 2)
