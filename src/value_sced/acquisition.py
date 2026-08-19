"""Leakage-safe acquisition primitives.

This module deliberately deals only in candidate identifiers and features.  Oracle
labels live behind :class:`OracleLabelStore` in ``experiment.py`` and are revealed
only after a batch has been selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from value_sced.grid import Grid, build_dc_matrices, without_line


def _validate_batch(remaining: Sequence[int], batch_size: int) -> np.ndarray:
    indices = np.asarray(remaining, dtype=int)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError("remaining indices must be a one-dimensional unique sequence")
    if not 0 < batch_size <= len(indices):
        raise ValueError("batch_size must be positive and no larger than the pool")
    return indices


def initial_target_indices(candidate_count: int, count: int, seed: int) -> tuple[int, ...]:
    """Return the shared initial target set for every strategy in one seed."""

    if not 0 < count <= candidate_count:
        raise ValueError("initial count must be positive and no larger than the pool")
    return tuple(int(i) for i in np.random.default_rng(seed).choice(
        candidate_count, size=count, replace=False
    ))


def select_random(
    remaining: Sequence[int], batch_size: int, rng: np.random.Generator
) -> tuple[int, ...]:
    """Select a uniform batch without replacement."""

    indices = _validate_batch(remaining, batch_size)
    return tuple(int(i) for i in rng.choice(indices, size=batch_size, replace=False))


def ensemble_disagreement(member_predictions: np.ndarray) -> np.ndarray:
    """Return mean squared Euclidean disagreement from the ensemble mean."""

    predictions = np.asarray(member_predictions, dtype=float)
    if predictions.ndim != 3 or predictions.shape[0] < 2:
        raise ValueError("member_predictions must have shape (members, samples, outputs)")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("member predictions must be finite")
    mean = predictions.mean(axis=0, keepdims=True)
    return np.square(predictions - mean).sum(axis=2).mean(axis=0)


def _estimated_flows(grid: Grid, demands: np.ndarray, generation: np.ndarray) -> np.ndarray:
    """Estimate DC flows, assigning prediction imbalance to the reference bus."""

    bus_index = {bus: index for index, bus in enumerate(grid.buses)}
    injections = -demands.copy()
    for column, generator in enumerate(grid.generators):
        injections[:, bus_index[generator.bus]] += generation[:, column]
    injections[:, bus_index[grid.reference_bus]] -= injections.sum(axis=1)
    matrices = build_dc_matrices(grid)
    non_reference = [i for i, bus in enumerate(grid.buses) if bus != grid.reference_bus]
    angles = np.zeros_like(injections)
    angles[:, non_reference] = np.linalg.solve(
        matrices.bus_susceptance[np.ix_(non_reference, non_reference)],
        injections[:, non_reference].T,
    ).T
    return angles @ matrices.branch_susceptance.T


def operational_criticality(
    grid: Grid,
    demands_mw: np.ndarray,
    mean_generation_mw: np.ndarray,
    contingency_lines: Sequence[int],
) -> np.ndarray:
    """Maximum estimated line-loading ratio over base and contingency states."""

    demands = np.asarray(demands_mw, dtype=float)
    generation = np.asarray(mean_generation_mw, dtype=float)
    expected = (len(demands), len(grid.generators))
    if demands.ndim != 2 or demands.shape[1] != len(grid.buses):
        raise ValueError("demands must have one column per grid bus")
    if generation.shape != expected or not np.all(np.isfinite(demands)) or not np.all(np.isfinite(generation)):
        raise ValueError("generation must be finite with one column per generator")
    states = (grid,) + tuple(without_line(grid, int(line)) for line in contingency_lines)
    ratios = []
    for state in states:
        flows = _estimated_flows(state, demands, generation)
        limits = np.asarray([line.limit_mw for line in state.lines])
        ratios.append(np.max(np.abs(flows) / limits, axis=1))
    return np.max(np.stack(ratios), axis=0)


def minmax_normalise(scores: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1], or to zero when their range is zero."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite vector")
    span = float(values.max() - values.min()) if len(values) else 0.0
    return np.zeros_like(values) if span <= 1e-12 else (values - values.min()) / span


def decision_sensitive_scores(
    uncertainty: np.ndarray, criticality: np.ndarray, decision_lambda: float = 1.0
) -> np.ndarray:
    """Combine per-round normalised uncertainty and operational criticality."""

    if decision_lambda < 0.0:
        raise ValueError("decision_lambda must be non-negative")
    uncertainty_values = np.asarray(uncertainty, dtype=float)
    criticality_values = np.asarray(criticality, dtype=float)
    if uncertainty_values.shape != criticality_values.shape:
        raise ValueError("uncertainty and criticality must have the same shape")
    return minmax_normalise(uncertainty_values) * (
        1.0 + decision_lambda * minmax_normalise(criticality_values)
    )


def select_highest_scores(
    remaining: Sequence[int], scores: np.ndarray, batch_size: int, rng: np.random.Generator
) -> tuple[int, ...]:
    """Select highest scores with reproducible random ordering within exact ties."""

    indices = _validate_batch(remaining, batch_size)
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(indices),) or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite value for each remaining candidate")
    tie_breakers = rng.random(len(indices))
    order = np.lexsort((tie_breakers, -values))
    return tuple(int(i) for i in indices[order[:batch_size]])


def select_uncertainty(
    remaining: Sequence[int], member_predictions: np.ndarray,
    batch_size: int, rng: np.random.Generator,
) -> tuple[int, ...]:
    """Select candidates with the greatest ensemble disagreement."""

    return select_highest_scores(
        remaining, ensemble_disagreement(member_predictions), batch_size, rng
    )


def select_decision_sensitive(
    remaining: Sequence[int], member_predictions: np.ndarray, demands_mw: np.ndarray,
    grid: Grid, contingency_lines: Sequence[int], batch_size: int,
    rng: np.random.Generator, decision_lambda: float = 1.0,
) -> tuple[int, ...]:
    """Select by uncertainty weighted by predicted operational criticality."""

    uncertainty = ensemble_disagreement(member_predictions)
    predictions = np.asarray(member_predictions, dtype=float)
    criticality = operational_criticality(
        grid, demands_mw, predictions.mean(axis=0), contingency_lines
    )
    scores = decision_sensitive_scores(uncertainty, criticality, decision_lambda)
    return select_highest_scores(remaining, scores, batch_size, rng)


def source_standardisation(source_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the frozen diversity transformation on source inputs only."""

    values = np.asarray(source_features, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("source_features must be a non-empty matrix")
    scale = values.std(axis=0)
    return values.mean(axis=0), np.where(scale > 1e-12, scale, 1.0)


def select_diversity(
    candidate_features: np.ndarray,
    remaining: Sequence[int],
    labelled_indices: Sequence[int],
    labelled_external_features: np.ndarray,
    batch_size: int,
    source_mean: np.ndarray,
    source_scale: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Select a farthest-first batch, with seeded random tie breaking.

    ``labelled_external_features`` holds labelled source points (and may be empty).
    Target points already labelled are addressed by ``labelled_indices``.
    """

    available = list(int(i) for i in _validate_batch(remaining, batch_size))
    candidates = (np.asarray(candidate_features, dtype=float) - source_mean) / source_scale
    if candidates.ndim != 2:
        raise ValueError("candidate_features must be a matrix")
    external = np.asarray(labelled_external_features, dtype=float)
    if external.size:
        external = (np.atleast_2d(external) - source_mean) / source_scale
    else:
        external = np.empty((0, candidates.shape[1]))
    centres = [*external, *(candidates[int(i)] for i in labelled_indices)]
    if not centres:
        raise ValueError("diversity acquisition requires at least one labelled centre")

    selected: list[int] = []
    for _ in range(batch_size):
        centre_matrix = np.asarray(centres)
        distances = np.asarray([
            np.linalg.norm(centre_matrix - candidates[index], axis=1).min()
            for index in available
        ])
        tied = np.flatnonzero(np.isclose(distances, distances.max(), rtol=1e-12, atol=1e-12))
        position = int(rng.choice(tied))
        chosen = available.pop(position)
        selected.append(chosen)
        centres.append(candidates[chosen])
    return tuple(selected)


@dataclass
class PoolState:
    """Bookkeeping for disjoint labelled, selected and remaining target cases."""

    candidate_count: int
    labelled: list[int]
    remaining: list[int]

    @classmethod
    def initialise(cls, candidate_count: int, initial: Sequence[int]) -> "PoolState":
        initial_values = [int(i) for i in initial]
        if len(set(initial_values)) != len(initial_values):
            raise ValueError("initial indices must be unique")
        if any(i < 0 or i >= candidate_count for i in initial_values):
            raise IndexError("initial index outside candidate pool")
        chosen = set(initial_values)
        return cls(candidate_count, initial_values, [i for i in range(candidate_count) if i not in chosen])

    def acquire(self, batch: Sequence[int]) -> None:
        """Move a complete batch from remaining to labelled, rejecting duplicates."""

        values = [int(i) for i in batch]
        if len(set(values)) != len(values) or not set(values).issubset(self.remaining):
            raise ValueError("batch must contain distinct remaining candidates")
        selected = set(values)
        self.labelled.extend(values)
        self.remaining = [i for i in self.remaining if i not in selected]
