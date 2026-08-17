"""Budgeted acquisition loop with an explicit oracle-label boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from value_sced.acquisition import PoolState


Label = TypeVar("Label")


@dataclass
class OracleLabelStore(Generic[Label]):
    """Keep precomputed labels inaccessible until their identifiers are selected."""

    _labels: Mapping[int, Label]
    revealed: set[int]

    def __init__(self, labels: Mapping[int, Label]):
        self._labels = dict(labels)
        self.revealed = set()

    def reveal(self, selected: Sequence[int]) -> dict[int, Label]:
        requested = [int(i) for i in selected]
        if len(set(requested)) != len(requested):
            raise ValueError("cannot reveal a duplicate candidate")
        if self.revealed.intersection(requested):
            raise ValueError("candidate label has already been revealed")
        if not set(requested).issubset(self._labels):
            raise KeyError("selected candidate has no oracle label")
        self.revealed.update(requested)
        return {i: self._labels[i] for i in requested}


def run_acquisition_rounds(
    state: PoolState,
    labels: OracleLabelStore[Label],
    select_batch: Callable[[PoolState, int], Sequence[int]],
    *,
    batch_size: int,
    rounds: int,
    on_reveal: Callable[[int, Mapping[int, Label]], None] | None = None,
) -> list[tuple[int, ...]]:
    """Select then reveal each batch; scoring never receives the label store."""

    history: list[tuple[int, ...]] = []
    for round_number in range(1, rounds + 1):
        batch = tuple(int(i) for i in select_batch(state, batch_size))
        # Validate and update pool membership before the only label access.
        state.acquire(batch)
        revealed = labels.reveal(batch)
        if on_reveal is not None:
            on_reveal(round_number, revealed)
        history.append(batch)
    return history
