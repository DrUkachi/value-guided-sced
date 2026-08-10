"""Tests for physical topology and DC matrix invariants."""

import numpy as np

from value_sced.grid import (
    build_dc_matrices,
    build_source_grid,
    build_target_grid,
    is_connected,
)


def test_source_and_target_topologies() -> None:
    source = build_source_grid()
    target = build_target_grid()

    assert len(source.buses) == 6
    assert len(source.lines) == 10
    assert len(source.generators) == 4
    assert len(target.lines) == 9
    assert 8 in {line.number for line in source.lines}
    assert 8 not in {line.number for line in target.lines}
    assert is_connected(source)
    assert is_connected(target)


def test_dc_matrix_shapes_and_conservation() -> None:
    grid = build_source_grid()
    matrices = build_dc_matrices(grid)

    assert matrices.incidence.shape == (10, 6)
    assert matrices.branch_susceptance.shape == (10, 6)
    assert matrices.bus_susceptance.shape == (6, 6)
    assert np.allclose(matrices.incidence.sum(axis=1), 0.0)
    assert np.allclose(matrices.bus_susceptance.sum(axis=1), 0.0)
