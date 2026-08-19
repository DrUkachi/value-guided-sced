"""Tests for secure minimum-L1 dispatch repair."""

import numpy as np

from value_sced.grid import build_source_grid
from value_sced.oracle import solve_economic_dispatch
from value_sced.repair import repair_dispatch


def test_feasible_dispatch_requires_no_repair() -> None:
    grid = build_source_grid()
    demand = np.array([0.0, 40.0, 35.0, 45.0, 50.0, 30.0])
    optimum = solve_economic_dispatch(grid, demand, (3, 4))
    repaired = repair_dispatch(grid, demand, np.asarray(optimum.generation_mw), (3, 4))
    assert repaired.success
    assert repaired.repair_magnitude_mw <= 1e-7


def test_repair_restores_balance_and_security() -> None:
    grid = build_source_grid()
    demand = np.array([0.0, 40.0, 35.0, 45.0, 50.0, 30.0])
    repaired = repair_dispatch(grid, demand, np.array([170.0, 20.0, -5.0, 0.0]), (3, 4))
    assert repaired.success
    generation = np.asarray(repaired.generation_mw)
    assert np.isclose(generation.sum(), demand.sum(), atol=1e-7)
    assert all(
        generator.minimum_mw - 1e-7 <= output <= generator.maximum_mw + 1e-7
        for output, generator in zip(generation, grid.generators, strict=True)
    )
