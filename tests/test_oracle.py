"""Tests for DC economic-dispatch feasibility and known solutions."""

import numpy as np

from value_sced.grid import build_source_grid
from value_sced.oracle import solve_economic_dispatch


def test_source_dispatch_solves_successfully() -> None:
    grid = build_source_grid()
    demand = np.array([0.0, 40.0, 35.0, 45.0, 50.0, 30.0])

    result = solve_economic_dispatch(grid, demand)
    generation = np.asarray(result.generation_mw)

    assert result.success
    assert result.solver_status == 0
    assert np.isfinite(result.objective_cost)
    assert np.isclose(generation.sum(), demand.sum(), atol=1e-7)
    for output, generator in zip(generation, grid.generators, strict=True):
        assert generator.minimum_mw - 1e-7 <= output
        assert output <= generator.maximum_mw + 1e-7
