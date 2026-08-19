"""Tests for feasible, congested, and infeasible DC-SCED cases."""

from dataclasses import replace

import numpy as np

from value_sced.grid import build_source_grid
from value_sced.oracle import audit_dispatch, solve_economic_dispatch


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


def test_security_constraints_cover_both_locked_outages() -> None:
    grid = build_source_grid()
    result = solve_economic_dispatch(
        grid,
        np.array([0.0, 40.0, 35.0, 45.0, 50.0, 30.0]),
        (3, 4),
    )

    assert result.success
    assert len(result.contingency_line_flows_mw) == 2
    for numbers, flows in zip(
        result.contingency_line_numbers,
        result.contingency_line_flows_mw,
        strict=True,
    ):
        limits = {line.number: line.limit_mw for line in grid.lines}
        assert all(abs(flow) <= limits[number] + 1e-7 for number, flow in zip(numbers, flows, strict=True))

    audit = audit_dispatch(grid, np.array([0.0, 40.0, 35.0, 45.0, 50.0, 30.0]), result, (3, 4))
    assert audit.passes(1e-7)
    assert audit.state_names == ("base", "outage_line_3", "outage_line_4")
    assert max(audit.maximum_nodal_balance_residual_mw) <= 1e-7
    assert max(audit.maximum_line_limit_violation_mw) == 0.0


def test_congested_case_binds_a_tightened_line_limit() -> None:
    grid = build_source_grid()
    tight_lines = tuple(
        replace(line, limit_mw=30.0) if line.number == 1 else line
        for line in grid.lines
    )
    congested_grid = replace(grid, lines=tight_lines)
    result = solve_economic_dispatch(
        congested_grid,
        np.array([0.0, 40.0, 35.0, 45.0, 50.0, 30.0]),
    )

    assert result.success
    assert np.isclose(abs(result.line_flows_mw[0]), 30.0, atol=1e-7)


def test_excess_demand_is_deliberately_infeasible() -> None:
    grid = build_source_grid()
    demand = np.full(len(grid.buses), 100.0)

    result = solve_economic_dispatch(grid, demand, (3, 4))

    assert not result.success
    assert np.isnan(result.objective_cost)
