"""Deterministic minimum-L1 feasibility repair for surrogate dispatches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from value_sced.grid import Grid, build_dc_matrices, without_line


@dataclass(frozen=True)
class RepairResult:
    success: bool
    generation_mw: tuple[float, ...]
    repair_magnitude_mw: float
    operating_cost: float
    solver_message: str


def repair_dispatch(
    grid: Grid,
    demand_mw: np.ndarray,
    predicted_generation_mw: np.ndarray,
    contingency_lines: tuple[int, ...] = (),
) -> RepairResult:
    """Find the secure dispatch closest to a prediction in L1 distance.

    A tiny fixed cost coefficient breaks ties reproducibly without materially
    changing the primary minimum-repair objective.
    """

    demand = np.asarray(demand_mw, dtype=float)
    prediction = np.asarray(predicted_generation_mw, dtype=float)
    number_buses = len(grid.buses)
    number_generators = len(grid.generators)
    if demand.shape != (number_buses,) or prediction.shape != (number_generators,):
        raise ValueError("demand or prediction has an invalid shape")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("prediction must be finite")

    states = (grid,) + tuple(without_line(grid, line) for line in contingency_lines)
    number_states = len(states)
    # Variables are generation, absolute deviations, then one angle vector/state.
    generation_slice = slice(0, number_generators)
    deviation_slice = slice(number_generators, 2 * number_generators)
    angle_offset = 2 * number_generators
    number_variables = angle_offset + number_states * number_buses
    costs = np.asarray([generator.cost_per_mwh for generator in grid.generators])
    objective = np.zeros(number_variables)
    objective[deviation_slice] = 1.0
    objective[generation_slice] = 1e-9 * costs

    bus_index = {bus: index for index, bus in enumerate(grid.buses)}
    generator_map = np.zeros((number_buses, number_generators))
    for column, generator in enumerate(grid.generators):
        generator_map[bus_index[generator.bus], column] = 1.0

    equality_rows, equality_rhs = [], []
    inequality_rows, inequality_rhs = [], []
    # |g - prediction| <= t.
    deviation_positive = np.zeros((number_generators, number_variables))
    deviation_positive[:, generation_slice] = np.eye(number_generators)
    deviation_positive[:, deviation_slice] = -np.eye(number_generators)
    deviation_negative = np.zeros((number_generators, number_variables))
    deviation_negative[:, generation_slice] = -np.eye(number_generators)
    deviation_negative[:, deviation_slice] = -np.eye(number_generators)
    inequality_rows.extend((deviation_positive, deviation_negative))
    inequality_rhs.extend((prediction, -prediction))

    for state_index, state in enumerate(states):
        matrices = build_dc_matrices(state)
        angle_start = angle_offset + state_index * number_buses
        nodal = np.zeros((number_buses, number_variables))
        nodal[:, generation_slice] = generator_map
        nodal[:, angle_start : angle_start + number_buses] = -matrices.bus_susceptance
        reference = np.zeros((1, number_variables))
        reference[0, angle_start + bus_index[grid.reference_bus]] = 1.0
        equality_rows.extend((nodal, reference))
        equality_rhs.extend((demand, np.array([0.0])))
        flow = np.zeros((len(state.lines), number_variables))
        flow[:, angle_start : angle_start + number_buses] = matrices.branch_susceptance
        limits = np.asarray([line.limit_mw for line in state.lines])
        inequality_rows.extend((flow, -flow))
        inequality_rhs.extend((limits, limits))

    bounds = [
        (generator.minimum_mw, generator.maximum_mw) for generator in grid.generators
    ] + [(0.0, None)] * number_generators + [(None, None)] * (number_states * number_buses)
    solution = linprog(
        objective,
        A_ub=np.vstack(inequality_rows),
        b_ub=np.concatenate(inequality_rhs),
        A_eq=np.vstack(equality_rows),
        b_eq=np.concatenate(equality_rhs),
        bounds=bounds,
        method="highs",
    )
    if not solution.success or solution.x is None:
        return RepairResult(False, tuple(float("nan") for _ in grid.generators), float("nan"), float("nan"), str(solution.message))
    generation = solution.x[generation_slice]
    return RepairResult(
        True,
        tuple(float(value) for value in generation),
        float(np.abs(generation - prediction).sum()),
        float(costs @ generation),
        str(solution.message),
    )
