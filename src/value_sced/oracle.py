"""SciPy-based linear DC economic-dispatch oracle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import linprog

from value_sced.grid import Grid, build_dc_matrices


@dataclass(frozen=True)
class DispatchResult:
    """Auditable output from one DC dispatch solve."""

    success: bool
    solver_status: int
    solver_message: str
    objective_cost: float
    runtime_seconds: float
    generation_mw: tuple[float, ...]
    bus_angles: tuple[float, ...]
    line_flows_mw: tuple[float, ...]
    total_demand_mw: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def solve_economic_dispatch(grid: Grid, demand_mw: np.ndarray) -> DispatchResult:
    """Minimise linear generation cost subject to nodal DC-flow constraints."""

    demand = np.asarray(demand_mw, dtype=float)
    if demand.shape != (len(grid.buses),):
        raise ValueError(f"demand must have shape ({len(grid.buses)},)")
    if np.any(demand < 0.0) or not np.all(np.isfinite(demand)):
        raise ValueError("demand must be finite and non-negative")

    matrices = build_dc_matrices(grid)
    number_generators = len(grid.generators)
    number_buses = len(grid.buses)
    bus_index = {bus: index for index, bus in enumerate(grid.buses)}
    generator_map = np.zeros((number_buses, number_generators))
    for column, generator in enumerate(grid.generators):
        generator_map[bus_index[generator.bus], column] = 1.0

    costs = np.array([generator.cost_per_mwh for generator in grid.generators])
    objective = np.concatenate((costs, np.zeros(number_buses)))
    nodal_balance = np.hstack((generator_map, -matrices.bus_susceptance))
    reference_angle = np.zeros((1, number_generators + number_buses))
    reference_angle[0, number_generators + bus_index[grid.reference_bus]] = 1.0
    equality_matrix = np.vstack((nodal_balance, reference_angle))
    equality_rhs = np.concatenate((demand, [0.0]))

    zero_generation = np.zeros((len(grid.lines), number_generators))
    flow_matrix = np.hstack((zero_generation, matrices.branch_susceptance))
    inequality_matrix = np.vstack((flow_matrix, -flow_matrix))
    limits = np.array([line.limit_mw for line in grid.lines])
    inequality_rhs = np.concatenate((limits, limits))
    bounds = [
        (generator.minimum_mw, generator.maximum_mw)
        for generator in grid.generators
    ] + [(None, None)] * number_buses

    started = perf_counter()
    solution = linprog(
        objective,
        A_ub=inequality_matrix,
        b_ub=inequality_rhs,
        A_eq=equality_matrix,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
    )
    runtime = perf_counter() - started
    if solution.x is None:
        generation = np.full(number_generators, np.nan)
        angles = np.full(number_buses, np.nan)
        flows = np.full(len(grid.lines), np.nan)
    else:
        generation = solution.x[:number_generators]
        angles = solution.x[number_generators:]
        flows = matrices.branch_susceptance @ angles

    return DispatchResult(
        success=bool(solution.success),
        solver_status=int(solution.status),
        solver_message=str(solution.message),
        objective_cost=float(solution.fun) if solution.fun is not None else float("nan"),
        runtime_seconds=float(runtime),
        generation_mw=tuple(float(value) for value in generation),
        bus_angles=tuple(float(value) for value in angles),
        line_flows_mw=tuple(float(value) for value in flows),
        total_demand_mw=float(demand.sum()),
    )
