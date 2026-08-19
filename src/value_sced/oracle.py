"""SciPy-based linear DC economic-dispatch oracle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import linprog

from value_sced.grid import Grid, build_dc_matrices, without_line


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
    contingency_line_numbers: tuple[tuple[int, ...], ...]
    contingency_line_flows_mw: tuple[tuple[float, ...], ...]
    total_demand_mw: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DispatchAudit:
    """Independent residual and constraint checks for a dispatch result."""

    objective_recalculation_error: float
    generation_bound_violation_mw: float
    state_names: tuple[str, ...]
    maximum_nodal_balance_residual_mw: tuple[float, ...]
    maximum_line_limit_violation_mw: tuple[float, ...]
    maximum_line_loading_ratio: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def passes(self, tolerance_mw: float) -> bool:
        """Return whether every dimensional audit residual is within tolerance."""

        return (
            self.objective_recalculation_error <= tolerance_mw
            and self.generation_bound_violation_mw <= tolerance_mw
            and max(self.maximum_nodal_balance_residual_mw) <= tolerance_mw
            and max(self.maximum_line_limit_violation_mw) <= tolerance_mw
        )


def audit_dispatch(
    grid: Grid,
    demand_mw: np.ndarray,
    result: DispatchResult,
    contingency_lines: tuple[int, ...] = (),
) -> DispatchAudit:
    """Reconstruct physical residuals without relying on the solver success flag."""

    demand = np.asarray(demand_mw, dtype=float)
    if demand.shape != (len(grid.buses),):
        raise ValueError(f"demand must have shape ({len(grid.buses)},)")
    if len(contingency_lines) != len(result.contingency_line_flows_mw):
        raise ValueError("contingency list does not match the dispatch result")

    generation = np.asarray(result.generation_mw)
    bus_index = {bus: index for index, bus in enumerate(grid.buses)}
    generated_at_bus = np.zeros(len(grid.buses))
    for output, generator in zip(generation, grid.generators, strict=True):
        generated_at_bus[bus_index[generator.bus]] += output

    states = (grid,) + tuple(without_line(grid, number) for number in contingency_lines)
    state_flows = (result.line_flows_mw,) + result.contingency_line_flows_mw
    balance_residuals = []
    limit_violations = []
    loading_ratios = []
    for state, flows_tuple in zip(states, state_flows, strict=True):
        flows = np.asarray(flows_tuple)
        if flows.shape != (len(state.lines),):
            raise ValueError("line-flow vector does not match its network state")
        incidence = build_dc_matrices(state).incidence
        nodal_residual = generated_at_bus - demand - incidence.T @ flows
        limits = np.asarray([line.limit_mw for line in state.lines])
        balance_residuals.append(float(np.max(np.abs(nodal_residual))))
        limit_violations.append(float(np.max(np.maximum(np.abs(flows) - limits, 0.0))))
        loading_ratios.append(float(np.max(np.abs(flows) / limits)))

    lower = np.asarray([generator.minimum_mw for generator in grid.generators])
    upper = np.asarray([generator.maximum_mw for generator in grid.generators])
    bound_violation = np.maximum(lower - generation, generation - upper)
    costs = np.asarray([generator.cost_per_mwh for generator in grid.generators])
    reconstructed_cost = float(costs @ generation)

    return DispatchAudit(
        objective_recalculation_error=abs(reconstructed_cost - result.objective_cost),
        generation_bound_violation_mw=float(np.max(np.maximum(bound_violation, 0.0))),
        state_names=("base",) + tuple(f"outage_line_{number}" for number in contingency_lines),
        maximum_nodal_balance_residual_mw=tuple(balance_residuals),
        maximum_line_limit_violation_mw=tuple(limit_violations),
        maximum_line_loading_ratio=tuple(loading_ratios),
    )


def solve_economic_dispatch(
    grid: Grid,
    demand_mw: np.ndarray,
    contingency_lines: tuple[int, ...] = (),
) -> DispatchResult:
    """Solve preventive linear DC-SCED for the base case and locked outages."""

    demand = np.asarray(demand_mw, dtype=float)
    if demand.shape != (len(grid.buses),):
        raise ValueError(f"demand must have shape ({len(grid.buses)},)")
    if np.any(demand < 0.0) or not np.all(np.isfinite(demand)):
        raise ValueError("demand must be finite and non-negative")

    states = (grid,) + tuple(without_line(grid, number) for number in contingency_lines)
    state_matrices = tuple(build_dc_matrices(state) for state in states)
    matrices = state_matrices[0]
    number_generators = len(grid.generators)
    number_buses = len(grid.buses)
    bus_index = {bus: index for index, bus in enumerate(grid.buses)}
    generator_map = np.zeros((number_buses, number_generators))
    for column, generator in enumerate(grid.generators):
        generator_map[bus_index[generator.bus], column] = 1.0

    costs = np.array([generator.cost_per_mwh for generator in grid.generators])
    number_states = len(states)
    number_variables = number_generators + number_states * number_buses
    objective = np.concatenate((costs, np.zeros(number_states * number_buses)))
    equality_rows = []
    equality_rhs_parts = []
    inequality_rows = []
    inequality_rhs_parts = []
    for state_index, (state, state_matrix) in enumerate(zip(states, state_matrices, strict=True)):
        angle_start = number_generators + state_index * number_buses
        nodal = np.zeros((number_buses, number_variables))
        nodal[:, :number_generators] = generator_map
        nodal[:, angle_start : angle_start + number_buses] = -state_matrix.bus_susceptance
        reference = np.zeros((1, number_variables))
        reference[0, angle_start + bus_index[grid.reference_bus]] = 1.0
        equality_rows.extend((nodal, reference))
        equality_rhs_parts.extend((demand, np.array([0.0])))

        flow = np.zeros((len(state.lines), number_variables))
        flow[:, angle_start : angle_start + number_buses] = state_matrix.branch_susceptance
        limits = np.array([line.limit_mw for line in state.lines])
        inequality_rows.extend((flow, -flow))
        inequality_rhs_parts.extend((limits, limits))
    equality_matrix = np.vstack(equality_rows)
    equality_rhs = np.concatenate(equality_rhs_parts)
    inequality_matrix = np.vstack(inequality_rows)
    inequality_rhs = np.concatenate(inequality_rhs_parts)
    bounds = [
        (generator.minimum_mw, generator.maximum_mw)
        for generator in grid.generators
    ] + [(None, None)] * (number_states * number_buses)

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
        all_angles = np.full((number_states, number_buses), np.nan)
        flows = np.full(len(grid.lines), np.nan)
        contingency_flows = tuple(
            tuple(float("nan") for _ in state.lines) for state in states[1:]
        )
    else:
        generation = solution.x[:number_generators]
        all_angles = solution.x[number_generators:].reshape(number_states, number_buses)
        flows = matrices.branch_susceptance @ all_angles[0]
        contingency_flows = tuple(
            tuple(float(value) for value in state_matrix.branch_susceptance @ state_angles)
            for state_matrix, state_angles in zip(state_matrices[1:], all_angles[1:], strict=True)
        )

    return DispatchResult(
        success=bool(solution.success),
        solver_status=int(solution.status),
        solver_message=str(solution.message),
        objective_cost=float(solution.fun) if solution.fun is not None else float("nan"),
        runtime_seconds=float(runtime),
        generation_mw=tuple(float(value) for value in generation),
        bus_angles=tuple(float(value) for value in all_angles[0]),
        line_flows_mw=tuple(float(value) for value in flows),
        contingency_line_numbers=tuple(
            tuple(line.number for line in state.lines) for state in states[1:]
        ),
        contingency_line_flows_mw=contingency_flows,
        total_demand_mw=float(demand.sum()),
    )
