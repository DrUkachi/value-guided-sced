"""Six-bus network data, topology variants, and DC power-flow matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Line:
    """A directed branch used to define flow sign conventions."""

    number: int
    from_bus: int
    to_bus: int
    reactance: float
    limit_mw: float


@dataclass(frozen=True)
class Generator:
    """Linear-cost generator data."""

    number: int
    bus: int
    minimum_mw: float
    maximum_mw: float
    cost_per_mwh: float


@dataclass(frozen=True)
class Grid:
    """Immutable network definition."""

    buses: tuple[int, ...]
    lines: tuple[Line, ...]
    generators: tuple[Generator, ...]
    reference_bus: int = 1


@dataclass(frozen=True)
class DCPowerFlowMatrices:
    """Matrices satisfying ``flow = Bf @ theta`` and ``injection = Bbus @ theta``."""

    incidence: np.ndarray
    branch_susceptance: np.ndarray
    bus_susceptance: np.ndarray


def _generators() -> tuple[Generator, ...]:
    return (
        Generator(1, 1, 0.0, 140.0, 12.0),
        Generator(2, 2, 0.0, 100.0, 18.0),
        Generator(3, 3, 0.0, 80.0, 25.0),
        Generator(4, 6, 0.0, 60.0, 35.0),
    )


def _lines() -> tuple[Line, ...]:
    # Parameters are a documented synthetic benchmark, not a named test system.
    return (
        Line(1, 1, 2, 0.10, 100.0),
        Line(2, 1, 4, 0.12, 100.0),
        Line(3, 1, 5, 0.15, 100.0),
        Line(4, 2, 3, 0.10, 100.0),
        Line(5, 2, 4, 0.08, 100.0),
        Line(6, 2, 5, 0.11, 100.0),
        Line(7, 3, 5, 0.09, 100.0),
        Line(8, 3, 6, 0.13, 100.0),
        Line(9, 4, 5, 0.10, 100.0),
        Line(10, 5, 6, 0.12, 100.0),
    )


def build_source_grid() -> Grid:
    """Return the intact six-bus, ten-line source topology."""

    return Grid(tuple(range(1, 7)), _lines(), _generators())


def build_target_grid() -> Grid:
    """Return the target topology with line 8 removed."""

    source = build_source_grid()
    return Grid(
        source.buses,
        tuple(line for line in source.lines if line.number != 8),
        source.generators,
        source.reference_bus,
    )


def is_connected(grid: Grid) -> bool:
    """Check undirected connectivity using a graph traversal."""

    adjacency = {bus: set() for bus in grid.buses}
    for line in grid.lines:
        adjacency[line.from_bus].add(line.to_bus)
        adjacency[line.to_bus].add(line.from_bus)
    visited: set[int] = set()
    pending = [grid.buses[0]]
    while pending:
        bus = pending.pop()
        if bus not in visited:
            visited.add(bus)
            pending.extend(adjacency[bus] - visited)
    return visited == set(grid.buses)


def build_dc_matrices(grid: Grid) -> DCPowerFlowMatrices:
    """Construct branch-incidence, branch-flow, and nodal matrices."""

    bus_index = {bus: index for index, bus in enumerate(grid.buses)}
    incidence = np.zeros((len(grid.lines), len(grid.buses)))
    for row, line in enumerate(grid.lines):
        incidence[row, bus_index[line.from_bus]] = 1.0
        incidence[row, bus_index[line.to_bus]] = -1.0
    susceptance = np.array([1.0 / line.reactance for line in grid.lines])
    branch_susceptance = susceptance[:, None] * incidence
    bus_susceptance = incidence.T @ branch_susceptance
    return DCPowerFlowMatrices(incidence, branch_susceptance, bus_susceptance)
