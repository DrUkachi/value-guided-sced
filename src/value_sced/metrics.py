"""Predictive, raw operational, and post-repair evaluation metrics."""

from __future__ import annotations

import numpy as np

from value_sced.grid import Grid, build_dc_matrices, without_line
from value_sced.repair import repair_dispatch


def _line_overload(
    grid: Grid, demand: np.ndarray, generation: np.ndarray
) -> float:
    """Return maximum overload after assigning imbalance to the reference bus."""

    bus_index = {bus: index for index, bus in enumerate(grid.buses)}
    injections = -np.asarray(demand, dtype=float).copy()
    for output, generator in zip(generation, grid.generators, strict=True):
        injections[bus_index[generator.bus]] += output
    injections[bus_index[grid.reference_bus]] -= injections.sum()
    matrices = build_dc_matrices(grid)
    non_reference = [index for index, bus in enumerate(grid.buses) if bus != grid.reference_bus]
    angles = np.zeros(len(grid.buses))
    angles[non_reference] = np.linalg.solve(
        matrices.bus_susceptance[np.ix_(non_reference, non_reference)],
        injections[non_reference],
    )
    flows = matrices.branch_susceptance @ angles
    limits = np.asarray([line.limit_mw for line in grid.lines])
    return float(np.maximum(np.abs(flows) - limits, 0.0).max())


def evaluate_predictions(
    grid: Grid,
    demands_mw: np.ndarray,
    predictions_mw: np.ndarray,
    oracle_generation_mw: np.ndarray,
    oracle_costs: np.ndarray,
    *,
    contingency_lines: tuple[int, ...],
    balance_tolerance_mw: float,
    limit_tolerance_mw: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Evaluate predictions and repair each using one common secure procedure."""

    demands = np.asarray(demands_mw, dtype=float)
    predictions = np.asarray(predictions_mw, dtype=float)
    truth = np.asarray(oracle_generation_mw, dtype=float)
    errors = predictions - truth
    lower = np.asarray([generator.minimum_mw for generator in grid.generators])
    upper = np.asarray([generator.maximum_mw for generator in grid.generators])

    rows: list[dict[str, object]] = []
    for index, (demand, prediction, optimum, optimum_cost) in enumerate(
        zip(demands, predictions, truth, oracle_costs, strict=True)
    ):
        balance_error = float(abs(prediction.sum() - demand.sum()))
        generator_violation = float(
            np.maximum(np.maximum(lower - prediction, prediction - upper), 0.0).max()
        )
        base_overload = _line_overload(grid, demand, prediction)
        contingency_overload = max(
            _line_overload(without_line(grid, line), demand, prediction)
            for line in contingency_lines
        )
        raw_feasible = (
            balance_error <= balance_tolerance_mw
            and generator_violation <= limit_tolerance_mw
            and base_overload <= limit_tolerance_mw
            and contingency_overload <= limit_tolerance_mw
        )
        base_safe = (
            balance_error <= balance_tolerance_mw
            and generator_violation <= limit_tolerance_mw
            and base_overload <= limit_tolerance_mw
        )
        repair = repair_dispatch(grid, demand, prediction, contingency_lines)
        regret = repair.operating_cost - float(optimum_cost) if repair.success else float("nan")
        rows.append(
            {
                "sample_index": index,
                "prediction_mw": prediction.tolist(),
                "oracle_generation_mw": optimum.tolist(),
                "balance_error_mw": balance_error,
                "generator_limit_violation_mw": generator_violation,
                "base_line_overload_mw": base_overload,
                "contingency_line_overload_mw": contingency_overload,
                "raw_feasible": bool(raw_feasible),
                "false_safe": bool(base_safe and contingency_overload > limit_tolerance_mw),
                "repaired_generation_mw": list(repair.generation_mw),
                "repair_magnitude_mw": repair.repair_magnitude_mw,
                "repaired_cost_regret": regret,
            }
        )

    regrets = np.asarray([row["repaired_cost_regret"] for row in rows], dtype=float)
    repair_magnitudes = np.asarray([row["repair_magnitude_mw"] for row in rows], dtype=float)
    metrics = {
        "sample_count": float(len(rows)),
        "dispatch_mae_mw": float(np.abs(errors).mean()),
        "dispatch_rmse_mw": float(np.sqrt(np.square(errors).mean())),
        "maximum_generator_error_mw": float(np.abs(errors).max()),
        "mean_power_balance_error_mw": float(np.mean([row["balance_error_mw"] for row in rows])),
        "mean_generator_limit_violation_mw": float(np.mean([row["generator_limit_violation_mw"] for row in rows])),
        "mean_base_line_overload_mw": float(np.mean([row["base_line_overload_mw"] for row in rows])),
        "mean_contingency_line_overload_mw": float(np.mean([row["contingency_line_overload_mw"] for row in rows])),
        "raw_feasible_percentage": float(100.0 * np.mean([row["raw_feasible"] for row in rows])),
        "false_safe_percentage": float(100.0 * np.mean([row["false_safe"] for row in rows])),
        "mean_repair_magnitude_mw": float(np.mean(repair_magnitudes)),
        "mean_repaired_cost_regret": float(np.mean(regrets)),
        "median_repaired_cost_regret": float(np.median(regrets)),
        "p95_repaired_cost_regret": float(np.percentile(regrets, 95)),
        "worst_repaired_cost_regret": float(np.max(regrets)),
    }
    return metrics, rows
