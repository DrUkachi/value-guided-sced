"""Solve and save the first verified source-topology dispatch scenario."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from value_sced.grid import build_source_grid
from value_sced.oracle import solve_economic_dispatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "results" / "raw" / "verified_source_dispatch.json"
DEMAND_MW = np.array([0.0, 40.0, 35.0, 45.0, 50.0, 30.0])
TOLERANCE_MW = 1e-7


def main() -> None:
    grid = build_source_grid()
    result = solve_economic_dispatch(grid, DEMAND_MW)
    generation = np.asarray(result.generation_mw)

    assert result.success, result.solver_message
    assert np.isfinite(result.objective_cost)
    assert abs(generation.sum() - DEMAND_MW.sum()) <= TOLERANCE_MW
    for output, generator in zip(generation, grid.generators, strict=True):
        assert generator.minimum_mw - TOLERANCE_MW <= output
        assert output <= generator.maximum_mw + TOLERANCE_MW

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": "verified_source_dispatch",
        "topology": "source",
        "demand_mw": DEMAND_MW.tolist(),
        **result.to_dict(),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved verified dispatch to {OUTPUT_PATH}")
    print(f"Generation (MW): {generation.tolist()}")
    print(f"Cost: {result.objective_cost:.2f}; runtime: {result.runtime_seconds:.6f} s")


if __name__ == "__main__":
    main()
