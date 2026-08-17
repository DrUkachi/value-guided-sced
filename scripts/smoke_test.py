"""Run the smallest end-to-end network and dispatch verification."""

from __future__ import annotations

import numpy as np

from value_sced.grid import build_source_grid, build_target_grid, is_connected
from value_sced.oracle import audit_dispatch, solve_economic_dispatch


def main() -> None:
    source = build_source_grid()
    target = build_target_grid()
    assert len(source.lines) == 10
    assert 8 not in {line.number for line in target.lines}
    assert is_connected(source) and is_connected(target)

    demand = np.array([0.0, 40.0, 35.0, 45.0, 50.0, 30.0])
    result = solve_economic_dispatch(source, demand, (3, 4))
    generation = np.asarray(result.generation_mw)
    assert result.success
    assert np.isfinite(result.objective_cost)
    assert np.isclose(generation.sum(), demand.sum(), atol=1e-7)
    assert all(
        generator.minimum_mw - 1e-7 <= output <= generator.maximum_mw + 1e-7
        for output, generator in zip(generation, source.generators, strict=True)
    )
    assert audit_dispatch(source, demand, result, (3, 4)).passes(1e-7)
    print(
        "Smoke test passed: "
        f"dispatch={generation.tolist()}, cost={result.objective_cost:.2f}"
    )


if __name__ == "__main__":
    main()
