"""Operating-condition generation and leakage-safe dataset partitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class OperatingCondition:
    """Quantities known before dispatch for one scenario."""

    gross_demand_mw: tuple[float, ...]
    renewable_mw: tuple[float, ...]

    @property
    def net_demand_mw(self) -> np.ndarray:
        return np.asarray(self.gross_demand_mw) - np.asarray(self.renewable_mw)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["net_demand_mw"] = self.net_demand_mw.tolist()
        return payload


def sample_operating_condition(
    rng: np.random.Generator,
    total_range_mw: tuple[float, float],
    demand_profile: np.ndarray,
    demand_concentration: float,
    renewable_bus_indices: tuple[int, ...],
    renewable_fraction_range: tuple[float, float],
) -> OperatingCondition:
    """Sample gross demand and renewable injections from the frozen distribution."""

    profile = np.asarray(demand_profile, dtype=float)
    positive = profile > 0.0
    shares = np.zeros_like(profile)
    shares[positive] = rng.dirichlet(profile[positive] * demand_concentration)
    total = rng.uniform(*total_range_mw)
    gross = total * shares
    renewable = np.zeros_like(gross)
    for index in renewable_bus_indices:
        fraction = rng.uniform(*renewable_fraction_range)
        renewable[index] = fraction * gross[index]
    return OperatingCondition(
        tuple(float(value) for value in gross),
        tuple(float(value) for value in renewable),
    )
