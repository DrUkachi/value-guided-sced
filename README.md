# Value-Guided SCED

Research code for studying which target-topology operating conditions should receive expensive security-constrained economic dispatch (SCED) labels when adapting a learned dispatch surrogate after an unseen line outage.

The scientific contract is in [`docs/protocol.md`](docs/protocol.md), and all frozen machine-readable choices are in [`configs/base.yaml`](configs/base.yaml). The repository includes a synthetic six-bus DC network and a first linear economic-dispatch solver; no model training is included yet.

## Setup

Requires Python 3.10 or newer.

```bash
python -m pip install -e ".[dev]"
python scripts/check_environment.py
python scripts/solve_verified_dispatch.py
pytest
```

## Planned workflow

1. Generate source, target-candidate and untouched target-test operating conditions.
2. Solve the source and selected target conditions with the SCED oracle.
3. Train a compact MLP ensemble.
4. Compare random, diversity, uncertainty and decision-sensitive acquisition under equal budgets.
5. Report predictive error, raw security feasibility, repair magnitude and repaired-cost regret.
