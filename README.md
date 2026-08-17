# Value-Guided SCED

Research code for studying which target-topology operating conditions should receive expensive security-constrained economic dispatch (SCED) labels when adapting a learned dispatch surrogate after an unseen line outage.

The scientific contract is in [`docs/protocol.md`](docs/protocol.md), and all frozen machine-readable choices are in [`configs/base.yaml`](configs/base.yaml). The repository includes a synthetic six-bus DC network and a first linear economic-dispatch solver; no model training is included yet.

## Setup

Requires Python 3.10 or newer.

```bash
python -m pip install -e ".[dev]"
```

## Smallest smoke test

After setup, verify both topologies and solve the base dispatch with one command:

```bash
python scripts/smoke_test.py
```

For the complete local verification, run `python scripts/check_environment.py`, `python scripts/solve_verified_dispatch.py`, and `pytest`.

Run `python scripts/run_experiment.py` to reproduce the budget-zero source-only
MLP baseline, paired topology-shift diagnostic, feasibility repair, and initial
results under `results/processed/`.

## Planned workflow

1. Generate source, target-candidate and untouched target-test operating conditions.
2. Solve the source and selected target conditions with the SCED oracle.
3. Train a compact MLP ensemble.
4. Compare random, diversity, uncertainty and decision-sensitive acquisition under equal budgets.
5. Report predictive error, raw security feasibility, repair magnitude and repaired-cost regret.

The frozen Tuesday datasets are generated with `uv run python scripts/generate_data.py`.
The command refuses to overwrite an existing dataset; `--force` is reserved for a
documented protocol change.

Freeze the common initial target cases for all experiment seeds with
`uv run python scripts/freeze_initial_labels.py`. Acquisition selection sees only
candidate identifiers and features; the label store reveals oracle outputs after
each selected batch.
