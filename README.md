# Value-Guided SCED

Research code for studying which target-topology operating conditions should
receive expensive security-constrained economic dispatch (SCED) labels when
adapting a learned dispatch surrogate after an unseen line outage.

The implementation includes a synthetic six-bus DC network, a verified SCED
oracle, an MLP ensemble, deterministic feasibility repair, and random,
diversity, uncertainty, and decision-sensitive acquisition strategies.

## Setup

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
```

## Verification

```bash
python scripts/check_environment.py
python scripts/smoke_test.py
pytest
```

## Experiment workflow

The scripts support dataset generation, source-only evaluation, budgeted
acquisition experiments, aggregation, analysis, and experiment freezing. The
generated datasets and study outputs are intentionally not included in this
source-only repository.

Start by generating the configured datasets:

```bash
python scripts/generate_data.py
```

Then freeze the initial target selections and run an experiment seed:

```bash
python scripts/freeze_initial_labels.py
python scripts/run_experiment.py --seed 42
```
