# Experiment-Change Log

Record every change to the frozen experimental contract here before implementing it. Each entry must state the date, old decision, new decision, reason, and whether target-test results informed the change.

## 2026-08-11 — Freeze operating-condition sampling and oracle tolerances

- **Old decision:** Operating-condition ranges and numerical feasibility tolerances were unresolved.
- **New decision:** Sample total gross demand uniformly from 140–370 MW; distribute it over buses 2–6 with a Dirichlet distribution centred on the documented base profile with concentration 80; sample renewable injection independently at buses 4 and 5 from 0–35% of local gross demand. Use seed 42. Fix balance and limit tolerances at `1e-7` MW and binding tolerance at `1e-5` MW.
- **Reason:** The initial pre-freeze 140–300 MW trial produced no binding security constraints (maximum observed loading was about 81%), which would not exercise the SCED component. The upper bound was raised before ML work, while remaining below aggregate generation capacity; infeasible random draws are rejected and counted.
- **Target-test information used:** An aggregate pre-freeze engineering diagnostic included generated target cases. No surrogate, acquisition result, label-efficiency metric, or per-case target-test result was used. The trial files were replaced before being declared the experimental dataset.

## 2026-08-11 — Preventive security formulation

- **Old decision:** The oracle enforced only base-case DC line limits.
- **New decision:** One preventive generator dispatch must satisfy the base topology and the locked line-3 and line-4 outage states, with separate voltage-angle variables in each state.
- **Reason:** This implements the frozen DC-SCED security-contingency contract.
- **Target-test information used:** No.

## 2026-08-11 — Correct synthetic line ratings before ML experiments

- **Old decision:** Every synthetic line had a 100 MW rating.
- **New decision:** Every synthetic line has a 75 MW rating; topology, reactances, generator data, and contingency identities are unchanged.
- **Reason:** Pre-freeze oracle diagnostics showed that even the expanded operating range produced no binding constraints, so source and target labels would mostly collapse to the same uncongested merit-order rule. A 75 MW uniform rating retains a transparent synthetic benchmark while exercising security-constrained redispatch.
- **Target-test information used:** Only aggregate pre-freeze loading and binding-frequency diagnostics were inspected. No ML model, acquisition method, or target-test performance metric existed. All trial datasets were replaced after this correction.

## 10 August 2026 — Shortened schedule

- **Old decision:** Five final experiment seeds: 11, 22, 33, 44 and 55.
- **New decision:** Three final experiment seeds: 11, 22 and 33.
- **Reason:** Align the frozen contract with the shortened study schedule and its explicit three-seed requirement.
- **Target-test information used:** No. No target-test experiment has been run.
- **Files affected:** `configs/base.yaml`, `docs/protocol.md`, and `paper/main.tex`.

## 2026-08-12 — Source-only baseline split and features

- **Old decision:** The precise source train/validation split and model feature layout were unresolved.
- **New decision:** Use a deterministic 80/20 split of the 220 source cases with seed 42. Represent each input by the six gross-demand values followed by the six renewable-generation values; predict the four generator outputs. Fit all standardisation and early stopping using source data only.
- **Reason:** A source validation set is required for early stopping without leaking target-test information, and the feature layout directly implements the frozen operating-condition definition.
- **Target-test information used:** No.
- **Files affected:** `configs/base.yaml`, `src/value_sced/surrogate.py`, and `scripts/run_experiment.py`.

## 2026-08-12 — Paired topology-shift diagnostic

- **Old decision:** Source-domain and zero-shot target performance would be reported on their independently sampled evaluation cases.
- **New decision:** Retain those evaluations and additionally solve the source-topology oracle for the exact 80 target-test operating conditions, producing a paired source-versus-target topology diagnostic.
- **Reason:** The paired comparison holds operating conditions fixed and therefore attributes changes in error and operational regret to removal of line 8 rather than sampling variation.
- **Target-test information used:** The first aggregate baseline was run before this diagnostic was added. No target result was used to tune the model, repair, features, training schedule, or acquisition method; the diagnostic changes evaluation design only and is recorded transparently.
- **Files affected:** `scripts/run_experiment.py` and processed baseline outputs.

## Entry template

- **Date:** YYYY-MM-DD
- **Old decision:**
- **New decision:**
- **Reason:**
- **Target-test information used:** Yes/No, with explanation if yes.
- **Files affected:**
