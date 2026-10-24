# Research Plan: 10–20 August 2026

## Project

**Working title:** *Which Operating Conditions Are Worth Simulating? Value-Guided Data Acquisition for Security-Constrained Redispatch under Topology Shift*

**Experiment freeze:** 18 August 2026  
**Complete paper:** 20 August 2026

## 1. What this project demonstrates

The experiment asks a focused question:

> When simulations are costly, can we select operating conditions according to their value for a downstream grid-control decision rather than sampling only for statistical coverage?

The concrete downstream task is security-constrained economic dispatch (SCED). A machine-learning surrogate predicts dispatch decisions from operating conditions. The training grid and target grid differ structurally because one transmission line is removed in the target topology. Candidate target-grid cases are then selected for expensive SCED labelling using four acquisition strategies:

1. Random sampling
2. Diversity or coverage sampling
3. Predictive uncertainty sampling
4. Decision-sensitive value-guided sampling

The main comparison is label efficiency: how quickly each strategy improves operational performance on an untouched target-grid test set.

## 2. Minimum defensible research claim

The paper will not claim to solve data generation for real electricity grids. It will provide a controlled proof of concept showing whether decision-sensitive acquisition is more useful than generic sampling when:

- labels come from a constrained optimisation oracle;
- the topology changes between source and target domains;
- model errors have unequal operational consequences; and
- evaluation uses feasibility, repair cost and regret—not prediction error alone.

If the proposed method does not outperform the baselines, the result is still useful: it identifies when uncertainty or decision sensitivity fails to translate into operational value.

## 3. Experimental contract

Freeze these choices on 10 August. Change them only to fix an error, and record every change.

| Component | Frozen choice |
|---|---|
| Grid | Synthetic 6-bus transmission system |
| Network | 10 lines, 4 generators |
| Source topology | All 10 lines available |
| Target topology | Line 8 removed |
| Security contingencies | Outage of line 3 or line 4 |
| Optimisation oracle | Linear DC-SCED using SciPy `linprog` |
| Source labelled cases | 220 |
| Target candidate pool | 160 |
| Target test set | 80, held out before acquisition |
| Initial target labels | 12 |
| Acquisition batches | 4 batches of 12 |
| Maximum target labels | 60 |
| Surrogate | Small MLP ensemble |
| Acquisition methods | Random, diversity, uncertainty, decision-sensitive |
| Core random seeds | 3 |
| Optional extension | Seeds 4–5 only if all core outputs exist by 17 August |

### Inputs and outputs

Define one scenario vector from quantities known before dispatch, such as:

- bus demands;
- renewable injections or availability;
- generator availability; and
- the topology indicator.

The prediction target is the SCED generator dispatch. Do not include SCED outputs, post-dispatch line flows or objective values among the model inputs.

### Structural shift

The source and target domains use the same variable definitions but different network structures:

- **Source:** line 8 is in service.
- **Target:** line 8 is out of service.

Demonstrate the shift with engineering quantities rather than a generic distribution-distance score:

1. Show that the source and target line-incidence or susceptance matrices differ.
2. Compare line-loading patterns under matched or similarly distributed operating conditions.
3. Report how frequently the binding constraint changes.
4. Report the zero-shot degradation of the source-trained surrogate on the target topology.

### Expensive label

For each selected target operating condition, the SCED oracle returns:

- optimal generator dispatch;
- operating cost;
- feasibility status;
- base-case line flows; and
- contingency line flows.

Record wall-clock solve time. This allows the paper to discuss simulation cost honestly even though the small DC-SCED itself is inexpensive.

## 4. Machine-learning role

Machine learning is central but deliberately simple. The MLP ensemble learns a mapping from operating conditions to dispatch. Its purposes are to:

- expose transfer failure under topology shift;
- provide predictive uncertainty;
- estimate which unlabelled target cases may be valuable; and
- test whether targeted SCED labels improve downstream decisions faster than untargeted labels.

The paper is not about inventing a new neural architecture. A simple model keeps the contribution centred on data selection and decision-oriented validation.

## 5. Acquisition methods

At every acquisition round, score only the current unlabelled target pool. Select 12 cases, query SCED, add the resulting labels to the training data and retrain the surrogate from the same protocol.

### Random

Uniformly sample from the remaining target pool. Repeat per seed.

### Diversity

Select cases that cover the input space, using farthest-first sampling or k-means distance in standardised scenario space.

### Uncertainty

Use ensemble disagreement in predicted dispatch. A simple score is the mean variance across generator outputs.

### Decision-sensitive value guidance

Combine uncertainty with an inexpensive operational-sensitivity proxy available without the true SCED label. Start with:

\[
s(x)=u(x)\,[1+\lambda r(x)],
\]

where (u(x)) is ensemble disagreement and (r(x)) is a normalised proxy for operational risk, such as predicted proximity to generator or line limits after a deterministic feasibility-repair step.

The scoring rule must not use the candidate's hidden SCED solution, objective value or test-set results. If the risk proxy becomes unreliable or too complex, use a clearly labelled uncertainty-times-proximity baseline and state that it is a preliminary value heuristic.

## 6. Evaluation

Evaluate every acquisition method at 12, 24, 36, 48 and 60 target labels on the same untouched target test set.

### Primary metrics

1. **Post-repair operating-cost regret** relative to SCED
2. **Raw feasibility rate** before repair
3. **Repair magnitude or repair cost**
4. **Label efficiency**, measured by performance versus target labels acquired

### Secondary metrics

- generator-dispatch MAE;
- generator limit violations;
- line limit violations in the base case;
- line limit violations under the two contingencies;
- SCED solve time; and
- acquisition/retraining time.

Report mean and uncertainty across three seeds. Show individual seeds where space permits. Avoid making a strong superiority claim from a single seed.

## 7. Scope guardrails

Do not add any of the following before 20 August:

- AC optimal power flow;
- reinforcement learning;
- a second grid;
- GNN-versus-Transformer comparisons;
- deep generative models;
- broad hyperparameter searches;
- Real-E as a core dependency; or
- claims about deployment or wide-area protection.

Real-E can appear in the motivation or future-work discussion: real measurement correlations could later inform scenario construction and shift diagnostics. It should not delay this controlled proof of concept.

## 8. Day-by-day execution plan

### Monday, 10 August — Freeze the study and start the oracle

**Outcome:** a frozen protocol, runnable repository and one valid base dispatch.

#### Research contract

- [ ] Write the research question in one sentence.
- [ ] Freeze the source and target topology definitions.
- [ ] Freeze scenario counts, acquisition budgets, metrics and three seeds.
- [ ] Define the four acquisition strategies in pseudocode.
- [ ] Write hypotheses H1–H4.
- [ ] Write leakage rules and an experiment-change log.

#### Repository and paper

- [ ] Create the repository structure and environment file.
- [ ] Add one base configuration file containing all frozen constants.
- [ ] Create a paper skeleton with title, abstract placeholders and section headings.
- [ ] Add a README with one command for the smallest smoke test.

#### Additional work required because the schedule is now shorter

- [ ] Encode the six buses, ten lines, four generators, costs, capacities and line limits.
- [ ] Implement separate source- and target-topology builders.
- [ ] Add an automated assertion that target line 8 is unavailable and both grids remain connected.
- [ ] Create the DC power-flow or nodal-angle constraint matrices.
- [ ] Solve one base economic-dispatch case with `scipy.optimize.linprog`.
- [ ] Verify power balance, generator bounds and solver status in a unit test.
- [ ] Save the dispatch, objective value and runtime as the first reproducible result.
- [ ] Commit or checkpoint the working state and record unresolved issues for 11 August.

**Stop condition:** Do not spend today tuning the MLP or polishing prose. A single correct optimisation solve is more valuable.

### Tuesday, 11 August — Finish SCED and generate the datasets

**Outcome:** a tested oracle and frozen source, target-pool and target-test datasets.

- [x] Add base-case line-flow constraints.
- [x] Add the two line-outage security constraints.
- [x] Test feasible, congested and deliberately infeasible cases.
- [x] Generate 220 labelled source cases.
- [x] Generate 160 unlabelled target-pool cases while retaining hidden labels for offline evaluation only.
- [x] Generate and permanently freeze 80 target-test cases.
- [x] Store scenario ID, split, seed, topology and oracle metadata.
- [x] Plot scenario distributions and compare source/target loading and binding constraints.

### Wednesday, 12 August — Build the ML and evaluation baseline

**Outcome:** a source-trained surrogate with zero-shot target results.

- [x] Train the MLP ensemble on source cases.
- [x] Evaluate source-domain performance.
- [x] Evaluate zero-shot target performance.
- [x] Implement deterministic feasibility repair.
- [x] Compute MAE, raw feasibility, repair, post-repair regret and violations.
- [x] Save the first table demonstrating topology-shift degradation.

### Thursday, 13 August — Build the acquisition loop

**Outcome:** one common, leakage-safe loop for all strategies.

- [x] Freeze the 12 initial target labels per seed.
- [x] Implement pool bookkeeping and batch acquisition.
- [x] Implement random acquisition.
- [x] Implement diversity acquisition.
- [x] Confirm that the test set and hidden candidate labels never enter scoring.
- [x] Run a two-round toy smoke test.

### Friday, 14 August — Implement value signals

**Outcome:** uncertainty and decision-sensitive scoring are operational.

- [x] Implement ensemble-disagreement scoring.
- [x] Implement the operational-risk proxy.
- [x] Implement combined decision-sensitive scoring.
- [x] Inspect the top ten selected cases from each method.
- [x] Check score scales, duplicates and deterministic tie handling.

### Saturday, 15 August — Complete one end-to-end seed

**Outcome:** every method reaches the 60-label budget for one seed.

- [x] Run all five label budgets for all four methods.
- [x] Produce provisional learning curves and a results table.
- [x] Audit leakage, split identity and retraining consistency.
- [x] Fix only correctness and reproducibility problems.
- [x] Write a short failure log.

### Sunday, 16 August — Run the three core seeds

**Outcome:** the main experimental matrix is running or complete.

- [x] Launch seeds 1–3 using frozen configurations.
- [x] Log failures without silently changing the protocol.
- [x] Re-run only failed configurations after documented fixes.
- [x] Aggregate completed outputs into one tidy results table.

### Monday, 17 August — Analyse the main results

**Outcome:** final main tables and figures exist.

- [x] Complete missing seed runs.
- [x] Compute mean, standard deviation and confidence intervals where appropriate.
- [x] Produce regret-versus-label and feasibility-versus-label curves.
- [x] Produce the structural-shift diagnostic figure or table.
- [x] Compare operational metrics with dispatch MAE.
- [x] Draft the principal findings without overstating them.

### Tuesday, 18 August — Minimal ablations and experiment freeze

**Outcome:** experiments are frozen by the end of the day.

- [x] Ablate the operational-risk term: uncertainty alone versus combined score.
- [x] Compare raw predictions with repaired decisions.
- [x] Analyse one failure mode or counterexample.
- [x] Run a final reproducibility check from saved configuration.
- [x] Freeze datasets, results, figures and commit identifier.
- [x] Move all uncompleted ideas to future work.

### Wednesday, 19 August — Write the full draft

**Outcome:** a complete paper draft can be read from beginning to end.

- [x] Finalise Methods and Experimental Setup first.
- [x] Write Results directly from frozen tables and figures.
- [x] Write Introduction, Related Work, Discussion and Limitations.
- [x] Complete the abstract last.
- [x] Add references and verify every factual claim.
- [x] Clean the repository and reproduction instructions.

### Thursday, 20 August — Finalize and archive the paper

**Outcome:** a complete paper draft and archived experiment artifacts.

- [x] Revise for one clear contribution and remove unsupported claims.
- [x] Check that every figure is readable and every metric is defined.
- [x] Export the paper to PDF and archive the exact experiment configuration.

## 9. Hypotheses

- **H1 — Shift:** A source-trained surrogate performs worse on the target topology than on the source topology.
- **H2 — Operational mismatch:** Dispatch MAE alone does not reliably rank models by feasibility or regret.
- **H3 — Label efficiency:** Targeted acquisition improves target operational performance faster than random sampling.
- **H4 — Value guidance:** Combining uncertainty with an operational-risk proxy is at least as useful as uncertainty alone at small label budgets.

H4 is deliberately testable and may be rejected. A negative result must be reported rather than tuned away.

## 10. Expected paper structure

1. Abstract
2. Introduction and research question
3. Related work
4. Problem formulation
5. Experimental design
6. Acquisition strategies
7. Decision-oriented evaluation
8. Results
9. Failure analysis and limitations
10. Connection to the proposed PhD
11. Conclusion

## 11. Definition of success by 20 August

The sprint succeeds if it produces:

- a reproducible DC-SCED oracle;
- a documented source-to-target topology shift;
- four acquisition strategies evaluated under one fair protocol;
- three-seed learning curves for operational regret and feasibility;
- at least one honest ablation or failure analysis;
- a complete draft paper;
- a clean repository or reproducibility bundle.

A statistically decisive positive result is not required. A well-controlled negative result, clear limitations and a credible next-step research programme are academically defensible.

## 12. Emergency fallback

If implementation slips by more than one day:

1. Keep only random, uncertainty and decision-sensitive acquisition.
2. Retain three seeds but reduce candidate cases before reducing test integrity.
3. Keep all five label budgets if possible; otherwise report 12, 36 and 60.
4. Preserve the held-out test set, leakage rules and operational metrics.
5. Never replace missing evidence with a stronger claim.

The non-negotiable deliverable is one reproducible end-to-end experiment plus a clear paper narrative.
