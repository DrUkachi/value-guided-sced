# Interview narrative and rehearsal sheet

## Three-minute version

My project asks a simple data-selection question: if power-system simulations are expensive, which operating conditions are actually worth simulating? I make that concrete using security-constrained economic dispatch. A neural surrogate predicts generator dispatch, but the network changes when one transmission line is removed. I then have a limited budget for obtaining new target-topology SCED labels.

I compare four selection strategies under exactly the same budget: random sampling, diversity, ensemble uncertainty, and a decision-sensitive heuristic. The heuristic starts from uncertainty and increases the score when the predicted dispatch appears close to a line limit in the base network or either of two contingencies. Importantly, it uses only candidate inputs and model predictions; hidden SCED solutions and the test set remain unavailable.

The controlled experiment uses a synthetic six-bus grid, 220 source labels, 160 target candidates, an untouched 80-case target test set, and three final seeds. Each strategy starts with the same 12 target labels and acquires four batches of 12. I evaluate dispatch error, raw security feasibility, the amount of deterministic feasibility repair, and repaired operating-cost regret.

The result is useful because it is not the result I originally hoped for. At 60 labels, uncertainty sampling has the lowest mean MAE, 1.387 MW, and the lowest mean repaired-cost regret, 21.06. Adding predicted operational criticality gives 1.425 MW and 22.42, so it does not improve uncertainty. In one counterexample, the weighted method incurs 54.253 more regret on a test case. The proxy can inherit model error and direct labels to the wrong places.

My conclusion is therefore modest: targeted labels help, operational evaluation changes what we learn from the experiment, but a decision-aware acquisition score is only as good as its estimate of downstream value. The PhD extension would replace this heuristic with calibrated value-of-information methods and test them across larger grids, multiple topology changes, AC or dynamic simulators, and more realistic scenario distributions.

## Seven-minute version

### 0:00--1:00 — Motivation and question

Power-system machine learning often assumes that more simulated training cases are uniformly useful. That assumption becomes questionable when labels require constrained optimization or dynamic simulation, and when the learned output is used for an operational decision. My question is: after the network changes, which target operating conditions should receive a limited number of new simulation labels?

I study this through preventive security-constrained economic dispatch. The source is a synthetic six-bus, ten-line grid. The target removes line 8, while every dispatch must still respect the base network and locked outages of lines 3 and 4. This gives a physically defined structural shift rather than an abstract distribution shift.

### 1:00--2:15 — Gap and hypothesis

A generic active learner might query uncertain cases or cases that improve input-space coverage. Those are sensible baselines, but an error near an operational limit may matter more than the same error in an uncongested case. I therefore tested whether uncertainty weighted by predicted network criticality could acquire more valuable labels.

This is related to decision-focused learning, but it is not a decision-focused training loss. It is a pool-based acquisition rule. The candidate's demand and renewable inputs are visible; its secure dispatch, cost, and flows remain hidden until selection.

### 2:15--3:45 — Experimental design

The oracle is a linear DC-SCED. I generated 220 labelled source cases, 160 target candidates, and 80 permanently held-out target cases. The surrogate is a five-member MLP ensemble. Every method begins from the same 12 target labels within a seed and adds four batches of 12, producing budgets from 12 to 60 labels.

The four strategies are random, farthest-first diversity, ensemble disagreement, and the combined score. For the combined score I estimate base and contingency line-loading ratios from the ensemble-mean dispatch, normalize criticality and uncertainty over the remaining pool, and multiply uncertainty by one plus criticality. The coefficient was frozen at one and was never tuned on the test set.

Fairness controls matter here: all methods use the same source data, initial labels, architecture, retraining schedule, repair algorithm, seeds, and test cases. Candidate labels were precomputed only for convenience and are isolated behind a label store. Automated tests check that neither candidate labels nor test data enter acquisition.

### 3:45--5:15 — Metrics and results

I report prediction and operational metrics separately. Raw feasibility requires balance, generator, base-line, and contingency-line violations all to be within the frozen tolerance. Because an unconstrained regressor rarely satisfies exact equalities, I also apply the same deterministic linear repair to every prediction. Repair magnitude is the total absolute change in generator dispatch; regret is the repaired dispatch cost minus the oracle cost.

All methods share a mean MAE of 1.922 MW and regret of 28.13 at the common 12-label start. At 60 labels, uncertainty is best on average: 1.387 MW MAE and 21.06 regret. Decision-sensitive acquisition reaches 1.425 and 22.42; random reaches 1.582 and 23.23; diversity reaches 1.504 and 23.24. Exact raw feasibility remains zero under the deliberately stringent tolerance, so continuous violations and repair magnitude are more informative.

The topology penalty is present but modest in the matched diagnostic: removing line 8 increases MAE by 0.077 MW and regret by 0.55 on identical cases. This is a single debugging-seed diagnostic, not a precise three-seed estimate.

### 5:15--6:15 — Negative result and limitations

The operational weighting does not beat uncertainty. Its effect varies by seed and budget, and a concrete seed-33 case has 54.253 more repaired regret than uncertainty. My interpretation is that predicted criticality is endogenous to an imperfect model. Multiplying by it can amplify error rather than identify true downstream value.

The evidence is intentionally bounded: six buses, DC physics, linear costs, one target outage, two contingencies, three seeds, and a cheap oracle. It demonstrates a research protocol and failure mechanism, not deployment readiness or a universal method ranking.

### 6:15--7:00 — PhD extension

The next research step is to estimate value rather than assume a proxy. I would calibrate uncertainty and constraint-risk estimates using validation data, target expected repair or regret reduction, and test multiple structural shifts. I would then scale to larger grids, AC optimal power flow and dynamic simulators, with scenario distributions informed by real measurements. The core research programme is about allocating simulation effort according to downstream decision value, with negative results treated as evidence about when the value estimator fails.

## Likely questions and concise answers

### What is novel?

The defensible novelty is the controlled combination of topology-shift adaptation, equal-budget target-case acquisition, and decision-oriented SCED evaluation. I do not claim the multiplicative score is a new general algorithm; it is a falsifiable first proxy whose failure motivates principled value-of-information work.

### If the SCED oracle is cheap, why discuss expensive labels?

This six-bus linear oracle is intentionally cheap so the protocol can be audited and repeated. It stands in for richer AC, dynamic-security, or co-simulation labels. The current experiment tests selection logic, not a wall-clock speedup claim; scaling must separately demonstrate computational benefit.

### Why is raw feasibility zero?

The precommitted tolerance is $10^{-7}$ MW and the MLP is unconstrained, so exact balance alone is difficult. Zero is therefore a diagnostic of direct deployability, not a useful ranking by itself. I report continuous violations, repair magnitude, and repaired regret alongside it.

### Does the negative result invalidate the idea?

It invalidates the frozen operational-weighting hypothesis in this setting, which is valuable. It shows that an operational proxy derived from an imperfect prediction can amplify error. The broader question remains open and now has a concrete failure mechanism to address.

### Why only three seeds?

The sprint prioritized a complete leakage-controlled matrix. Three seeds make the intervals wide, so rankings are suggestive. A confirmatory study needs more seeds and independently regenerated datasets.

### How does this fit a PhD group?

The project connects power-system simulation, constrained optimization, active learning, and trustworthy evaluation. I can contribute a working reproducible baseline immediately, then extend it toward the group's strongest simulators, grid models, and uncertainty or decision-making methods.

### Why not use a graph neural network?

The experiment isolates data acquisition, so a small MLP avoids confounding the result with architectural novelty. A topology-aware GNN is a sensible extension once the acquisition claim is established across several topologies.

### Could repair hide a bad surrogate?

Yes. That is why raw violations and repair magnitude are reported separately from repaired regret. A model cannot be called operationally strong merely because the optimizer repairs it cheaply.

## Two-pass rehearsal record

### Pass 1 — content audit

- Target: deliver the three-minute version without equations.
- Checkpoints: question by 0:30; four methods by 1:15; result by 2:15; limitation and extension by 3:00.
- Questions to tighten after speaking: Can I explain preventive SCED in one sentence? Can I distinguish decision-sensitive acquisition from decision-focused training without jargon?

### Pass 2 — evidence audit

- Target: deliver the seven-minute version while advancing exactly five slides.
- Checkpoints: slide 2 by 1:00; slide 3 by 2:15; slide 4 by 3:45; slide 5 by 6:15.
- Questions to tighten after speaking: Why is the topology penalty modest? What evidence would justify a stronger superiority claim? How would expected value of information be estimated without candidate labels?

These are structured rehearsal passes, not a claim that the candidate has spoken them aloud. Record actual duration and difficult transitions after each live rehearsal.

## Artifact archive

- Frozen experiment commit: `16896e0ecb411fd07b16f4b3d430f88e5606bc7f`
- Exact configuration: `configs/base.yaml`
- SHA-256 artifact inventory: `results/processed/artifact_manifest.json`
- Paper: `paper/main.pdf`
- Slides: `interview/slides.pdf`
