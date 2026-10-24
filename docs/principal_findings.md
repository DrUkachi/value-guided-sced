# Principal findings

This note reports the finalized core experiment: three frozen seeds (11, 22, and 33), five target-label budgets, and four acquisition methods. Intervals are wide with three seeds, so the results are descriptive rather than strong evidence of a general ranking.

1. All methods shared the same initial 12 target labels and therefore had the same budget-12 result: dispatch MAE 1.922 MW and mean repaired-cost regret 28.13.
2. At 60 labels, uncertainty sampling had the lowest mean dispatch MAE (1.387 MW) and mean repaired-cost regret (21.06) across the three seeds.
3. Raw security feasibility was 0% for every method at the frozen `1e-7` MW tolerances. Predictions required the common security repair; repair magnitude and repaired-cost regret provide the informative continuous outcomes.
4. The paired seed-42 diagnostic increased target-topology MAE by 0.077 MW and mean regret by 0.55 on the same operating cases. This debugging-seed result is not part of the three-seed estimate.
5. Prediction accuracy and operational outcomes did not rank all method-budget cells identically, so they remain separate reported metrics.

The preregistered H5 predicted that decision-sensitive acquisition would achieve lower security violations and repaired-cost regret per target-oracle call. The frozen experiment did not support H5: uncertainty sampling was strongest on both reported endpoint means, while no method achieved exact raw feasibility. The initial protocol and subsequent implementation decisions are linked from the [README](../README.md#findings) and documented in the [experimental protocol](protocol.md) and [change log](change_log.md).
