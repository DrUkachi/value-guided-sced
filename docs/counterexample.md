# Counterexample: operational weighting can hurt

The largest per-case disadvantage of the combined decision-sensitive score relative to uncertainty occurred for seed 33, budget 24, scenario `target_test-0045`.

| Quantity | Uncertainty | Decision-sensitive |
|---|---:|---:|
| Repaired-cost regret | 0.000 | 54.253 |
| Repair magnitude (MW) | 18.238 | 18.357 |
| Raw balance error (MW) | 0.820 | 0.976 |
| Contingency overload (MW) | 0.000 | 0.000 |

The operationally weighted model incurred 54.253 more regret on this case. This does not show that the criticality term always hurts; it shows that a criticality estimate derived from imperfect model predictions can redirect labels in a way that worsens particular unseen cases. The aggregate ablation must therefore be reported alongside this counterexample.
