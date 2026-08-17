# Principal findings draft

This draft reports means across the three frozen final seeds (11, 22 and 33). With only three seeds, the 95% t-intervals are wide and method rankings should be treated as suggestive rather than definitive.

1. All four strategies shared the same initial 12 target labels and therefore the same budget-12 result: dispatch MAE 1.922 MW and mean repaired-cost regret 28.13.
2. At 60 labels, **uncertainty** had the lowest mean dispatch MAE (1.387 MW), while **uncertainty** had the lowest mean repaired-cost regret (21.06).
3. Uncertainty and decision-sensitive acquisition were the strongest average endpoint methods, but no strategy won every seed or every metric. This does not support claiming a universally best acquisition rule.
4. Exact raw feasibility remained 0% under the frozen 1e-7 MW tolerances. The more informative continuous diagnostic is repair magnitude; learned predictions still require the common security repair.
5. The paired seed-42 diagnostic increased target-topology MAE by 0.077 MW and mean regret by 0.55 on identical operating cases. This isolates a modest structural-shift penalty but is a debugging-seed diagnostic, not a three-seed estimate.
6. MAE and operational metrics are related but not interchangeable: their rankings differ in some seed-budget cells, supporting separate reporting of predictive error, repair burden and regret.

These findings should remain provisional until the planned ablations and final reproducibility freeze are complete.
