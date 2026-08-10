# Glossary

This glossary is a compact companion to the frozen definitions in [the experimental protocol](protocol.md).

- **Source topology:** The original six-bus grid with line 8 in operation.
- **Target topology:** The changed grid in which line 8 has been removed.
- **Operating condition:** $x=[d_1,\ldots,d_n,r_1,\ldots,r_m]$, containing bus demand and renewable generation.
- **Candidate sample:** A target-topology input whose optimal dispatch label is hidden from the acquisition method.
- **Oracle label:** The secure optimum $g^*(x)=\operatorname{SCED}(x,T_{\mathrm{target}})$.
- **Oracle call:** One SCED optimisation for one selected target operating condition.
- **Target test sample:** An operating condition and oracle solution used only for final evaluation.
- **Acquisition round:** Select 12 candidates, reveal their labels, add them to training data, retrain and evaluate.
