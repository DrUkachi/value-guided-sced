# Failure log

- Initial launch failed before training because the hidden-label loader expected feature fields that are intentionally absent from the leakage-safe label file. Fixed by joining oracle labels to the separately loaded features after verifying scenario-ID order.
- Initial aggregation wrote the CSV but could not create the figure because Matplotlib selected a Tk GUI backend in the headless environment. Fixed by explicitly selecting the non-interactive `Agg` backend.
- The corrected experiment completed twice without training, acquisition, oracle, repair, or evaluation failures; both output hashes matched exactly.
