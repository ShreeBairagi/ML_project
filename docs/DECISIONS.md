# Decisions

## Decision Entry

Decision:
Why:
Alternative:
Why rejected:
Expected benefit:
Experiment:
Result:
Trade-off:
Failure mode:
What would change our decision:

## Decision 001 — Primary Prediction Task



Decision:

The primary supervised task is next-cycle discharge-capacity prediction.



At discharge cycle t, the model predicts the recorded discharge capacity at cycle t+1 using only information available up to and including cycle t.



Why:

This gives the project a clear regression task before introducing derived targets such as SOH, RUL, or failure-within-N-cycles classification.



It also makes prediction-time leakage easier to define and test.



Alternative:

Predict current-cycle capacity, predict RUL directly, or make failure classification the primary task.



Why rejected:

Current-cycle capacity prediction risks becoming trivial if current-cycle capacity or directly equivalent information is included.



RUL and failure classification require additional decisions about EOL thresholds, censoring, and classification horizons before the basic supervised pipeline can be built.



Expected benefit:

A simple and defensible first prediction problem that supports MAE, RMSE, and R² evaluation and can later feed regress-then-threshold experiments.



Experiment:

Compare approved regression models and baselines using the project's split protocols.



Result:



Trade-off:

Next-cycle prediction is a shorter-horizon problem than full RUL prediction and does not by itself answer when the battery will fail.



Failure mode:

Leakage occurs if information from cycle t+1 or later is used to construct features for the prediction made at cycle t.



What would change our decision:

If the available measurements cannot support a meaningful next-cycle prediction setup, or if the project requirements explicitly prioritize direct RUL prediction as the main task.

