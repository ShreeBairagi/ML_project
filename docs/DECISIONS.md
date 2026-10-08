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

## Decision 002 — Prediction Moment and Available Information



Decision:

The prediction is made at the end of discharge cycle t.



At that point, all measurements recorded during discharge cycle t are considered available.



The model predicts the recorded discharge capacity of discharge cycle t+1.



Allowed information may include only data available up to and including cycle t.



Why:

This creates a clear prediction-time boundary and makes leakage checks explicit.



Alternative:

Predict during cycle t before the full discharge curve is available, or predict current-cycle capacity.



Why rejected:

Predicting during the cycle would require a different partial-curve problem definition.



Predicting current-cycle capacity could become trivial if measurements from the same completed discharge cycle directly reveal that capacity.



Expected benefit:

Clear rules for feature construction and leakage auditing.



Experiment:

All future supervised experiments must construct features from cycles at or before t and targets from cycle t+1.



Result:



Trade-off:

The model assumes a complete current discharge cycle has already been observed before predicting the next one.



Failure mode:

Any feature using measurements from cycle t+1 or later creates future-information leakage.



What would change our decision:

If the project later changes to early-warning prediction before a discharge cycle is complete.

