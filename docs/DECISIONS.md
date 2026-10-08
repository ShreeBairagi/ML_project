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

## Decision 003 — Allowed Information at Prediction Time



Decision:

For a prediction made at the end of discharge cycle t, the model may use only information that is available by that moment.



Allowed information may include:



\- battery identifier only where the experiment design explicitly permits it

\- discharge cycle index t

\- measurements from discharge cycle t

\- measurements from earlier cycles

\- recorded discharge capacity at cycle t

\- previously observed impedance measurements whose timestamps occur at or before cycle t



The model must not use:



\- any measurement from discharge cycle t+1 or later

\- capacity from cycle t+1 or later

\- future impedance measurements

\- future threshold-crossing information

\- final battery lifetime

\- values computed using the full battery trajectory

\- statistics fitted using held-out/test data



Why:

The project predicts next-cycle capacity, so every input must represent information that would actually exist when the prediction is made.



Alternative:

Allow information from the full battery trajectory during feature construction.



Why rejected:

That would give the model access to future information and create data leakage.



Expected benefit:

A clear feature-availability rule that can be tested and defended during evaluation and viva.



Experiment:

Before using any feature group, record when the feature becomes available and verify that it uses data only from cycles at or before t.



Result:



Trade-off:

Some potentially useful whole-life statistics cannot be used because they would not exist at prediction time.



Failure mode:

A feature may appear harmless but still leak future information if it is calculated using later cycles, future capacity values, or global statistics from the complete battery trajectory.



What would change our decision:

If the project task changes from forecasting to retrospective analysis.

## Decision 004 — Primary Evaluation Split



Decision:

The primary reported evaluation split is leave-one-battery-out (LOBO).



For each outer evaluation fold:



\- one battery is held out completely for final testing

\- the remaining batteries are used for training

\- the held-out battery is not used for fitting preprocessing, selecting features, choosing hyperparameters, or setting decision thresholds



Random-by-cycle splitting is used only as a deliberately wrong leakage demonstration.



Time-ordered within-battery evaluation is kept as a secondary experiment for studying temporal generalization.



Why:

The project should estimate how well a model generalizes to an unseen battery rather than merely to unseen cycles from batteries it has already observed.



Alternative:

Use random train/test splitting across all discharge cycles as the main evaluation.



Why rejected:

Neighboring cycles from the same battery are highly related. Random cycle splitting can place very similar observations from one battery in both training and test sets and produce overly optimistic performance.



Expected benefit:

A more realistic and defensible estimate of cross-battery generalization.



Experiment:

Compare the same model under:

\- random-by-cycle split, clearly labeled WRONG

\- leave-one-battery-out

\- time-ordered within-battery evaluation



Result:



Trade-off:

With only four real batteries, each LOBO fold trains on only three batteries, so estimates may be unstable and cannot support broad population-level claims.



Failure mode:

Leakage occurs if any information from the held-out battery is used during training, preprocessing, hyperparameter selection, feature selection, or threshold selection.



What would change our decision:

If the intended deployment scenario changes to predicting future cycles only for a battery that has already been partially observed, then time-ordered prefix-informed evaluation may become the primary evaluation instead.

## Decision 005 — Held-Out Battery Information: Prefix-Informed Evaluation



Decision:

For the primary next-cycle prediction task, held-out-battery evaluation is prefix-informed.



When battery B is held out as the outer test battery, the model may use information from that battery only up to the current prediction cycle t when constructing the test example.



It may not use any information from cycle t+1 or later.



The held-out battery must still remain excluded from:



\- model fitting

\- preprocessing fitting

\- feature selection

\- hyperparameter selection

\- ensemble-weight selection

\- threshold selection



Why:

The project predicts the next discharge-cycle capacity after observing the current cycle. Therefore, information from the held-out battery's past and current cycles is genuinely available at prediction time.



Alternative:

Cold-start evaluation where no measurements from the held-out battery are allowed at all.



Why rejected:

Cold-start evaluation represents a different problem: predicting an unseen battery without observing any of its own history. That is not the primary next-cycle forecasting scenario defined in Decision 001 and Decision 002.



Expected benefit:

The evaluation matches the actual prediction moment while still testing whether learned relationships generalize across batteries.



Experiment:

For each LOBO fold, train only on the other batteries. Construct each held-out-battery test example using information available through cycle t and predict capacity at cycle t+1.



Result:



Trade-off:

This evaluation measures forecasting for an unseen battery after some of its history has been observed. It does not measure zero-history prediction for a completely new battery.



Failure mode:

Leakage occurs if later cycles from the held-out battery are used to construct earlier test examples, preprocessing statistics, features, thresholds, or model-selection decisions.



What would change our decision:

If the intended deployment scenario becomes prediction for a completely unseen battery before any of its measurements are available, cold-start evaluation should be used instead.

## Decision 006 — End-of-Life Threshold



Decision:

Use 1.4 Ah recorded discharge capacity as the project’s operational end-of-life threshold.



A battery is considered to have reached EOL at the first observed discharge cycle where recorded capacity is less than or equal to 1.4 Ah.



The threshold is treated as an externally defined experimental criterion based on approximately 30% capacity fade from a 2.0 Ah rated capacity.



It is not derived from each battery's observed maximum capacity.



Why:

Using one fixed threshold makes RUL and failure-label definitions consistent across cells and matches the project brief's stated EOL convention.



Alternative:

Define EOL as 70% of each battery's first observed capacity or maximum observed capacity.



Why rejected:

That would create a different threshold for every battery and would mix the SOH normalization decision with the EOL criterion.



It could also change whether particular cells are considered to have reached EOL.



Expected benefit:

A simple, consistent threshold that can be clearly applied to every battery and defended in experiments.



Experiment:

Use the recorded discharge-capacity sequence to identify the first observed cycle at which capacity is less than or equal to 1.4 Ah.



Result:



Trade-off:

The rated 2.0 Ah reference differs from the observed starting and maximum capacities in the dataset, so this threshold should not be interpreted as exactly 70% of each cell's observed initial capacity.



Failure mode:

A battery may never cross 1.4 Ah within the recorded observation window. In that case, the true EOL cycle is not observed and must not be fabricated.



What would change our decision:

If the course, dataset documentation, or project requirements specify a different EOL convention, or if we explicitly decide to study a per-cell normalized EOL definition as a separate experiment.

## Decision 007 — RUL Definition and Censoring



Decision:

Remaining Useful Life (RUL) is defined as the number of discharge cycles from the current discharge cycle t until the first observed discharge cycle where recorded capacity is less than or equal to the project EOL threshold of 1.4 Ah.



For a battery with observed EOL cycle e:



RUL(t) = e - t



Exact RUL labels are created only for cycles where the battery's EOL crossing is observed in the recorded data.



If a battery never reaches the EOL threshold during the observation window, its true EOL cycle is unknown.



Such a battery is treated as censored for exact-RUL evaluation.



Do not:

\- replace its EOL with the final recorded cycle;

\- assign RUL = 0 at the final recorded observation;

\- extrapolate an EOL cycle and treat that extrapolation as ground truth.



Why:

The true failure cycle of a non-crossing battery is not present in the dataset, so assigning an exact EOL would fabricate a target.



Alternative:

Use the final recorded discharge cycle as EOL for batteries that never cross 1.4 Ah.



Why rejected:

The final recorded cycle only indicates the end of observation, not necessarily the physical end of life.



Expected benefit:

RUL metrics are calculated only where exact ground truth is actually observed.



Experiment:

For each battery, determine whether an observed 1.4 Ah crossing exists.



Use exact-RUL regression metrics only for batteries/cycles whose EOL is observed under this definition.



Report censored batteries separately rather than silently dropping or relabeling them.



Result:



Trade-off:

This reduces the amount of real data available for exact-RUL evaluation and may exclude some batteries from particular RUL metrics.



Failure mode:

Treating the last observed cycle of a censored battery as true EOL would create false labels and misleading RUL errors.



What would change our decision:

If the project later introduces an approved method specifically designed for censored time-to-event data, or if additional observations reveal the true EOL crossing for currently censored batteries.

