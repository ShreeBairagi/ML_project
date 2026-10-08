# Experiment Log

## Experiment Entry

Experiment:
Question:
Prediction:
Configuration:
Result file:
Observed numbers:
Conclusion:

Experiment:
Baseline comparison under different split strategies

Question:
How do the naive last-observed-value baseline and exponential baseline behave under random-by-cycle, LOBO, and time-ordered evaluation?

Prediction:
I expect random-by-cycle evaluation to produce more optimistic errors than leave-one-battery-out because cycles from the same battery can appear in both train and test sets.

I expect the last-observed-value baseline to be competitive for next-cycle prediction because adjacent discharge capacities are often similar.

I do not know whether the exponential baseline will outperform it.


## Experiment — Ridge Leakage Demonstration

Experiment:
Compare Ridge regression under correct and deliberately incorrect evaluation/preprocessing setups.

Question:
How much do random cycle splitting and preprocessing fitted before splitting inflate Ridge performance compared with grouped battery evaluation and training-only preprocessing?

Prediction:
I expect Ridge evaluated with random-by-cycle splitting to show lower prediction error than leave-one-battery-out evaluation because cycles from the same battery can appear in both training and test sets.

I also expect fitting the scaler using all data before splitting to produce more optimistic results than fitting the scaler using training data only.

I do not know how large either difference will be.

Configuration:
Pending experiment implementation.

Result file:

Observed numbers:

Conclusion:

## Experiment — Single Models under Leave-One-Battery-Out

Experiment:
Compare several syllabus-approved regression models for next-cycle discharge-capacity prediction under leave-one-battery-out evaluation.

Question:
How differently do linear, regularized linear, instance-based, tree-based, and kernel-based models behave when generalizing to a held-out battery?

Prediction:
I expect the models to show different error patterns across held-out batteries.

I expect scale-sensitive models such as KNN and SVR to require scaling, while tree-based models should not depend strongly on feature scale.

I do not know which model will have the lowest error, and I will not choose a winner before seeing the results.

Configuration:
Pending implementation.

Result file:

Observed numbers:

Conclusion:

## Experiment — Ensemble Models and Error Correlation

Experiment:
Compare individual regressors with simple ensemble methods under leave-one-battery-out evaluation and inspect how correlated their prediction errors are.

Question:
Can combining models with different error patterns improve next-cycle capacity prediction compared with the individual models?

Prediction:
I expect models with less-correlated residual errors to be more useful in an ensemble than models making nearly identical errors.

I expect simple averaging or weighted averaging to sometimes improve robustness, but I do not know whether an ensemble will outperform every individual model on every held-out battery.

Configuration:
Pending implementation.

Result file:

Observed numbers:

Conclusion:

## Experiment — Classification, Unequal Costs, and Calibration

Experiment:
Compare a direct binary classifier with a regress-then-threshold approach for predicting whether the next cycle is below the operational EOL threshold.

Question:
How do direct classification and regress-then-threshold behave when false negatives and false positives have unequal costs?

Prediction:
I expect the preferred decision threshold to depend on the relative cost assigned to false negatives and false positives.

I also expect that a model with strong overall accuracy may still be poorly calibrated or produce an undesirable error tradeoff under unequal costs.

I do not know which approach will perform better before running the experiment.

Configuration:
- **Task**: Predict `capacity <= 1.4 Ah` for the next cycle.
- **Models**: `RandomForestClassifier` (direct) vs `RandomForestRegressor` (regress-then-threshold).
- **Evaluation Split**: Leave-One-Battery-Out (LOBO).
- **Cost Setup**: High penalty for False Negatives (FN cost = 5, late failure prediction) and low penalty for False Positives (FP cost = 1, early failure prediction).
- **Thresholds**: Classification varied decision probability (0.1 to 0.9); Regression varied capacity threshold (1.30 to 1.50).

Result file:
`results/tables/classification_costs_results.csv`

Observed numbers:
- **Direct Classification Costs**: Best cost was at probability threshold 0.50 (average cost 10.00 across batteries). Lowering threshold to 0.30 eliminated some FNs but introduced many FPs on non-degraded cycles, increasing average cost to 16.50.
- **Regress-then-Threshold Costs**: Best cost was at capacity threshold 1.40 (average cost 8.50). Raising the threshold to 1.45 to be "safe" avoided all FNs but drastically increased FPs (e.g., 24 FPs for battery B0007, which never truly reached EOL), bringing average cost up to 19.25.
- **Accuracy vs. Cost**: At capacity threshold 1.45, regress-then-threshold still had ~85-90% accuracy but an undesirable cost trade-off due to excessive early warnings (FPs).

Conclusion: