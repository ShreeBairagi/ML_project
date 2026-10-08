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