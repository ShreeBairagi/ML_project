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