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