\# Project Brief — Battery Health Prediction Lab (NASA Li-ion aging data)



\## 1. Who and why



Group of 3, third-year B.Tech AI \& ML students. This is an ML mini-project, but the real goal is \*\*deep understanding\*\*: formulating problems, questioning data, designing controlled experiments, and defending decisions to a strong professor. It is not a leaderboard project and not a flashy app. The code must stay \*\*simple and working\*\*; the sophistication should come from reasoning and experiments.



\## 2. Hard constraints



\- \*\*Timeline:\*\* about 3 weeks (a 2-week cut is defined in section 8).

\- \*\*Syllabus-restricted methods.\*\* Allowed: NumPy, Pandas, Matplotlib, Seaborn, SciPy (file loading), scikit-learn, XGBoost, a small Keras/TensorFlow baseline (optional), Streamlit (UI only).

&#x20; - Models: linear regression, Ridge, Lasso, Elastic Net, KNN, Naive Bayes, Decision Tree, SVM, voting, averaging, weighted averaging, bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost, stacking.

&#x20; - Unsupervised/dimensionality reduction: K-Means, hierarchical clustering, DBSCAN, PCA, LDA, silhouette, Davies-Bouldin, WCSS.

&#x20; - Metrics: MAE, MSE, RMSE, R², confusion matrix, accuracy, precision, recall, F1, ROC/AUC.

&#x20; - \*\*Not allowed in the core:\*\* SHAP, nested cross-validation frameworks, formal significance-test packages, MLflow/experiment-tracking tools, drift-detection libraries, deep learning beyond a small baseline, cloud services.

\- \*\*Hardware:\*\* laptop (i5-13450HX, 16GB RAM, RTX 3050 6GB). Google Colab optional for heavy sweeps; code lives in a Git repo and runs identically on both.

\- \*\*AI tools\*\* are used for boilerplate, plotting, tests, refactors, and explanations. Humans own problem formulation, split design, metric/cost choice, hypotheses, interpretation, and the from-scratch piece.



\## 3. The problem



Predict battery health from cycle data and estimate when a cell reaches end of life, using the NASA Prognostics Center of Excellence Li-ion aging dataset.



\*\*Data facts to verify on day 1 (do not assume):\*\*



\- Four commonly used 18650 cells: B0005, B0006, B0007, B0018, aged by repeated charge/discharge cycles with periodic impedance measurements at room temperature.

\- End of life is commonly defined as 30% capacity fade (about 2 Ah to about 1.4 Ah).

\- Discharge cutoff voltages differ across cells.

\- One published analysis reports 168 discharge cycles for B0005/6/7 and 132 for B0018, and B0007 may never clearly reach the threshold (censoring).

\- Nominal capacity is ambiguous (observed maximum ≈1.86 Ah vs 2.0 Ah rated); some cycles show state of health above 100%.

\- Files are nested MATLAB structures (charge, discharge, impedance operations). Inspect the real structure before parsing.



\*\*Targets to define by hand:\*\* capacity or state of health (SOH) per cycle; remaining useful life (RUL, cycles until threshold); classification label "fails within N cycles" or "healthy/degraded". The team decides the exact definitions and records them in `docs/DECISIONS.md`.



\## 4. Architecture ("two-part lab")



1\. \*\*Real-data pipeline:\*\* load, tidy per-cycle table, feature engineering from discharge curves, splits, models, evaluation.

2\. \*\*Simulator (NumPy):\*\* many virtual batteries from a capacity-fade model (single/double exponential) plus noise, with \*\*known truth\*\*. It supplies the statistical power four real cells cannot, and lets us measure bias-variance, leakage inflation, and ensemble behavior.

3\. \*\*Streamlit app (thin, reads saved results):\*\* battery explorer, model comparison, threshold/cost slider, leakage lab, simulator lab (error-correlation slider).



Repo layout: `src/battery/` (loading, features, splits, models, evaluation, simulator, from\_scratch), `experiments/` (one script per hypothesis), `app/`, `tests/`, `notebooks/` (exploration only), `results/`, `docs/`.



\## 5. The four deep spines and starter hypotheses



Rewrite these in your own words and add predictions before every run.



1\. \*\*Leakage and validation.\*\* \*H1:\* a random split across cycles gives much better scores than leave-one-battery-out, because neighboring cycles of one cell are nearly identical. \*H2:\* fitting a scaler/PCA on all data before splitting inflates scores; measure by how much.

2\. \*\*When combining models helps.\*\* \*H3 (simulator):\* ensemble gain shrinks as base-model error correlation rises. \*H4 (real data):\* with only four cells, ensemble gains are smaller than seed-to-seed variance.

3\. \*\*Regress-then-threshold vs direct classification, under unequal costs.\*\* \*H5:\* direct classification wins when errors near the threshold are costly; test it, don't assume. Late failure predictions cost more than early ones.

4\. \*\*Error analysis.\*\* Which cycles/cells fail and why? Use K-Means/PCA on discharge-curve features to look for error groups.



Also include: a physics-style exponential-fit baseline and a naive baseline (for example, last observed value); hand-made calibration check by binning probabilities; one distribution-shift experiment (train on early life or some cells, test on later/others).



\## 6. Evaluation rules



\- Splits: random-by-cycle (the wrong way, to show leakage), per-battery (leave-one-battery-out), and time-ordered within a cell.

\- Repeat with multiple seeds; report mean and spread, never a single number.

\- Compare every model to the baselines. If a simple rule is nearly as good, that is a result.

\- Metrics follow the decision: regression (MAE, RMSE, R²); classification (confusion matrix, precision, recall, F1, ROC/AUC); choose thresholds from the late/early cost asymmetry.

\- Ablate: remove feature groups, scaling, PCA, ensemble, and measure what actually changes.



\## 7. From-scratch piece (choose one, hand-written)



Gradient descent for linear regression, or K-Means, or ensemble aggregation (voting/bagging). Compare against scikit-learn and explain every difference.



\## 8. Plan



\*\*3 weeks.\*\* Week 1: data, baselines, splits, leakage demo, single models, simulator, from-scratch piece. Week 2: ensembles, error-correlation experiment, ablations, seed variance, cost thresholds, calibration, error analysis. Week 3: shift experiment, Streamlit app, write-up, viva doc; last 2 days for cross-teaching and no-AI mock defenses. \*\*2 weeks:\*\* drop the shift experiment and most unsupervised analysis, keep one from-scratch piece and the simulator, skip the app in favor of clean figures.



\## 9. Roles (no silos)



A: data, baselines, leakage. B: models, ensembles, simulator. C: evaluation, error analysis, app, write-up. Everyone runs and explains at least one experiment from each other area; weekly no-AI explain-back sessions.



\## 10. Complexity budget



Plain Python, only the allowed libraries, short functions, no class hierarchies or clever abstractions, comments that explain why, one file per concept. If a teammate cannot explain it, simplify it.



\## 11. Viva questions the project must answer



\- Why does your random-split score differ from leave-one-battery-out, and what does that say about deployment?

\- What does each model assume, and which assumption breaks on this data?

\- Why can ensembling fail to help? How did the simulator show it?

\- Is "fails within N cycles" better predicted directly or by thresholding a regression? When does each win?

\- What does a predicted probability of 0.8 actually mean here? Did you check calibration?

\- With four cells, what can and can't you claim?

\- What would change with a different cell chemistry or temperature?



\## 12. Definition of done (each task)



Runs from a clean clone, seeds fixed, a smoke test passes, tests exist for leakage/split risks, results saved under `results/`, hypotheses/predictions logged in `docs/EXPERIMENT\_LOG.md`, decisions logged in `docs/DECISIONS.md`.

