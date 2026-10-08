\# Agent Rules — Battery Health Prediction Lab



\## Context



Read `docs/PROJECT\_BRIEF.md` first. It is the source of truth for project goals, syllabus constraints, scientific questions, and schedule.



Also read:



\- `docs/DECISIONS.md` for human-made definitions and methodological decisions;

\- `docs/EXPERIMENT\_LOG.md` for hypotheses, predictions, experiment status, and human conclusions;

\- `README.md` for current project status;

\- `docs/PROMPTS.md` for task-specific coding-agent prompts.



Never treat a starter hypothesis as a fact.



\---



\## Authority order



If instructions appear inconsistent, use this order:



1\. explicit instruction from the user for the current task;

2\. `AGENTS.md`;

3\. `docs/PROJECT\_BRIEF.md`;

4\. `docs/DECISIONS.md` for recorded project decisions;

5\. `docs/PROMPTS.md`;

6\. existing implementation.



If two scientific definitions conflict, do not silently choose one.

Report the conflict.



\---



\## Hard constraints



\### Syllabus only



Use only the libraries, models, metrics, and methods allowed by `docs/PROJECT\_BRIEF.md`.



Do not introduce another package, framework, experiment tracker, explainability package, validation framework, cloud service, or model without explicit approval and an entry in `docs/DECISIONS.md`.



Pin dependency versions.



\### Keep the implementation explainable



Use plain Python.



Prefer:

\- short functions;

\- direct data flow;

\- ordinary modules;

\- explicit arguments;

\- readable sklearn Pipelines.



Avoid:

\- class hierarchies;

\- generic framework layers;

\- factories unless genuinely necessary;

\- metaprogramming;

\- async code;

\- configuration systems for a tiny number of experiments;

\- abstractions created only because they might be useful later.



Comments should explain why, especially around leakage and evaluation, rather than narrating obvious syntax.



If a teammate cannot explain the implementation in a viva, simplify it.



\### Reproducibility



Fix and record random seeds whenever randomness exists.



Do not pretend a seed matters for a deterministic operation.



The project must run from a fresh clone using documented commands.



Raw data is never committed.



Raw data lives under:



`data/raw/`



and must be gitignored.



Provide or document a reproducible download/verification procedure, but do not silently replace the intended dataset with another battery dataset.



\---



\# Scientific boundaries



\## Human-owned decisions



The human team owns:



\- prediction-task formulation;

\- prediction moment;

\- target definitions;

\- SOH normalization/reference capacity;

\- EOL threshold;

\- RUL definition;

\- treatment of censored/non-EOL cells;

\- classification horizon N;

\- train/test split philosophy;

\- information allowed at prediction time;

\- asymmetric error costs;

\- hypotheses;

\- pre-run predictions;

\- scientific interpretations;

\- conclusions.



If one of these is missing and required by the current task, stop before the experiment and ask for the decision.



Do not infer it from a paper, notebook, blog, or common convention.



\---



\# Protected hand-written files



Do not edit these files unless the user explicitly changes ownership:



\- `src/battery/targets.py`

\- `src/battery/splits.py`

\- `src/battery/features.py`

\- `src/battery/simulator.py`

\- `src/battery/from\_scratch/`

\- hypotheses in `docs/EXPERIMENT\_LOG.md`

\- predictions in `docs/EXPERIMENT\_LOG.md`

\- conclusions in `docs/EXPERIMENT\_LOG.md`



Read them when necessary.



If a protected file appears incorrect or blocks the task:

1\. identify the exact issue;

2\. explain why it matters;

3\. suggest the smallest human-made change;

4\. do not edit it.



Update this protected-file list if human-owned files are added later.



\---



\# Dataset rules



Do not encode an expected dataset fact merely because it appears in the project brief.



The downloaded files are authoritative for observed properties such as:



\- actual battery IDs present;

\- operation counts;

\- cycle counts;

\- field availability;

\- observed capacity ranges;

\- observed threshold crossings;

\- operation ordering.



Inspect the real `.mat` structure before implementing the production parser.



Do not change parsing code merely to force observed data to match an expected count from a publication or secondary source.



Keep separate concepts separate unless the human team explicitly makes them equivalent:



\- rated capacity;

\- observed initial capacity;

\- maximum observed capacity;

\- SOH denominator;

\- EOL threshold.



\---



\# Prediction-time rule



Every supervised feature must have a defined availability time.



A feature is allowed only if it would actually exist at the declared prediction moment.



Do not use:

\- future-cycle measurements;

\- later capacity values;

\- future threshold crossing information;

\- statistics computed using future test observations;

\- the target itself or a deterministic transformation of it as an input.



A feature can be non-leaking for one task and leaking for another.

Judge it against the project's declared prediction moment.



\---



\# Censoring rule



Never fabricate an exact EOL for a battery that does not reach the defined EOL threshold during observation.



Never silently use the final recorded cycle as true EOL unless the human team explicitly defines and justifies that convention in `docs/DECISIONS.md`.



Before an RUL or failure-horizon experiment, verify that its labels are well defined for every included sample under the human-written censoring decision.



\---



\# Leakage rules — never violate



\## Split before fit



No learned preprocessing may be fitted using information from the final evaluation data.



This applies to:



\- scalers;

\- imputers;

\- PCA;

\- feature selection;

\- target-dependent transformations;

\- learned encoders;

\- clustering when its fitted representation is used as part of a supervised prediction pipeline.



Fit these using training data only.



Where appropriate, keep preprocessing and estimator together in an `sklearn.pipeline.Pipeline`.



\## Battery grouping



Never randomly mix cycles from the same battery across train and test for a reported deployment-style result.



Random-by-cycle evaluation is allowed only in an experiment explicitly named and labeled as a leakage demonstration.



Never present its score as the project's main model performance.



\## Test-set isolation



Never use the final held-out data to choose:



\- hyperparameters;

\- features;

\- feature groups;

\- PCA components;

\- probability thresholds;

\- regression decision thresholds;

\- ensemble weights;

\- early stopping choices;

\- model family.



Any selection step must occur using training data only.



\## Stacking



Stacking meta-model inputs for training must be generated from out-of-fold predictions.



For the real-cell problem, those inner folds must respect battery/group boundaries.



Never train a base learner on a sample and then use its prediction for that same training sample as a meta-model training feature.



The outer held-out battery must remain untouched until final evaluation.



\---



\# Evaluation rules



Always compare eligible learned models with the approved baselines.



Save raw numerical evidence under `results/` before making claims.



Do not claim that a model improved unless the saved numbers support the statement.



Do not manufacture repeated measurements.



Use multiple seeds only where randomness actually exists.



For deterministic LOBO evaluation, report variation across held-out batteries rather than pretending different seeds create independent evidence.



Never silently average away per-battery failures.



Retain per-cell/per-fold results.



Use only metrics allowed by the project brief.



A metric may be calculated only when its target and evaluation samples are valid under the project's target/censoring decisions.



\---



\# Experiment protocol



Before every experiment:



1\. identify the question;

2\. identify what changes;

3\. identify what remains fixed;

4\. identify the evaluation split;

5\. identify information available at prediction time;

6\. verify training-only fitting;

7\. require the human user's prediction to be written in `docs/EXPERIMENT\_LOG.md`.



Do not run the experiment before the prediction exists.



After running:



\- save raw results;

\- record configuration and seed where relevant;

\- report measured values accurately;

\- do not invent missing values;

\- do not write the human's conclusion;

\- do not retroactively rewrite the prediction.



You may flag a suspicious result, such as unexpectedly perfect performance, and recommend a leakage/data-quality check.



\---



\# Baseline rules



A baseline must obey the same information-availability assumptions as the model it is compared against.



In particular, a last-observed-value baseline may use a held-out battery's previous capacity only if the documented deployment scenario permits observed history from that battery.



An exponential baseline must have an explicit documented rule for how parameters are obtained for a held-out battery.



Do not choose that rule yourself.



\---



\# Hyperparameter policy



This is not a leaderboard project.



Do not perform large searches merely to improve scores.



Use:

\- documented defaults;

\- a small manually justified configuration;

\- or a small predeclared grid when the experiment specifically requires model selection.



Any tuning must occur entirely inside the training data.



Do not use the test battery repeatedly as feedback for manual tuning.



\---



\# Simulator rules



`src/battery/simulator.py` is human-owned.



Do not alter simulator parameters to make a hypothesis succeed.



Keep simulator results clearly separate from real-data results.



Do not imply that improved synthetic-data performance proves improved performance on real batteries.



The simulator is primarily for controlled experiments with known truth.



\---



\# Unsupervised-analysis rules



PCA, K-Means, hierarchical clustering, and DBSCAN may be used only within the syllabus and project scope.



Do not give a cluster a physical interpretation merely because its members have similar errors.



Do not treat PCA directions as physical mechanisms without supporting evidence.



Exploratory associations are not causal explanations.



\---



\# Results and interpretation



The agent may:



\- compute results;

\- save results;

\- describe exactly what a number represents;

\- point out anomalous or suspicious output;

\- prepare tables and plots.



The agent must not:



\- invent results;

\- change hypotheses after seeing results;

\- write the student's scientific conclusion;

\- declare a model "best" unless explicitly asked to perform a purely numerical ranking;

\- turn correlation into causation;

\- generalize four cells to all lithium-ion batteries;

\- claim robustness to chemistries, temperatures, or operating regimes that were not tested.



\---



\# Workflow



Work on one task at a time.



Before coding:

\- give a short plan;

\- identify files to change;

\- explain why;

\- explain what could break;

\- state the tests.



Wait for human approval before implementation.



One logical task should normally correspond to one small commit.



Never push.



Never run destructive commands without explicit approval.



Do not erase or overwrite raw results to hide an unsuccessful run.



When something fails:

1\. show the real error or failing test;

2\. state your current hypothesis;

3\. propose the smallest fix;

4\. then fix it after approval when required by the workflow.



Do not silently make unrelated refactors while fixing a bug.



\---



\# Git and tool switching



Never push on behalf of the user.



Before suggesting a commit:

\- show or summarize the diff;

\- make sure the task is internally complete;

\- run relevant tests.



Do not let two coding agents edit the repository concurrently.



Before switching agents/tools:

1\. finish or abandon the current atomic task explicitly;

2\. review its diff;

3\. commit approved work if appropriate;

4\. update project status.



A new agent should be able to continue by reading:

\- `AGENTS.md`;

\- `docs/PROJECT\_BRIEF.md`;

\- `docs/DECISIONS.md`;

\- `docs/EXPERIMENT\_LOG.md`;

\- `README.md`.



\---



\# Definition of done for a coding task



A task is complete only when applicable conditions are satisfied:



\- implementation runs;

\- randomness is controlled where relevant;

\- smoke test passes;

\- relevant unit tests pass;

\- leakage/split tests pass;

\- raw experiment output is saved;

\- experiment configuration is recoverable;

\- README phase status reflects reality;

\- experiment-log status reflects reality;

\- required decision-log entries exist;

\- no protected file was modified without permission.



Do not mark planned work as done.



\---



\# Teaching mode



When asked to explain, use:



1\. mechanism;

2\. assumptions;

3\. failure modes.



Prefer reasoning about the actual project code over generic textbook definitions.



When asked for a quiz:

\- ask one question at a time;

\- wait for the user's answer;

\- give feedback;

\- then ask the next question.



When acting as a professor, challenge unsupported claims and ask how the student knows them from the experiment.



Do not substitute polished wording for genuine understanding.

