# 🔋 Battery Health Prediction Lab

Machine Learning investigation into Lithium-ion battery aging using the NASA Prognostics Center of Excellence (PCoE) dataset and a synthetic degradation simulator.

---

## 📌 Project Overview
This project focuses on **deep methodological understanding** and rigorous ML principles rather than superficial metric chasing:
1. **Real Data Pipeline**: End-to-end processing and modeling of NASA Li-ion 18650 battery run-to-failure cycles (B0005, B0006, B0007, B0018).
2. **Synthetic Degradation Simulator**: Controlled mathematical simulator to study bias-variance trade-offs, data leakage inflation, and ensemble error-correlation dynamics.
3. **Four Research Spines**:
   - **Leakage & Validation**: Comparing random-by-cycle splits against Leave-One-Battery-Out (LOBO).
   - **Ensemble Dynamics**: Quantifying when model combinations help vs. base error correlation.
   - **Asymmetric Cost Evaluation**: Direct classification vs. regress-then-threshold under heavy late-failure penalties.
   - **Error Analysis**: Unsupervised clustering of discharge curves to diagnose failure modes.
4. **Streamlit UI**: Thin dashboard reading saved numerical artifacts from `results/`.

---

## 🛠️ Repository Layout
```text
├── data/
│   ├── raw/                 # Raw .mat files (gitignored)
│   └── processed/           # Structured Parquet tables
├── docs/
│   ├── PROJECT_BRIEF.md     # Source of truth for syllabus, goals, and constraints
│   ├── DECISIONS.md         # Human-owned definitions, SOH/EOL formulas, and conventions
│   ├── EXPERIMENT_LOG.md    # Hypotheses, pre-run predictions, and measured results
│   ├── PROMPTS.md           # Task prompt templates for coding agents
│   └── RESULTS_SUMMARY.md   # Final factual numerical results summary
├── experiments/             # Standalone reproducible experiment scripts
├── app/                     # Streamlit dashboard application
├── notebooks/               # Exploratory notebooks (non-production)
├── results/
│   ├── figures/             # Output plots and visualizations
│   └── tables/              # Numerical metrics (CSV/JSON)
├── src/battery/             # Core Python package
│   ├── loading.py           # MATLAB .mat ingestion and schema extraction
│   ├── features.py          # Discharge/charge feature extractors (Protected)
│   ├── splits.py            # LOBO, temporal, and leakage splitters (Protected)
│   ├── targets.py           # SOH, RUL, and failure label targets (Protected)
│   ├── models.py            # Estimators, baselines, and sklearn Pipelines
│   ├── evaluation.py        # Metrics, cost functions, calibration utilities
│   ├── simulator.py         # Synthetic degradation simulator (Protected)
│   └── from_scratch/        # Hand-written implementations (Protected)
├── tests/                   # Pytest suite
├── requirements.txt         # Pinned syllabus-only dependencies
└── AGENTS.md                # Agent behavioral and scientific guardrails
```

---

## 🚀 Environment Setup

### 1. Prerequisites
- Python 3.11.x

### 2. Create Virtual Environment & Install Dependencies
```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Run Verification Tests
```powershell
.venv\Scripts\python.exe -m pytest -v
```

### 4. Reproducibility Commands (Experiments & App)

Run the major experiments using these commands:
- **Baselines**: `.venv\Scripts\python.exe experiments\run_baselines.py`
- **Ridge Leakage**: `.venv\Scripts\python.exe experiments\run_ridge_leakage.py`
- **Phase 4 Single Models**: `.venv\Scripts\python.exe experiments\run_phase4_single_models.py`
- **Simulator Demo**: `.venv\Scripts\python.exe experiments\run_simulator_demo.py`
- **Phase 6 Ensembles**: `.venv\Scripts\python.exe experiments\run_phase6_ensembles.py`
- **Phase 7 Classification**: `.venv\Scripts\python.exe experiments\run_phase7_classification.py`
- **Phase 8 Error Analysis**: `.venv\Scripts\python.exe experiments\run_phase8_error_analysis.py`

*Note: Verify the exact script names under `experiments/` before running.*

**Run the Streamlit app:**
```powershell
.venv\Scripts\streamlit.exe run app\app.py
```
*Note: Experiment result artifacts must already exist under `results/` for the app to function properly.*

---

## 📊 Phase Status
- [x] **Phase 1: Environment & Repository Skeleton** (Python 3.11, pinned syllabus dependencies, skeleton, templates, smoke tests)
- [x] **Phase 2: Data Ingestion & NASA .mat Exploration**
- [x] **Phase 3: Baseline Models & Split Infrastructure (LOBO)**
- [x] **Phase 4: Deep Spines & Experiments (H1 to H6)**
- [x] **Phase 5: Simulator Lab & From-Scratch Piece**
- [x] **Phase 6: Streamlit UI & Viva Preparation**
- [x] **Phase 7: Classification**
- [x] **Phase 8: Error Analysis**
- [x] **Phase 9: Result generation**
- [x] **Phase 10: Final Audit and Cleanup**

## Important Limitations
- The dataset contains only four real batteries, limiting population-level generalization.
- B0007 does not reach the operational EOL threshold (1.4 Ah) in the given observation window.
- Random-cycle splitting is included purely as a pedagogical demonstration of data leakage, not a recommended evaluation metric.
