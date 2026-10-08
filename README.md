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
│   └── PROMPTS.md           # Task prompt templates for coding agents
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
.venv\Scripts\pytest
```

---

## 📊 Phase Status
- [x] **Phase 1: Environment & Repository Skeleton** (Python 3.11, pinned syllabus dependencies, skeleton, templates, smoke tests)
- [ ] **Phase 2: Data Ingestion & NASA .mat Exploration**
- [ ] **Phase 3: Baseline Models & Split Infrastructure (LOBO)**
- [ ] **Phase 4: Deep Spines & Experiments (H1 to H6)**
- [ ] **Phase 5: Simulator Lab & From-Scratch Piece**
- [ ] **Phase 6: Streamlit UI & Viva Preparation**
