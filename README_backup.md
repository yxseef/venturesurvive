# VentureSurvive — Startup Survival Prediction with Strict 6-Month Snapshot

VentureSurvive is a **research-grade machine learning project** that predicts startup success using **only information available in the first 6 months** of a startup's life. The project implements **rigorous temporal validation** to ensure no target leakage and provides comprehensive drift diagnostics for production-level monitoring.

## 🎯 Key Innovation

Unlike traditional startup survival models that use lifetime aggregates, VentureSurvive enforces a **strict 6-month snapshot constraint**:
- **Features**: Only derivable from information available at `t ≤ founded_at + 6 months`
- **Target**: Long-term survival (≥5 years) or successful exit (acquired/IPO)
- **Validation**: Temporal split aligned with snapshot dates
- **Monitoring**: Population Stability Index (PSI) and residual shift diagnostics


## 1. Research Problem & Solution

### 🚨 Critical Challenge: Target Leakage
Most startup survival models suffer from **target leakage** by using:
- Lifetime funding totals (`funding_total_usd`, `funding_rounds`)
- Last funding dates (`last_funding_at`)
- Company age at observation (`years_alive`)

These features contain **future information** unavailable at prediction time, leading to **overly optimistic performance** that doesn't generalize.

### ✅ Our Solution: Strict Snapshot Methodology
VentureSurvive implements **rigorous academic temporal rigor**:

- **6-Month Snapshot**: Prediction time = `founded_at + 6 months`
- **Feature Censoring**: Only use events occurring `≤ snapshot_date`
- **Target Definition**: Success = (survived ≥5 years) OR (acquired/IPO)
- **Leakage Guards**: Automated validation prevents future information usage


## 2. Setup & Run Instructions

### 🚀 Quick Start

Follow these steps to set up the environment and run the complete pipeline:

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the complete ML pipeline
python main.py

# 5. Generate all visualizations
python plots.py
```
### 📦 Model Artifacts

Trained models are saved locally to the `models/` directory when running the pipeline.

> **Note**: Model artifacts (`*.joblib`) are intentionally **not tracked by git** to keep the repository lightweight and compatible with GitHub size limits.  
> All models can be fully regenerated end-to-end by running `python main.py`.

### 🔧 Prerequisites

- **Python 3.10+** (tested with Python 3.12)
- **Git** (for cloning the repository)
- **~2GB disk space** (for data + models + results)

### 📋 Environment Details

The project uses a **minimal, production-ready dependency stack**:
- Core ML: `scikit-learn`, `lightgbm`, `numpy`, `pandas`
- Visualization: `matplotlib`, `seaborn`
- Utilities: `scipy`, `joblib`

All dependencies are pinned to specific versions in `requirements.txt` for reproducibility.

### ⚡ What Happens When You Run

1. **Data Preprocessing** (`main.py`):
   - Loads raw startup data
   - Applies strict 6-month snapshot filtering
   - Removes leakage-prone features
   - Saves cleaned dataset to `data/processed/`

2. **Model Training** (`main.py`):
   - Temporal train/test split (80/20 by snapshot_date)
   - Trains 4 models: Logistic Regression, Random Forest (baseline + tuned), LightGBM
   - Saves trained models to `models/`
   - Generates comprehensive evaluation metrics

3. **Visualization Generation** (`plots.py`):
   - Creates ROC/PR curves, confusion matrices
   - Generates feature importance plots
   - Produces drift diagnostics (PSI, residual shift)
   - Saves all plots to `results/`

### 📊 Expected Output

After successful execution, you'll find:
- **Models(generated locally)**: `models/*.joblib` (4 trained models, not tracked by git)
- **Metrics**: `results/metrics_summary.csv` (performance table)
- **Plots**: `results/*.png` (10+ visualizations)
- **Diagnostics**: `results/*_by_year.csv` (drift analysis)

### 🐛 Common Issues

**Memory Error**: Reduce dataset size by modifying `REBUILD_CLEAN_DATASET = False` in `main.py`

**LightGBM Warnings**: Expected feature name warnings - these don't affect functionality

**Empty Test Set**: Auto-cutoff prevents this - if it occurs, check data quality in `data/processed/`

## 3. Data & Preprocessing

### 📊 Dataset Structure
```
data/
├── startups_raw.csv              # Raw Crunchbase-like export
└── processed/
    └── startups_clean.csv        # Cleaned + snapshot-safe features
```

### 🔧 Key Raw Variables
- **Status**: `acquired`, `ipo`, `closed`, `operating`
- **Funding**: `funding_total_usd`, `funding_rounds`
- **Dates**: `founded_at`, `first_funding_at`, `last_funding_at`
- **Geography**: `country_code`, `region`, `city`
- **Industry**: `category_list` (pipe-separated)

### ⚡ Preprocessing Pipeline (`src/preprocess.py`)
1. **Date Validation**: Remove future/outlier dates (post-2025, pre-1980)
2. **Eligibility Filtering**: Keep only companies observable for ≥5 years
3. **Target Engineering**: Compute `success` using survival + exit events
4. **Leakage Prevention**: Drop `last_funding_at`, `years_alive`, lifetime aggregates
5. **Snapshot Creation**: Add `snapshot_date = founded_at + 6 months`

**Result**: Dataset size varies with filtering (see `results/metrics_summary.csv` for latest run statistics)


## 4. Target Definition

### 🎯 Binary Success Target
The `success` variable captures **long-term survival or clear exits**:

```
success = 1 if (years_alive ≥ 5) OR (status ∈ {acquired, ipo})
success = 0 otherwise
```

### ⏰ Temporal Constraint
- **Prediction Time**: `snapshot_date = founded_at + 6 months`
- **Observation Window**: Must be observable for ≥5 years after snapshot
- **Target Engineering**: Uses `last_funding_at` ONLY for label construction, never as feature

### 📈 Success Rate Distribution
- **Train (pre-cutoff)**: Higher success rate (survivorship bias in early cohorts)
- **Test (post-cutoff)**: Lower success rate (more realistic, includes recent failures)

This **temporal shift** is expected and properly quantified in our diagnostics (see `results/residual_shift_by_year.csv`).


## 5. Feature Engineering (Snapshot-Safe)

### 🚫 Forbidden Features (Prevent Leakage)
```
❌ last_funding_at          # Future information
❌ funding_total_usd        # Lifetime aggregate  
❌ funding_rounds           # Lifetime aggregate
❌ years_alive             # Post-snapshot outcome
❌ survived_5y             # Target leakage
```

### ✅ Allowed Features (t ≤ 6 months)
#### Time-Based Features
- `funded_within_6m`: Binary indicator of first funding ≤ snapshot
- `age_at_first_funding_days`: Days from founding to first funding (censored)
- `snapshot_horizon_days`: Exact 6-month window length
- `first_funding_missing`: Missingness indicator

#### Geographic Features
- `is_us`, `is_uk`, `is_eu`: Binary country indicators
- Missingness flags for geographic variables

#### Category Features
- `category_main`: First industry category from pipe-separated list

### 🛡️ Leakage Prevention
```python
# Automated validation in train.py
forbidden = {"last_funding_at", "years_alive", "funding_total_usd"}
assert not any(col in X_train.columns for col in forbidden)
```

4. **Temporal train/test split** (`src/split.py`)
   - Split at a cutoff date using `snapshot_date` (founded_at + 6 months):
     - **Train**: `snapshot_date < cutoff_date`
     - **Test**: `snapshot_date ≥ cutoff_date`
   - Auto-cutoff based on quantiles prevents empty splits after filtering.

5. **Modeling** (`src/models.py`)
   - Common preprocessing via `ColumnTransformer`:
     - numeric pipeline: median imputation + standard scaling;
     - categorical pipeline: most-frequent imputation + one-hot encoding.
   - Models:
     - `make_logistic_regression_pipeline` — baseline linear classifier;
     - `make_random_forest_baseline_pipeline` — default Random Forest;
     - `make_random_forest_tuned_pipeline` — Random Forest with tuned
       hyperparameters (via `RandomizedSearchCV`);
     - `make_lightgbm_pipeline` — gradient boosting model (optional,
       requires `lightgbm`).

6. **Training orchestration** (`src/train.py`)
   - `run_modeling_pipeline` performs the full workflow:
     1. load cleaned data,
     2. assemble features and target `success`,
     3. perform temporal split into train/test,
     4. fit Logistic Regression, Random Forest (baseline, tuned), LightGBM,
     5. evaluate each model on the test set,
     6. save trained pipelines to the `models/` directory.

7. **Evaluation and plots** (`src/evaluate.py`, `plots.py`)
   - Compute standard metrics: accuracy, precision, recall, F1, ROC AUC.
   - Plot ROC and Precision-Recall curves.
   - Plot confusion matrices (normalized).
   - Compute permutation-based feature importance.
   - `plots.py` generates comparison figures and saves them under `results/`.

8. **Exploratory Data Analysis** (`src/eda.py`)
   - Summary statistics and dataset info.
   - Distributions of `status`, `years_alive`, and `success`.
   - Relationships between `success` and funding/geographic variables.


## 6. Modeling & Validation

### 🤖 Model Pipeline
1. **Preprocessing**: Median imputation + scaling (numeric), most-frequent + one-hot (categorical)
2. **Models**: Logistic Regression, Random Forest (baseline/tuned), LightGBM
3. **Cross-Validation**: TimeSeriesSplit (n_splits=3) for hyperparameter tuning
4. **Evaluation**: ROC-AUC, PR-AUC, Brier score

### 🔄 Temporal Validation Strategy
```python
# Split aligned with snapshot_date (founded_at + 6 months)
# Auto-cutoff based on snapshot_date quantile (default 80/20 split)
cutoff_date = snapshot_date.quantile(0.80)  # e.g., 2008-11-01

# Train: snapshot_date < cutoff_date (past cohorts)
# Test:  snapshot_date ≥ cutoff_date (future cohorts)
```

**Note**: The split uses `snapshot_date` (not `first_funding_at`) to align with the prediction time. Auto-cutoff prevents empty splits after filtering.

### 📊 Performance Results
See `results/metrics_summary.csv` for the latest performance metrics. Example from a recent run:

| Model                    | ROC-AUC | PR-AUC | Accuracy | Brier Score |
|--------------------------|---------|--------|----------|-------------|
| Logistic Regression      | 0.66    | 0.65   | 0.59     | 0.28        |
| Random Forest (baseline) | 0.65    | 0.63   | 0.56     | 0.30        |
| Random Forest (tuned)    | 0.67    | 0.65   | 0.60     | 0.23        |
| LightGBM                 | 0.66    | 0.65   | 0.59     | 0.29        |

**Note**: See `results/metrics_summary.csv` for exact latest run statistics.

### 📈 Visual Results
Key visualizations are generated automatically:
- `results/roc_comparison.png` - ROC curves comparison
- `results/pr_comparison.png` - Precision-Recall curves  
- `results/confusion_matrices.png` - Confusion matrices for all models
- `results/feature_importance.png` - Feature importance ranking
- `results/residual_shift_by_year.png` - Success rate by snapshot year

## 7. Drift Diagnostics & Monitoring

### 📈 Population Stability Index (PSI)
Automated drift detection on numeric features (see `results/covariate_shift_psi_numeric.csv`):

```
Top drifted features (PSI):
age_at_first_funding_days: 0.635  (moderate drift)
snapshot_horizon_days:     0.238  (light drift)
funded_within_6m:          0.000  (stable)
```

### 📊 Residual Shift Analysis
Year-by-year success rate comparison between train/test cohorts:
- **Export**: `results/residual_shift_by_year.csv`
- **Visualization**: `results/residual_shift_by_year.png`
- **Insight**: Quantifies temporal distribution shift (expected in survival analysis)

### 🛡️ Monitoring-Style Diagnostics
```python
# Automated checks in pipeline
assert "snapshot_date" not in X_train.columns  # No leakage
assert train_df["snapshot_date"].is_monotonic_increasing  # CV validity
```

## 8. Usage & Reproducibility

### 🚀 Quick Start
```bash
# Clone and setup
git clone <repository>
cd venturesurvive
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run full pipeline
python main.py

# Generate visualizations
python plots.py
```

### 📋 What the Pipeline Does
1. **Data preprocessing** with strict temporal constraints
2. **Feature engineering** (6-month snapshot-safe)
3. **Temporal train/test split** (auto-cutoff based on quantiles)
4. **Model training** (LR, RF, LightGBM) with TimeSeriesSplit CV
5. **Drift diagnostics** (PSI + residual shift analysis)
6. **Performance evaluation** and model saving

### 📊 Outputs Generated
```
models/                          # Trained model pipelines (generated locally, not versioned)
├── log_reg_baseline.joblib
├── random_forest_baseline.joblib
├── random_forest_tuned.joblib
└── lightgbm_baseline.joblib

results/                         # Diagnostics & metrics
├── metrics_summary.csv
├── residual_shift_by_year.csv
├── covariate_shift_psi_numeric.csv
├── roc_comparison.png
├── pr_comparison.png
├── confusion_matrices.png
├── metrics_comparison.png
├── feature_importance.png
├── psi_top_numeric.png
└── cohort_shift_by_year.png
```

### 🔧 Advanced Usage
```python
# Custom cutoff date
from src.train import run_modeling_pipeline
results = run_modeling_pipeline(cutoff_date="2012-01-01")

# Load and evaluate models
from src.evaluate import evaluate_classification
metrics = evaluate_classification(model, X_test, y_test)
```


## 9. Project Structure

```
venturesurvive/
├── data/
│   ├── startups_raw.csv            # Raw dataset (not versioned)
│   └── processed/
│       └── startups_clean.csv      # Cleaned + snapshot-safe dataset
├── models/                         # Trained model pipelines
├── results/                        # Diagnostics, plots, metrics
├── src/
│   ├── __init__.py
│   ├── config.py                   # Paths, constants, SNAPSHOT_MONTHS=6
│   ├── data.py                     # Data loading + date utilities
│   ├── preprocess.py               # Filtering + target engineering
│   ├── features.py                 # 6-month snapshot feature engineering
│   ├── split.py                    # Temporal train/test split
│   ├── models.py                   # Model + preprocessing pipelines
│   ├── evaluate.py                 # Metrics + evaluation utilities
│   ├── train.py                    # End-to-end training + drift diagnostics
│   └── utils.py                    # Logging + generic utilities
├── main.py                         # CLI entry point
├── requirements.txt
└── README.md
```

## 10. Academic Rigor & Limitations

### ✅ Strengths
- **No target leakage**: Strict 6-month snapshot constraint
- **Temporal validation**: Realistic train/test split
- **Drift diagnostics**: PSI and residual shift monitoring
- **Reproducible**: Fixed random seeds, automated pipeline

### ⚠️ Limitations
- **Temporal shift**: Expected in survival analysis
- **Single dataset**: No external validation yet
- **Feature scope**: Limited to basic company attributes
- **Survival proxy**: Uses funding dates as lifetime approximation

### 🎯 Research Contributions
1. **Methodology**: Demonstrates importance of strict temporal constraints
2. **Diagnostics**: Monitoring-style drift detection for ML systems
3. **Benchmark**: Realistic performance baseline for early-stage prediction

## 11. Citation

If you use this work in research, please cite:

```bibtex
@software{venturesurvive2025,
  title={VentureSurvive: Startup Survival Prediction with Strict 6-Month Snapshot},
  author={VentureSurvive Team},
  year={2025},
  url={https://github.com/yourusername/venturesurvive}
}
```