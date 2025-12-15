# VentureSurvive — Startup Survival Prediction for VC Risk Assessment

VentureSurvive is a research-oriented project that studies **startup survival** using
public venture data. The goal is to build **reproducible**, **interpretable** models
that estimate the probability that a startup will still be alive (or will have
reached a successful exit) after at least five years. The target audience is
primarily academics and data scientists interested in startup dynamics and
venture capital risk assessment.


## 1. Research Problem

Early-stage investors and researchers often face two challenges:

- **Data sparsity and censoring**: many startups disappear silently without a
  clear "closed" event, and observation windows are limited.
- **Noisy status labels**: raw company status fields (e.g. `acquired`, `ipo`,
  `closed`, `operating`) mix truly failed startups, still-surviving startups,
  and success stories.

This project defines a more meaningful **binary survival target** and builds a
full machine learning pipeline to predict it, with an emphasis on:

- clear **target definition** based on survival horizon and exit events,
- explicit **feature construction** (funding history, geography, categories,
  timing of events),
- rigorous **temporal validation** (train/test split in time),
- **interpretable evaluation** and feature importance analysis.


## 2. Data

The project uses a startups dataset in CSV form (e.g. Crunchbase-like export),
stored under:

- `data/startups_raw.csv` — raw dataset,
- `data/processed/startups_clean.csv` — cleaned and feature-enriched dataset
  produced by the preprocessing pipeline.

Key raw variables include (non-exhaustive):

- `status`: categorical company status (`acquired`, `ipo`, `closed`, ...),
- `funding_total_usd`: total funding raised (string/float),
- `funding_rounds`: number of funding rounds,
- `country_code`, `region`, `city`, `state_code`: geography,
- `category_list`: pipe-separated list of industries,
- `founded_at`, `first_funding_at`, `last_funding_at`: key dates.

The **cleaned dataset** adds engineered variables such as:

- `years_alive`: approximate company lifetime in years,
- `survived_5y`: indicator of survival for at least 5 years,
- `success`: final binary target used for modeling,
- time-based, funding-based and geographic features (see below).


## 3. Target Definition

The main binary target `success` is defined to capture **long-term survival or
clear exits**, instead of raw status alone.

Let:

- `years_alive` be the time (in years) between `founded_at` and the last
  observed funding date (or first funding date, when `last_funding_at` is
  missing),
- `status ∈ {acquired, ipo, closed, ...}` be the status field.

Then:

- `survived_5y = 1` if `years_alive ≥ 5`, else `0`.
- `success = 1` if `survived_5y = 1` **or** `status ∈ {acquired, ipo}`.
- `success = 0` otherwise.

Thus, a company is considered **successful** if it either:

1. survives at least 5 years (proxy for resilience), or
2. reaches a clear exit event (`acquired` or IPO).

This definition is implemented in `src/data.py` / `src/preprocess.py` and
ensures a consistent, reproducible target across notebooks and scripts.


## 4. Methodology and Pipeline

The project follows a **code-first**, modular design. All business logic lives
in `src/`, while notebooks are thin orchestration/visualization clients.

### 4.1 Overall pipeline

1. **Data loading** (`src/data.py`)
   - Load raw CSV (`load_raw_data`).
   - Convert relevant columns to appropriate types (dates, numerics).

2. **Preprocessing & target engineering** (`src/preprocess.py`)
   - Filter to modeling-relevant statuses: `acquired`, `ipo`, `closed`.
   - Convert date columns to pandas `datetime`.
   - Compute `years_alive`, `survived_5y`, and `success`.
   - Save the cleaned dataset to `data/processed/startups_clean.csv`.

3. **Feature engineering** (`src/features.py`)
   - Time-based features (in days/years):
     - `age_at_first_funding_days`,
     - `time_between_first_last_days`,
     - `time_to_first_funding_years`,
     - `company_age_at_last_funding_years`,
     - `avg_round_interval_years`.
   - Funding-based features:
     - numeric-safe `funding_total_usd_num`,
     - `funding_per_round`, `log_funding_total`, `log_funding_per_round`,
     - `high_total_funding`, `high_funding_per_round`, `has_multiple_rounds`.
   - Geographic & category features:
     - `is_us`, `is_uk`, `is_eu`, missingness flags,
     - `category_main` extracted from `category_list`.
   - Remove obvious identifiers and leakage-prone columns
     (`permalink`, `name`, `homepage_url`, `category_list`, etc.).

4. **Temporal train/test split** (`src/split.py`)
   - Split at a fixed cutoff date (default `2013-01-01`) using
     `first_funding_at`:
     - **Train**: startups with first funding strictly before cutoff.
     - **Test**: startups with first funding on or after cutoff.
   - This simulates a realistic “train on the past, test on the future” setup.

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


## 5. Experiments and Results

Experiments are run via:

```bash
python main.py
```

using a fixed random seed (`RANDOM_STATE = 42`) and a temporal cutoff date of
`2013-01-01`. On the current dataset (≈13k startups), the following test
performances are obtained:

| Model                     | Accuracy | Precision | Recall | F1    | ROC AUC |
|---------------------------|----------|-----------|--------|-------|---------|
| Logistic Regression       | 0.80     | 0.69      | 0.72   | 0.70  | **0.87** |
| Random Forest (baseline)  | 0.78     | 0.65      | 0.71   | 0.68  | 0.85    |
| Random Forest (tuned)     | 0.80     | 0.69      | 0.71   | 0.70  | 0.86    |
| LightGBM (baseline)       | 0.79     | 0.68      | 0.71   | 0.70  | 0.86    |

These numbers are reproduced programmatically by the `run_modeling_pipeline`
function and summarized again in `plots.py`.


### 5.1 Feature importance (Logistic Regression)

Permutation feature importance (computed on the test set) indicates that the
most predictive variables include:

- **`category_main`** — high-level industry category;
- **funding intensity features** — `log_funding_per_round`,
  `log_funding_total`;
- **time-based features** — `age_at_first_funding_days`,
  `time_to_first_funding_years`;
- **geographic indicators** — `is_us`, `country_code`, and missingness flags.

These results suggest that both **industry** and **funding dynamics** (level
and timing) are important drivers of long-term startup survival.


## 6. Reproducibility and Usage

### 6.1 Environment setup

The project uses Python 3 and a standard scientific stack (`pandas`,
`numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `lightgbm`). Dependencies
are listed in `requirements.txt`.

Typical setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Running the pipeline

1. **Preprocess data and train models** (end-to-end):

   ```bash
   python main.py
   ```

   This will:

   - load raw / cleaned data,
   - assemble features,
   - perform temporal split,
   - train all models,
   - output metrics to the console,
   - save trained models under `models/`.

2. **Generate evaluation plots**:

   ```bash
   python plots.py
   ```

   This will:

   - load the trained models from `models/`,
   - recompute the test set from the cleaned data,
   - generate and save the following figures under `results/`:
     - `roc_comparison.png` — ROC curves for all models,
     - `pr_comparison.png` — Precision-Recall curves,
     - `confusion_matrices.png` — normalized confusion matrices,
     - `metrics_comparison.png` — bar chart of main metrics,
     - `feature_importance.png` — top feature importances.
   - save a CSV table of permutation importances as
     `feature_importance.csv`.


## 7. Project Structure

```text
venturesurvive/
├── data/
│   ├── startups_raw.csv            # raw dataset (not versioned)
│   └── processed/
│       └── startups_clean.csv      # cleaned + engineered dataset
├── models/                         # saved trained model pipelines
├── results/                        # figures and CSVs from evaluation
├── notebooks/                      # EDA and modeling notebooks (thin clients)
├── src/
│   ├── __init__.py
│   ├── config.py                   # paths, constants, random seed
│   ├── data.py                     # data loading + date/target helpers
│   ├── preprocess.py               # filtering + target engineering
│   ├── features.py                 # feature engineering utilities
│   ├── split.py                    # temporal train/test split
│   ├── models.py                   # model + preprocessing pipelines
│   ├── evaluate.py                 # metrics + plotting helpers
│   ├── eda.py                      # EDA summaries and plots
│   ├── train.py                    # end-to-end training orchestration
│   └── utils.py                    # logging and generic utilities
├── main.py                         # CLI entry point: train models
├── plots.py                        # CLI entry point: generate plots
├── requirements.txt
└── README.md
```


## 8. Limitations and Future Work

- **Censoring and observation window**: the proxy used for company lifetime
  (based on funding dates) may underestimate survival for bootstrapped or
  late-filing companies.
- **Label noise**: the `status` field may itself be noisy or incomplete,
  especially for still-operating but low-visibility startups.
- **External covariates**: current features focus on company-level attributes;
  macroeconomic conditions and market-level signals are not yet included.

Possible extensions include:

- more refined survival modeling (e.g. Cox models, competing risks,
  explicit censoring),
- richer text-based features from company descriptions or websites,
- cross-market generalization studies (e.g. training on US, testing on EU).


## 9. Citation

If you build upon this work in an academic or industrial context, you can cite
it informally as:

> *VentureSurvive: Startup Survival Prediction for VC Risk Assessment*.
> GitHub repository, 2025.

