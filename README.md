# Student Score Prediction

A tabular regression project that estimates maths scores from reading/writing scores and five student background fields. Includes exploratory notebooks, cross-validated model selection, and a local Flask prediction form.

## Problem

Explore how observed exam scores and background variables relate to maths performance. This is an educational modelling exercise; reading and writing results must already be available at prediction time.

## Key Features

- Dataset schema, category, missing-value, duplicate, and score-range checks.
- Seven regression model families compared using training-only cross-validation.
- Preprocessing and prediction saved together in one fitted pipeline.
- Flask form with server-side input validation and model-availability handling.
- EDA and exploratory model-comparison notebooks; regression tests for the pipeline and form.

## Architecture

```text
Bundled CSV -> validation -> 800 training / 200 test rows
Training rows -> 3-fold CV [imputation -> encoding/scaling -> regressor]
Best mean CV R2 -> refit on 800 rows -> held-out evaluation + mean baseline
Saved fitted pipeline -> Flask form -> estimated maths score
```

## Dataset

`notebook/data/stud.csv` contains **1,000 rows and 8 columns**, with no missing values or duplicate rows in the audited copy.

- Target: `math_score` (observed range 0â€“100).
- Numeric predictors: `reading_score`, `writing_score`.
- Categorical predictors: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`.
- Split: 80/20, `random_state=42`, without stratification.
- Preprocessing: median numeric imputation and standard scaling; most-frequent categorical imputation, one-hot encoding, and scaling without centring. Unseen categories are ignored by the encoder; the web form accepts the documented categories only.

The original dataset source and redistribution licence are **unverified**: no attribution or licence was supplied in the repository. The exact audited file hash is recorded in [the results](reports/metrics.json).

## Methodology / Models

The training script compares linear regression, decision trees, random forests, gradient boosting, AdaBoost, XGBoost, and CatBoost. Linear regression provides a simple linear reference; tree models explore nonlinear relationships. The original search grids are retained except for an unsupported decision-tree criterion.

Each candidate uses three shuffled training folds with seed 42. Preprocessing is fitted inside each fold. Selection uses mean validation R2, then the chosen pipeline is refitted on all training rows. The test set is evaluated after selection, alongside a training-mean dummy baseline. Stochastic estimators use fixed seeds.

The model notebook also explores Ridge, Lasso, and nearest neighbours. Its repeated held-out comparisons are exploratory and are not the authoritative model-selection procedure.

## Evaluation / Results

Verified locally with Python 3.11.9 on Windows using the pinned dependencies. The selected model was **Linear Regression**, with mean training CV R2 **0.8678** (fold standard deviation **0.0097**).

| Model | Held-out R2 | MAE (score points) | RMSE (score points) |
|---|---:|---:|---:|
| Selected linear regression | 0.8804 | 4.2148 | 5.3940 |
| Training-mean baseline | -0.0170 | 12.3399 | 15.7316 |

Full metrics, selected parameters, dataset hash, and library versions: [reports/metrics.json](reports/metrics.json). These are regression metrics, not classification accuracy. CV scores were used for selection and are not independent performance estimates. The original project already inspected this test split; these results are not a fresh external validation.

## Project Structure

```text
app.py                         Flask entry point
src/config.py                  Shared schema and checkout-relative paths
src/components/                Data ingestion, preprocessing, training
src/pipeline/predict_pipeline.py  Saved-pipeline inference and form validation
src/utils.py                   CV search and persistence
notebook/                      EDA, exploratory models, source dataset
reports/metrics.json            Verified run snapshot
reports/audit.md                Audit findings and verification record
templates/                     HTML pages
tests/                         Regression tests
artifacts/                     Locally generated files; excluded from Git
```

## Technology Stack

Python, pandas, NumPy, scikit-learn, XGBoost, CatBoost, Flask/Jinja, and HTML. Notebooks use JupyterLab, Matplotlib, and Seaborn. Tests use Python's built-in `unittest`.

## Installation

Use Python 3.11. From a terminal:

```shell
git clone https://github.com/manu-nani18/Grade-Prediction.git
cd Grade-Prediction
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux, then:

```shell
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install keeps paths tied to this checkout. Direct dependency versions are pinned; transitive dependencies are not fully locked. Only Windows/Python 3.11 was tested.

## Usage

Run from the repository root with the environment active:

```shell
python -m src.components.data_ingestion
python app.py
```

Training runs the full search and creates `artifacts/model.pkl`, split CSVs, and `artifacts/metrics.json`. Allow several minutes, depending on hardware. Open **http://127.0.0.1:5000** and follow the form link. Train before submitting a prediction; old two-file model artifacts are incompatible with the corrected single-pipeline format. Only load pickle files you generated from trusted code.

```shell
python -m unittest discover -s tests -v
python -m pip check
```

For notebooks:

```shell
python -m pip install -r requirements-notebooks.txt
cd notebook
jupyter lab
```

Execute cells in order using the same environment. Notebook outputs are cleared in Git to avoid stale results and large embedded plots. Regeneration writes local results; `reports/metrics.json` is the audited snapshot, not automatically overwritten by later runs.

## Limitations

- Small, single-dataset exercise with a previously inspected random test split; no external or temporal validation.
- Dataset provenance and redistribution licence need confirmation before public redistribution.
- Gender and ethnicity are predictors. No fairness analysis or justification for use in real educational decisions is provided.
- Reading and writing scores are contemporaneous predictors, so this is not an early-warning system.
- No causal claims, uncertainty estimates, or guaranteed 0â€“100 prediction bounds.
- Local Flask development server only; no verified deployment, authentication, monitoring, or production infrastructure.

## Future Improvements

Confirm dataset attribution, assess subgroup errors and feature ablations, and evaluate on independently collected data before considering broader use.

## Author

Mohan Ramamurthy

MSc Data Science, University of Hertfordshire

## License

No repository licence file was present. No licence has been assumed or added; the author must choose one and separately confirm dataset redistribution terms.
