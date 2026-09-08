# AURELIS ML Studio

**Low-code machine learning, refined.**

A production-oriented Streamlit workspace for exploratory data analysis, dataset preparation and AutoML experiments. AURELIS ML Studio is designed to turn a raw CSV dataset into a reproducible machine-learning experiment without requiring the user to write model-training code.

## What it does

- Imports CSV datasets with explicit size and encoding safeguards.
- Profiles dataset shape, missing values, duplicate rows and column cardinality.
- Cleans missing values and duplicate observations before modeling.
- Lets the user choose a **classification** or **regression** workflow and an explicit target column.
- Uses PyCaret cross-validation to compare candidate models automatically.
- Reports hold-out performance with task-appropriate metrics.
- Supports hyperparameter tuning and optional finalization for deployment-oriented training.
- Exports the selected model together with machine-readable experiment metadata.
- Preserves experiment state across Streamlit reruns with `st.session_state`.
- Provides a dark, restrained AURELIS visual system intended for an analytical workspace rather than a tutorial demo.

## Architecture

```text
CSV upload
   ↓
Input validation
   ↓
Data quality profile
   ↓
Cleaning / normalization
   ↓
Task + target selection
   ↓
PyCaret experiment setup
   ↓
Cross-validated AutoML comparison
   ↓
Hold-out evaluation
   ↓
Optional hyperparameter tuning
   ↓
Optional model finalization
   ↓
Model + JSON metadata export
```

## Run locally

From the application directory:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

For hosted environments, the application exposes the standard Streamlit process through the repository `Procfile`.

## Data expectations

The input must be a CSV with at least two columns and one data row. One column is selected as the target; the remaining columns are treated as model inputs. Categorical feature columns are passed to PyCaret as categorical data rather than forcibly converted with an ordinal `LabelEncoder`, allowing the AutoML framework to select an appropriate preprocessing pipeline.

The uploader enforces a 200 MB per-file limit and attempts UTF-8, UTF-8-SIG, CP1250 and Latin-1 decoding. Streamlit itself documents that upload limits can be configured globally and that file-type checks are best-effort, so validation is also performed by the application. citeturn688132search1

## Modeling

The application initializes a PyCaret experiment and uses `compare_models` for automated model comparison. PyCaret documents `compare_models` as a cross-validation based model selection mechanism and `predict_model` as the hold-out/unseen-data prediction interface. citeturn767220search0

Classification experiments are ranked on **Accuracy** during the initial model comparison and tuned on **F1**. Regression experiments are ranked and tuned on **R²**. The UI also reports secondary metrics so that model quality is not reduced to a single number.

## State and reproducibility

Streamlit reruns the application on widget interaction. The project therefore stores the active dataset, selected target, task and trained model objects in session state, using a fixed experiment seed of `42`. Streamlit documents `st.session_state` as the mechanism for persisting state between reruns and widget interactions. citeturn688132search0

## Operational notes

The repository originally contained an old demonstration implementation, duplicated application code and a large generated log file. The application has now been consolidated around one primary `app.py` entry point. Generated logs should not be versioned as source artifacts; operational logging should be handled by the hosting environment instead.

The dependency set should be kept aligned with the deployed Python runtime. The historical environment used very old package pins, including Streamlit 0.89.0, pandas 1.3.3, NumPy 1.19.5, scikit-learn 0.24.2 and PyCaret 2.3.4. Those versions are not treated as a design target for the rebuilt application; the runtime should be validated against a current, mutually compatible stack before production deployment.

## Project identity

**AURELIS AI**

*Intelligence, refined.*  
*Inteligencja. Precyzja. Forma.*

The machine-learning studio is positioned as a serious analytical instrument: minimal interface, explicit assumptions, reproducible experiments and portable artifacts.
