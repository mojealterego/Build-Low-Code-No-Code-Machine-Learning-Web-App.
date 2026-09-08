"""AURELIS ML Studio: a low-code AutoML workspace built with Streamlit."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

try:
    from ydata_profiling import ProfileReport
except ImportError:  # pragma: no cover
    ProfileReport = None

try:
    from streamlit_pandas_profiling import st_profile_report
except ImportError:  # pragma: no cover
    st_profile_report = None

from pycaret.classification import compare_models, finalize_model, predict_model, save_model, setup, tune_model
from pycaret.regression import (
    compare_models as compare_models_reg,
    finalize_model as finalize_model_reg,
    predict_model as predict_model_reg,
    save_model as save_model_reg,
    setup as setup_reg,
    tune_model as tune_model_reg,
)

APP_TITLE = "AURELIS ML Studio"
APP_TAGLINE = "Low-code machine learning, refined."
MAX_UPLOAD_MB = 200
RANDOM_STATE = 42


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --obsidian: #0B0B0D;
            --gold: #D4AF37;
            --snow: #F8F7F4;
            --burgundy: #5A0F24;
            --royal: #173B8F;
            --green: #123C2A;
        }
        .stApp {
            background:
                radial-gradient(circle at 85% 0%, rgba(212,175,55,.12), transparent 32%),
                linear-gradient(180deg, #111114 0%, #0B0B0D 72%);
        }
        [data-testid="stSidebar"] {
            background: #0E0E11;
            border-right: 1px solid rgba(212,175,55,.20);
        }
        .aurelis-hero {
            border: 1px solid rgba(212,175,55,.24);
            background: linear-gradient(135deg, rgba(212,175,55,.10), rgba(90,15,36,.10));
            border-radius: 20px;
            padding: 28px 30px;
            margin-bottom: 20px;
        }
        .aurelis-kicker {
            color: #D4AF37;
            text-transform: uppercase;
            letter-spacing: .22em;
            font-size: .72rem;
            font-weight: 700;
        }
        .aurelis-title {
            margin: 6px 0 4px;
            color: #F8F7F4;
            font-family: Georgia, 'Times New Roman', serif;
            font-size: clamp(2rem, 5vw, 3.6rem);
            line-height: 1;
        }
        .aurelis-subtitle { color: rgba(248,247,244,.76); margin: 0; }
        .metric-card {
            border: 1px solid rgba(212,175,55,.18);
            background: rgba(255,255,255,.03);
            border-radius: 14px;
            padding: 15px 16px;
            min-height: 94px;
        }
        .metric-label {
            color: rgba(248,247,244,.58);
            font-size: .76rem;
            text-transform: uppercase;
            letter-spacing: .08em;
        }
        .metric-value {
            color: #F8F7F4;
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 7px;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid rgba(212,175,55,.34);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #D4AF37;
            color: #D4AF37;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialise_state() -> None:
    defaults: dict[str, Any] = {
        "raw_data": None,
        "prepared_data": None,
        "target": None,
        "task": "Classification",
        "source_name": None,
        "best_model": None,
        "tuned_model": None,
        "best_predictions": None,
        "tuned_predictions": None,
        "model_name": "aurelis_model",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def read_csv_safely(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Plik przekracza limit {MAX_UPLOAD_MB} MB.")

    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1250", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError("Nie udało się odczytać kodowania CSV. Zapisz plik jako UTF-8 i spróbuj ponownie.") from last_error


def upload_data() -> pd.DataFrame | None:
    uploaded = st.file_uploader(
        "Wybierz zbiór danych CSV",
        type=["csv"],
        max_upload_size=MAX_UPLOAD_MB,
        help="Obsługiwane kodowania: UTF-8, UTF-8-SIG, CP1250 i Latin-1.",
    )
    if uploaded is None:
        return st.session_state.raw_data

    if st.session_state.source_name == uploaded.name:
        return st.session_state.raw_data

    try:
        data = read_csv_safely(uploaded)
    except Exception as exc:
        st.error(f"Nie udało się wczytać danych: {exc}")
        return None

    if data.empty or data.shape[1] < 2:
        st.error("CSV musi zawierać co najmniej jeden wiersz i dwie kolumny.")
        return None

    st.session_state.raw_data = data
    st.session_state.source_name = uploaded.name
    st.session_state.prepared_data = None
    st.session_state.best_model = None
    st.session_state.tuned_model = None
    st.session_state.best_predictions = None
    st.session_state.tuned_predictions = None
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()].copy()

    for column in cleaned.columns:
        series = cleaned[column]
        if not series.isna().any():
            continue
        if pd.api.types.is_numeric_dtype(series):
            replacement = series.median()
            cleaned[column] = series.fillna(0 if pd.isna(replacement) else replacement)
        else:
            mode = series.mode(dropna=True)
            cleaned[column] = series.fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return cleaned.drop_duplicates().reset_index(drop=True)


def prepare_for_pycaret(data: pd.DataFrame, target: str) -> pd.DataFrame:
    prepared = data.copy()
    for column in prepared.columns:
        if column == target:
            continue
        if pd.api.types.is_object_dtype(prepared[column]) or pd.api.types.is_string_dtype(prepared[column]):
            prepared[column] = prepared[column].astype(str)
    return prepared


def safe_model_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())[:80] or "aurelis_model"


def render_header() -> None:
    st.markdown(
        f"""
        <div class="aurelis-hero">
            <div class="aurelis-kicker">AURELIS AI · MACHINE LEARNING</div>
            <div class="aurelis-title">{APP_TITLE}</div>
            <p class="aurelis-subtitle">{APP_TAGLINE} Przygotuj dane, zdiagnozuj ich jakość i wytrenuj modele AutoML bez pisania pipeline'u od zera.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(data: pd.DataFrame) -> None:
    missing = int(data.isna().sum().sum())
    duplicates = int(data.duplicated().sum())
    memory_mb = data.memory_usage(deep=True).sum() / (1024**2)
    cards = [
        ("Wiersze", f"{len(data):,}"),
        ("Kolumny", f"{data.shape[1]:,}"),
        ("Braki", f"{missing:,}"),
        ("Duplikaty", f"{duplicates:,}"),
        ("Pamięć", f"{memory_mb:.2f} MB"),
    ]
    columns = st.columns(len(cards))
    for column, (label, value) in zip(columns, cards):
        column.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )


def render_quality(data: pd.DataFrame, target: str) -> None:
    quality = pd.DataFrame(
        {
            "Typ": data.dtypes.astype(str),
            "Braki": data.isna().sum(),
            "Unikalne": data.nunique(dropna=False),
            "Braków %": (data.isna().mean() * 100).round(2),
        }
    )
    st.dataframe(quality, use_container_width=True)
    st.caption(f"Target: **{target}** · Duplikaty przed czyszczeniem: **{data.duplicated().sum():,}**")


def render_eda(data: pd.DataFrame) -> None:
    if ProfileReport is None or st_profile_report is None:
        st.info("Pełny raport EDA jest niedostępny w tym środowisku. Profil tabelaryczny pozostaje dostępny powyżej.")
        return
    with st.expander("Pełny raport EDA", expanded=False):
        try:
            profile = ProfileReport(data, minimal=True, explorative=True)
            st_profile_report(profile)
        except Exception as exc:
            st.warning(f"Nie udało się wygenerować raportu EDA: {exc}")


def normalise_predictions(predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    result = predictions.copy()
    label_column = "prediction_label" if "prediction_label" in result.columns else "Label"
    if target in result.columns:
        result["y_true"] = result[target]
    elif "Label" in result.columns:
        result["y_true"] = result["Label"]
    else:
        raise ValueError("PyCaret nie zwrócił kolumny z wartościami referencyjnymi.")
    if label_column not in result.columns:
        raise ValueError("PyCaret nie zwrócił kolumny predykcji.")
    result["y_pred"] = result[label_column]
    return result


def regression_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y_true = pd.to_numeric(predictions["y_true"], errors="coerce")
    y_pred = pd.to_numeric(predictions["y_pred"], errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    if not valid.any():
        return {}
    return {
        "R²": float(r2_score(y_true[valid], y_pred[valid])),
        "MAE": float(mean_absolute_error(y_true[valid], y_pred[valid])),
        "RMSE": float(np.sqrt(mean_squared_error(y_true[valid], y_pred[valid]))),
    }


def classification_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y_true = predictions["y_true"].astype(str)
    y_pred = predictions["y_pred"].astype(str)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1 weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "F1 macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def render_model_metrics(predictions: pd.DataFrame, task: str) -> None:
    metrics = classification_metrics(predictions) if task == "Classification" else regression_metrics(predictions)
    if not metrics:
        st.warning("Nie udało się wyliczyć metryk hold-out.")
        return
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, f"{value:.4f}")


def run_automl(data: pd.DataFrame, target: str, task: str, test_size: float) -> tuple[Any, pd.DataFrame]:
    common = {
        "data": data,
        "target": target,
        "train_size": 1 - test_size,
        "session_id": RANDOM_STATE,
        "verbose": False,
        "n_jobs": -1,
    }
    if task == "Classification":
        setup(**common)
        model = compare_models(sort="Accuracy", n_select=1, turbo=True, verbose=False)
        prediction = predict_model(model, verbose=False)
    else:
        setup_reg(**common)
        model = compare_models_reg(sort="R2", n_select=1, turbo=True, verbose=False)
        prediction = predict_model_reg(model, verbose=False)
    return model, normalise_predictions(prediction, target)


def tune_selected_model(model: Any, task: str) -> tuple[Any, pd.DataFrame]:
    if task == "Classification":
        tuned = tune_model(model, optimize="F1", verbose=False)
        prediction = predict_model(tuned, verbose=False)
    else:
        tuned = tune_model_reg(model, optimize="R2", verbose=False)
        prediction = predict_model_reg(tuned, verbose=False)
    return tuned, normalise_predictions(prediction, st.session_state.target)


def export_model(model: Any, task: str, model_name: str, metadata: dict[str, Any]) -> tuple[bytes, bytes, str]:
    with TemporaryDirectory(prefix="aurelis_export_") as temp_dir:
        base_path = Path(temp_dir) / safe_model_name(model_name)
        if task == "Classification":
            result = save_model(model, str(base_path), verbose=False)
        else:
            result = save_model_reg(model, str(base_path), verbose=False)
        model_path = Path(result[0]) if isinstance(result, tuple) else Path(f"{base_path}.pkl")
        metadata_bytes = json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")
        return model_path.read_bytes(), metadata_bytes, model_path.name


def sidebar_controls(data: pd.DataFrame | None) -> tuple[float, bool]:
    with st.sidebar:
        st.markdown("### AURELIS AI")
        st.caption("Intelligence, refined.")
        st.divider()
        test_size = st.slider("Rozmiar zbioru testowego", 0.10, 0.40, 0.20, 0.05)
        run_requested = st.button("Uruchom AutoML", type="primary", use_container_width=True, disabled=data is None)
        st.divider()
        st.caption("PyCaret · seed 42 · równoległość CPU: -1")
    return test_size, run_requested


def main() -> None:
    configure_page()
    initialise_state()
    render_header()

    data = upload_data()
    test_size, run_requested = sidebar_controls(data)

    if data is None:
        st.info("Wgraj plik CSV, aby rozpocząć analizę.")
        return

    render_metrics(data)
    cleaned = clean_data(data)
    target_default = st.session_state.target if st.session_state.target in cleaned.columns else cleaned.columns[-1]

    st.subheader("Konfiguracja eksperymentu")
    task = st.radio("Typ zadania", ["Classification", "Regression"], horizontal=True, key="task_selector")
    target = st.selectbox("Zmienna docelowa (target)", list(cleaned.columns), index=list(cleaned.columns).index(target_default))
    st.session_state.target = target
    st.session_state.task = task

    prepared = prepare_for_pycaret(cleaned, target)
    st.session_state.prepared_data = prepared

    tab_data, tab_quality, tab_models = st.tabs(["Dane", "Jakość i EDA", "Modele"])
    with tab_data:
        st.dataframe(cleaned.head(100), use_container_width=True)
        st.caption(f"Źródło: {st.session_state.source_name or 'CSV'} · Po czyszczeniu: {len(cleaned):,} wierszy")
    with tab_quality:
        render_quality(cleaned, target)
        render_eda(cleaned)

    with tab_models:
        if run_requested:
            with st.spinner("Uruchamiam eksperyment AutoML…"):
                try:
                    model, predictions = run_automl(prepared, target, task, test_size)
                    st.session_state.best_model = model
                    st.session_state.best_predictions = predictions
                    st.session_state.tuned_model = None
                    st.session_state.tuned_predictions = None
                    st.success("AutoML zakończony. Wybrano najlepszy model na podstawie cross-validation.")
                except Exception as exc:
                    st.error(f"AutoML nie powiódł się: {exc}")

        model = st.session_state.best_model
        predictions = st.session_state.best_predictions
        if model is None or predictions is None:
            st.info("Konfiguracja jest gotowa. Uruchom AutoML z panelu po lewej.")
            return

        st.subheader("Najlepszy model")
        st.code(str(model), language="text")
        render_model_metrics(predictions, task)
        st.dataframe(predictions.head(50), use_container_width=True)

        col_tune, col_name = st.columns([1, 2])
        with col_tune:
            if st.button("Strojenie hiperparametrów", use_container_width=True):
                with st.spinner("Strojenie modelu…"):
                    try:
                        tuned, tuned_predictions = tune_selected_model(model, task)
                        st.session_state.tuned_model = tuned
                        st.session_state.tuned_predictions = tuned_predictions
                        st.success("Model został dostrojony.")
                    except Exception as exc:
                        st.error(f"Strojenie nie powiodło się: {exc}")
        with col_name:
            st.session_state.model_name = st.text_input(
                "Nazwa artefaktu",
                value=st.session_state.model_name,
                help="Używana jako nazwa eksportowanego pliku modelu.",
            )

        tuned_model = st.session_state.tuned_model
        tuned_predictions = st.session_state.tuned_predictions
        if tuned_model is not None and tuned_predictions is not None:
            st.divider()
            st.subheader("Model po tuningu")
            render_model_metrics(tuned_predictions, task)
            st.dataframe(tuned_predictions.head(50), use_container_width=True)
            if st.button("Finalizuj model", use_container_width=True):
                with st.spinner("Finalizuję model na pełnym zbiorze treningowym…"):
                    try:
                        if task == "Classification":
                            st.session_state.tuned_model = finalize_model(tuned_model)
                        else:
                            st.session_state.tuned_model = finalize_model_reg(tuned_model)
                        st.success("Model został sfinalizowany.")
                    except Exception as exc:
                        st.error(f"Finalizacja nie powiodła się: {exc}")

        st.divider()
        st.subheader("Eksport")
        export_target = st.session_state.tuned_model or st.session_state.best_model
        if export_target is not None:
            metadata = {
                "application": APP_TITLE,
                "source": st.session_state.source_name,
                "task": task,
                "target": target,
                "model_name": safe_model_name(st.session_state.model_name),
                "random_state": RANDOM_STATE,
            }
            try:
                model_bytes, metadata_bytes, model_filename = export_model(
                    export_target, task, st.session_state.model_name, metadata
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Pobierz model (.pkl)",
                        data=model_bytes,
                        file_name=model_filename,
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
                with c2:
                    st.download_button(
                        "Pobierz metadane (.json)",
                        data=metadata_bytes,
                        file_name=f"{safe_model_name(st.session_state.model_name)}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
            except Exception as exc:
                st.warning(f"Eksport modelu nie jest obecnie dostępny: {exc}")


if __name__ == "__main__":
    main()
