from __future__ import annotations

import io
import json
import re
import unicodedata
import wave
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils.database import (
    authenticate_user,
    complete_reminder,
    create_reminder,
    delete_reminder,
    get_prediction_history,
    get_unread_notification_count,
    get_user_notifications,
    get_user_reminders,
    initialize_database,
    mark_all_notifications_read,
    mark_notification_read,
    process_due_reminders,
    register_user,
    save_prediction,
)
from utils.weather_api import get_weather_forecast, search_locations, weather_code_label

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False


# -----------------------------------------------------------------------------
# Smart Farming Assistant
# This app reads the CSV files that are actually present in ./data.
# Model inputs and widgets are built from each dataset's real columns.
# -----------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

st.set_page_config(
    page_title="Smart Farming Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .hero {
            padding: 1.35rem 1.5rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #113b2b 0%, #206044 60%, #47845b 100%);
            color: white;
            margin-bottom: 1rem;
        }
        .hero h1 {margin: 0; font-size: 2.25rem;}
        .hero p {margin: .45rem 0 0 0; font-size: 1rem; opacity: .94;}
        .small-note {font-size: .9rem; opacity: .85;}
        div[data-testid="stMetric"] {border: 1px solid rgba(120,120,120,.2); padding: 12px; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -- Column and dataset detection ------------------------------------------------

def normalise_name(value: Any) -> str:
    """Normalise a column name for matching without changing the source CSV."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def read_csv_safely(path_text: str) -> pd.DataFrame:
    """Read common CSV encodings/delimiters safely."""
    path = Path(path_text)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    if df.shape[1] == 1:
        # Handle a semicolon/tab separated file whose extension is still .csv.
        try:
            alternative = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
            if alternative.shape[1] > 1:
                df = alternative
        except Exception:
            pass
    df.columns = [str(column).strip() for column in df.columns]
    # Remove accidental completely empty rows/columns, but do not change the data otherwise.
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df


@st.cache_data(show_spinner=False)
def load_csv_cached(path_text: str, modified_time: float, file_size: int) -> pd.DataFrame:
    # modified_time/file_size are cache keys: when the CSV changes, it reloads.
    return read_csv_safely(path_text)


def find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = {normalise_name(column): str(column) for column in df.columns}
    for alias in aliases:
        if normalise_name(alias) in lookup:
            return lookup[normalise_name(alias)]
    return None


CROP_TARGET_ALIASES = ["label", "crop_label", "recommended_crop", "crop_name", "crop"]
FERTILIZER_TARGET_ALIASES = [
    "fertilizer_name", "fertilizer_name_recommended", "recommended_fertilizer",
    "fertilizer_label", "fertilizer", "label",
]
YIELD_TARGET_ALIASES = ["yield", "crop_yield", "crop_yield_per_hectare", "yield_tonnes_per_hectare"]


def classify_dataset(df: pd.DataFrame, filename: str) -> str:
    """Identify one of the expected datasets from columns first, then filename."""
    n = find_column(df, ["N", "nitrogen"])
    p = find_column(df, ["P", "phosphorus", "phosphorous"])
    k = find_column(df, ["K", "potassium"])
    temp = find_column(df, ["temperature", "temparature", "temp"])
    humidity = find_column(df, ["humidity"])
    ph = find_column(df, ["ph", "soil_ph", "ph_value"])
    rainfall = find_column(df, ["rainfall", "annual_rainfall", "precipitation"])
    crop_target = find_column(df, CROP_TARGET_ALIASES)

    normal_cols = {normalise_name(c) for c in df.columns}
    file_key = normalise_name(Path(filename).stem)

    # Classic crop recommendation data has N/P/K + temperature + humidity + pH + rainfall + crop label.
    if all([n, p, k, temp, humidity, ph, rainfall, crop_target]):
        return "crop"
    if "crop_recommendation" in file_key or "crop_recommend" in file_key:
        return "crop"

    yield_target = find_column(df, YIELD_TARGET_ALIASES)
    has_area = find_column(df, ["area", "cultivated_area"]) is not None
    has_crop = find_column(df, ["crop", "crop_name"]) is not None
    if yield_target and (has_area or has_crop or "crop_yield" in file_key or "yield" in file_key):
        return "yield"

    fertilizer_target = find_column(df, FERTILIZER_TARGET_ALIASES)
    fertilizer_context = any(
        find_column(df, aliases) is not None
        for aliases in [
            ["soil_type", "soiltype"],
            ["crop_type", "croptype"],
            ["moisture", "soil_moisture"],
        ]
    )
    if fertilizer_target and (fertilizer_context or "fertilizer" in file_key):
        # Avoid treating the numeric Fertilizer amount in a yield dataset as a target.
        target_norm = normalise_name(fertilizer_target)
        if target_norm not in {"fertilizer", "fertilizer_used"} or fertilizer_context:
            return "fertilizer"

    # Common filename fallback, while still showing the CSV schema for checking.
    if "fertilizer" in file_key:
        return "fertilizer"
    if "yield" in file_key:
        return "yield"
    return "other"


def scan_datasets() -> list[dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    found: list[dict[str, Any]] = []
    for path in sorted((item for item in DATA_DIR.iterdir() if item.is_file() and item.suffix.lower() == ".csv"), key=lambda item: item.name.lower()):
        try:
            stat = path.stat()
            frame = load_csv_cached(str(path), stat.st_mtime, stat.st_size)
            role = classify_dataset(frame, path.name)
            found.append({
                "name": path.name,
                "path": str(path),
                "role": role,
                "df": frame,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            })
        except Exception as exc:
            found.append({
                "name": path.name,
                "path": str(path),
                "role": "error",
                "df": pd.DataFrame(),
                "mtime": 0.0,
                "size": 0,
                "error": str(exc),
            })
    return found


def datasets_by_role(all_data: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    result: dict[str, dict[str, Any] | None] = {"crop": None, "fertilizer": None, "yield": None}
    for item in all_data:
        role = item["role"]
        if role in result and result[role] is None:
            result[role] = item
    return result


# -- Model preparation ------------------------------------------------------------

def convert_numeric_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking text columns while leaving normal categories intact."""
    data = frame.copy()
    for column in data.columns:
        if data[column].dtype == "object":
            text_values = data[column].astype(str).str.replace(",", "", regex=False).str.strip()
            converted = pd.to_numeric(text_values, errors="coerce")
            non_empty = data[column].notna().sum()
            if non_empty > 0 and converted.notna().sum() / non_empty >= 0.9:
                data[column] = converted
    return data


def choose_target(df: pd.DataFrame, role: str) -> str | None:
    if role == "crop":
        # Prefer the common 'label' field in the standard crop recommendation dataset.
        return find_column(df, ["label", "crop_label", "recommended_crop", "crop_name", "crop"])
    if role == "fertilizer":
        return find_column(df, ["fertilizer_name", "fertilizer_name_recommended", "recommended_fertilizer", "fertilizer_label", "fertilizer", "label"])
    if role == "yield":
        return find_column(df, YIELD_TARGET_ALIASES)
    return None


def make_features(df: pd.DataFrame, target: str, role: str) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    data[target] = data[target].astype(str).str.strip()
    mask = df[target].notna() & (data[target] != "") & (data[target].str.lower() != "nan")
    data = data.loc[mask].copy()

    if role == "yield":
        # Production mathematically determines many recorded yield values. Excluding it
        # reduces target leakage and makes the prediction a more useful estimate.
        leakage_cols = [c for c in data.columns if normalise_name(c) in {"production", "crop_production", "yield_per_hectare"} and c != target]
        data = data.drop(columns=leakage_cols, errors="ignore")
        target_values = pd.to_numeric(data[target].astype(str).str.replace(",", "", regex=False), errors="coerce")
        data = data.loc[target_values.notna()].copy()
        y = target_values.loc[data.index].astype(float)
    else:
        y = data[target].astype(str)

    X = data.drop(columns=[target], errors="ignore")
    # Drop generic row-number columns; they are not helpful as real-world model inputs.
    id_like = [c for c in X.columns if normalise_name(c) in {"unnamed_0", "index", "id", "serial_no", "s_no"}]
    X = X.drop(columns=id_like, errors="ignore")
    X = convert_numeric_strings(X)
    X = X.dropna(axis=1, how="all")
    # Remove constant columns: they cannot help a model and may complicate the UI.
    constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant, errors="ignore")
    return X, y


def build_pipeline(X: pd.DataFrame, task: str) -> Pipeline:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    transformers = []
    if numeric_columns:
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("numeric", numeric_pipe, numeric_columns))
    if categorical_columns:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_columns))

    if not transformers:
        raise ValueError("Dataset mein model train karne ke liye usable input columns nahi mile.")

    preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
    if task == "regression":
        estimator = RandomForestRegressor(n_estimators=160, random_state=42, n_jobs=-1, min_samples_leaf=1)
    else:
        estimator = RandomForestClassifier(n_estimators=160, random_state=42, n_jobs=-1, class_weight=None)
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def try_validation_score(X: pd.DataFrame, y: pd.Series, task: str) -> dict[str, float] | None:
    if len(X) < 10 or y.nunique() < 2:
        return None
    try:
        stratify = None
        if task == "classification":
            counts = y.value_counts()
            # Stratification needs at least two records in every class and enough test slots.
            if counts.min() >= 2 and int(np.ceil(len(y) * 0.2)) >= y.nunique():
                stratify = y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        test_model = build_pipeline(X_train, task)
        test_model.fit(X_train, y_train)
        predictions = test_model.predict(X_test)
        if task == "regression":
            return {
                "r2": float(r2_score(y_test, predictions)),
                "mae": float(mean_absolute_error(y_test, predictions)),
            }
        return {"accuracy": float(accuracy_score(y_test, predictions))}
    except Exception:
        return None


@st.cache_resource(show_spinner="Dataset se Machine Learning model train ho raha hai...")
def train_model_cached(path_text: str, modified_time: float, file_size: int, role: str):
    df = read_csv_safely(path_text)
    target = choose_target(df, role)
    if target is None:
        raise ValueError(f"{role.title()} dataset ka target column automatically identify nahi hua.")
    X, y = make_features(df, target, role)
    if len(X) < 5:
        raise ValueError("Model training ke liye kam se kam 5 valid records chahiye.")
    if X.shape[1] == 0:
        raise ValueError("Target ke alawa usable feature columns nahi mile.")
    if role != "yield" and y.nunique() < 2:
        raise ValueError("Classification dataset mein kam se kam 2 alag target labels hone chahiye.")
    task = "regression" if role == "yield" else "classification"
    score = try_validation_score(X, y, task)
    model = build_pipeline(X, task)
    model.fit(X, y)
    return model, list(X.columns), target, score, task


# -- Reusable UI helpers ----------------------------------------------------------
def role_title(role: str) -> str:
    return {
        "crop": "Crop Recommendation",
        "fertilizer": "Fertilizer Prediction",
        "yield": "Crop Yield Prediction",
        "other": "Unrecognised CSV",
        "error": "CSV read error",
    }.get(role, role.title())


def get_dataset_summary(item: dict[str, Any]) -> dict[str, Any]:
    frame = item["df"]
    return {
        "Dataset": item["name"],
        "Detected role": role_title(item["role"]),
        "Rows": len(frame),
        "Columns": len(frame.columns),
        "Missing cells": int(frame.isna().sum().sum()) if not frame.empty else 0,
    }


def show_missing_dataset(role: str, item: dict[str, Any] | None, all_data: list[dict[str, Any]]) -> None:
    st.warning(f"{role_title(role)} ke liye matching CSV automatically identify nahi hui.")
    st.write("`data` folder mein maujood CSV files:")
    if not all_data:
        st.error("`data` folder mein koi `.csv` file nahi mili. Apni CSV files `Smart Farming Assistant/data/` mein rakho.")
        return
    for ds in all_data:
        st.markdown(f"**{ds['name']}** — {role_title(ds['role'])}")
        if not ds["df"].empty:
            st.caption("Columns: " + ", ".join(map(str, ds["df"].columns)))
        elif ds.get("error"):
            st.caption("Read error: " + ds["error"])
    st.info("CSV ke actual column names dekhkar hi feature/model choose kiya jayega. Agar detection galat ho, column headers share karo; guess karke wrong model use nahi karenge.")


def create_input_form(df: pd.DataFrame, feature_columns: list[str], key_prefix: str) -> tuple[dict[str, Any], bool]:
    """Build controls directly from the model dataset's real feature columns."""
    user_values: dict[str, Any] = {}
    with st.form(f"{key_prefix}_prediction_form"):
        st.write("Input values apne dataset ke observed columns aur values ke according select karo.")
        left, right = st.columns(2)
        for index, column in enumerate(feature_columns):
            series = df[column] if column in df.columns else pd.Series(dtype="object")
            target_col = left if index % 2 == 0 else right
            with target_col:
                clean_series = series.dropna()
                if pd.api.types.is_numeric_dtype(series):
                    values = pd.to_numeric(clean_series, errors="coerce").dropna()
                    if values.empty:
                        user_values[column] = st.number_input(str(column), value=0.0, key=f"{key_prefix}_{index}")
                        continue
                    minimum = float(values.min())
                    maximum = float(values.max())
                    default = float(values.median())
                    if minimum == maximum:
                        user_values[column] = st.number_input(
                            str(column), value=default, key=f"{key_prefix}_{index}"
                        )
                    else:
                        is_integer = bool(np.all(np.isclose(values.to_numpy(dtype=float) % 1, 0)))
                        step = 1.0 if is_integer else max((maximum - minimum) / 100.0, 0.01)
                        fmt = "%.0f" if is_integer else "%.3f"
                        user_values[column] = st.number_input(
                            str(column),
                            min_value=minimum,
                            max_value=maximum,
                            value=min(max(default, minimum), maximum),
                            step=step,
                            format=fmt,
                            help=f"Dataset mein observed range: {minimum:g} se {maximum:g}",
                            key=f"{key_prefix}_{index}",
                        )
                else:
                    unique_values = clean_series.astype(str).unique().tolist()
                    unique_values = [value for value in unique_values if value.strip() and value.lower() != "nan"]
                    unique_values = sorted(unique_values)
                    if 0 < len(unique_values) <= 100:
                        # Default to the most common observed category, when available.
                        counts = clean_series.astype(str).value_counts()
                        default_value = str(counts.index[0]) if not counts.empty else unique_values[0]
                        default_index = unique_values.index(default_value) if default_value in unique_values else 0
                        user_values[column] = st.selectbox(
                            str(column), unique_values, index=default_index,
                            help="Options isi CSV dataset mein se liye gaye hain.", key=f"{key_prefix}_{index}"
                        )
                    else:
                        common = clean_series.astype(str).mode()
                        default_value = str(common.iloc[0]) if not common.empty else ""
                        user_values[column] = st.text_input(str(column), value=default_value, key=f"{key_prefix}_{index}")
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)
    return user_values, submitted


def render_model_page(role: str, item: dict[str, Any] | None, all_data: list[dict[str, Any]], user_id: int) -> None:
    st.header(role_title(role))
    if item is None:
        show_missing_dataset(role, item, all_data)
        return

    df = item["df"]
    target = choose_target(df, role)
    if target is None:
        st.error("Is CSV mein expected target column identify nahi hua. Neeche actual columns diye gaye hain.")
        st.write(list(df.columns))
        st.info("Code dataset ke hisaab se fields banata hai. Target column galat guess nahi kiya jayega.")
        return

    st.caption(f"Source: `data/{item['name']}` · {len(df):,} rows · {len(df.columns)} columns · Target: `{target}`")
    if role == "yield":
        st.info("Yield model `Production` column ko input se exclude karta hai, kyunki isse target leakage ho sakta hai. Prediction historical dataset ke pattern ka estimate hai, guaranteed real-world yield nahi.")
    elif role == "crop":
        st.info("Crop suggestions dataset mein maujood labels aur features se seekhe gaye hain. Result farming guarantee nahi hai; local soil testing/agronomy advice ko bhi consider karein.")
    else:
        st.info("Fertilizer output isi dataset ke labels ko predict karta hai. Real fertilizer dose ke liye soil test aur local agricultural guidance zaroori hai.")

    try:
        model, feature_columns, target, score, task = train_model_cached(
            item["path"], item["mtime"], item["size"], role
        )
        X_preview, _ = make_features(df, target, role)
        inputs, submitted = create_input_form(X_preview, feature_columns, role)
        if submitted:
            input_frame = pd.DataFrame([inputs], columns=feature_columns)
            prediction = model.predict(input_frame)[0]
            try:
                save_prediction(
                    user_id=user_id,
                    module=role,
                    dataset_name=item["name"],
                    target_name=target,
                    predicted_value=prediction,
                    inputs=inputs,
                )
                st.caption("Prediction tumhare account ki history mein save ho gayi hai.")
            except Exception as save_error:
                st.warning(f"Prediction hui, lekin history save nahi ho saki: {save_error}")

            if role == "crop" and hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_frame)[0]
                classes = model.named_steps["model"].classes_
                sorted_indices = np.argsort(probabilities)[::-1][: min(3, len(classes))]
                st.success(f"Recommended crop: **{prediction}**")
                st.subheader("Top crop suggestions")
                cols = st.columns(len(sorted_indices))
                for place, index in enumerate(sorted_indices):
                    cols[place].metric(
                        f"#{place + 1} {classes[index]}",
                        f"{probabilities[index] * 100:.1f}% model score",
                    )
                st.caption("Scores represent the model's class probabilities, not a guarantee that a crop will succeed.")
            elif role == "fertilizer":
                st.success(f"Dataset-based fertilizer prediction: **{prediction}**")
            else:
                st.success(f"Estimated yield: **{float(prediction):,.4f}** (dataset ke target units mein)")

            result = pd.DataFrame([inputs])
            result[f"Predicted {target}"] = [prediction]
            csv_bytes = result.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download this prediction (CSV)",
                data=csv_bytes,
                file_name=f"{role}_prediction.csv",
                mime="text/csv",
            )

        if score:
            st.divider()
            st.subheader("Hold-out validation (20% test split)")
            if "accuracy" in score:
                st.metric("Validation accuracy", f"{score['accuracy'] * 100:.2f}%")
            else:
                metric_cols = st.columns(2)
                metric_cols[0].metric("R²", f"{score['r2']:.3f}")
                metric_cols[1].metric("Mean absolute error", f"{score['mae']:.4f}")
            st.caption("Yeh score isi dataset ke held-out records par hai; real farm performance alag ho sakti hai.")
        else:
            st.caption("Dataset chhota/uneven hai, isliye reliable hold-out metric calculate nahi hua. Model phir bhi available records par train hua.")

    except Exception as exc:
        st.error(f"Model abhi train nahi ho saka: {exc}")
        st.write("Dataset ke actual columns:", list(df.columns))
        st.info("Kisi target ya feature column ka format alag ho sakta hai. Error aur column names dekhkar mapping sahi ki ja sakti hai.")


def render_overview(all_data: list[dict[str, Any]], role_map: dict[str, dict[str, Any] | None]) -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🌱 Smart Farming Assistant</h1>
          <p>Crop recommendation · Fertilizer prediction · Crop yield estimation</p>
          <p class="small-note">Dashboard tumhare <b>data/</b> folder mein maujood CSV files ko read karta hai.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_rows = sum(len(ds["df"]) for ds in all_data if not ds["df"].empty)
    total_files = len(all_data)
    detected = sum(1 for ds in all_data if ds["role"] in {"crop", "fertilizer", "yield"})
    metrics = st.columns(3)
    metrics[0].metric("CSV files found", total_files)
    metrics[1].metric("Rows across CSV files", f"{total_rows:,}")
    metrics[2].metric("Recognised datasets", f"{detected}/3")

    st.subheader("Your dataset status")
    if not all_data:
        st.warning("Abhi `data` folder mein koi CSV file nahi mili.")
        st.code("Smart Farming Assistant/\n├── app.py\n└── data/\n    ├── crop_recommendation.csv\n    ├── fertilizer_prediction.csv\n    └── crop_yield.csv")
        return

    summaries = [get_dataset_summary(item) for item in all_data]
    st.dataframe(pd.DataFrame(summaries), use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Available modules")
        for role, icon, label in [
            ("crop", "🌾", "Crop Recommendation"),
            ("fertilizer", "🧪", "Fertilizer Prediction"),
            ("yield", "📈", "Crop Yield Prediction"),
        ]:
            if role_map[role] is not None:
                st.success(f"{icon} {label} — dataset found")
            else:
                st.warning(f"{icon} {label} — dataset not recognised yet")
    with col_b:
        st.subheader("Data checks")
        for item in all_data:
            if item["df"].empty:
                st.error(f"{item['name']}: empty file or could not be read")
            else:
                missing = int(item["df"].isna().sum().sum())
                if missing:
                    st.warning(f"{item['name']}: {missing:,} missing cells")
                else:
                    st.success(f"{item['name']}: no blank cells found")

    st.info("Live Weather, Personal Reminders, Alarms aur Notifications app mein available hain. Reminder alerts ke liye app/browser open rehna chahiye.")


def render_dataset_explorer(all_data: list[dict[str, Any]]) -> None:
    st.header("Dataset Explorer")
    if not all_data:
        st.warning("`data` folder mein CSV file nahi mili.")
        return
    readable = [ds for ds in all_data if not ds["df"].empty]
    if not readable:
        st.error("CSV files read nahi ho pa rahi hain.")
        return
    chosen_name = st.selectbox("Dataset select karo", [ds["name"] for ds in readable])
    chosen = next(ds for ds in readable if ds["name"] == chosen_name)
    df = chosen["df"]
    st.caption(f"Detected role: {role_title(chosen['role'])} · {len(df):,} rows · {len(df.columns)} columns")
    left, right, third = st.columns(3)
    left.metric("Rows", f"{len(df):,}")
    right.metric("Columns", len(df.columns))
    third.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    tab1, tab2, tab3 = st.tabs(["Data preview", "Column information", "Missing values"])
    with tab1:
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        st.caption("Preview mein first 100 rows dikhayi gayi hain.")
    with tab2:
        info = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Data type": [str(df[c].dtype) for c in df.columns],
            "Non-empty values": [int(df[c].notna().sum()) for c in df.columns],
            "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
        })
        st.dataframe(info, use_container_width=True, hide_index=True)
    with tab3:
        missing_df = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        }).sort_values("Missing values", ascending=False)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download selected dataset as CSV", csv_data, file_name=chosen["name"], mime="text/csv")
    try:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Dataset")
            info.to_excel(writer, index=False, sheet_name="Columns")
        st.download_button(
            "Export dataset + column info to Excel",
            data=excel_buffer.getvalue(),
            file_name=f"{Path(chosen['name']).stem}_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as exc:
        st.caption(f"Excel export available nahi hua: {exc}. `openpyxl` install check karo.")


def render_auth_screen() -> None:
    """Simple account screen. Passwords are hashed before they are stored."""
    st.markdown(
        """
        <div class="hero">
          <h1>🌱 Smart Farming Assistant</h1>
          <p>Create your own account to keep your predictions private.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    register_tab, login_tab = st.tabs(["Create account", "Login"])

    with register_tab:
        st.subheader("New user registration")
        with st.form("register_form", clear_on_submit=False):
            new_username = st.text_input("Username", placeholder="e.g. farmer_jaipur")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password (minimum 8 characters)", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            register_clicked = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if register_clicked:
            if new_password != confirm_password:
                st.error("Password aur confirm password match nahi karte.")
            else:
                ok, message = register_user(new_username, new_email, new_password)
                (st.success if ok else st.error)(message)

    with login_tab:
        st.subheader("Login to your account")
        with st.form("login_form"):
            identifier = st.text_input("Username or email")
            password = st.text_input("Password", type="password")
            login_clicked = st.form_submit_button("Login", type="primary", use_container_width=True)
        if login_clicked:
            account = authenticate_user(identifier, password)
            if account:
                st.session_state["authenticated_user"] = account
                st.rerun()
            else:
                st.error("Login details sahi nahi hain. Username/email aur password check karo.")
    st.caption("Demo note: Account records is project ke local SQLite database mein save hote hain.")


def render_personal_history(user_id: int) -> None:
    st.header("📚 My Prediction History")
    records = get_prediction_history(user_id)
    if not records:
        st.info("Abhi tumhare account mein koi prediction save nahi hai. Pehle Crop, Fertilizer ya Yield page par prediction karo.")
        return

    history = pd.DataFrame(records)
    visible = history.drop(columns=["inputs_json"], errors="ignore")
    st.dataframe(visible, use_container_width=True, hide_index=True)
    st.download_button(
        "Download my history as CSV",
        data=visible.to_csv(index=False).encode("utf-8-sig"),
        file_name="my_prediction_history.csv",
        mime="text/csv",
    )
    selection_labels = {
        str(row["id"]): f"#{row['id']} · {row['module']} · {row['predicted_value']} · {row['created_at']}"
        for _, row in history.iterrows()
    }
    chosen_id = st.selectbox("Prediction details dekho", list(selection_labels), format_func=lambda key: selection_labels[key])
    selected = next(record for record in records if str(record["id"]) == chosen_id)
    st.subheader("Inputs used for this prediction")
    try:
        st.json(json.loads(selected["inputs_json"]))
    except (TypeError, json.JSONDecodeError):
        st.code(selected["inputs_json"])



# -- Personal reminders, alarms and notifications ---------------------------------

IST = ZoneInfo("Asia/Kolkata")


def make_alarm_wav() -> bytes:
    """Build a short alarm tone locally; no sound file download is needed."""
    sample_rate = 22050
    duration = 0.65
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Two-tone, pulsing sine beep with a gentle fade-in/out.
    tone = np.sin(2 * np.pi * 880 * t)
    pulse = (np.sin(2 * np.pi * 3 * t) > -0.15).astype(float)
    fade_len = max(1, int(sample_rate * 0.025))
    envelope = np.ones_like(t)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    samples = (tone * pulse * envelope * 0.45 * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())
    return buffer.getvalue()


def render_reminders_page(user_id: int) -> None:
    st.header("⏰ Personal Reminders, Alarms & Notifications")
    st.write("Yahan banaye gaye reminders sirf tumhare logged-in account se linked rahenge.")
    st.info(
        "Reminder time Asia/Kolkata (IST) ke hisaab se save hota hai. App/browser open rehne par "
        "app har 30 seconds mein due reminders check karta hai. App band ho to sound/push notification "
        "nahi aayegi; dobara open karne par overdue reminder detect hoga."
    )

    if st.button("🔊 Test alarm sound", key="test_alarm_sound"):
        st.audio(make_alarm_wav(), format="audio/wav", autoplay=True)
        st.caption("Agar browser autoplay block kare, audio player ka Play button dabao.")

    now = datetime.now(IST)
    default_due = now + timedelta(minutes=30)
    st.subheader("➕ Create a reminder")
    with st.form("create_personal_reminder", clear_on_submit=True):
        title = st.text_input("Reminder title", placeholder="e.g. Water the wheat field")
        description = st.text_area(
            "Details (optional)",
            placeholder="e.g. Check soil moisture before irrigation",
            height=80,
        )
        date_col, time_col = st.columns(2)
        due_date = date_col.date_input(
            "Reminder date (IST)", value=default_due.date(), min_value=now.date()
        )
        due_time = time_col.time_input(
            "Reminder time (IST)", value=default_due.time().replace(second=0, microsecond=0)
        )
        submitted = st.form_submit_button("Save reminder", type="primary", use_container_width=True)

    if submitted:
        due_datetime = datetime.combine(due_date, due_time).replace(tzinfo=IST)
        if not title.strip():
            st.error("Reminder title likhna zaroori hai.")
        elif due_datetime <= datetime.now(IST):
            st.error("Future date/time select karo. Reminder time abhi se aage hona chahiye.")
        else:
            try:
                reminder_id = create_reminder(
                    user_id,
                    title,
                    description,
                    due_datetime.strftime("%Y-%m-%d %H:%M"),
                )
                st.success(f"Reminder save ho gaya. Reminder ID: {reminder_id}")
                st.rerun()
            except Exception as exc:
                st.error(f"Reminder save nahi ho saka: {exc}")

    st.divider()
    st.subheader("📅 My active reminders")
    reminders = get_user_reminders(user_id)
    if not reminders:
        st.info("Abhi koi active reminder nahi hai. Upar form se pehla reminder banao.")
    else:
        for item in reminders:
            due_text = str(item["due_at"])
            try:
                due_dt = datetime.strptime(due_text, "%Y-%m-%d %H:%M").replace(tzinfo=IST)
            except ValueError:
                due_dt = None
            is_due = item["status"] == "notified" or (due_dt is not None and due_dt <= datetime.now(IST))
            badge = "🔔 DUE / ALERT SENT" if is_due else "🕒 UPCOMING"
            with st.container(border=True):
                details_col, complete_col, delete_col, alarm_col = st.columns([5, 1.4, 1, 1.5])
                details_col.markdown(f"**{item['title']}**  \n{badge} · {due_text} IST")
                if item.get("description"):
                    details_col.caption(item["description"])
                if complete_col.button("✅ Done", key=f"rem_done_{item['id']}", use_container_width=True):
                    complete_reminder(user_id, int(item["id"]))
                    st.success("Reminder completed.")
                    st.rerun()
                if delete_col.button("🗑️", key=f"rem_delete_{item['id']}", help="Delete reminder"):
                    delete_reminder(user_id, int(item["id"]))
                    st.rerun()
                if is_due and alarm_col.button("🔊 Alarm", key=f"rem_alarm_{item['id']}", use_container_width=True):
                    st.audio(make_alarm_wav(), format="audio/wav", autoplay=True)

    st.divider()
    st.subheader("🔔 My notifications")
    unread_count = get_unread_notification_count(user_id)
    notif_col, action_col = st.columns([4, 1.5])
    notif_col.caption(f"Unread notifications: {unread_count}")
    if unread_count and action_col.button("Mark all as read", key="mark_all_notifications_read"):
        mark_all_notifications_read(user_id)
        st.rerun()

    notifications = get_user_notifications(user_id, unread_only=False, limit=50)
    if not notifications:
        st.info("Abhi tumhare account mein koi notification nahi hai.")
    else:
        for item in notifications:
            with st.container(border=True):
                first, second = st.columns([5, 1.4])
                icon = "🔔" if not item["is_read"] else "✓"
                first.markdown(f"**{icon} {item['kind'].title()}** · {item['created_at']}")
                first.write(item["message"])
                if not item["is_read"] and second.button("Mark read", key=f"notif_read_{item['id']}"):
                    mark_notification_read(user_id, int(item["id"]))
                    st.rerun()


# -- Account-scoped Excel reports --------------------------------------------------

def render_excel_reports(user_id: int, username: str) -> None:
    """Export only the signed-in user's saved records to an Excel workbook."""
    st.header("📗 My Reports & Excel Export")
    st.write(
        "Yahan se apni prediction history, reminders aur notifications ko ek Excel workbook mein download kar sakte ho. "
        "Report mein sirf current logged-in account ke records honge."
    )

    prediction_records = get_prediction_history(user_id, limit=2000)
    reminder_records = get_user_reminders(user_id, include_completed=True)
    notification_records = get_user_notifications(user_id, unread_only=False, limit=500)

    predictions_df = pd.DataFrame(prediction_records)
    reminders_df = pd.DataFrame(reminder_records)
    notifications_df = pd.DataFrame(notification_records)

    # Keep this report focused on the current account and omit internal user IDs.
    reminders_df = reminders_df.drop(columns=["user_id"], errors="ignore")
    notifications_df = notifications_df.drop(columns=["user_id"], errors="ignore")

    metrics = st.columns(3)
    metrics[0].metric("Saved predictions", len(predictions_df))
    metrics[1].metric("Reminders", len(reminders_df))
    metrics[2].metric("Notifications", len(notifications_df))

    tab_predictions, tab_reminders, tab_notifications = st.tabs(
        ["Prediction History", "Reminders", "Notifications"]
    )
    with tab_predictions:
        if predictions_df.empty:
            st.info("Abhi koi prediction history save nahi hai.")
        else:
            display_predictions = predictions_df.drop(columns=["inputs_json"], errors="ignore")
            st.dataframe(display_predictions, use_container_width=True, hide_index=True)
            st.caption("Input details workbook ke 'Prediction Inputs' sheet mein bhi include hongi.")
    with tab_reminders:
        if reminders_df.empty:
            st.info("Abhi koi reminder save nahi hai.")
        else:
            st.dataframe(reminders_df, use_container_width=True, hide_index=True)
    with tab_notifications:
        if notifications_df.empty:
            st.info("Abhi koi notification save nahi hai.")
        else:
            st.dataframe(notifications_df, use_container_width=True, hide_index=True)

    summary_df = pd.DataFrame([
        {"Item": "Account", "Value": username},
        {"Item": "Export generated (IST)", "Value": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")},
        {"Item": "Prediction records included", "Value": len(predictions_df)},
        {"Item": "Reminder records included", "Value": len(reminders_df)},
        {"Item": "Notification records included", "Value": len(notifications_df)},
        {"Item": "Privacy scope", "Value": "Only the current logged-in user's saved records"},
    ])

    # Include a readable table and a separate input-details sheet.
    prediction_table = predictions_df.copy()
    if not prediction_table.empty and "inputs_json" in prediction_table.columns:
        input_details_df = prediction_table[["id", "module", "created_at", "inputs_json"]].copy()
        input_details_df = input_details_df.rename(columns={
            "id": "Prediction ID",
            "module": "Module",
            "created_at": "Created At",
            "inputs_json": "Inputs (JSON)",
        })
        prediction_table = prediction_table.drop(columns=["inputs_json"])
    else:
        input_details_df = pd.DataFrame(columns=["Prediction ID", "Module", "Created At", "Inputs (JSON)"])

    try:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Export Summary")
            prediction_table.to_excel(writer, index=False, sheet_name="Prediction History")
            input_details_df.to_excel(writer, index=False, sheet_name="Prediction Inputs")
            reminders_df.to_excel(writer, index=False, sheet_name="Reminders")
            notifications_df.to_excel(writer, index=False, sheet_name="Notifications")
        st.download_button(
            "⬇️ Download my complete Excel report",
            data=excel_buffer.getvalue(),
            file_name="smart_farming_my_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Excel report generate nahi ho saka: {exc}")
        st.info("Check karo ki openpyxl installed ho: python -m pip install openpyxl")

    st.caption(
        "Note: Excel report mein saved records hi aayenge. Agar koi prediction/reminder nahi banaya hai, "
        "toh related sheet blank rahegi. Passwords ya password hashes report mein include nahi hote."
    )


# -- Weather page (Open-Meteo live API) -------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def cached_location_search(city: str) -> list[dict[str, Any]]:
    return search_locations(city)


@st.cache_data(ttl=900, show_spinner=False)
def cached_weather_forecast(latitude: float, longitude: float) -> dict[str, Any]:
    return get_weather_forecast(latitude, longitude)


def make_daily_forecast_table(payload: dict[str, Any]) -> pd.DataFrame:
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    rows = []
    for i, day in enumerate(dates):
        def item(key: str, default: Any = None) -> Any:
            values = daily.get(key, [])
            return values[i] if i < len(values) else default

        code = item("weather_code")
        rows.append({
            "Date": day,
            "Condition": weather_code_label(code),
            "Min temp (°C)": item("temperature_2m_min"),
            "Max temp (°C)": item("temperature_2m_max"),
            "Precipitation (mm)": item("precipitation_sum"),
            "Precipitation chance (%)": item("precipitation_probability_max"),
            "Max wind (km/h)": item("wind_speed_10m_max"),
        })
    return pd.DataFrame(rows)


def make_weather_alerts(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Transparent app-defined thresholds, not official meteorological warnings."""
    alerts: list[dict[str, str]] = []
    for _, row in forecast_df.iterrows():
        date = str(row.get("Date", ""))
        try:
            chance = float(row["Precipitation chance (%)"])
        except (TypeError, ValueError):
            chance = float("nan")
        try:
            rain = float(row["Precipitation (mm)"])
        except (TypeError, ValueError):
            rain = float("nan")
        try:
            high_temp = float(row["Max temp (°C)"])
        except (TypeError, ValueError):
            high_temp = float("nan")
        try:
            wind = float(row["Max wind (km/h)"])
        except (TypeError, ValueError):
            wind = float("nan")

        if pd.notna(chance) and chance >= 70:
            alerts.append({"Date": date, "Type": "High precipitation chance", "Details": f"Forecast precipitation chance is {chance:.0f}% (project threshold: 70%)."})
        if pd.notna(rain) and rain >= 40:
            alerts.append({"Date": date, "Type": "High daily precipitation estimate", "Details": f"Forecast precipitation is {rain:.1f} mm (project threshold: 40 mm/day)."})
        if pd.notna(high_temp) and high_temp >= 40:
            alerts.append({"Date": date, "Type": "High temperature", "Details": f"Forecast maximum is {high_temp:.1f}°C (project threshold: 40°C)."})
        if pd.notna(wind) and wind >= 50:
            alerts.append({"Date": date, "Type": "Strong wind", "Details": f"Forecast maximum wind is {wind:.1f} km/h (project threshold: 50 km/h)."})
    return pd.DataFrame(alerts, columns=["Date", "Type", "Details"])


def render_weather_page() -> None:
    st.header("🌦️ Live Weather & 7-Day Forecast")
    st.write("Apne shehar ka naam enter karo. Weather live Open-Meteo API se aayega; is feature ko kisi local weather CSV ki zaroorat nahi hai.")

    with st.form("weather_location_form"):
        city_name = st.text_input("City / town in India", value=st.session_state.get("weather_city_input", "Jaipur"), placeholder="e.g. Jaipur")
        search_clicked = st.form_submit_button("Search location", type="primary", use_container_width=True)

    if search_clicked:
        st.session_state["weather_city_input"] = city_name
        st.session_state.pop("weather_result", None)
        try:
            with st.spinner("Location search ho rahi hai..."):
                results = cached_location_search(city_name)
            st.session_state["weather_locations"] = results
            if not results:
                st.warning("Is naam ki Indian location nahi mili. City ka spelling check karke dobara try karo.")
        except requests.RequestException as exc:
            st.error(f"Location API se connect nahi ho saka: {exc}")
        except Exception as exc:
            st.error(f"Location search mein problem: {exc}")

    locations = st.session_state.get("weather_locations", [])
    if not locations:
        st.caption("Try: Jaipur, Alwar, Delhi, Kota ya Jodhpur.")
        return

    def location_label(location: dict[str, Any]) -> str:
        parts = [str(location.get("name", "Unknown"))]
        for field in ["admin1", "country"]:
            value = location.get(field)
            if value and value not in parts:
                parts.append(str(value))
        return ", ".join(parts)

    selected_location = st.selectbox(
        "Correct location select karo",
        locations,
        format_func=location_label,
        key="weather_location_choice",
    )
    if st.button("Get current weather + 7-day forecast", type="primary", use_container_width=True):
        try:
            with st.spinner("Weather forecast load ho raha hai..."):
                payload = cached_weather_forecast(float(selected_location["latitude"]), float(selected_location["longitude"]))
            st.session_state["weather_result"] = {"location": selected_location, "payload": payload}
        except requests.RequestException as exc:
            st.error(f"Weather API se connect nahi ho saka: {exc}")
        except Exception as exc:
            st.error(f"Weather data process nahi ho saka: {exc}")

    result = st.session_state.get("weather_result")
    if not result:
        return
    saved_location = result["location"]
    # Avoid accidentally showing old weather after changing the selected location.
    if (float(saved_location["latitude"]) != float(selected_location["latitude"]) or
            float(saved_location["longitude"]) != float(selected_location["longitude"])):
        st.info("Selected location ke liye forecast dekhne ko upar wala button dobara dabao.")
        return

    payload = result["payload"]
    current = payload.get("current", {})
    current_units = payload.get("current_units", {})
    st.subheader(f"Current conditions — {location_label(saved_location)}")
    st.caption(f"Updated at {current.get('time', 'time unavailable')} · Timezone: {payload.get('timezone', 'local time')}")
    current_cols = st.columns(5)
    current_cols[0].metric("Temperature", f"{current.get('temperature_2m', '—')} {current_units.get('temperature_2m', '°C')}")
    current_cols[1].metric("Feels like", f"{current.get('apparent_temperature', '—')} {current_units.get('apparent_temperature', '°C')}")
    current_cols[2].metric("Humidity", f"{current.get('relative_humidity_2m', '—')}%")
    current_cols[3].metric("Wind", f"{current.get('wind_speed_10m', '—')} {current_units.get('wind_speed_10m', 'km/h')}")
    current_cols[4].metric("Condition", weather_code_label(current.get("weather_code")))
    st.caption(f"Current precipitation: {current.get('precipitation', '—')} {current_units.get('precipitation', 'mm')} · Wind gust: {current.get('wind_gusts_10m', '—')} {current_units.get('wind_gusts_10m', 'km/h')}")

    forecast_df = make_daily_forecast_table(payload)
    if forecast_df.empty:
        st.warning("API ne daily forecast return nahi kiya.")
        return

    st.divider()
    st.subheader("7-day forecast")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    if {"Date", "Min temp (°C)", "Max temp (°C)"}.issubset(forecast_df.columns):
        chart_df = forecast_df.set_index("Date")[["Min temp (°C)", "Max temp (°C)"]].apply(pd.to_numeric, errors="coerce")
        st.subheader("Temperature trend")
        st.line_chart(chart_df)

    st.subheader("⚠️ Weather risk flags")
    st.caption("Yeh simple project thresholds hain, official government weather alerts nahi. Kheti ke decisions se pehle local official forecast/advisory check karo.")
    alerts_df = make_weather_alerts(forecast_df)
    if alerts_df.empty:
        st.success("Is forecast mein app ke configured warning thresholds trigger nahi hue. Iska matlab zero weather risk ki guarantee nahi hai.")
    else:
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Download weather report")
    csv_data = forecast_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download forecast CSV", csv_data, file_name="weather_forecast_7_day.csv", mime="text/csv")
    try:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            forecast_df.to_excel(writer, index=False, sheet_name="7-day forecast")
            alerts_df.to_excel(writer, index=False, sheet_name="Project risk flags")
        st.download_button(
            "Download weather report (Excel)",
            data=excel_buffer.getvalue(),
            file_name="smart_farming_weather_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as exc:
        st.caption(f"Excel report generate nahi hua: {exc}")

    st.caption("Weather data: Open-Meteo. Location lookup data is based on GeoNames. API usage terms attribution ke saath follow karein.")


# -- App entry point --------------------------------------------------------------
initialize_database()
current_user = st.session_state.get("authenticated_user")
if not current_user:
    render_auth_screen()
    st.stop()

# Refresh the app periodically while the browser tab is open so reminders become due.
if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=30_000, key="smart_farming_reminder_refresh")
else:
    st.warning("Automatic reminder checking ke liye dependency install karo: python -m pip install streamlit-autorefresh")

newly_due_titles = process_due_reminders(
    int(current_user["id"]), datetime.now(IST).strftime("%Y-%m-%d %H:%M")
)
if newly_due_titles:
    for due_title in newly_due_titles:
        st.toast(f"Reminder due: {due_title}", icon="⏰")
    st.warning("⏰ Ek ya zyada reminders due ho gaye hain. Reminders & Alarms page par details dekho.")
    st.audio(make_alarm_wav(), format="audio/wav", autoplay=True)

all_datasets = scan_datasets()
role_map = datasets_by_role(all_datasets)

with st.sidebar:
    st.title("🌱 Smart Farming")
    st.caption(f"Logged in as **{current_user['username']}**")
    if st.button("Logout", use_container_width=True):
        st.session_state.pop("authenticated_user", None)
        st.rerun()
    st.divider()
    selected_page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Crop Recommendation",
            "Fertilizer Prediction",
            "Crop Yield Prediction",
            "Live Weather & Warnings",
            "Reminders & Alarms",
            "Dataset Explorer",
            "My Prediction History",
            "My Reports & Excel",
        ],
    )
    st.divider()
    st.caption("Data folder")
    st.code(str(DATA_DIR), language="text")
    if st.button("Refresh dataset list", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

if selected_page == "Dashboard":
    render_overview(all_datasets, role_map)
elif selected_page == "Crop Recommendation":
    render_model_page("crop", role_map["crop"], all_datasets, int(current_user["id"]))
elif selected_page == "Fertilizer Prediction":
    render_model_page("fertilizer", role_map["fertilizer"], all_datasets, int(current_user["id"]))
elif selected_page == "Crop Yield Prediction":
    render_model_page("yield", role_map["yield"], all_datasets, int(current_user["id"]))
elif selected_page == "Live Weather & Warnings":
    render_weather_page()
elif selected_page == "Reminders & Alarms":
    render_reminders_page(int(current_user["id"]))
elif selected_page == "Dataset Explorer":
    render_dataset_explorer(all_datasets)
elif selected_page == "My Prediction History":
    render_personal_history(int(current_user["id"]))
elif selected_page == "My Reports & Excel":
    render_excel_reports(int(current_user["id"]), str(current_user["username"]))
