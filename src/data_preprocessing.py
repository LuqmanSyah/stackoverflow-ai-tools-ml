"""Utilities for loading and preprocessing Stack Overflow survey data.

The functions in this module are intentionally small and explicit so the
workflow is easy to explain in a course report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


RANDOM_STATE = 42

TARGET_COLUMN = "AI_Usage"
TARGET_SOURCE_COLUMN = "AISelect"

CANDIDATE_FEATURES = [
    "Age",
    "Country",
    "EdLevel",
    "DevType",
    "Employment",
    "RemoteWork",
    "YearsCode",
    "YearsCodePro",
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith",
    "ToolsTechHaveWorkedWith",
]

NUMERIC_FEATURES = ["YearsCode", "YearsCodePro"]

MULTI_SELECT_FEATURES = [
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith",
    "ToolsTechHaveWorkedWith",
]

AI_LEAKAGE_PREFIXES = (
    "AI",
    "AITool",
    "AIModels",
    "AIAgent",
)

AI_LEAKAGE_COLUMNS = {
    "AISelect",
    "AISent",
    "AIAcc",
    "AIBen",
    "AIComplex",
    "AIThreat",
    "AIFrustration",
    "AIExplain",
    "AIAgents",
    "AIAgentChange",
    "AIAgent_Uses",
    "AIHuman",
    "AIOpen",
    "LearnCodeAI",
    "AILearnHow",
}


def find_dataset_path(project_root: str | Path, year: int = 2025) -> Path:
    """Find the survey CSV for a given year (supports survey_{year}.csv and survey_results_public_{year}.csv)."""
    root = Path(project_root)
    candidates = [
        root / "Survey" / "packages" / "archive" / str(year) / "results.csv",
        root / "data" / f"survey_results_public_{year}.csv",
        root / f"survey_results_public_{year}.csv",
        root / "data" / f"survey_{year}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = sorted(root.rglob(f"*{year}*.csv"))
    if matches:
        return matches[0]

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Dataset tahun {year} tidak ditemukan. Lokasi yang dicek:\n{searched}"
    )


def load_survey_data(data_path: str | Path) -> pd.DataFrame:
    """Load survey data and treat Stack Overflow's literal 'NA' as missing."""
    return pd.read_csv(data_path, low_memory=False, na_values=["NA", ""])


def inspect_candidate_columns(
    df: pd.DataFrame, candidate_features: Iterable[str] = CANDIDATE_FEATURES
) -> pd.DataFrame:
    """Return a simple availability table for the requested feature columns."""
    columns = set(df.columns)
    return pd.DataFrame(
        [{"column": column, "available": column in columns} for column in candidate_features]
    )


def create_ai_usage_target(
    df: pd.DataFrame, source_col: str = TARGET_SOURCE_COLUMN
) -> pd.Series:
    """Create binary AI usage target from AISelect.

    Mapping:
    - 1: answer starts with "Yes"
    - 0: answer starts with "No"
    - missing/other: NaN and removed before modeling
    """
    if source_col not in df.columns:
        similar = [col for col in df.columns if "ai" in col.lower()]
        raise KeyError(
            f"Kolom target '{source_col}' tidak ditemukan. "
            f"Kolom yang mengandung 'AI': {similar}"
        )

    values = df[source_col].astype("string").str.strip().str.lower()
    target = pd.Series(np.nan, index=df.index, dtype="float")
    target.loc[values.str.startswith("yes", na=False)] = 1
    target.loc[values.str.startswith("no", na=False)] = 0
    return target


def clean_years_code_value(value) -> float:
    """Convert Stack Overflow coding-experience text into numeric years."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text == "Less than 1 year":
        return 0.0
    if text == "More than 50 years":
        return 51.0

    try:
        return float(text)
    except ValueError:
        return np.nan


def is_ai_leakage_column(column: str) -> bool:
    """Detect AI-related columns that should not be used as input features."""
    return column in AI_LEAKAGE_COLUMNS or any(
        column.startswith(prefix) for prefix in AI_LEAKAGE_PREFIXES
    )


def select_available_features(
    df: pd.DataFrame, candidate_features: Iterable[str] = CANDIDATE_FEATURES
) -> tuple[list[str], list[str]]:
    """Select only available non-leakage features and report missing ones."""
    available = []
    missing = []

    for column in candidate_features:
        if column not in df.columns:
            missing.append(column)
            continue
        if is_ai_leakage_column(column):
            continue
        available.append(column)

    return available, missing


def prepare_model_data(
    df: pd.DataFrame,
    candidate_features: Iterable[str] = CANDIDATE_FEATURES,
    target_source_col: str = TARGET_SOURCE_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Create target, clean selected features, and drop rows with empty target."""
    data = df.copy()
    data[TARGET_COLUMN] = create_ai_usage_target(data, target_source_col)
    data = data.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    available_features, missing_features = select_available_features(data, candidate_features)

    for column in NUMERIC_FEATURES:
        if column in data.columns:
            data[column] = data[column].apply(clean_years_code_value)

    X = data[available_features].copy()
    y = data[TARGET_COLUMN].copy()

    metadata = {
        "available_features": available_features,
        "missing_features": missing_features,
        "target_source_col": target_source_col,
        "target_distribution": y.value_counts().sort_index().to_dict(),
        "n_rows_after_target_drop": len(data),
    }
    return X, y, metadata


def flatten_to_strings(values) -> pd.Series:
    """Convert a one-column array from ColumnTransformer into text values."""
    return pd.Series(np.asarray(values).ravel()).fillna("Unknown").astype(str)


def split_semicolon_tokens(text: str) -> list[str]:
    """Split Stack Overflow multi-select answers into clean tokens."""
    if text is None or pd.isna(text):
        return ["Unknown"]

    tokens = [token.strip() for token in str(text).split(";") if token.strip()]
    return tokens if tokens else ["Unknown"]


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible with older and newer scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(
    X: pd.DataFrame,
    min_token_frequency: int = 20,
) -> tuple[ColumnTransformer, dict]:
    """Build preprocessing pipeline for numeric, categorical, and multi-select data."""
    numeric_features = [column for column in NUMERIC_FEATURES if column in X.columns]
    multi_select_features = [column for column in MULTI_SELECT_FEATURES if column in X.columns]
    categorical_features = [
        column
        for column in X.columns
        if column not in numeric_features and column not in multi_select_features
    ]

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    for column in multi_select_features:
        transformers.append(
            (
                f"{column}_tokens",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        (
                            "flatten",
                            FunctionTransformer(flatten_to_strings, validate=False),
                        ),
                        (
                            "vectorizer",
                            CountVectorizer(
                                tokenizer=split_semicolon_tokens,
                                token_pattern=None,
                                binary=True,
                                min_df=min_token_frequency,
                            ),
                        ),
                    ]
                ),
                [column],
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    feature_groups = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "multi_select_features": multi_select_features,
    }
    return preprocessor, feature_groups
