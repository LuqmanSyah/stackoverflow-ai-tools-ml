"""Model training utilities for the Stack Overflow AI tools project."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data_preprocessing import RANDOM_STATE


def build_classification_models(preprocessor) -> dict[str, Pipeline]:
    """Create baseline classification models with the same preprocessing."""
    return {
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Linear SVM": Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        max_iter=5000,
                    ),
                ),
            ]
        ),
    }


def train_classification_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    test_size: float = 0.2,
) -> tuple[dict[str, Pipeline], dict]:
    """Split data, train Random Forest and Linear SVM, then return fitted models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_classification_models(preprocessor)
    for model in models.values():
        model.fit(X_train, y_train)

    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    return models, split_data


def run_kmeans_analysis(
    X: pd.DataFrame,
    preprocessor,
    k_values: range = range(2, 11),
    n_svd_components: int = 50,
    silhouette_sample_size: int = 5000,
) -> dict:
    """Run elbow and silhouette analysis, then fit final K-Means.

    TruncatedSVD keeps clustering practical on one-hot and multi-label features.
    """
    X_processed = clone(preprocessor).fit_transform(X)

    max_components = min(n_svd_components, X_processed.shape[0] - 1, X_processed.shape[1] - 1)
    if max_components < 2:
        X_cluster = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
        svd = None
    else:
        svd = TruncatedSVD(n_components=max_components, random_state=RANDOM_STATE)
        X_cluster = svd.fit_transform(X_processed)

    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(X_cluster)

    metrics = []
    labels_by_k = {}

    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_cluster)
        labels_by_k[k] = labels

        sample_size = min(silhouette_sample_size, X_cluster.shape[0])
        silhouette = silhouette_score(
            X_cluster,
            labels,
            sample_size=sample_size,
            random_state=RANDOM_STATE,
        )
        metrics.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette,
            }
        )

    metrics_df = pd.DataFrame(metrics)
    best_k = int(metrics_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])

    final_model = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    final_labels = final_model.fit_predict(X_cluster)

    return {
        "X_cluster": X_cluster,
        "svd": svd,
        "scaler": scaler,
        "metrics": metrics_df,
        "best_k": best_k,
        "labels": final_labels,
        "model": final_model,
    }


def _top_values(series: pd.Series, top_n: int = 3) -> str:
    counts = series.dropna().astype(str).value_counts().head(top_n)
    return "; ".join(f"{idx} ({count})" for idx, count in counts.items())


def _top_multiselect_values(series: pd.Series, top_n: int = 5) -> str:
    exploded = (
        series.dropna()
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
    )
    counts = exploded[exploded != ""].value_counts().head(top_n)
    return "; ".join(f"{idx} ({count})" for idx, count in counts.items())


def summarize_clusters(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Create an interpretable cluster summary table from original features."""
    data = X.copy()
    data["Cluster"] = labels

    rows = []
    for cluster_id, group in data.groupby("Cluster"):
        row = {
            "Cluster": cluster_id,
            "Jumlah_Data": len(group),
            "Persentase": len(group) / len(data) * 100,
        }

        for column in ["Age", "Country", "EdLevel", "DevType", "Employment", "RemoteWork"]:
            if column in group.columns:
                row[f"{column}_Dominan"] = _top_values(group[column])

        for column in ["YearsCode", "YearsCodePro"]:
            if column in group.columns:
                row[f"{column}_Rata_Rata"] = pd.to_numeric(group[column], errors="coerce").mean()

        for column in [
            "LanguageHaveWorkedWith",
            "DatabaseHaveWorkedWith",
            "PlatformHaveWorkedWith",
            "WebframeHaveWorkedWith",
            "ToolsTechHaveWorkedWith",
        ]:
            if column in group.columns:
                row[f"{column}_Dominan"] = _top_multiselect_values(group[column])

        rows.append(row)

    return pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)
