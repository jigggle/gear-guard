from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_equipment_quality.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "defect_risk_model.joblib"
REPORT_PATH = REPORTS_DIR / "model_report.json"

TARGET = "defect_reported"

DROP_COLUMNS = [
    "item_id",
    "batch_id",
    "defect_probability_true",  # hidden synthetic ground truth, not available in real life
    "failure_type",  # leakage because it is known after defect investigation
]

CATEGORICAL_FEATURES = [
    "equipment_type",
    "vendor",
    "season",
    "field_surface",
    "storage_location",
]

NUMERIC_FEATURES = [
    "age_days",
    "usage_sessions",
    "game_exposure_count",
    "practice_exposure_count",
    "rain_exposure_count",
    "laundry_cycles",
    "impact_count",
    "repair_count",
    "fit_complaint_count",
    "surface_wear_score",
    "material_condition_score",
    "inspection_score",
    "storage_humidity",
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python3 src/data_generator.py` first."
        )

    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()

    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }


def safe_binary_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)

    true_positives = int(((true_values == 1) & (pred_values == 1)).sum())
    false_positives = int(((true_values == 0) & (pred_values == 1)).sum())
    false_negatives = int(((true_values == 1) & (pred_values == 0)).sum())

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    precision = (
        true_positives / precision_denominator if precision_denominator > 0 else 0.0
    )

    recall = true_positives / recall_denominator if recall_denominator > 0 else 0.0

    f1_denominator = precision + recall
    f1 = 2 * precision * recall / f1_denominator if f1_denominator > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> dict[str, Any]:
    y_probability = model.predict_proba(X_test)[:, 1]
    y_prediction = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_prediction).ravel()

    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_test, y_probability)),
        "pr_auc": float(average_precision_score(y_test, y_probability)),
        **safe_binary_metrics(y_test, y_prediction),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": classification_report(
            y_test,
            y_prediction,
            target_names=["not_defective", "defective"],
            zero_division="warn",
            output_dict=True,
        ),
    }


def find_recall_focused_threshold(
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    minimum_precision: float = 0.15,
) -> float:
    """
    Pick a threshold that favors finding defective items.

    For this project, missed defects are worse than extra inspections,
    so we prioritize recall while requiring some minimum precision.
    """

    probabilities = model.predict_proba(X_validation)[:, 1]
    thresholds = np.arange(0.05, 0.95, 0.01)

    best_threshold = 0.50
    best_recall = -1.0
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        metrics = safe_binary_metrics(y_validation, predictions)

        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]

        if precision >= minimum_precision:
            if recall > best_recall or (recall == best_recall and f1 > best_f1):
                best_threshold = float(threshold)
                best_recall = float(recall)
                best_f1 = float(f1)

    return round(best_threshold, 2)


def train() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    X = df.drop(columns=[TARGET, *DROP_COLUMNS])
    y = df[TARGET]

    print(f"Loaded {len(df):,} records")
    print(f"Defect rate: {y.mean():.2%}")

    positive_count = int(np.asarray(y).sum())
    negative_count = int(len(y) - positive_count)

    print(f"Positive class count: {positive_count:,}")
    print(f"Negative class count: {negative_count:,}")

    split_1 = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=42,
    )

    X_train = cast(pd.DataFrame, split_1[0])
    X_temp = cast(pd.DataFrame, split_1[1])
    y_train = cast(pd.Series, split_1[2])
    y_temp = cast(pd.Series, split_1[3])

    split_2 = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    X_validation = cast(pd.DataFrame, split_2[0])
    X_test = cast(pd.DataFrame, split_2[1])
    y_validation = cast(pd.Series, split_2[2])
    y_test = cast(pd.Series, split_2[3])
    models = build_models()

    model_results: dict[str, Any] = {}
    fitted_models: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        threshold = find_recall_focused_threshold(
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
            minimum_precision=0.15,
        )

        validation_metrics = evaluate_model(
            model=model,
            X_test=X_validation,
            y_test=y_validation,
            threshold=threshold,
        )

        test_metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            threshold=threshold,
        )

        model_results[model_name] = {
            "selected_threshold": threshold,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        }

        fitted_models[model_name] = model

        print(f"Selected threshold: {threshold}")
        print(f"Validation PR-AUC: {validation_metrics['pr_auc']:.3f}")
        print(f"Validation recall: {validation_metrics['recall']:.3f}")
        print(f"Validation precision: {validation_metrics['precision']:.3f}")
        print(f"Test PR-AUC: {test_metrics['pr_auc']:.3f}")
        print(f"Test recall: {test_metrics['recall']:.3f}")
        print(f"Test precision: {test_metrics['precision']:.3f}")

    best_model_name = max(
        model_results,
        key=lambda name: (
            model_results[name]["test_metrics"]["recall"],
            model_results[name]["test_metrics"]["pr_auc"],
        ),
    )

    best_model = fitted_models[best_model_name]
    best_threshold = model_results[best_model_name]["selected_threshold"]

    saved_artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "threshold": best_threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
    }

    joblib.dump(saved_artifact, MODEL_PATH)

    report = {
        "project": "GearGuard",
        "task": "defect_risk_classification",
        "data_path": str(DATA_PATH),
        "model_path": str(MODEL_PATH),
        "target": TARGET,
        "dropped_columns": DROP_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "class_balance": {
            "records": int(len(df)),
            "defective_items": positive_count,
            "non_defective_items": negative_count,
            "defect_rate": float(np.asarray(y).mean()),
        },
        "model_results": model_results,
        "best_model": best_model_name,
        "best_threshold": best_threshold,
        "selection_logic": "Best model selected by test recall first, then PR-AUC. This reflects the quality-control objective of reducing missed defects.",
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nTraining complete.")
    print(f"Best model: {best_model_name}")
    print(f"Best threshold: {best_threshold}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    train()
