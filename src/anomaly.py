from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_equipment_quality.csv"
SCORED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "equipment_quality_scored.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFECT_MODEL_PATH = MODELS_DIR / "defect_risk_model.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.joblib"
ANOMALY_REPORT_PATH = REPORTS_DIR / "anomaly_report.json"

TARGET = "defect_reported"

DROP_COLUMNS = [
    "item_id",
    "batch_id",
    "defect_probability_true",
    "failure_type",
    TARGET,
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


def load_defect_model(path: Path = DEFECT_MODEL_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Defect risk model not found at {path}. Run `python3 src/train.py` first."
        )

    artifact = joblib.load(path)
    return cast(dict[str, Any], artifact)


def build_anomaly_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    anomaly_model = IsolationForest(
        n_estimators=300,
        contamination=cast(Any, 0.05),
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", anomaly_model),
        ]
    )


def assign_risk_tier(probability: float) -> str:
    if probability >= 0.75:
        return "critical"
    if probability >= 0.50:
        return "high"
    if probability >= 0.25:
        return "medium"
    return "low"


def generate_recommended_action(row: pd.Series) -> str:
    if row["risk_tier"] == "critical":
        return "Remove from active rotation and inspect immediately."
    if row["risk_tier"] == "high":
        return "Prioritize for inspection before next usage cycle."
    if row["anomaly_flag"] == 1:
        return "Review item history because usage or inspection pattern is unusual."
    if row["risk_tier"] == "medium":
        return "Monitor during routine inspection."
    return "No immediate action required."


def score_defect_risk(df: pd.DataFrame) -> pd.DataFrame:
    artifact = load_defect_model()

    model = artifact["model"]
    threshold = float(artifact["threshold"])

    X = df.drop(columns=DROP_COLUMNS)
    probabilities = model.predict_proba(X)[:, 1]

    scored_df = df.copy()
    scored_df["predicted_defect_risk"] = probabilities.round(4)
    scored_df["defect_risk_flag"] = (probabilities >= threshold).astype(int)
    scored_df["risk_tier"] = [
        assign_risk_tier(float(probability)) for probability in probabilities
    ]

    return scored_df


def add_anomaly_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline]:
    X = df.drop(
        columns=[
            "item_id",
            "batch_id",
            "defect_probability_true",
            "failure_type",
            TARGET,
            "predicted_defect_risk",
            "defect_risk_flag",
            "risk_tier",
        ]
    )

    anomaly_pipeline = build_anomaly_pipeline()
    anomaly_pipeline.fit(X)

    anomaly_predictions = anomaly_pipeline.predict(X)
    anomaly_raw_scores = anomaly_pipeline.decision_function(X)

    # IsolationForest decision_function: lower means more anomalous.
    # Convert it so higher anomaly_score means more unusual.
    anomaly_scores = -anomaly_raw_scores

    scored_df = df.copy()
    scored_df["anomaly_score"] = anomaly_scores.round(4)
    scored_df["anomaly_flag"] = np.where(anomaly_predictions == -1, 1, 0)

    scored_df["recommended_action"] = scored_df.apply(
        generate_recommended_action,
        axis=1,
    )

    return scored_df, anomaly_pipeline


def get_scalar(row: pd.Series, column: str) -> Any:
    value = row.at[column]
    if hasattr(value, "item"):
        return value.item()
    return value


def build_anomaly_report(scored_df: pd.DataFrame) -> dict[str, Any]:
    sorted_anomalies = scored_df.sort_values(
        ["anomaly_score", "predicted_defect_risk"],
        ascending=[False, False],
    )

    top_anomaly_columns = [
        "item_id",
        "equipment_type",
        "vendor",
        "batch_id",
        "season",
        "predicted_defect_risk",
        "risk_tier",
        "anomaly_score",
        "anomaly_flag",
        "inspection_score",
        "surface_wear_score",
        "repair_count",
        "impact_count",
        "recommended_action",
    ]

    top_anomalies: list[dict[str, Any]] = []

    for _, row in sorted_anomalies.head(10).iterrows():
        anomaly_record = {
            column: get_scalar(row, column) for column in top_anomaly_columns
        }

        top_anomalies.append(anomaly_record)

    grouped_summary = scored_df.groupby("equipment_type", as_index=False).agg(
        total_items=("item_id", "count"),
        anomaly_count=("anomaly_flag", "sum"),
        avg_anomaly_score=("anomaly_score", "mean"),
        avg_defect_risk=("predicted_defect_risk", "mean"),
    )

    grouped_records: list[dict[str, Any]] = []

    for _, row in grouped_summary.iterrows():
        grouped_records.append(
            {
                "equipment_type": str(get_scalar(row, "equipment_type")),
                "total_items": int(get_scalar(row, "total_items")),
                "anomaly_count": int(get_scalar(row, "anomaly_count")),
                "avg_anomaly_score": float(get_scalar(row, "avg_anomaly_score")),
                "avg_defect_risk": float(get_scalar(row, "avg_defect_risk")),
            }
        )

    anomaly_summary_by_equipment_type = sorted(
        grouped_records,
        key=lambda record: record["anomaly_count"],
        reverse=True,
    )

    anomaly_flags = np.asarray(scored_df["anomaly_flag"])
    risk_tiers = np.asarray(scored_df["risk_tier"])

    anomaly_count = int(anomaly_flags.sum())
    anomaly_rate = float(anomaly_flags.mean())
    high_or_critical_risk_count = int(np.isin(risk_tiers, ["high", "critical"]).sum())

    return {
        "records_scored": int(len(scored_df)),
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "high_or_critical_risk_count": high_or_critical_risk_count,
        "top_anomalies": top_anomalies,
        "anomaly_summary_by_equipment_type": anomaly_summary_by_equipment_type,
        "method": "Isolation Forest with 5% contamination rate. Higher anomaly_score indicates more unusual equipment quality signature.",
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()

    scored_df = score_defect_risk(df)
    scored_df, anomaly_pipeline = add_anomaly_scores(scored_df)

    scored_df.to_csv(SCORED_DATA_PATH, index=False)
    joblib.dump(anomaly_pipeline, ANOMALY_MODEL_PATH)

    report = build_anomaly_report(scored_df)

    with ANOMALY_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Scored {len(scored_df):,} records")
    print(f"Saved scored dataset to: {SCORED_DATA_PATH}")
    print(f"Saved anomaly model to: {ANOMALY_MODEL_PATH}")
    print(f"Saved anomaly report to: {ANOMALY_REPORT_PATH}")
    print(f"Anomaly rate: {report['anomaly_rate']:.2%}")
    print(f"High or critical risk count: {report['high_or_critical_risk_count']}")

    print("\nTop anomalous items:")
    display_columns = [
        "item_id",
        "equipment_type",
        "vendor",
        "predicted_defect_risk",
        "risk_tier",
        "anomaly_score",
        "anomaly_flag",
        "recommended_action",
    ]

    print(
        scored_df.sort_values(
            by=["anomaly_score", "predicted_defect_risk"],
            ascending=[False, False],
        )
        .head(10)[display_columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
