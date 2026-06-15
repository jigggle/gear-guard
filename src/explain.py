from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "equipment_quality_scored.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "defect_risk_model.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
ITEM_EXPLANATIONS_PATH = REPORTS_DIR / "item_explanations.json"

TARGET = "defect_reported"

DROP_COLUMNS = [
    "item_id",
    "batch_id",
    "defect_probability_true",
    "failure_type",
    TARGET,
    "predicted_defect_risk",
    "defect_risk_flag",
    "risk_tier",
    "anomaly_score",
    "anomaly_flag",
    "recommended_action",
]


def load_scored_data(path: Path = SCORED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Scored dataset not found at {path}. Run `python3 src/anomaly.py` first."
        )

    return pd.read_csv(path)


def load_model_artifact(path: Path = MODEL_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python3 src/train.py` first."
        )

    artifact = joblib.load(path)
    return cast(dict[str, Any], artifact)


def get_scalar(row: pd.Series, column: str) -> Any:
    value = row.at[column]
    if hasattr(value, "item"):
        return value.item()
    return value


def get_feature_names(model: Pipeline) -> list[str]:
    preprocessor = cast(Any, model.named_steps["preprocessor"])
    feature_names = preprocessor.get_feature_names_out()
    return [str(feature_name) for feature_name in feature_names]


def get_model_coefficients(model: Pipeline) -> np.ndarray:
    classifier = cast(Any, model.named_steps["model"])

    if hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_[0], dtype=float)
        return coefficients

    if hasattr(classifier, "feature_importances_"):
        coefficients = np.asarray(classifier.feature_importances_, dtype=float)
        return coefficients

    raise TypeError("Model does not expose coefficients or feature importances.")


def clean_feature_name(feature_name: str) -> str:
    cleaned = feature_name

    if cleaned.startswith("numeric__"):
        cleaned = cleaned.replace("numeric__", "")

    if cleaned.startswith("categorical__"):
        cleaned = cleaned.replace("categorical__", "")

    return cleaned


def build_global_feature_importance(
    feature_names: list[str],
    coefficients: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for feature_name, coefficient in zip(feature_names, coefficients):
        records.append(
            {
                "feature": clean_feature_name(feature_name),
                "raw_feature": feature_name,
                "importance": round(float(abs(coefficient)), 6),
                "direction": "increases_risk" if coefficient > 0 else "decreases_risk",
                "coefficient": round(float(coefficient), 6),
            }
        )

    sorted_records = sorted(
        records,
        key=lambda record: float(record["importance"]),
        reverse=True,
    )

    return pd.DataFrame(sorted_records)


def get_item_contributions(
    model: Pipeline,
    row_df: pd.DataFrame,
    feature_names: list[str],
    coefficients: np.ndarray,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    preprocessor = cast(Any, model.named_steps["preprocessor"])
    transformed = preprocessor.transform(row_df)

    if hasattr(transformed, "toarray"):
        transformed_array = np.asarray(transformed.toarray(), dtype=float)
    else:
        transformed_array = np.asarray(transformed, dtype=float)

    row_values = transformed_array[0]
    contributions = row_values * coefficients

    records: list[dict[str, Any]] = []

    for feature_name, contribution, transformed_value, coefficient in zip(
        feature_names,
        contributions,
        row_values,
        coefficients,
    ):
        records.append(
            {
                "feature": clean_feature_name(feature_name),
                "raw_feature": feature_name,
                "contribution": round(float(contribution), 6),
                "absolute_contribution": round(float(abs(contribution)), 6),
                "transformed_value": round(float(transformed_value), 6),
                "coefficient": round(float(coefficient), 6),
                "effect": "pushes_risk_up" if contribution > 0 else "pushes_risk_down",
            }
        )

    sorted_records = sorted(
        records,
        key=lambda record: float(record["absolute_contribution"]),
        reverse=True,
    )

    return sorted_records[:top_n]


def generate_explanation_summary(
    item_id: str,
    equipment_type: str,
    risk: float,
    risk_tier: str,
    top_contributions: list[dict[str, Any]],
) -> str:
    risk_drivers = [
        str(record["feature"])
        for record in top_contributions
        if str(record["effect"]) == "pushes_risk_up"
    ]

    if len(risk_drivers) == 0:
        driver_text = "no strong positive risk drivers"
    else:
        driver_text = ", ".join(risk_drivers[:4])

    return (
        f"{item_id} ({equipment_type}) was flagged as {risk_tier} risk "
        f"with predicted defect probability of {risk:.1%}. "
        f"The strongest risk drivers were: {driver_text}. "
        f"Recommended action: review inspection history, verify physical condition, "
        f"and compare against similar items from the same equipment type or vendor."
    )


def build_item_explanations(
    df: pd.DataFrame,
    model: Pipeline,
    feature_names: list[str],
    coefficients: np.ndarray,
    max_items: int = 15,
) -> list[dict[str, Any]]:
    candidate_rows: list[dict[str, Any]] = []

    for row_position, (_, row) in enumerate(df.iterrows()):
        candidate_rows.append(
            {
                "index": row_position,
                "item_id": str(get_scalar(row, "item_id")),
                "equipment_type": str(get_scalar(row, "equipment_type")),
                "vendor": str(get_scalar(row, "vendor")),
                "batch_id": str(get_scalar(row, "batch_id")),
                "season": str(get_scalar(row, "season")),
                "predicted_defect_risk": float(
                    get_scalar(row, "predicted_defect_risk")
                ),
                "risk_tier": str(get_scalar(row, "risk_tier")),
                "anomaly_score": float(get_scalar(row, "anomaly_score")),
                "anomaly_flag": int(get_scalar(row, "anomaly_flag")),
                "recommended_action": str(get_scalar(row, "recommended_action")),
            }
        )

    high_risk_rows = [
        record
        for record in candidate_rows
        if record["risk_tier"] in ["critical", "high"]
    ]

    sorted_candidates = sorted(
        high_risk_rows,
        key=lambda record: (
            float(record["predicted_defect_risk"]),
            float(record["anomaly_score"]),
        ),
        reverse=True,
    )

    explanations: list[dict[str, Any]] = []

    for record in sorted_candidates[:max_items]:
        row_index = int(record["index"])
        row_df = df.iloc[[row_index]].drop(columns=DROP_COLUMNS)

        top_contributions = get_item_contributions(
            model=model,
            row_df=row_df,
            feature_names=feature_names,
            coefficients=coefficients,
            top_n=8,
        )

        summary = generate_explanation_summary(
            item_id=str(record["item_id"]),
            equipment_type=str(record["equipment_type"]),
            risk=float(record["predicted_defect_risk"]),
            risk_tier=str(record["risk_tier"]),
            top_contributions=top_contributions,
        )

        explanations.append(
            {
                "item_id": record["item_id"],
                "equipment_type": record["equipment_type"],
                "vendor": record["vendor"],
                "batch_id": record["batch_id"],
                "season": record["season"],
                "predicted_defect_risk": record["predicted_defect_risk"],
                "risk_tier": record["risk_tier"],
                "anomaly_score": record["anomaly_score"],
                "anomaly_flag": record["anomaly_flag"],
                "recommended_action": record["recommended_action"],
                "top_contributions": top_contributions,
                "summary": summary,
            }
        )

    return explanations


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scored_data()
    artifact = load_model_artifact()

    model = cast(Pipeline, artifact["model"])
    model_name = str(artifact["model_name"])

    feature_names = get_feature_names(model)
    coefficients = get_model_coefficients(model)

    feature_importance_df = build_global_feature_importance(
        feature_names=feature_names,
        coefficients=coefficients,
    )

    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    explanations = build_item_explanations(
        df=df,
        model=model,
        feature_names=feature_names,
        coefficients=coefficients,
        max_items=15,
    )

    output = {
        "model_name": model_name,
        "method": (
            "For the selected logistic regression model, explanations are computed "
            "from standardized feature values multiplied by model coefficients. "
            "This provides directional feature attribution for item-level risk."
        ),
        "item_explanations": explanations,
    }

    with ITEM_EXPLANATIONS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"Loaded {len(df):,} scored records")
    print(f"Model used for explanation: {model_name}")
    print(f"Saved feature importance to: {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved item explanations to: {ITEM_EXPLANATIONS_PATH}")

    print("\nTop global risk features:")
    print(feature_importance_df.head(10).to_string(index=False))

    print("\nExample item explanation:")
    if explanations:
        print(explanations[0]["summary"])


if __name__ == "__main__":
    main()
