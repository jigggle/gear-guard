from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "equipment_quality_scored.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

DRIFT_REPORT_CSV_PATH = REPORTS_DIR / "drift_report.csv"
DRIFT_REPORT_JSON_PATH = REPORTS_DIR / "drift_report.json"

BASELINE_SEASON = "Fall_2024"
CURRENT_SEASON = "Spring_2025"

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
    "predicted_defect_risk",
    "anomaly_score",
]


def load_scored_data(path: Path = SCORED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Scored dataset not found at {path}. Run `python3 src/anomaly.py` first."
        )

    return pd.read_csv(path)


def get_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, df.loc[:, column]).astype(float)


def to_float_array(values: Any) -> np.ndarray:
    series = pd.Series(values)
    return np.asarray(series.dropna(), dtype=float)


def calculate_psi(
    baseline_values: Any,
    current_values: Any,
    bins: int = 10,
) -> float:
    baseline_array = to_float_array(baseline_values)
    current_array = to_float_array(current_values)

    if len(baseline_array) == 0 or len(current_array) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(baseline_array, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 3:
        return 0.0

    baseline_counts, _ = np.histogram(baseline_array, bins=bin_edges)
    current_counts, _ = np.histogram(current_array, bins=bin_edges)

    baseline_pct = baseline_counts / max(baseline_counts.sum(), 1)
    current_pct = current_counts / max(current_counts.sum(), 1)

    epsilon = 1e-6
    baseline_pct = np.where(baseline_pct == 0, epsilon, baseline_pct)
    current_pct = np.where(current_pct == 0, epsilon, current_pct)

    psi_values = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)

    return float(np.sum(psi_values))


def classify_drift(psi: float, p_value: float) -> str:
    if psi >= 0.25 or p_value < 0.01:
        return "major_drift"
    if psi >= 0.10 or p_value < 0.05:
        return "moderate_drift"
    return "stable"


def interpret_drift(feature: str, baseline_mean: float, current_mean: float) -> str:
    direction = "increased" if current_mean > baseline_mean else "decreased"

    feature_explanations = {
        "usage_sessions": "usage volume changed between seasons",
        "impact_count": "contact or collision exposure changed",
        "repair_count": "repair frequency changed",
        "fit_complaint_count": "fit-related complaints shifted",
        "surface_wear_score": "visible wear patterns changed",
        "material_condition_score": "material condition shifted",
        "inspection_score": "inspection outcomes changed",
        "storage_humidity": "storage environment changed",
        "predicted_defect_risk": "overall model-estimated quality risk shifted",
        "anomaly_score": "unusual equipment signatures became more or less common",
    }

    base_message = feature_explanations.get(feature, "feature distribution changed")

    return f"{feature} {direction}; {base_message}."


def build_drift_report(df: pd.DataFrame) -> pd.DataFrame:
    baseline_mask = cast(pd.Series, df["season"] == BASELINE_SEASON)
    current_mask = cast(pd.Series, df["season"] == CURRENT_SEASON)

    baseline_df = cast(pd.DataFrame, df.loc[baseline_mask].copy())
    current_df = cast(pd.DataFrame, df.loc[current_mask].copy())

    if baseline_df.empty:
        raise ValueError(f"No records found for baseline season: {BASELINE_SEASON}")

    if current_df.empty:
        raise ValueError(f"No records found for current season: {CURRENT_SEASON}")

    drift_records: list[dict[str, Any]] = []

    for feature in NUMERIC_FEATURES:
        baseline_values = get_numeric_series(baseline_df, feature).dropna()
        current_values = get_numeric_series(current_df, feature).dropna()

        baseline_array = to_float_array(baseline_values)
        current_array = to_float_array(current_values)

        baseline_mean = float(baseline_array.mean())
        current_mean = float(current_array.mean())

        psi = calculate_psi(baseline_array, current_array)

        ks_result = ks_2samp(baseline_array, current_array)
        ks_statistic_raw, p_value_raw = cast(tuple[float, float], ks_result)

        ks_statistic = float(ks_statistic_raw)
        p_value = float(p_value_raw)

        drift_status = classify_drift(psi=psi, p_value=p_value)

        drift_records.append(
            {
                "feature": feature,
                "baseline_season": BASELINE_SEASON,
                "current_season": CURRENT_SEASON,
                "baseline_mean": round(baseline_mean, 4),
                "current_mean": round(current_mean, 4),
                "mean_change": round(current_mean - baseline_mean, 4),
                "psi": round(psi, 4),
                "ks_statistic": round(ks_statistic, 4),
                "ks_p_value": round(p_value, 6),
                "drift_status": drift_status,
                "interpretation": interpret_drift(
                    feature=feature,
                    baseline_mean=baseline_mean,
                    current_mean=current_mean,
                ),
            }
        )

    report_df = pd.DataFrame(drift_records)

    status_rank = {
        "major_drift": 2,
        "moderate_drift": 1,
        "stable": 0,
    }

    drift_statuses = [str(value) for value in report_df["drift_status"].tolist()]
    report_df["status_rank"] = [status_rank.get(status, 0) for status in drift_statuses]

    report_df = report_df.sort_values(
        by=["status_rank", "psi", "ks_statistic"],
        ascending=[False, False, False],
    )

    report_df = report_df.drop(columns=["status_rank"])

    return report_df


def build_json_summary(drift_report: pd.DataFrame) -> dict[str, Any]:
    drift_status_values = np.asarray(drift_report["drift_status"])

    major_drift_count = int((drift_status_values == "major_drift").sum())
    moderate_drift_count = int((drift_status_values == "moderate_drift").sum())
    stable_count = int((drift_status_values == "stable").sum())

    top_drifted_features: list[dict[str, Any]] = []

    for _, row in drift_report.head(10).iterrows():
        top_drifted_features.append(
            {
                "feature": str(row.at["feature"]),
                "drift_status": str(row.at["drift_status"]),
                "psi": float(row.at["psi"]),
                "ks_statistic": float(row.at["ks_statistic"]),
                "ks_p_value": float(row.at["ks_p_value"]),
                "interpretation": str(row.at["interpretation"]),
            }
        )

    return {
        "baseline_season": BASELINE_SEASON,
        "current_season": CURRENT_SEASON,
        "features_checked": int(len(drift_report)),
        "major_drift_count": major_drift_count,
        "moderate_drift_count": moderate_drift_count,
        "stable_count": stable_count,
        "top_drifted_features": top_drifted_features,
        "method": "Drift is measured with Population Stability Index and two-sample Kolmogorov-Smirnov tests.",
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scored_data()
    drift_report = build_drift_report(df)

    drift_report.to_csv(DRIFT_REPORT_CSV_PATH, index=False)

    json_summary = build_json_summary(drift_report)

    with DRIFT_REPORT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(json_summary, file, indent=2)

    print(f"Loaded {len(df):,} scored records")
    print(f"Baseline season: {BASELINE_SEASON}")
    print(f"Current season: {CURRENT_SEASON}")
    print(f"Saved drift report CSV to: {DRIFT_REPORT_CSV_PATH}")
    print(f"Saved drift report JSON to: {DRIFT_REPORT_JSON_PATH}")

    print("\nDrift summary:")
    print(f"Major drift features: {json_summary['major_drift_count']}")
    print(f"Moderate drift features: {json_summary['moderate_drift_count']}")
    print(f"Stable features: {json_summary['stable_count']}")

    print("\nTop drifted features:")
    print(
        drift_report[
            [
                "feature",
                "drift_status",
                "baseline_mean",
                "current_mean",
                "psi",
                "ks_statistic",
                "ks_p_value",
                "interpretation",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
