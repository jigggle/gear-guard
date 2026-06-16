from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "equipment_quality_scored.csv"
DRIFT_REPORT_PATH = PROJECT_ROOT / "reports" / "drift_report.csv"
ITEM_EXPLANATIONS_PATH = PROJECT_ROOT / "reports" / "item_explanations.json"

SOP_PATH = PROJECT_ROOT / "docs" / "equipment_inspection_sop.md"
DRIFT_GUIDE_PATH = PROJECT_ROOT / "docs" / "drift_response_guide.md"

OUTPUT_PATH = PROJECT_ROOT / "reports" / "quality_triage_summary.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return cast(dict[str, Any], json.load(file))


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return path.read_text(encoding="utf-8")


def get_scalar(row: pd.Series, column: str) -> Any:
    value = row.at[column]
    if hasattr(value, "item"):
        return value.item()
    return value


def numeric_mean(df: pd.DataFrame, column: str) -> float:
    values = np.asarray(df[column], dtype=float)
    return float(values.mean())


def mask_count(mask: Any) -> int:
    return int(np.asarray(mask).sum())


def sort_dataframe(
    df: pd.DataFrame,
    by: str | list[str],
    ascending: bool | list[bool],
) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        cast(Any, df).sort_values(by=by, ascending=ascending),
    )


def get_top_risk_items(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    sorted_df = sort_dataframe(
        df,
        by=["predicted_defect_risk", "anomaly_score"],
        ascending=[False, False],
    )

    return cast(pd.DataFrame, sorted_df.head(limit))


def get_top_drift_features(drift_report: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    status_rank = {
        "major_drift": 2,
        "moderate_drift": 1,
        "stable": 0,
    }

    report = drift_report.copy()

    drift_status_values = [str(value) for value in report["drift_status"].tolist()]
    report["status_rank"] = [
        status_rank.get(status, 0) for status in drift_status_values
    ]

    sorted_report = sort_dataframe(
        report,
        by=["status_rank", "ks_statistic", "psi"],
        ascending=[False, False, False],
    )

    return cast(pd.DataFrame, sorted_report.head(limit))


def extract_excerpt(text: str, start_heading: str, max_lines: int = 20) -> str:
    lines = text.splitlines()

    try:
        start_index = lines.index(start_heading)
    except ValueError:
        return "\n".join(lines[:max_lines])

    excerpt = lines[start_index : start_index + max_lines]
    return "\n".join(excerpt)


def build_triage_summary() -> str:
    scored_df = pd.read_csv(SCORED_DATA_PATH)
    drift_report = pd.read_csv(DRIFT_REPORT_PATH)
    explanations = load_json(ITEM_EXPLANATIONS_PATH)

    inspection_sop = load_text(SOP_PATH)
    drift_guide = load_text(DRIFT_GUIDE_PATH)

    top_risk_items = get_top_risk_items(scored_df, limit=10)
    top_drift_features = get_top_drift_features(drift_report, limit=5)

    total_items = int(len(scored_df))
    defect_rate = numeric_mean(scored_df, "defect_reported")
    anomaly_rate = numeric_mean(scored_df, "anomaly_flag")
    high_critical_count = mask_count(scored_df["risk_tier"].isin(["high", "critical"]))

    major_drift_count = mask_count(drift_report["drift_status"] == "major_drift")
    moderate_drift_count = mask_count(drift_report["drift_status"] == "moderate_drift")

    item_explanations = cast(
        list[dict[str, Any]],
        explanations.get("item_explanations", []),
    )

    lines: list[str] = []

    lines.append("# GearGuard Quality Triage Summary")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "GearGuard identified high-risk equipment items, anomalous quality signatures, "
        "and drifted quality signals across the simulated football equipment inventory."
    )
    lines.append("")
    lines.append(f"- Total equipment records reviewed: **{total_items:,}**")
    lines.append(f"- Actual defect rate: **{defect_rate:.2%}**")
    lines.append(f"- Anomaly rate: **{anomaly_rate:.2%}**")
    lines.append(f"- High or critical risk items: **{high_critical_count:,}**")
    lines.append(f"- Major drift features: **{major_drift_count}**")
    lines.append(f"- Moderate drift features: **{moderate_drift_count}**")
    lines.append("")

    lines.append("## Highest-Risk Equipment Items")
    lines.append("")
    lines.append(
        "| Item ID | Type | Vendor | Season | Predicted Risk | Risk Tier | Anomaly Score | Recommended Action |"
    )
    lines.append("|---|---|---|---|---:|---|---:|---|")

    for _, row in top_risk_items.iterrows():
        item_id = str(get_scalar(row, "item_id"))
        equipment_type = str(get_scalar(row, "equipment_type"))
        vendor = str(get_scalar(row, "vendor"))
        season = str(get_scalar(row, "season"))
        predicted_risk = float(get_scalar(row, "predicted_defect_risk"))
        risk_tier = str(get_scalar(row, "risk_tier"))
        anomaly_score = float(get_scalar(row, "anomaly_score"))
        recommended_action = str(get_scalar(row, "recommended_action"))

        lines.append(
            f"| {item_id} "
            f"| {equipment_type} "
            f"| {vendor} "
            f"| {season} "
            f"| {predicted_risk:.1%} "
            f"| {risk_tier} "
            f"| {anomaly_score:.4f} "
            f"| {recommended_action} |"
        )

    lines.append("")
    lines.append("## Top Drifted Quality Signals")
    lines.append("")
    lines.append(
        "| Feature | Drift Status | Baseline Mean | Current Mean | KS Statistic | PSI | Interpretation |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|")

    for _, row in top_drift_features.iterrows():
        feature = str(get_scalar(row, "feature"))
        drift_status = str(get_scalar(row, "drift_status"))
        baseline_mean = float(get_scalar(row, "baseline_mean"))
        current_mean = float(get_scalar(row, "current_mean"))
        ks_statistic = float(get_scalar(row, "ks_statistic"))
        psi = float(get_scalar(row, "psi"))
        interpretation = str(get_scalar(row, "interpretation"))

        lines.append(
            f"| {feature} "
            f"| {drift_status} "
            f"| {baseline_mean:.4f} "
            f"| {current_mean:.4f} "
            f"| {ks_statistic:.4f} "
            f"| {psi:.4f} "
            f"| {interpretation} |"
        )

    lines.append("")
    lines.append("## Example Item-Level Investigation Notes")
    lines.append("")

    for item in item_explanations[:5]:
        item_id = str(item["item_id"])
        equipment_type = str(item["equipment_type"])
        vendor = str(item["vendor"])
        batch_id = str(item["batch_id"])
        season = str(item["season"])
        predicted_defect_risk = float(item["predicted_defect_risk"])
        risk_tier = str(item["risk_tier"])
        anomaly_score = float(item["anomaly_score"])
        recommended_action = str(item["recommended_action"])
        summary = str(item["summary"])

        lines.append(f"### {item_id} — {equipment_type}")
        lines.append("")
        lines.append(f"- Vendor: **{vendor}**")
        lines.append(f"- Batch: **{batch_id}**")
        lines.append(f"- Season: **{season}**")
        lines.append(f"- Predicted defect risk: **{predicted_defect_risk:.1%}**")
        lines.append(f"- Risk tier: **{risk_tier}**")
        lines.append(f"- Anomaly score: **{anomaly_score:.4f}**")
        lines.append(f"- Recommended action: **{recommended_action}**")
        lines.append("")
        lines.append(summary)
        lines.append("")

    lines.append("## Retrieved Inspection Guidance")
    lines.append("")
    lines.append(
        "The following operational guidance was referenced from local documentation:"
    )
    lines.append("")
    lines.append("- `docs/equipment_inspection_sop.md`")
    lines.append("- `docs/drift_response_guide.md`")
    lines.append("")
    lines.append("### Inspection SOP Excerpt")
    lines.append("")
    lines.append("```text")
    lines.append(extract_excerpt(inspection_sop, "## Risk Tiers", max_lines=25))
    lines.append("```")
    lines.append("")
    lines.append("### Drift Response Excerpt")
    lines.append("")
    lines.append("```text")
    lines.append(extract_excerpt(drift_guide, "## Drift Severity", max_lines=25))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "This summary is generated from model outputs, anomaly scores, drift reports, "
        "and local inspection guidance. It is intended for decision support and should "
        "not replace manual review by trained equipment staff."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary = build_triage_summary()
    OUTPUT_PATH.write_text(summary, encoding="utf-8")

    print(f"Saved quality triage summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
