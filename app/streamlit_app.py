from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "equipment_quality_scored.csv"
MODEL_REPORT_PATH = PROJECT_ROOT / "reports" / "model_report.json"
ANOMALY_REPORT_PATH = PROJECT_ROOT / "reports" / "anomaly_report.json"
DRIFT_REPORT_PATH = PROJECT_ROOT / "reports" / "drift_report.csv"
ITEM_EXPLANATIONS_PATH = PROJECT_ROOT / "reports" / "item_explanations.json"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "feature_importance.csv"


st.set_page_config(
    page_title="GearGuard",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()

    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def pct(value: float) -> str:
    return f"{value:.1%}"


def risk_badge(risk_tier: str) -> str:
    risk_tier = str(risk_tier)

    if risk_tier == "critical":
        return "🔴 Critical"
    if risk_tier == "high":
        return "🟠 High"
    if risk_tier == "medium":
        return "🟡 Medium"
    return "🟢 Low"


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


df = load_csv(SCORED_DATA_PATH)
model_report = load_json(MODEL_REPORT_PATH)
anomaly_report = load_json(ANOMALY_REPORT_PATH)
drift_report = load_csv(DRIFT_REPORT_PATH)
explanations_report = load_json(ITEM_EXPLANATIONS_PATH)
feature_importance = load_csv(FEATURE_IMPORTANCE_PATH)

st.title("GearGuard")
st.caption(
    "AI quality risk monitoring for football equipment operations — defect prediction, "
    "anomaly detection, drift monitoring, and explainable triage."
)

tabs = st.tabs(
    [
        "Overview",
        "Risk Scoring",
        "Anomaly Detection",
        "Drift Monitor",
        "Item Explanation",
        "Model Report",
    ]
)

with tabs[0]:
    st.subheader("Equipment Quality Overview")

    total_items = len(df)
    defect_rate = numeric_mean(df, "defect_reported")
    avg_predicted_risk = numeric_mean(df, "predicted_defect_risk")
    anomaly_rate = numeric_mean(df, "anomaly_flag")

    high_critical_count = mask_count(df["risk_tier"].isin(["high", "critical"]))

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Items", f"{total_items:,}")
    col2.metric("Actual Defect Rate", pct(defect_rate))
    col3.metric("Avg Predicted Risk", pct(avg_predicted_risk))
    col4.metric("Anomaly Rate", pct(anomaly_rate))
    col5.metric("High/Critical Items", f"{high_critical_count:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        defect_by_type = cast(
            pd.DataFrame,
            df.groupby("equipment_type", as_index=False).agg(
                total_items=("item_id", "count"),
                defect_rate=("defect_reported", "mean"),
                avg_predicted_risk=("predicted_defect_risk", "mean"),
            ),
        )

        defect_by_type = sort_dataframe(
            defect_by_type,
            by="defect_rate",
            ascending=False,
        )

        fig = px.bar(
            defect_by_type,
            x="equipment_type",
            y="defect_rate",
            title="Actual Defect Rate by Equipment Type",
            labels={
                "equipment_type": "Equipment Type",
                "defect_rate": "Defect Rate",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        risk_by_vendor = cast(
            pd.DataFrame,
            df.groupby("vendor", as_index=False).agg(
                total_items=("item_id", "count"),
                avg_predicted_risk=("predicted_defect_risk", "mean"),
                anomaly_rate=("anomaly_flag", "mean"),
            ),
        )

        risk_by_vendor = sort_dataframe(
            risk_by_vendor,
            by="avg_predicted_risk",
            ascending=False,
        )

        fig = px.bar(
            risk_by_vendor,
            x="vendor",
            y="avg_predicted_risk",
            title="Average Predicted Defect Risk by Vendor",
            labels={
                "vendor": "Vendor",
                "avg_predicted_risk": "Average Predicted Risk",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        defect_by_type,
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    st.subheader("Risk Scoring")

    risk_tier_options = ["critical", "high", "medium", "low"]
    selected_tiers = st.multiselect(
        "Filter by risk tier",
        options=risk_tier_options,
        default=["critical", "high"],
    )

    filtered_mask = df["risk_tier"].isin(selected_tiers)
    filtered_df = cast(pd.DataFrame, df.loc[filtered_mask].copy())

    filtered_df = sort_dataframe(
        filtered_df,
        by=["predicted_defect_risk", "anomaly_score"],
        ascending=[False, False],
    )

    st.metric("Filtered Items", f"{len(filtered_df):,}")

    display_columns = [
        "item_id",
        "equipment_type",
        "vendor",
        "batch_id",
        "season",
        "predicted_defect_risk",
        "risk_tier",
        "anomaly_score",
        "inspection_score",
        "surface_wear_score",
        "repair_count",
        "recommended_action",
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    fig = px.histogram(
        df,
        x="predicted_defect_risk",
        nbins=40,
        color="risk_tier",
        title="Predicted Defect Risk Distribution",
        labels={"predicted_defect_risk": "Predicted Defect Risk"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Anomaly Detection")

    col1, col2, col3 = st.columns(3)
    col1.metric("Anomaly Count", f"{anomaly_report['anomaly_count']:,}")
    col2.metric("Anomaly Rate", pct(float(anomaly_report["anomaly_rate"])))
    col3.metric(
        "High/Critical Risk Items",
        f"{anomaly_report['high_or_critical_risk_count']:,}",
    )

    st.caption(str(anomaly_report["method"]))

    anomaly_mask = df["anomaly_flag"] == 1
    anomalous_df = cast(pd.DataFrame, df.loc[anomaly_mask].copy())

    anomalous_df = sort_dataframe(
        anomalous_df,
        by=["anomaly_score", "predicted_defect_risk"],
        ascending=[False, False],
    )

    st.dataframe(
        anomalous_df[
            [
                "item_id",
                "equipment_type",
                "vendor",
                "batch_id",
                "predicted_defect_risk",
                "risk_tier",
                "anomaly_score",
                "inspection_score",
                "surface_wear_score",
                "repair_count",
                "recommended_action",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    fig = px.histogram(
        df,
        x="anomaly_score",
        nbins=40,
        color="anomaly_flag",
        title="Anomaly Score Distribution",
        labels={"anomaly_score": "Anomaly Score"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Drift Monitor")

    major_count = int((drift_report["drift_status"] == "major_drift").sum())
    moderate_count = int((drift_report["drift_status"] == "moderate_drift").sum())
    stable_count = int((drift_report["drift_status"] == "stable").sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Major Drift Features", major_count)
    col2.metric("Moderate Drift Features", moderate_count)
    col3.metric("Stable Features", stable_count)

    st.dataframe(
        drift_report,
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        drift_report.head(10),
        x="feature",
        y="ks_statistic",
        color="drift_status",
        title="Top Drifted Features by KS Statistic",
        labels={
            "feature": "Feature",
            "ks_statistic": "KS Statistic",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Item Explanation")

    explanations = explanations_report["item_explanations"]

    item_ids = [item["item_id"] for item in explanations]

    selected_item_id = st.selectbox(
        "Select a high-risk item",
        options=item_ids,
    )

    selected_explanation = next(
        item for item in explanations if item["item_id"] == selected_item_id
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Item ID", selected_explanation["item_id"])
    col2.metric("Equipment", selected_explanation["equipment_type"])
    col3.metric("Risk Tier", risk_badge(selected_explanation["risk_tier"]))
    col4.metric(
        "Predicted Risk",
        pct(float(selected_explanation["predicted_defect_risk"])),
    )

    st.info(selected_explanation["summary"])

    contributions_df = pd.DataFrame(selected_explanation["top_contributions"])

    st.dataframe(
        contributions_df[
            [
                "feature",
                "contribution",
                "absolute_contribution",
                "effect",
                "coefficient",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        contributions_df.sort_values("absolute_contribution", ascending=True),
        x="contribution",
        y="feature",
        orientation="h",
        color="effect",
        title=f"Top Risk Contributions for {selected_item_id}",
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("Model Report")

    best_model = model_report["best_model"]
    best_threshold = float(model_report["best_threshold"])
    class_balance = model_report["class_balance"]
    test_metrics = model_report["model_results"][best_model]["test_metrics"]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Best Model", best_model)
    col2.metric("Threshold", f"{best_threshold:.2f}")
    col3.metric("Test Recall", pct(float(test_metrics["recall"])))
    col4.metric("Test Precision", pct(float(test_metrics["precision"])))
    col5.metric("Test PR-AUC", f"{float(test_metrics['pr_auc']):.3f}")

    st.caption(model_report["selection_logic"])

    st.write("Class balance")
    st.json(class_balance)

    st.write("Top Global Feature Importance")
    st.dataframe(
        feature_importance.head(20),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        feature_importance.head(15).sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        color="direction",
        title="Top Global Feature Importance",
    )
    st.plotly_chart(fig, use_container_width=True)
