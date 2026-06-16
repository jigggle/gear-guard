from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def get_scalar(row: pd.Series, column: str) -> Any:
    value = row.at[column]
    if hasattr(value, "item"):
        return value.item()
    return value


def numeric_quantile(df: pd.DataFrame, column: str, q: float) -> float:
    values = np.asarray(df[column], dtype=float)
    return float(np.quantile(values, q))


def build_risk_counts(df: pd.DataFrame) -> pd.DataFrame:
    risk_order = ["critical", "high", "medium", "low"]
    records: list[dict[str, Any]] = []

    for risk_tier in risk_order:
        count = mask_count(df["risk_tier"] == risk_tier)
        records.append(
            {
                "risk_tier": risk_tier,
                "item_count": count,
            }
        )

    return pd.DataFrame(records)


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

    st.subheader("Equipment Type × Vendor Quality Heatmap")

    heatmap_df = cast(
        pd.DataFrame,
        cast(Any, df).pivot_table(
            index="equipment_type",
            columns="vendor",
            values="predicted_defect_risk",
            aggfunc="mean",
        ),
    )

    fig = px.imshow(
        heatmap_df,
        text_auto=True,
        aspect="auto",
        title="Average Predicted Defect Risk by Equipment Type and Vendor",
        labels={
            "x": "Vendor",
            "y": "Equipment Type",
            "color": "Avg Predicted Risk",
        },
    )

    fig.update_traces(
        texttemplate="%{z:.1%}",
        hovertemplate=(
            "Equipment Type: %{y}<br>"
            "Vendor: %{x}<br>"
            "Avg Predicted Risk: %{z:.1%}<extra></extra>"
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

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

    st.subheader("Risk vs Anomaly Quadrant")

    risk_threshold = 0.50
    anomaly_threshold = numeric_quantile(df, "anomaly_score", 0.95)

    fig = px.scatter(
        df,
        x="predicted_defect_risk",
        y="anomaly_score",
        color="risk_tier",
        hover_data=[
            "item_id",
            "equipment_type",
            "vendor",
            "batch_id",
            "season",
            "inspection_score",
            "surface_wear_score",
            "repair_count",
            "recommended_action",
        ],
        title="Predicted Defect Risk vs Anomaly Score",
        labels={
            "predicted_defect_risk": "Predicted Defect Risk",
            "anomaly_score": "Anomaly Score",
            "risk_tier": "Risk Tier",
        },
    )

    fig.add_vline(
        x=risk_threshold,
        line_dash="dash",
        annotation_text="Risk threshold",
    )

    fig.add_hline(
        y=anomaly_threshold,
        line_dash="dash",
        annotation_text="Top anomaly band",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Items in the upper-right region combine high predicted defect risk with unusual quality signatures."
    )

    st.subheader("Inspection Workload by Risk Tier")

    risk_counts = build_risk_counts(df)

    fig = px.funnel(
        risk_counts,
        x="item_count",
        y="risk_tier",
        title="Inspection Workload Funnel",
        labels={
            "item_count": "Item Count",
            "risk_tier": "Risk Tier",
        },
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

    major_count = mask_count(drift_report["drift_status"] == "major_drift")
    moderate_count = mask_count(drift_report["drift_status"] == "moderate_drift")
    stable_count = mask_count(drift_report["drift_status"] == "stable")

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

    st.subheader("Baseline vs Current Quality Signal Shift")

    top_drift = drift_report.head(8).copy()

    fig = go.Figure()

    for _, row in top_drift.iterrows():
        feature = str(get_scalar(row, "feature"))
        baseline_mean = float(get_scalar(row, "baseline_mean"))
        current_mean = float(get_scalar(row, "current_mean"))
        drift_status = str(get_scalar(row, "drift_status"))

        fig.add_trace(
            go.Scatter(
                x=[baseline_mean, current_mean],
                y=[feature, feature],
                mode="lines+markers",
                name=feature,
                hovertemplate=(
                    "Feature: %{y}<br>"
                    "Value: %{x:.4f}<br>"
                    f"Drift status: {drift_status}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Baseline vs Current Feature Means",
        xaxis_title="Feature Mean",
        yaxis_title="Feature",
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

    st.subheader("Risk Contribution Waterfall")

    waterfall_df = sort_dataframe(
        contributions_df,
        by="absolute_contribution",
        ascending=False,
    ).head(8)

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * len(waterfall_df),
            x=waterfall_df["feature"],
            y=waterfall_df["contribution"],
            text=[f"{value:.3f}" for value in waterfall_df["contribution"]],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"Top Feature Contributions for {selected_item_id}",
        xaxis_title="Feature",
        yaxis_title="Contribution to Risk Score",
    )

    st.plotly_chart(fig, use_container_width=True)

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

    contributions_sorted = sort_dataframe(
        contributions_df,
        by="absolute_contribution",
        ascending=True,
    )

    fig = px.bar(
        contributions_sorted,
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

    top_feature_importance = sort_dataframe(
        cast(pd.DataFrame, feature_importance.head(15)),
        by="importance",
        ascending=True,
    )

    fig = px.bar(
        top_feature_importance,
        x="importance",
        y="feature",
        orientation="h",
        color="direction",
        title="Top Global Feature Importance",
    )
    st.plotly_chart(fig, use_container_width=True)
