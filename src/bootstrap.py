"""Build the synthetic dataset, model and report artifacts on first run.

Streamlit Community Cloud starts from a fresh clone where generated data and
models (gitignored) do not exist. ensure_artifacts() runs the same pipeline as
`make all`, once, and is a no-op when the artifacts are already present.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    PROJECT_ROOT / "data" / "processed" / "equipment_quality_scored.csv",
    PROJECT_ROOT / "reports" / "model_report.json",
    PROJECT_ROOT / "reports" / "anomaly_report.json",
    PROJECT_ROOT / "reports" / "drift_report.csv",
    PROJECT_ROOT / "reports" / "item_explanations.json",
    PROJECT_ROOT / "reports" / "feature_importance.csv",
]

STEPS = [
    "data_generator.py",
    "db.py",
    "train.py",
    "anomaly.py",
    "drift.py",
    "explain.py",
    "summary.py",
]


def artifacts_ready() -> bool:
    return all(path.exists() for path in REQUIRED)


def ensure_artifacts() -> None:
    if artifacts_ready():
        return

    for step in STEPS:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / step)],
            cwd=PROJECT_ROOT,
            check=True,
        )

    if not artifacts_ready():
        raise RuntimeError("GearGuard pipeline finished but artifacts are missing.")
