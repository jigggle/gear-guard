from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_equipment_quality.csv"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "gear_guard.sqlite"


def load_csv_to_sqlite(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run `python src/data_generator.py` first."
        )

    df = pd.read_csv(csv_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        df.to_sql("equipment_quality", conn, if_exists="replace", index=False)

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_equipment_type
            ON equipment_quality (equipment_type);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_vendor
            ON equipment_quality (vendor);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_season
            ON equipment_quality (season);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_defect_reported
            ON equipment_quality (defect_reported);
            """
        )

    print(f"Loaded {len(df):,} records into {db_path}")
    print("Table created: equipment_quality")


def run_quality_checks(db_path: Path = DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        total_records = pd.read_sql_query(
            """
            SELECT COUNT(*) AS total_records
            FROM equipment_quality;
            """,
            conn,
        )

        defect_summary = pd.read_sql_query(
            """
            SELECT
                defect_reported,
                COUNT(*) AS record_count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM equipment_quality), 2) AS pct_of_total
            FROM equipment_quality
            GROUP BY defect_reported;
            """,
            conn,
        )

        defect_by_type = pd.read_sql_query(
            """
            SELECT
                equipment_type,
                COUNT(*) AS total_items,
                SUM(defect_reported) AS defective_items,
                ROUND(AVG(defect_reported) * 100, 2) AS defect_rate_pct
            FROM equipment_quality
            GROUP BY equipment_type
            ORDER BY defect_rate_pct DESC;
            """,
            conn,
        )

        defect_by_vendor = pd.read_sql_query(
            """
            SELECT
                vendor,
                COUNT(*) AS total_items,
                SUM(defect_reported) AS defective_items,
                ROUND(AVG(defect_reported) * 100, 2) AS defect_rate_pct
            FROM equipment_quality
            GROUP BY vendor
            ORDER BY defect_rate_pct DESC;
            """,
            conn,
        )

    print("\nTotal records:")
    print(total_records)

    print("\nDefect summary:")
    print(defect_summary)

    print("\nDefect rate by equipment type:")
    print(defect_by_type)

    print("\nDefect rate by vendor:")
    print(defect_by_vendor)


def main() -> None:
    load_csv_to_sqlite()
    run_quality_checks()


if __name__ == "__main__":
    main()
