from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_RECORDS = 5000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_equipment_quality.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_equipment_quality_data(
    n_records: int = N_RECORDS, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generate a synthetic sports equipment quality dataset.

    The dataset simulates equipment-room quality signals such as usage,
    repairs, inspection scores, vendor batches, environmental exposure,
    and reported defects.

    The target variable is defect_reported.
    """

    rng = np.random.default_rng(seed)

    equipment_types = np.array(
        [
            "helmet",
            "shoulder_pads",
            "cleats",
            "practice_jersey",
            "game_jersey",
            "football",
            "gloves",
            "training_equipment",
        ]
    )

    vendors = np.array(["Vendor_A", "Vendor_B", "Vendor_C", "Vendor_D"])
    seasons = np.array(["Fall_2024", "Spring_2025", "Fall_2025"])
    field_surfaces = np.array(["natural_grass", "artificial_turf", "indoor"])
    storage_locations = np.array(["main_equipment_room", "aux_storage", "travel_trunk"])

    equipment_type = rng.choice(
        equipment_types,
        size=n_records,
        p=[0.16, 0.12, 0.18, 0.14, 0.10, 0.12, 0.08, 0.10],
    )

    vendor = rng.choice(vendors, size=n_records, p=[0.35, 0.30, 0.22, 0.13])
    season = rng.choice(seasons, size=n_records, p=[0.35, 0.30, 0.35])
    field_surface = rng.choice(field_surfaces, size=n_records, p=[0.45, 0.45, 0.10])
    storage_location = rng.choice(
        storage_locations, size=n_records, p=[0.70, 0.20, 0.10]
    )

    batch_number = rng.integers(1, 21, size=n_records)
    batch_id = np.array([f"{v}_B{b:02d}" for v, b in zip(vendor, batch_number)])

    base_age = {
        "helmet": 420,
        "shoulder_pads": 500,
        "cleats": 160,
        "practice_jersey": 300,
        "game_jersey": 260,
        "football": 130,
        "gloves": 120,
        "training_equipment": 360,
    }

    age_days = (
        np.array(
            [
                max(10, rng.normal(base_age[item], base_age[item] * 0.35))
                for item in equipment_type
            ]
        )
        .round()
        .astype(int)
    )

    usage_multiplier = {
        "helmet": 1.15,
        "shoulder_pads": 1.05,
        "cleats": 1.25,
        "practice_jersey": 1.40,
        "game_jersey": 0.55,
        "football": 1.30,
        "gloves": 1.20,
        "training_equipment": 0.95,
    }

    usage_sessions = np.array(
        [
            rng.poisson(lam=max(5, age / 7 * usage_multiplier[item]))
            for age, item in zip(age_days, equipment_type)
        ]
    )

    game_exposure_count = rng.poisson(
        lam=np.where(
            np.isin(
                equipment_type, ["helmet", "shoulder_pads", "cleats", "game_jersey"]
            ),
            usage_sessions * 0.18,
            usage_sessions * 0.06,
        )
    )

    practice_exposure_count = np.maximum(usage_sessions - game_exposure_count, 0)

    rain_exposure_count = rng.poisson(
        lam=np.where(field_surface == "indoor", 0.2, usage_sessions * 0.08)
    )

    laundry_cycles = rng.poisson(
        lam=np.where(
            np.isin(equipment_type, ["practice_jersey", "game_jersey", "gloves"]),
            usage_sessions * 0.65,
            np.where(
                equipment_type == "football",
                usage_sessions * 0.05,
                usage_sessions * 0.12,
            ),
        )
    )

    impact_count = rng.poisson(
        lam=np.where(
            np.isin(equipment_type, ["helmet", "shoulder_pads"]),
            usage_sessions * 3.2,
            np.where(
                equipment_type == "football",
                usage_sessions * 1.1,
                usage_sessions * 0.25,
            ),
        )
    )

    vendor_quality_adjustment = {
        "Vendor_A": 3.0,
        "Vendor_B": 0.5,
        "Vendor_C": -4.0,
        "Vendor_D": -1.5,
    }

    season_adjustment = {
        "Fall_2024": 1.5,
        "Spring_2025": -1.0,
        "Fall_2025": 0.0,
    }

    humidity_base = np.where(
        storage_location == "main_equipment_room",
        52,
        np.where(storage_location == "aux_storage", 60, 66),
    )
    storage_humidity = np.clip(rng.normal(humidity_base, 7), 35, 85).round(1)

    surface_wear_score = (
        15
        + usage_sessions * 0.22
        + rain_exposure_count * 0.75
        + laundry_cycles * 0.10
        + np.where(field_surface == "artificial_turf", 4.0, 0.0)
        + rng.normal(0, 8, n_records)
    )
    surface_wear_score = np.clip(surface_wear_score, 0, 100).round(1)

    repair_count = rng.poisson(
        lam=np.clip(
            usage_sessions * 0.015
            + impact_count * 0.002
            + np.where(surface_wear_score > 70, 0.6, 0.0),
            0.05,
            5,
        )
    )

    fit_complaint_count = rng.poisson(
        lam=np.where(
            np.isin(equipment_type, ["helmet", "shoulder_pads", "cleats"]),
            0.25 + usage_sessions * 0.008 + repair_count * 0.15,
            0.05 + usage_sessions * 0.002,
        )
    )

    material_condition_score = (
        95
        - age_days * 0.035
        - usage_sessions * 0.08
        - surface_wear_score * 0.18
        - repair_count * 2.5
        + np.array([vendor_quality_adjustment[v] for v in vendor])
        + rng.normal(0, 6, n_records)
    )
    material_condition_score = np.clip(material_condition_score, 5, 100).round(1)

    inspection_score = (
        92
        - surface_wear_score * 0.28
        - repair_count * 3.2
        - fit_complaint_count * 2.2
        - rain_exposure_count * 0.20
        + np.array([season_adjustment[s] for s in season])
        + rng.normal(0, 7, n_records)
    )
    inspection_score = np.clip(inspection_score, 0, 100).round(1)

    risk_logit = (
        -4.1
        + usage_sessions * 0.012
        + impact_count * 0.004
        + repair_count * 0.45
        + fit_complaint_count * 0.33
        + surface_wear_score * 0.026
        - material_condition_score * 0.025
        - inspection_score * 0.032
        + rain_exposure_count * 0.030
        + np.where(storage_humidity > 65, 0.45, 0.0)
        + np.where(vendor == "Vendor_C", 0.60, 0.0)
        + np.where(np.isin(batch_number, [7, 13, 18]), 0.55, 0.0)
        + np.where(season == "Spring_2025", 0.25, 0.0)
        + np.where(field_surface == "artificial_turf", 0.20, 0.0)
    )

    defect_probability = sigmoid(risk_logit)
    defect_reported = rng.binomial(1, defect_probability)

    failure_type = np.full(n_records, "no_defect", dtype=object)

    defect_indices = np.where(defect_reported == 1)[0]
    for idx in defect_indices:
        scores = {
            "structural_wear": impact_count[idx] * 0.020 + repair_count[idx] * 0.70,
            "surface_damage": surface_wear_score[idx] * 0.040
            + rain_exposure_count[idx] * 0.070,
            "fit_issue": fit_complaint_count[idx] * 0.90,
            "hygiene_issue": laundry_cycles[idx] * 0.025
            + storage_humidity[idx] * 0.020,
            "missing_component": repair_count[idx] * 0.35 + rng.normal(0, 0.2),
        }

        labels = list(scores.keys())
        values = np.array(list(scores.values()))
        values = np.maximum(values, 0.01)
        probs = values / values.sum()

        failure_type[idx] = rng.choice(labels, p=probs)

    df = pd.DataFrame(
        {
            "item_id": [f"EQ-{i + 1:05d}" for i in range(n_records)],
            "equipment_type": equipment_type,
            "vendor": vendor,
            "batch_id": batch_id,
            "season": season,
            "age_days": age_days,
            "usage_sessions": usage_sessions,
            "game_exposure_count": game_exposure_count,
            "practice_exposure_count": practice_exposure_count,
            "rain_exposure_count": rain_exposure_count,
            "laundry_cycles": laundry_cycles,
            "impact_count": impact_count,
            "repair_count": repair_count,
            "fit_complaint_count": fit_complaint_count,
            "surface_wear_score": surface_wear_score,
            "material_condition_score": material_condition_score,
            "inspection_score": inspection_score,
            "storage_humidity": storage_humidity,
            "field_surface": field_surface,
            "storage_location": storage_location,
            "defect_probability_true": defect_probability.round(4),
            "defect_reported": defect_reported,
            "failure_type": failure_type,
        }
    )

    return df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate_equipment_quality_data()
    df.to_csv(OUTPUT_PATH, index=False)

    defect_rate = df["defect_reported"].mean()
    defect_by_equipment_type = cast(
        pd.DataFrame,
        df.groupby("equipment_type", as_index=False).agg(
            count=("defect_reported", "count"),
            defective_items=("defect_reported", "sum"),
            defect_rate=("defect_reported", "mean"),
        ),
    )

    defect_by_equipment_type = defect_by_equipment_type.sort_values(
        by="defect_rate",
        ascending=False,
    )

    print(f"Generated {len(df):,} records")
    print(f"Saved dataset to: {OUTPUT_PATH}")
    print(f"Defect rate: {defect_rate:.2%}")
    print("\nDefect count by equipment type:")
    print(defect_by_equipment_type)


if __name__ == "__main__":
    main()
