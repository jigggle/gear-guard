from src.data_generator import generate_equipment_quality_data


def test_data_generator_shape() -> None:
    df = generate_equipment_quality_data(n_records=100, seed=42)

    assert len(df) == 100
    assert "item_id" in df.columns
    assert "equipment_type" in df.columns
    assert "defect_reported" in df.columns


def test_defect_target_binary() -> None:
    df = generate_equipment_quality_data(n_records=100, seed=42)

    assert set(df["defect_reported"].unique()).issubset({0, 1})


def test_required_equipment_types_present() -> None:
    df = generate_equipment_quality_data(n_records=1000, seed=42)

    expected_types = {
        "helmet",
        "shoulder_pads",
        "cleats",
        "practice_jersey",
        "game_jersey",
        "football",
        "gloves",
        "training_equipment",
    }

    assert set(df["equipment_type"].unique()).issubset(expected_types)


def test_defect_rate_is_imbalanced() -> None:
    df = generate_equipment_quality_data(n_records=5000, seed=42)
    defect_rate = df["defect_reported"].mean()

    assert 0.01 <= defect_rate <= 0.10
