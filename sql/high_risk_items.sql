SELECT
    item_id,
    equipment_type,
    vendor,
    batch_id,
    season,
    usage_sessions,
    impact_count,
    repair_count,
    fit_complaint_count,
    surface_wear_score,
    material_condition_score,
    inspection_score,
    defect_reported,
    failure_type
FROM equipment_quality
ORDER BY
    defect_reported DESC,
    inspection_score ASC,
    surface_wear_score DESC,
    repair_count DESC
LIMIT 25;
