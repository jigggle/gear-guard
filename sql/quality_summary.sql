SELECT
    equipment_type,
    COUNT(*) AS total_items,
    SUM(defect_reported) AS defective_items,
    ROUND(AVG(defect_reported) * 100, 2) AS defect_rate_pct,
    ROUND(AVG(inspection_score), 2) AS avg_inspection_score,
    ROUND(AVG(surface_wear_score), 2) AS avg_surface_wear,
    ROUND(AVG(repair_count), 2) AS avg_repair_count
FROM equipment_quality
GROUP BY equipment_type
ORDER BY defect_rate_pct DESC;
