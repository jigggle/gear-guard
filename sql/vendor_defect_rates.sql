SELECT
    vendor,
    COUNT(*) AS total_items,
    SUM(defect_reported) AS defective_items,
    ROUND(AVG(defect_reported) * 100, 2) AS defect_rate_pct,
    ROUND(AVG(inspection_score), 2) AS avg_inspection_score,
    ROUND(AVG(material_condition_score), 2) AS avg_material_condition
FROM equipment_quality
GROUP BY vendor
ORDER BY defect_rate_pct DESC;
