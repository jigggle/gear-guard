# GearGuard Quality Triage Summary

## Executive Summary

GearGuard identified high-risk equipment items, anomalous quality signatures, and drifted quality signals across the simulated football equipment inventory.

- Total equipment records reviewed: **5,000**
- Actual defect rate: **3.34%**
- Anomaly rate: **5.00%**
- High or critical risk items: **754**
- Major drift features: **3**
- Moderate drift features: **0**

## Highest-Risk Equipment Items

| Item ID | Type | Vendor | Season | Predicted Risk | Risk Tier | Anomaly Score | Recommended Action |
|---|---|---|---|---:|---|---:|---|
| EQ-00206 | shoulder_pads | Vendor_A | Spring_2025 | 100.0% | critical | 0.0870 | Remove from active rotation and inspect immediately. |
| EQ-03630 | shoulder_pads | Vendor_C | Spring_2025 | 100.0% | critical | 0.0837 | Remove from active rotation and inspect immediately. |
| EQ-01463 | shoulder_pads | Vendor_C | Fall_2025 | 100.0% | critical | 0.0776 | Remove from active rotation and inspect immediately. |
| EQ-04529 | shoulder_pads | Vendor_C | Fall_2024 | 100.0% | critical | 0.0194 | Remove from active rotation and inspect immediately. |
| EQ-01197 | shoulder_pads | Vendor_C | Fall_2024 | 99.9% | critical | 0.0685 | Remove from active rotation and inspect immediately. |
| EQ-02120 | shoulder_pads | Vendor_C | Fall_2024 | 99.9% | critical | 0.0472 | Remove from active rotation and inspect immediately. |
| EQ-00239 | helmet | Vendor_B | Spring_2025 | 99.9% | critical | 0.0998 | Remove from active rotation and inspect immediately. |
| EQ-03872 | shoulder_pads | Vendor_D | Spring_2025 | 99.9% | critical | 0.1050 | Remove from active rotation and inspect immediately. |
| EQ-01364 | helmet | Vendor_A | Fall_2024 | 99.9% | critical | 0.0441 | Remove from active rotation and inspect immediately. |
| EQ-01225 | shoulder_pads | Vendor_B | Fall_2024 | 99.8% | critical | 0.0907 | Remove from active rotation and inspect immediately. |

## Top Drifted Quality Signals

| Feature | Drift Status | Baseline Mean | Current Mean | KS Statistic | PSI | Interpretation |
|---|---|---:|---:|---:|---:|---|
| inspection_score | major_drift | 80.2181 | 78.3046 | 0.0966 | 0.0707 | inspection_score decreased; inspection outcomes changed. |
| anomaly_score | major_drift | -0.0672 | -0.0627 | 0.0683 | 0.0364 | anomaly_score increased; unusual equipment signatures became more or less common. |
| predicted_defect_risk | major_drift | 0.2145 | 0.1820 | 0.0650 | 0.0349 | predicted_defect_risk decreased; overall model-estimated quality risk shifted. |
| laundry_cycles | stable | 12.1746 | 11.2246 | 0.0428 | 0.0145 | laundry_cycles decreased; feature distribution changed. |
| age_days | stable | 290.7699 | 284.1883 | 0.0381 | 0.0161 | age_days decreased; feature distribution changed. |

## Example Item-Level Investigation Notes

### EQ-00206 — shoulder_pads

- Vendor: **Vendor_A**
- Batch: **Vendor_A_B03**
- Season: **Spring_2025**
- Predicted defect risk: **100.0%**
- Risk tier: **critical**
- Anomaly score: **0.0870**
- Recommended action: **Remove from active rotation and inspect immediately.**

EQ-00206 (shoulder_pads) was flagged as critical risk with predicted defect probability of 100.0%. The strongest risk drivers were: repair_count, inspection_score, material_condition_score, game_exposure_count. Recommended action: review inspection history, verify physical condition, and compare against similar items from the same equipment type or vendor.

### EQ-03630 — shoulder_pads

- Vendor: **Vendor_C**
- Batch: **Vendor_C_B18**
- Season: **Spring_2025**
- Predicted defect risk: **100.0%**
- Risk tier: **critical**
- Anomaly score: **0.0837**
- Recommended action: **Remove from active rotation and inspect immediately.**

EQ-03630 (shoulder_pads) was flagged as critical risk with predicted defect probability of 100.0%. The strongest risk drivers were: usage_sessions, inspection_score, practice_exposure_count, material_condition_score. Recommended action: review inspection history, verify physical condition, and compare against similar items from the same equipment type or vendor.

### EQ-01463 — shoulder_pads

- Vendor: **Vendor_C**
- Batch: **Vendor_C_B02**
- Season: **Fall_2025**
- Predicted defect risk: **100.0%**
- Risk tier: **critical**
- Anomaly score: **0.0776**
- Recommended action: **Remove from active rotation and inspect immediately.**

EQ-01463 (shoulder_pads) was flagged as critical risk with predicted defect probability of 100.0%. The strongest risk drivers were: inspection_score, material_condition_score, usage_sessions, game_exposure_count. Recommended action: review inspection history, verify physical condition, and compare against similar items from the same equipment type or vendor.

### EQ-04529 — shoulder_pads

- Vendor: **Vendor_C**
- Batch: **Vendor_C_B07**
- Season: **Fall_2024**
- Predicted defect risk: **100.0%**
- Risk tier: **critical**
- Anomaly score: **0.0194**
- Recommended action: **Remove from active rotation and inspect immediately.**

EQ-04529 (shoulder_pads) was flagged as critical risk with predicted defect probability of 100.0%. The strongest risk drivers were: repair_count, inspection_score, usage_sessions, material_condition_score. Recommended action: review inspection history, verify physical condition, and compare against similar items from the same equipment type or vendor.

### EQ-01197 — shoulder_pads

- Vendor: **Vendor_C**
- Batch: **Vendor_C_B13**
- Season: **Fall_2024**
- Predicted defect risk: **99.9%**
- Risk tier: **critical**
- Anomaly score: **0.0685**
- Recommended action: **Remove from active rotation and inspect immediately.**

EQ-01197 (shoulder_pads) was flagged as critical risk with predicted defect probability of 99.9%. The strongest risk drivers were: inspection_score, usage_sessions, practice_exposure_count, fit_complaint_count. Recommended action: review inspection history, verify physical condition, and compare against similar items from the same equipment type or vendor.

## Retrieved Inspection Guidance

The following operational guidance was referenced from local documentation:

- `docs/equipment_inspection_sop.md`
- `docs/drift_response_guide.md`

### Inspection SOP Excerpt

```text
## Risk Tiers

### Critical

Recommended action:

- Remove item from active rotation immediately.
- Perform full physical inspection.
- Check prior repair history.
- Compare against similar items from the same vendor or batch.
- Do not return item to use until cleared.

### High

Recommended action:

- Prioritize item for inspection before next usage cycle.
- Review condition, repair history, fit complaints, and usage exposure.
- Monitor similar items from the same equipment type or vendor.

### Medium

Recommended action:

- Monitor during routine inspection.
```

### Drift Response Excerpt

```text
## Drift Severity

### Major Drift

Recommended action:

- Review the affected feature immediately.
- Check whether the shift is isolated or part of a broader quality pattern.
- Compare by equipment type, vendor, batch, and season.
- Review recent operational changes.

### Moderate Drift

Recommended action:

- Monitor the feature during the next reporting cycle.
- Check if related features are also moving.
- Review whether the shift affects defect risk or anomaly scores.

### Stable

Recommended action:

- No immediate action required.
- Continue routine monitoring.
```

## Notes

This summary is generated from model outputs, anomaly scores, drift reports, and local inspection guidance. It is intended for decision support and should not replace manual review by trained equipment staff.
