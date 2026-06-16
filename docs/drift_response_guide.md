# Drift Response Guide

## Purpose

This guide describes how to respond when GearGuard detects feature drift across seasons or operational windows.

Drift means the distribution of a quality signal has changed between a baseline period and a current period. Drift does not automatically mean a defect has occurred, but it should trigger review.

---

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

---

## Common Drift Interpretations

### Inspection Score Drift

A decrease in inspection scores may indicate:

- Increased equipment wear
- Lower maintenance quality
- Vendor or batch quality issues
- Changes in inspection standards
- Higher usage intensity

### Anomaly Score Drift

A change in anomaly scores may indicate:

- More unusual equipment patterns
- New vendor behavior
- Different usage conditions
- Under-inspection or inconsistent maintenance

### Predicted Defect Risk Drift

A change in predicted defect risk may indicate:

- Quality risk is shifting across the inventory
- Usage or repair patterns have changed
- The model is seeing different feature distributions than the baseline

---

## Investigation Steps

When drift is detected:

1. Identify the drifted feature.
2. Compare baseline and current means.
3. Review PSI and KS-test results.
4. Segment by equipment type.
5. Segment by vendor.
6. Check whether defect rate changed.
7. Check whether anomaly rate changed.
8. Review high-risk examples.
9. Decide whether inspection priorities need to change.
