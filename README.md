# RxReview

## Pharmacist-Led Diabetes Discharge Review Decision Support

RxReview is a healthcare decision-support prototype designed to help hospitals prioritize patients with diabetes for pharmacist-led medication reconciliation and discharge counseling when pharmacist capacity is limited.

The application combines:

- 30-day readmission risk prediction
- medication complexity
- pharmacist-capacity constraints
- cost/value analysis
- model comparison
- source-grounded discharge guidance using RAG
- responsible-use and governance considerations

The goal is not to replace clinical judgment. RxReview helps identify which patients may benefit most from additional pharmacist review.

---

## Business Problem

Hospitals often have limited pharmacist capacity and cannot provide intensive discharge review to every patient.

RxReview addresses the question:

> Which patients should receive pharmacist-led medication reconciliation and discharge counseling before discharge when pharmacist capacity is limited?

The primary stakeholder is a Director of Pharmacy or Manager of Transitions-of-Care Pharmacy Services.

Additional stakeholders include:

- Clinical pharmacists
- Discharge-planning nurses
- Care-management leaders

---

## RxReview Priority Score

Patients are ranked using:

**RxReview Pharmacist-Review Priority Score = Predicted Readmission Risk × Medication Complexity Weight**

The score combines:

1. Calibrated probability of 30-day readmission
2. Medication complexity based on medication burden, diabetes medications, medication changes, insulin use, and insulin changes

The Medication Complexity Weight is a project-defined operational measure and is not a clinically validated medication-safety scale.

---

## Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UCI Machine Learning Repository.

The dataset is downloaded programmatically using:

```python
from ucimlrepo import fetch_ucirepo

diabetes = fetch_ucirepo(id=296)
