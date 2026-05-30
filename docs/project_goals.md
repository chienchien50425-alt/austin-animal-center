# Austin Animal Center — Analytics Project Goals

## Project Overview

**Data sources**
| Dataset | Socrata ID | Rows (approx.) | Key columns |
|---------|-----------|----------------|-------------|
| Outcomes | `gsvs-ypi7` | ~170,000 | `animal_id`, `outcome_status`, `outcome_date`, `intake_date`, `type`, `primary_breed`, `sex`, `spayed_neutered`, `date_of_birth` |
| Intakes | `pyqf-r2dc` | ~190,000 | `animal_id`, `source_date`, `source_name`, `intake_health_condition`, `found_address`, `ispreviouslyspayedneutered` |

**Date range:** October 2013 – May 2025 (~12 years)  
**Join key:** `animal_id` (one animal can have multiple intake and outcome records)  
**Unit of analysis:** Varies by problem — see each section below

---

## Problem 1 — Adoption Outcome Prediction

**Business Question:** Which animals entering the shelter are least likely to be adopted, and why?

**Why It Matters:**  
Shelter resources (staff time, medical care, kennel space) are finite. Identifying animals at high risk of non-adoption early allows staff to intervene — increase promotional visibility, arrange foster placement, or flag for rescue transfer — before the animal becomes a long-stay case.

**Opportunity:**  
Build a scoring model that ranks incoming animals by adoption likelihood. A low score triggers a proactive action (e.g., social media feature, rescue outreach) within the first week of intake.

**Datasets Required:** Outcomes (primary) + Intakes (join on `animal_id` to get `intake_health_condition` and `source_name`)

**Key Columns:**
- Features: `type`, `primary_breed`, `sex`, `spayed_neutered`, `age_at_intake_days` (derived), `intake_health_condition`, `source_name`, `has_name` (derived)
- Target: `adopted` (binary, derived from `outcome_group`)
- Leakage risk: `spayed_neutered` (animals are often fixed *during* adoption), `euthanasia_reason` (dropped)

**How to Measure:**
- Model: Logistic Regression baseline → XGBoost
- Metrics: AUC-ROC (primary), Precision-Recall curve, F1-score
- Feature importance: SHAP values to explain which factors drive low adoption likelihood

---

## Problem 2 — Length of Stay Prediction

**Business Question:** How long will a given animal stay in the shelter before an outcome?

**Why It Matters:**  
Length of stay (LOS) is a direct driver of shelter operating cost and capacity. A dog occupying a kennel for 90 days costs significantly more than one adopted in 7 days. Predicting LOS enables better bed management and earlier escalation for long-stay animals.

**Opportunity:**  
Flag animals predicted to stay >30 days within their first week. Use prediction to prioritize intervention (foster recruitment, behavior training, promotional campaigns).

**Datasets Required:** Outcomes only (`length_of_stay_days` = `outcome_date` − `intake_date`)

**Key Columns:**
- Features: `type`, `primary_breed`, `sex`, `age_at_intake_days`, `spayed_neutered`, `intake_month` (derived), `outcome_group`
- Target: `length_of_stay_days` (regression) or LOS bucket — Short (<7d) / Medium (7–30d) / Long (>30d) (classification)
- Note: Records with negative LOS or missing `outcome_date` must be excluded

**How to Measure:**
- Regression: RMSE, MAE, Median Absolute Error
- Classification: F1-score per bucket, confusion matrix
- Baseline: predict median LOS for each species/breed group

---

## Problem 3 — Euthanasia Risk Profiling

**Business Question:** Which animals are at highest risk of being euthanized, and what characteristics drive that outcome?

**Why It Matters:**  
Euthanasia is the most severe outcome. Early identification of at-risk animals creates an opportunity to intervene — medical treatment, emergency rescue transfer, or behavioral rehabilitation — before the decision is made.

**Opportunity:**  
A risk score surfaced at intake could trigger immediate escalation to rescue partners. Even reducing euthanasia by 5–10% represents hundreds of animals per year.

**Datasets Required:** Outcomes (primary) + Intakes (join for `intake_health_condition` — critical feature)

**Key Columns:**
- Features: `type`, `primary_breed`, `sex`, `age_at_intake_days`, `intake_health_condition`, `source_name`, `spayed_neutered`
- Target: binary — Euthanized (1) vs. All other outcomes (0)
- Leakage: `euthanasia_reason` is perfect leakage — already dropped in cleaning
- Note: Do NOT filter to domestic-only for this analysis; wildlife euthanasia patterns are also relevant

**How to Measure:**
- Model: Logistic Regression → XGBoost with class weighting (euthanized class is minority)
- Primary metric: **Recall** on euthanized class (missing a high-risk animal is the costliest error)
- Secondary: Precision, AUC-ROC, confusion matrix
- SHAP analysis: identify top drivers (e.g., intake condition, age, species)

---

## Problem 4 — Seasonal & Temporal Intake Trends

**Business Question:** When do intake spikes occur, and are they predictable enough to support capacity planning?

**Why It Matters:**  
Every spring/summer, shelters experience a surge in kitten and puppy intakes ("kitten season"). Without anticipating this, shelters are caught understaffed, underfunded, and over capacity — leading to worse outcomes for all animals.

**Opportunity:**  
Quantify seasonal patterns to support staffing schedules, foster recruitment campaigns, and supply procurement. Identify whether trends are worsening or improving over the 12-year period.

**Datasets Required:** Intakes (primary — `source_date` is the intake timestamp); Outcomes can be layered in for outcome-rate-by-month analysis

**Key Columns:**
- `source_date` → extract month, week, year, day-of-week
- `type`, `primary_breed` (kitten/puppy surge vs. adult)
- `source_name` (Stray vs. Owner Surrender seasonal patterns differ)
- Outcomes: `outcome_group`, `adopted` layered by intake month

**How to Measure:**
- Monthly intake volume time series with year-over-year overlay
- Seasonal decomposition (trend + seasonality + residual)
- Adoption rate by intake month (does kitten season lower individual adoption probability?)
- Statistical test: is the seasonal pattern statistically significant?

---

## Problem 5 — Repeat Intake (Recidivism) Analysis

**Business Question:** How often do animals return to the shelter after leaving, and what predicts re-entry?

**Why It Matters:**  
An animal adopted and returned represents a failed placement — wasted resources, stress for the animal, and a harder second adoption. Understanding re-entry rates and patterns can improve adoption screening, post-adoption support, and owner education.

**Opportunity:**  
Identify characteristics of animals with high re-entry rates. Use findings to recommend targeted post-adoption check-ins or flag high-risk adopter-animal mismatches.

**Datasets Required:** Both — Intakes AND Outcomes, joined on `animal_id`, sorted by date

**Key Columns:**
- `animal_id` — group by this to find repeat animals
- `source_date` (Intakes), `outcome_date` (Outcomes) — needed to establish intake/outcome sequence
- `source_name` — did the animal come back as a Stray or Owner Surrender?
- `outcome_group` from first visit — what happened before re-entry?
- Time between visits: `source_date[visit 2]` − `outcome_date[visit 1]`
- Note: Join requires date-based matching to avoid mixing unrelated intake/outcome pairs

**How to Measure:**
- Re-entry rate: % of `animal_id`s with 2+ intake records
- Time-to-return distribution: median days between outcome and next intake
- Re-entry rate by first outcome type: do adopted animals return less than transferred animals?
- Re-entry rate trend over years

---

## Problem 6 — Intake Condition → Outcome Pipeline

**Business Question:** How does an animal's health condition at intake affect its ultimate outcome?

**Why It Matters:**  
`intake_health_condition` (Healthy, Injured, Sick, Treatable, etc.) exists only in the Intakes dataset — it is absent from Outcomes. This means the question cannot be answered without joining both datasets. Answering it quantifies the ROI of medical investment and identifies whether certain conditions are systematically under-served.

**Opportunity:**  
If "Treatable" animals have a disproportionately high euthanasia rate, that signals a resource gap — more veterinary capacity or rescue partnerships could shift those outcomes. This finding has direct policy implications.

**Datasets Required:** Both joined on `animal_id` — `intake_health_condition` from Intakes, `outcome_group` / `adopted` from Outcomes

**Key Columns:**
- `intake_health_condition` (Intakes): Healthy / Injured / Sick / Treatable — Rehabilitable / Treatable — Manageable / etc.
- `outcome_group`, `adopted`, `length_of_stay_days` (Outcomes)
- `type`, `age_at_intake_days` (control variables)
- Join logic: match each outcome record to its corresponding intake record by `animal_id` and nearest prior `source_date`

**How to Measure:**
- Outcome distribution (stacked bar) by `intake_health_condition`
- Adoption rate and euthanasia rate by condition group
- Logistic regression: condition's odds ratio for adoption after controlling for age, type, breed
- LOS by condition: do sick animals stay longer even when adopted?

---

## Problem 7 — Geographic Hotspot Analysis

**Business Question:** Which neighborhoods or areas generate the most stray animal intakes, and how do outcomes differ by location?

**Why It Matters:**  
If stray intakes cluster in specific zip codes, targeted community intervention (free microchipping events, TNR programs for cats, owner education) in those areas could reduce shelter intake volume at the source.

**Opportunity:**  
Provide data-driven recommendations for where to deploy community outreach resources. Overlay with outcome data to identify areas where animals come in but are rarely reclaimed by owners (suggesting low microchipping rates or transient populations).

**Datasets Required:** Intakes (primary — `found_address`); Outcomes layered in for reclaim-rate-by-area analysis

**Key Columns:**
- `found_address` (Intakes) — requires zip code extraction or geocoding
- `source_name` — filter to `Stray` for geographic relevance (Owner Surrender has home address, not found address)
- `type`, `primary_breed` — species distribution by area
- Outcomes: `outcome_group` (specifically `Reclaimed` rate — proxy for owner engagement)
- `source_date` — temporal patterns by area

**How to Measure:**
- Intake count per zip code / neighborhood (choropleth map)
- Stray-to-reclaim rate by area: low reclaim = low owner engagement = intervention priority
- Year-over-year change in intake volume by area (is the problem growing?)
- Species composition by area (cat-heavy vs. dog-heavy neighborhoods need different interventions)

---

## Notebook Map

| Notebook | Problem(s) | Primary dataset |
|----------|-----------|-----------------|
| `01_cleaning_eda.ipynb` | Shared cleaning + Outcomes EDA | Outcomes + Intakes (separate) |
| `02_join_intake_outcome.ipynb` | Join logic + validation | Both |
| `03_adoption_prediction.ipynb` | Problem 1 | Joined |
| `04_los_prediction.ipynb` | Problem 2 | Outcomes |
| `05_euthanasia_risk.ipynb` | Problem 3 | Joined |
| `06_seasonal_trends.ipynb` | Problem 4 | Intakes |
| `07_repeat_intake.ipynb` | Problem 5 | Both (date-sequenced) |
| `08_intake_condition.ipynb` | Problem 6 | Joined |
| `09_geographic.ipynb` | Problem 7 | Intakes + geocoding |
