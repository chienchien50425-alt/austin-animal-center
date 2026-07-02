# Predicting Long-Stay Shelter Animals — Austin Animal Center (2013–2025)

> Flagging the dogs and cats most likely to get stuck in the shelter, **on the day they
> arrive**, so staff can step in early instead of reacting weeks too late.

---

## 1. The problem

Austin Animal Center is a large open-intake municipal shelter. It has been facing constant overcapacity and prolonged kennel stays since 2021, due to a pandemic-related decline in adoptions and spay/neuter procedures. Animals that stay a long time consume kennel space, staff hours, and medical cost.

The question this project answers: using only what is known the day an animal walks in, can we predict whether it will become a long-stay (>30 days) case? A reliable early flag lets staff prioritise foster placement, targeted marketing, or behavioural support before an animal lingers.

This is framed as a binary classification task with target is_long_stay (1 = stay > 30 days).

---

## 2. The data

Everything here is built on Austin Animal Center's own public records, published on the
[City of Austin Open Data Portal](https://data.austintexas.gov/). The shelter logs two things:
every animal that comes **in** (an *intake* record) and every animal that leaves, for any
reason (an *outcome* record). This project uses the full history from **October 2013 to May
2025** — roughly 174,000 intake events and 174,000 outcome events.

After matching each departure back to the arrival record and narrowing to dogs and
cats, the analysis runs on **162,465 animal stays**.

---

## 3. What the project delivers

The result is an **early-warning flag**: on intake day, each dog or cat is scored for how
likely it is to become a long-stay case, using only what's actually known at that moment.

In plain terms, **the flag catches roughly 7 to 8 out of every 10 animals that would go on to
get stuck** — while they still have their best shot at a fast placement. That's the number
that matters operationally: the earlier a future long-stay animal is identified, the more
levers (foster, marketing, behavioural help) the shelter still has to pull.

The operating threshold is chosen to **maximise F1** — the point that best balances catching as many future long-stay animals as possible (recall) against the operational load of chasing false alarms (precision). At that balance point the flag happens to lean toward recall (~0.71–0.75) over precision (~0.40), which suits a triage tool: an unnecessary early foster nudge costs lower than a missed long-stay animal costs real kennel-weeks. The flag is built to rank and prioritise.

For the technically inclined, here are the underlying numbers, back-tested on a held-out
future year (2024) the models never saw during development:

| Species | Model | AUC | Precision | Recall | F1 |
|---------|-------|-----|-----------|--------|-----|
| Dog | XGBoost | 0.707 | 0.404 | 0.711 | 0.515 |
| Dog | Logistic Reg. | 0.650 | 0.375 | 0.707 | 0.490 |
| Cat | XGBoost | 0.750 | 0.418 | 0.753 | 0.538 |
| Cat | Logistic Reg. | 0.712 | 0.461 | 0.669 | 0.546 |

*Scored on the 2024 test year at the chosen threshold (To maximise F1 on
the earlier 2022–2023 data).*

| <img width="400" alt="Untitled design" src="https://github.com/user-attachments/assets/1678d1b4-5bb0-4e99-8961-c1ae543d0242" /> | <img width="750" alt="image" src="https://github.com/user-attachments/assets/49c9121b-08a2-44f7-bcb0-5d6c7dc3fc02" /> |
| :---: | :---: |
| Figure 1. Test AUC by fold (2020–2024), per species and model. | Figure 2. Confusion matrices on the 2024 test fold, per species and model (at each model's operating threshold). |

---

## 4. How it works — data, modelling, and the judgment calls

<img width="1108" height="576" alt="Screenshot 2026-06-28 at 19 55 28" src="https://github.com/user-attachments/assets/44cce23f-80a4-4a6f-b77f-b1c3f05a611a" />
Figure 3. The five-stage data-flow pipeline, from raw Austin Open Data exports to per-species modeling

### The project runs as a five-stage data-flow pipeline

Stage 1 · Raw Data Sources — Two CSV exports from the City of Austin Open Data Portal (2013-10-01 → 2025-05-05): an Intakes file (173,812 rows, one row per intake event) and an Outcomes file (173,775 rows, one row per outcome event).  

Stage 2 · Clean & Merge (01_cleaning) — Python/pandas notebook that aligns both exports to a shared schema and backward-merges intakes with outcomes into one row per outcome event.  

Stage 3 · Processed Dataset — Writes df_full_merged.csv to disk (162,465 rows × 26 columns, filtered to dogs & cats). This single file is the source every downstream notebook reads from.  

Stage 4 · EDA (02_eda) — Exploratory analysis; its findings inform the feature choices used in modeling.  

Stage 5 · Modeling (03_modeling) — Predicts is_long_stay separately per species using Logistic Regression and XGBoost. Outputs predictions only.  

### Key judgment calls   

- Modelling dogs and cats separately.
- Each outcome is matched to its arrival with a **backward `merge_asof`** on animal ID. Every departure is joined to its *most recent prior* intake. Drop outcomes with no matching intake (~0.5%) and outcomes re-claiming the same intake.
- Long stay is derived from outcome, so *every* outcome-derived column is a leakage -> remove from features. 
- Random split would cause data leakage in time-series data, so the model is trained on data up to year $Y-1$ and tested on year $Y$, walking forward from 2020 to 2024.
  | fold | train years | test year | role |
  |---|---|---|---|
  | 1 | 2013–2019 | 2020 | test |
  | 2 | 2013–2020 | 2021 | test |
  | 3 | 2013–2021 | 2022 | **validation** (threshold decisions) |
  | 4 | 2013–2022 | 2023 | **validation** (threshold decisions + feature selection) |
  | 5 | 2013–2023 | 2024 | **reference** (test + interpretability) |

- The 2020–2022 test folds drive no decisions; they trace test AUC over time. They uncover the dog model's 2023 drop as a structural result and show the cat model stays flat across the same years.
- In 2023, the dog long_stay rate jumps structurally (Dog +0.096 from 2022) for reasons the recorded fields don't explain. Pooling 2022 (0.219) and 2023 (0.315) places the cut-point between the pre- and post-jump regimes, making it more robust for 2024 deployment. Cats show no jump but use the same rule for parity.
- Feature selection of breed and spay/neuter is decided by ablation on single 2023 fold. This holds on one year because an ablation compares AUC with vs. without a feature, and AUC is a ranking measure largely unaffected by a base-rate shift and the 2023 jump is broad-based (a uniform lift, not a structural reshuffle).
  - Keep *Cat breed*:  Mutual information against the target was **0.001**, which argues for dropping it, but mutual information misses interaction effects. Validation-fold ablation moved XGBoost AUC by **+0.0013** with breed included, effectively neutral.
  - Keep *Spay/neuter vs. age*:  These two are correlated (r ≈ 0.41 dog, 0.59 cat), raising a
  redundancy worry. Dropping the spay/neuter slightly lowered validation AUC for both species (Δ ≈ −0.003) in LR, drop will slightly hurt performance.
- Handling imbalance  
Long-stay cases are the minority class (dog ≈ 0.17-0.33, cat ≈ 0.17-0.34). Re-weighting the positive class (`class_weight='balanced'` for LR and `scale_pos_weight` for XGBoost). 

### Features — how each input is selected, encoded, or cleaned before modeling
 
| Feature handling | Dog | Cat | Where decided |
|---|---|---|---|
| Selected feature set | intake reason, breed, health condition, sex, age, `is_sn`, `is_mix` | same seven | MI / Pearson screening |
| Dropped features | colour (primary/secondary/pattern), intake month | same | Near-floor MI (≤ 0.0013) |
| Breed encoding | top-20 + Other | top-4 + Other | Top-4 covers 97.7% of cats; ablation test kept it |
| `is_sn` (spay/neuter) | kept | kept | Collinearity ablation (Δ AUC ≈ −0.003) |
| Age (XGBoost) | raw `age_at_intake_days` | same | Tree model, scale-invariant |
| Age (Logistic Reg.) | 8 buckets: <2mo … 15yr+ | same | EDA |
| Health condition | keep Normal / Injured / Sick / Nursing / Neonatal, rest → Other | same | Keep the well-populated levels, merge <1000 categories into 'Other' |
| Encoder fitting | one-hot + top-N breeds, re-fit per fold on train only | same | Prevents fold-to-fold leakage |
 
### Model & validation — training, tuning, and back-test setup
 
| Setting | Dog | Cat | Where decided |
|---|---|---|---|
| Long-stay threshold (target) | > 30 days | > 30 days |  From shelters announcement |
| Operating threshold (XGBoost) | 0.417 | 0.455 | Max-F1 on pooled 2022–2023 |
| Operating threshold (Logistic Reg.) | 0.446 | 0.517 | Same |
| Class imbalance | per-fold `scale_pos_weight` (XGB) / balanced weights (LR) | same | Recomputed per fold as base rate drifts |
| XGBoost tuning grid | depth ∈ {3, 5}, lr ∈ {0.03, 0.1}, n_estimators ≤ 800, early stop 30 | same | Nested CV (inner = last train year) |
| Logistic Reg. | L2, C = 1.0, max_iter = 2000 | same | Fixed regularised baseline |
| Back-test folds | test years 2020–2024, expanding window | same | Rolling-origin |
| Validation / test fold | feature decisions on 2023; headline on 2024 | same | 2024 never used for selection/tuning |

---

## 5. What actually drives a long stay

Drivers are read from **SHAP values on the reference-fold model** (train ≤ 2023), not impurity
importance (biased toward high-cardinality features like breed):

<img align="left" width="550" src="https://github.com/user-attachments/assets/7726dbe5-1f14-48e7-89c0-d18aed45dce3">

**Figure 4. SHAP summary for the dog XGBoost model**  

Each dot is one dog; x-position is that feature's push on the prediction (right = toward long-stay, left = toward faster exit). Color encodes the feature's value: for age, red = older; for 0/1 features (breed, owner-surrender, spay/neuter), red = positive.  

Age ranks first, and its red/blue spread on both sides is the non-monotonic effect. Pit Bull, owner-surrender and injured push toward long-stay; small popular breeds (Chihuahua, Dachshund, Miniature Schnauzer, Miniature Poodle) push the other way.

<br clear="left"/>
<br><br>

<img align="left" width="550" src="https://github.com/user-attachments/assets/732c1b6e-ebf2-490a-a052-1ccf9222a188" /> 

**Figure 5. SHAP summary for the cat XGBoost model**  

Each dot is one cat; x-position is the feature's push (right = toward long-stay, left = toward faster exit); color is the feature's value (for age, red = older; for 0/1 features, red = present).  

Age ranks first with a non-monotonic red/blue spread. Sex_Unknown: present (red) pushes strongly left; These intakes are overwhelmingly newborn kittens (median age 22 days) transferred out on day 0 (~86% transfer, ~0% adoption) — likely too young to be sexed at intake, and routed straight to foster/rescue rather than entering the shelter pipeline. In contrast, Owner-surrender, nursing and injured push toward long-stay.


<br clear="left"/>
<br><br>

<img align="left" width="550" src="https://github.com/user-attachments/assets/1fe58e34-d395-4f07-b8aa-7a9aed40e1d8" />

**Figure 6. Long-stay rate by age at intake, dogs vs cats**  

In EDA we can see the age effect is *non-monotonic* for both dog and cat: the youngest puppies (under ~2 months) carry the highest long-stay risk (~27%), risk collapses in adolescence (~5% at 2–6 months), climbs again through prime adulthood, then falls for seniors. Very young kittens are highest-risk (~40%), and risk *rises* again into the senior years rather than falling.

<br clear="left"/>

---

## 6. Limitation


---


## 7. Lesson learn


---

## 8. Pipeline & repository

```
01_cleaning.ipynb  →  02_eda.ipynb  →  03_modeling.ipynb
```

| Notebook | What it does |
|----------|--------------|
| **01_cleaning** | Aligns both raw exports to a shared snake_case schema; normalises mixed-timezone dates; backward `merge_asof` join of each outcome to its most recent prior intake; drops orphans/duplicates; parses sex/breed/colour; engineers the `is_long_stay` target and intake-time features; filters to dogs & cats → `df_full_merged.csv` (162,465 rows). |
| **02_eda** | Per-species univariate, bivariate (long-stay rate by feature), and temporal analysis; correlation heatmaps and mutual-information screening against the target. EDA only — no modelling. |
| **03_modeling** | Time-aware folds; per-species Logistic Regression + XGBoost; feature ablations on the validation fold; back-test on 2024; threshold selection; confusion matrices; SHAP interpretation. |

```
.
├── data/
│   ├── raw dataset/        # two raw Austin Open Data exports (intakes, outcomes)
│   └── processed/
│       └── df_full_merged.csv   # cleaned, merged, dog/cat-only (162,465 rows)
├── notebooks/
│   ├── 01_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
└── README.md
```

**Reproducing:**

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
# run notebooks in order: 01 → 02 → 03
# 01 writes data/processed/df_full_merged.csv, which 02 and 03 both read.
```

---

*Data: City of Austin Open Data Portal. Independent portfolio project.*
