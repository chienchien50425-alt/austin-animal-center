# Challenges & Key Decisions

A log of the non-obvious problems hit while building the adoption model, and how each was
resolved. Every entry states the **tension**, the **decision**, and the **why** — these are
the calls a reviewer would otherwise question, so they are written down instead of buried in
code.

Scope: Problem 1 (adoption prediction). Pipeline
[01_cleaning](../notebooks/01_cleaning.ipynb) →
[02_eda](../notebooks/02_eda.ipynb) →
[03_modeling](../notebooks/03_modeling.ipynb), plus the side diagnostic
[survivorship_check](../notebooks/survivorship_check.ipynb).

---

## A. Defining the population & target

### 1. Which animals to remove — `Euthanasia Request` vs `Public Assist` intakes

**Tension.** Some intake reasons have extreme adoption rates. `Euthanasia Request` animals
are essentially never adopted; `Public Assist` dogs are adopted only ~17% of the time. Both
look like they "drag down" the model — should *both* be removed?

**Decision.** Remove **`Euthanasia Request`** (~243 rows); **keep `Public Assist`**.

**Why — they are not the same kind of row:**

| | `Euthanasia Request` | `Public Assist` |
|---|---|---|
| What it is | Owner surrenders the animal *specifically to be put down* | Owner needs help; animal often later reclaimed / returned to owner |
| Adoption rate | ≈ 0 (target fixed by construction) | Dogs 16.8% (n=8,339), Cats 35.2% (n=1,443) — low but real |
| Is it an adoption candidate? | No | Yes |
| Action | **Drop** | **Keep** |

- Keeping `Euthanasia Request` lets the model learn a **circular rule** ("intake reason =
  euthanasia → not adopted") that is true *by definition*. It inflates apparent accuracy
  while teaching nothing transferable. This is a **structural** exclusion — the target is
  fixed, not predicted.
- `Public Assist` animals *are* adoptable. Their low rate is **genuine signal recorded at
  intake** (before the outcome), which is exactly what the model should learn. Dropping them
  would discard real cases and bias the model toward an unrealistically easy population.

**Principle adopted:** *remove a group only when its target is undefined or fixed by
construction — not merely because its rate is low or extreme.* A low base rate is signal to
learn; a forced target is a row to exclude.

*Where:* filter in [03_modeling §2](../notebooks/03_modeling.ipynb); the per-reason adoption
breakdown that informed it is §9.2.

### 2. Survivorship bias / right-censoring — animals still in the shelter

**Tension.** The analysis table is built by a backward `merge_asof` on the **outcomes**
table, so only animals that *already have an outcome* appear. Animals still in the shelter
(no outcome yet) silently drop out. Since "slow to be adopted" correlates with "never
adopted," this could bias the target.

**Decision.** Quantify it before trusting any result; proceed without correction.

**Why.** The dedicated diagnostic
([survivorship_check](../notebooks/survivorship_check.ipynb)) found only **864 animals
(0.55%)** of 156k have no outcome, and they are **concentrated near the export cutoff** (the
normal "still being processed" tail), not spread across all years. The censored cohort is
tiny and time-localized, so the bias is negligible for a model trained on 2013–2024 — but it
is documented rather than assumed.

---

## B. Data & cleaning

### 3. Joining two tables where one animal has many records

**Tension.** An `animal_id` appears in *multiple* intake and *multiple* outcome events
(animals are surrendered, adopted, returned, re-surrendered). A naive merge mixes unrelated
intake/outcome pairs.

**Decision.** Backward `merge_asof` on `animal_id`, matching each outcome to its **nearest
prior** intake, so every outcome row carries the intake record that actually triggered it.

*Where:* [01_cleaning](../notebooks/01_cleaning.ipynb) → `data/processed/df_full_merged.csv`.

### 4. Parsing the free-text `color` field

**Tension.** `color` mixes a base color and a coat pattern in one messy string
("Brown Tabby", "Black/White", "Tricolor"), and dogs vs cats use different color
vocabularies (Brindle/Merle for dogs; Tabby/Calico/Tortie/Point for cats).

**Decision.** Split on `/`, then route through **species-specific** maps into a clean
`primary_color` (base) + `pattern`. Edge cases fixed along the way: `"Tricolor"` (the whole
string is a pattern keyword, leaving no base) and `"Brown Tabby"` (cat base color was
missing from the map).

*Where:* [01_cleaning](../notebooks/01_cleaning.ipynb). (Ultimately color was dropped as a
feature — see #7 — but the clean columns remain for EDA.)

---

## C. Feature selection

### 5. Leakage discipline — which columns can never be features

**Tension.** Several columns are only known *after* the outcome and would leak the answer:
`outcome_type`, `outcome_subtype`, `outcome_date`, `length_of_stay_days`, and
`spayed_neutered` *recorded at outcome* (animals are often fixed during adoption).

**Decision.** Maintain an explicit `LEAKAGE_COLS` block that is never selected as a feature;
use the **intake-time** spay/neuter flag (`is_previously_spayed_neutered`) only. Every
data-driven step (top-N breed lists, encoders, scalers) is fit on **training rows only**.

*Where:* [03_modeling §2, §6–§7](../notebooks/03_modeling.ipynb).

### 6. Pearson correlation can't score categorical features

**Tension.** Most predictors are categorical strings (`intake_reason`, `primary_breed`,
`primary_color`, …). Pearson correlation only handles numeric/binary columns, so the
correlation pass silently ignored the most important features.

**Decision.** Add a `mutual_info_classif` pass (ordinal-encoded, `discrete_features=True`)
to rank categorical features per species — used for *ranking*, not as proof of usefulness
(absolute MI values are all small here).

*Where:* [02_eda §4.3](../notebooks/02_eda.ipynb).

### 7. Is color actually useful? (No.)

**Tension.** Coat color *feels* predictive ("black dog syndrome"), and in cats it is
visually tied to sex (tortie/calico are almost always female).

**Decision.** **Drop** `primary_color`, `secondary_color`, `pattern` for both species.

**Why.** Mutual information was floor-level (< 0.002) and a with/without-color AUC test
showed no lift. In cats the apparent color signal is really the **sex** confound, which the
model already has. Carrying color added dimensions and a confound for zero gain.

*Where:* feature selection summarized in [03_modeling](../notebooks/03_modeling.ipynb)
overview.

### 8. `intake_reason` — strongest dog feature, or leakage?

**Tension.** `intake_reason` has the highest dog mutual information, but some reasons can
encode the outcome (the `Euthanasia Request` case from #1), so a big AUC contribution could
be *leakage* rather than signal.

**Decision.** **Keep it**, after an explicit with/without ablation on held-out test data —
and *do not auto-decide*; print the per-category breakdown for manual inspection.

**Why.** Removing it costs dogs ~0.05 AUC (LR +0.05 / XGB +0.06) and cats ~0.01 — a large,
consistent contribution. The breakdown shows the low-adoption categories (e.g. `Public
Assist` at 0.17) sit at *moderate* rates, not the 0/1 extremes that signal outcome-coupling,
and are recorded at intake. Verdict: legitimate intake-channel signal, not leakage.

*Where:* [03_modeling §9.1–§9.2](../notebooks/03_modeling.ipynb).

### 9. Cat `primary_breed` — when the metric and the model disagree

**Tension.** Mutual information for cat breed was 0.001 (≈ noise; nearly all cats are
Domestic Shorthair), which argues "drop it." But MI only measures *univariate* association.

**Decision.** **Keep** cat breed, decided on real held-out AUC rather than the MI score.

**Why.** A with/without ablation showed XGBoost *gains* from breed via interactions and rare
high-adoption breeds (Maine Coon, Snowshoe) that univariate MI averages away. The gain is
**small and fold-dependent** (≈ +0.015 on the 2023 fold, ≈ 0 on 2024) and never negative, so
it is kept as neutral-to-positive — but the fold-dependence is documented rather than
oversold.

*Where:* [03_modeling §9.3](../notebooks/03_modeling.ipynb).

### 10. `is_sn` × `age` collinearity in the linear model

**Tension.** Spay/neuter status and age are correlated (older animals are more often already
fixed). Collinearity is a classic worry for logistic regression.

**Decision.** **Drop `is_sn` from the LR** (keep it for XGBoost).

**Why.** Measured, the collinearity is **mild** — VIF 1.2 (dog) / 1.5 (cat), well under the
usual 5–10 flag — and with L2 regularization it **never moved the backtest AUC**. It only
inflates *coefficient variance*, distorting the §10.1 interpretation. Since dropping `is_sn`
from the LR costs ≈ 0 AUC (it is redundant with age for a linear model), removing it cleans
up the coefficients at no cost. Trees have no coefficient-variance issue, so XGBoost keeps it.

**Correction to a common assumption:** mild collinearity affects coefficient *reading*, not
predictive *performance* — so the headline AUCs were never in question.

*Where:* [03_modeling §9.4](../notebooks/03_modeling.ipynb).

---

## D. Modeling & evaluation

### 11. Dogs and cats don't behave alike

**Tension.** Pooling species into one model averages away opposite signals.

**Decision.** Train **separate** LR + XGBoost models per species.

**Why.** The drivers differ: **sex** dominates for cats (≈ the spay/neuter + tortie/calico
structure), **intake_reason** dominates for dogs, and age pushes in different directions.
Four models (LR/XGB × Dog/Cat) keep these legible.

### 12. Concept drift breaks the train/test split

**Tension.** Adoption rate drifts from ~0.42 (2014) to ~0.65 (2024). A random split would
put future, higher-adoption years into training and **leak the trend**, overstating
performance. Even a single 2013–22 / 2023–25 time split gives just one number and hides
year-to-year stability.

**Decision.** **Rolling-origin backtest** — expanding windows: for each test year *Y* in
2020–2025, train on all prior years (`< Y`), test on *Y*; report per-fold AUC plus a
mean ± std aggregate.

**Why.** This is the honest analogue of how the model is actually used (train on the past,
score the future). It also corrected an over-optimistic read: the single split gave Dog XGB
**0.762**, but the backtest mean is **0.741 ± 0.041** — the early folds (incl. a 2022 dog
trough) are harder and the single split happened to test only easier recent years. Cats are
stronger and steadier (XGB **0.810 ± 0.026**).

*Where:* [03_modeling §4, §7–§8](../notebooks/03_modeling.ipynb).

### 13. AUC alone is untrustworthy under drift

**Tension.** A model can rank well (good AUC) but output **miscalibrated** probabilities,
which matters because the use case is a *risk score*, not a ranking.

**Decision.** Report **calibration curves + Brier score**, pooled across backtest folds, not
just AUC.

*Where:* [03_modeling §8.2](../notebooks/03_modeling.ipynb).

---

## E. Tooling & reproducibility

### 14. Environment & stale notebook outputs

- **XGBoost / `libomp` load error** on macOS — resolved by running on a dedicated
  `aac-model` kernel with `DYLD_FALLBACK_LIBRARY_PATH` set, so `xgboost` + `shap` import
  cleanly.
- **Stale cell outputs** — after rewriting upstream cells, downstream outputs reflected an
  old state (`execution_count: None`). Fixed by re-executing the whole notebook end-to-end
  (`jupyter nbconvert --execute`) so every reported number matches the committed code.

---

## Recurring principle

Most of these decisions reduce to the same discipline: **don't let the model see what it
shouldn't, and don't trust a number you haven't stress-tested.** Remove rows only when the
target is undefined (not when it's merely hard); judge a feature by held-out AUC, not by a
single correlation/MI score; and evaluate the way the model will actually be used — past
predicting future.
