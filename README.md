# Austin Animal Center — Outcomes & Intakes Analytics

End-to-end analytics project on ~12 years (Oct 2013 – May 2025) of Austin Animal
Center data, covering shelter intakes and outcomes. The goal is to turn raw
shelter records into models and insights that support adoption, length-of-stay,
euthanasia-risk, and capacity-planning decisions.

See [docs/project_goals.md](docs/project_goals.md) for the full set of seven
business problems and the planned notebook map.

## Data Sources

| Dataset  | Socrata ID  | Rows (approx.) | Span |
|----------|-------------|----------------|------|
| Outcomes | `gsvs-ypi7` | ~173,000       | 2013-10-01 → 2025-05-05 |
| Intakes  | `pyqf-r2dc` | ~190,000       | 2013-10-01 → 2025-05-05 |

Two data vintages are used:
- **`data/raw/`** — full historical CSV exports (2013-10-01 → 2025-05-05)
  downloaded from the Austin open-data portal UI. Title-Case schema.
- **`data/new_system_data/`** — newer records pulled live via the Socrata API
  (after 2025-05-05), cached locally so re-runs don't re-download.

Derived data (`data/processed/`) is regenerable from the notebooks and is
git-ignored.

## Progress

**Week 1 — Data loading, cleaning & merge** ✅ (in progress)

- [x] Set up project structure, `requirements.txt`, and goal definitions for 7
      analytics problems ([docs/project_goals.md](docs/project_goals.md)).
- [x] Loaded both full historical exports (~173k outcomes, ~190k intakes) from
      the portal CSVs.
- [x] Built the live Socrata API loader with local caching for post-2025-05-05
      records ([notebooks/new_system_data.ipynb](notebooks/new_system_data.ipynb)).
- [x] Cleaning pipeline in
      [notebooks/01_cleaning_eda.ipynb](notebooks/01_cleaning_eda.ipynb):
  - Aligned Title-Case CSV schema → snake_case target schema.
  - Parsed mixed-ISO datetime fields into date-only columns.
  - Split `Sex upon Outcome` / `Sex upon Intake` into `sex` +
    `spayed_neutered` flags; split `Color` into primary/secondary; derived
    `ispreviouslyspayedneutered`.
  - Engineered baseline features: `length_of_stay_days`, `age_at_intake_days`,
    `is_adopted`.
  - Backward `merge_asof` of intakes ← outcomes on `animal_id` so each outcome
    carries its triggering intake record → `data/processed/df_full_merged.csv`.

**Next up**
- [ ] EDA pass on the merged frame (distributions, missingness, leakage checks).
- [ ] `02_join_intake_outcome.ipynb` — formalize and validate the join logic.
- [ ] Problem 1 — adoption-outcome prediction baseline.

## Repository Layout

```
austin-animal-outcomes/
├── data/
│   ├── raw/               # full historical CSV exports (committed)
│   ├── new_system_data/   # live Socrata API cache (committed)
│   └── processed/         # derived/merged frames (git-ignored)
├── docs/
│   └── project_goals.md   # 7 business problems + notebook map
├── notebooks/
│   ├── 01_cleaning_eda.ipynb
│   └── new_system_data.ipynb
├── outputs/
├── src/
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```
