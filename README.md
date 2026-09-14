# Marketing Mix Modeling

A from-scratch re-test of a Marketing Mix Modeling (MMM) project estimating how Google, Facebook, Email, and Affiliate spend drive weekly sales. This repo replaces an earlier version whose modeling had real, unresolved problems — a Bayesian model with R² as bad as −221 presented alongside its ROI numbers as if they were valid, four contradictory verdicts on Affiliate's ROI depending on which slide you read, and no consistent holdout across models. Everything here was re-run and re-validated on one shared evaluation setup; see [`reports/RESULTS.md`](reports/RESULTS.md) for the full write-up and [`notebooks/`](notebooks) for the work itself.

## Overview

**Business question:** which media channels actually drive weekly sales, and how should budget be allocated across them? MMM is used here to estimate each channel's contribution to sales from historical spend and sales data (as opposed to click-level attribution), while controlling for trend, seasonality, and other traffic.

## Data

- `data/raw/sample_media_spend_data.csv` — the original data: 3,051 weekly rows.
- `data/processed/cleaned_with_spend.csv` — the same data with simulated media spend added per channel (see below) and a few derived columns (year/month/season).

**This is panel data, not one time series**: 27 divisions, each observed weekly for 113 weeks (2018-01-06 to 2020-02-29). One division label ("Z") turned out to contain two distinct divisions' data under one name; it's split into `Z1`/`Z2` during loading (`mmm/data.py::fix_z_division`). Every model in this repo is trained and evaluated with that structure respected — adstock and lag features are computed *within* each division, never across division boundaries.

**Media spend is simulated, not real platform billing data.** It was derived as `impressions / 1000 * CPM`, using assumed CPM values per channel (Google $2, Email $1, Facebook $1.5, Affiliate $1.2). This is a reasonable approach for a learning project, but every ROI and dollar-contribution figure in this repo should be read as "what these models conclude from this simulated spend pattern," not as a validated real-world return. Affiliate is a small channel — about 1% of total simulated spend ($78K of $8.3M over the training period) — which turns out to matter a great deal for how confidently its effect can be estimated (see Key Insights).

Divisions differ in average weekly sales by roughly 20x, which is why every model here includes division fixed effects (or, for the Bayesian model, a proper hierarchical geo structure) rather than pooling divisions together.

## Methodology

| Model | Notebook | Notes |
|---|---|---|
| Naive baseline | `00_data_preparation.ipynb` | Predicts each division's own training-period average sales. The bar every other model needs to clear. |
| Ridge & Lasso | `01_ridge_lasso.ipynb` | Adstock + Hill saturation + division fixed effects + month dummies + log-transformed controls, tuned by walk-forward CV. |
| LightGBM | `02_lightgbm.ipynb` | Same transformed features; fit both with and without `sales_lag1` to isolate its leakage effect. |
| XGBoost | `03_xgboost.ipynb` | Finished (the original was an unfinished, effectively-random-split attempt); same feature set as LightGBM for a fair comparison. |
| LightweightMMM (Bayesian) | `04_lightweight_mmm_bayesian.ipynb` | Geo-hierarchical model, debugged from a −221 R² starting point — see below. |
| Comparison & insights | `05_model_comparison_and_insights.ipynb` | All of the above on one table, plus the cross-model ROI reconciliation. |
| Debugging history | `99_debugging_appendix_trial_and_error.ipynb` | The original, unedited trial-and-error notebook — kept for the real debugging work it shows, not as a source of results. |

**Transforms used consistently across every model:** geometric-decay adstock (θ tuned by CV, applied within each division's own week sequence), Hill saturation (γ calibrated from training weeks only, so no test-period information leaks into the transform), and month or Fourier seasonality terms (month dummies outperformed a smooth sin/cos term in CV — the data has a sharp, short holiday spike that a low-order Fourier term is too smooth to capture; see notebook 00).

**Shared evaluation:** every model is trained on the first 103 weeks and evaluated once on the last 10 weeks, held out identically across all models — a genuine change from the original notebooks, which mostly used in-sample fit or splits that (because the underlying panel structure wasn't recognized) weren't actually time-based at all.

**The Bayesian model, briefly:** the original attempts never achieved a positive R² (as bad as −221). This was root-caused to five stacked issues — pooling the panel into one series instead of using the library's own geo-hierarchical mode, inconsistent media/target scaling, no train/test split, uninformative flat priors, and a real JAX/library version incompatibility — and all five are fixed in notebook 04. The corrected model gets a genuine positive out-of-sample R² of 0.75, with honestly-reported convergence diagnostics (see Results). This is discussed in full in the notebook itself and in `reports/RESULTS.md`.

## How to reproduce

```bash
pip install -r requirements.txt
pip install --no-deps lightweight-mmm==0.1.9
```

`lightweight-mmm`'s own published metadata pins an old `matplotlib`/`seaborn`/`tensorflow` combination with no Windows wheels; installing it with everything else in one command makes pip's resolver fail or silently downgrade the rest of the stack. Installing it with `--no-deps` after the main requirements avoids that — see the comment block in `requirements.txt` for why, and `mmm/lmmm_compat.py` for the one small compatibility patch this project carries for it (a `jnp.where` keyword-argument incompatibility with modern JAX).

Then run the notebooks **in numeric order** from inside `notebooks/` (each one imports shared code from `../mmm`, and later notebooks read CSVs written by earlier ones):

```bash
jupyter notebook  # or: jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb notebooks/05*.ipynb
```

Notebook 04 (the Bayesian model) does real MCMC sampling and takes roughly 15-20 minutes on a CPU; everything else runs in well under a minute. No Google Drive mount, no hardcoded personal paths — every notebook reads from `../data/` with a relative path.

## Results

Every model, same holdout, same metrics — the comparison the original project never had:

| Model | R² | RMSE | MAPE |
|---|---:|---:|---:|
| Naive (division training-mean) | 0.885 | 57,232 | 21.9% |
| XGBoost (+ sales_lag1) | 0.880 | 58,519 | 19.8% |
| LightGBM (+ sales_lag1) | 0.857 | 63,762 | 21.0% |
| **Lasso** | **0.856** | **64,036** | **24.2%** |
| Ridge | 0.832 | 69,233 | 22.9% |
| XGBoost (spend-only) | 0.830 | 69,464 | 24.4% |
| LightGBM (spend-only) | 0.810 | 73,553 | 25.0% |
| LightweightMMM (Bayesian, geo) | 0.750 | 84,355 | 28.5% |

The naive baseline beats every model here — a genuinely surprising result explained in full in `reports/RESULTS.md` and `notebooks/05` (short version: the fixed holdout falls entirely inside an unusually sharp post-holiday demand trough, and with only ~2.2 years of history, no model can be shown to reliably beat a division-average baseline in general, only within specific holdout windows). `sales_lag1` rows are reported for comparison, not used for the ROI conclusions below, since last week's actual sales isn't a lever a media planner can pull.

## Key insights

- **Facebook and Email are the most reliably positive channels** — positive ROI across every re-estimation (Ridge, Lasso, LightGBM, XGBoost), and the top-ranked pair in three of four.
- **Google is a real, consistently positive driver, but never the most efficient per dollar** — it also carries ~65% of total media budget, so its absolute contribution is large while its ROI-per-dollar is unremarkable.
- **Affiliate's ROI cannot be reliably estimated from this dataset, and that is itself the finding.** Ridge estimates +1,055 ROI; LightGBM and XGBoost estimate strongly negative ROI (−72, −1,299); Lasso's cross-validated penalty shrinks its coefficient to exactly zero; the Bayesian posterior gives it the widest relative uncertainty of the four channels. Affiliate is ~1% of total spend — there is barely enough variation to identify an effect, so different models' regularization and priors resolve that near-zero signal differently. The actionable recommendation is to fund a real, sustained Affiliate test large enough to generate identifiable signal, rather than trust any single model's point estimate.

Full reconciliation, including why the four models disagree on Affiliate specifically and agree on everything else, is in `reports/RESULTS.md`.

## Limitations & caveats

- **Media spend is simulated** (CPM-derived), not real platform billing data — every dollar-ROI figure here describes what a model concludes from this simulated pattern, not a validated real-world return.
- **~2.2 years of history, one full holiday cycle observed twice** — not enough to reliably separate genuine media effects from "the same channels happen to get more budget right before the one big seasonal event we have two examples of." The holdout-sensitivity check in notebook 05 makes this concrete: which model looks best changes depending on which weeks fall in the evaluation window.
- **No experimental or geo-holdout spend variation** exists in the data, so no model here can fully separate correlation from a causal media effect. This is also why the Bayesian model's Hill-saturation curve *shape* is only weakly identified (r-hat up to ~1.4) even though its predictive accuracy is solid.
- **`organic_views`/`paid_views` controls may be partly caused by media spend themselves** (a "bad control" risk) — they measurably improve holdout fit and are kept, but some of the credit they absorb may rightfully belong to media channels.
- A real deployment would want actual platform spend data, several more years of history spanning multiple holiday cycles, and a geo lift-test or a documented, isolated budget change — especially before trusting any Affiliate-specific conclusion.

## Repository structure

```
data/
  raw/            original data
  processed/      cleaned data with simulated spend added
mmm/              shared code (data loading, transforms, metrics) used by every notebook
notebooks/        one notebook per modeling approach, numbered in narrative order
reports/          RESULTS.md (the distilled write-up), figures/, and per-model result CSVs
requirements.txt
LICENSE
```

## License

MIT — see [LICENSE](LICENSE).
