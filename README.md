# Marketing Mix Modeling

A Marketing Mix Modeling (MMM) project estimating how Google, Facebook, Email, and Affiliate spend drive weekly sales, using a panel-aware, holdout-validated modeling approach across five model families. See [`reports/RESULTS.md`](reports/RESULTS.md) for the full write-up and [`notebooks/`](notebooks) for the work itself.

## Overview

**Business question:** which media channels actually drive weekly sales, and how should budget be allocated across them? MMM estimates each channel's contribution from historical spend and sales data — as opposed to click-level attribution — while controlling for trend, seasonality, and division-level differences.

## Data

- `data/raw/sample_media_spend_data.csv` — the original data: 3,051 weekly rows.
- `data/processed/cleaned_with_spend.csv` — the same data with simulated media spend added per channel, plus derived year/month/season columns.

**This is panel data, not one time series:** 27 divisions, each observed weekly for 113 weeks (2018-01-06 to 2020-02-29). One division label ("Z") contained two distinct divisions under one name; it's split into `Z1`/`Z2` during loading (`mmm/data.py::fix_z_division`). Adstock and lag features are computed *within* each division's own week sequence, never across division boundaries — divisions differ in average weekly sales by roughly 20x, so every model includes division fixed effects (or, for the Bayesian model, a proper hierarchical geo structure).

**Media spend is simulated**, derived as `impressions / 1000 * CPM` using assumed CPM values per channel (Google $2, Email $1, Facebook $1.5, Affiliate $1.2). Every ROI and dollar-contribution figure below should be read as "what these models conclude from this simulated spend pattern," not a validated real-world return. Affiliate is a small channel — about 1% of total simulated spend ($78K of $8.3M over the training period) — which matters a great deal for how confidently its effect can be estimated (see Key Insights).

## Methodology

| Model | Notebook | Notes |
|---|---|---|
| Ridge & Lasso | `01_ridge_lasso.ipynb` | Adstock + Hill saturation + division fixed effects + month dummies, tuned by walk-forward CV. |
| LightGBM | `02_lightgbm.ipynb` | Same transformed features; fit both spend-only and with `sales_lag1`, to isolate its leakage effect. |
| XGBoost | `03_xgboost.ipynb` | Same feature set as LightGBM, for a fair comparison. |
| LightweightMMM (Bayesian) | `04_lightweight_mmm_bayesian.ipynb` | Geo-hierarchical model with full posterior uncertainty. |
| Comparison & insights | `05_model_comparison_and_insights.ipynb` | All models on one shared holdout, plus the cross-model ROI reconciliation. |

**Transforms used consistently across every model:** geometric-decay adstock (θ tuned by CV, applied within each division's own week sequence), Hill saturation (γ calibrated from training weeks only, so no test-period information leaks into the transform), and month seasonality dummies (which outperformed a smooth Fourier term in CV — the data has a sharp, short holiday spike a low-order sin/cos term is too smooth to capture).

**Shared evaluation:** every model is trained on the first 103 weeks and evaluated once on the same held-out final 10 weeks, so the comparison across models is genuinely apples-to-apples.

## How to reproduce

```bash
pip install -r requirements.txt
pip install --no-deps lightweight-mmm==0.1.9
```

`lightweight-mmm` pins an old matplotlib/seaborn/tensorflow combination with no Windows wheels; installing it separately with `--no-deps` avoids pip's resolver silently downgrading the rest of the stack (see the comment in `requirements.txt`, and `mmm/lmmm_compat.py` for the one small JAX compatibility patch this project carries).

Run the notebooks **in numeric order** from inside `notebooks/` — each imports shared code from `../mmm` and later notebooks read CSVs written by earlier ones:

```bash
jupyter notebook
```

Notebook 04 (the Bayesian model) runs real MCMC sampling and takes roughly 15–20 minutes on a CPU; everything else runs in under a minute. No Google Drive mount and no hardcoded personal paths — every notebook reads from `../data/` with a relative path.

## Results

Same holdout, same metrics, every model:

| Model | R² | RMSE | MAPE |
|---|---:|---:|---:|
| **Lasso** | **0.856** | **64,036** | **24.2%** |
| Ridge | 0.832 | 69,233 | 22.9% |
| XGBoost | 0.830 | 69,464 | 24.4% |
| LightGBM | 0.810 | 73,553 | 25.0% |
| LightweightMMM (Bayesian, geo) | 0.750 | 84,355 | 28.5% |

*For reference, adding `sales_lag1` (last week's actual sales) as a feature pushes XGBoost to 0.880 and LightGBM to 0.857 — but last week's sales isn't a lever a media planner can pull, so those variants are excluded from the table above and from the ROI conclusions below; see `notebooks/02`–`03`.*

## Key insights

Reconciled across Ridge, Lasso, LightGBM, and XGBoost's ROI estimates, plus the Bayesian model's posterior credible intervals:

| Channel | Ridge ROI | Lasso ROI | LightGBM ROI | XGBoost ROI | Bayesian ROI (90% interval) |
|---|---:|---:|---:|---:|---:|
| Google | 14.8 | 11.9 | 7.3 | 9.6 | 0.60 [0.55, 0.65] |
| Email | 98.9 | 166.0 | 51.8 | 56.3 | 0.48 [0.41, 0.58] |
| Facebook | 94.0 | 104.6 | 83.5 | 79.4 | 1.11 [0.48, 3.43] |
| Affiliate | 1054.7 | 0.0 | −71.8 | −1299.2 | 3.03 [0.16, 8.61] |

- **Facebook and Email are the most reliably positive channels** — positive ROI in every re-estimation, and the top-ranked pair in three of four non-Bayesian models. The Bayesian model agrees directionally but with much wider uncertainty on Facebook.
- **Google is a real, consistently positive driver, but never the most efficient per dollar.** It also carries roughly 65% of total media budget, so its absolute contribution is large while its ROI-per-dollar is unremarkable — and it's the channel every model, including the Bayesian one, agrees on most tightly (its 90% interval is barely half a point wide).
- **Affiliate's ROI cannot be reliably estimated from this dataset — and that is itself the finding.** Ridge estimates +1,055 ROI; LightGBM and XGBoost estimate strongly negative ROI; Lasso's cross-validated penalty shrinks its coefficient to exactly zero; and the Bayesian posterior gives it, by a wide margin, the widest credible interval of any channel (0.16 to 8.6 — a 50x spread). Affiliate is only ~1% of total spend, so there's barely enough variation in the data to pin down an effect; different models' regularization and priors resolve that weak signal differently. The actionable recommendation is to fund a real, sustained Affiliate test large enough to generate identifiable signal, rather than trust any single model's point estimate.

Full reconciliation, including why the models agree on everything except Affiliate, is in `reports/RESULTS.md`.

## Methodology note: baseline validation

As a validation step, every model was also compared against a trivial baseline — each division simply continuing at its own historical training-period average, with no media data at all. On this specific 10-week holdout, that baseline scores R²=0.885, ahead of every real model. This is not evidence the media models are wrong: the holdout sensitivity check in `notebooks/05` shows this apparent edge is a property of the *particular* 10-week window (which sits inside an unusually sharp post-holiday demand trough), not a general result — widen the holdout to 14 weeks and the same baseline's R² collapses to 0.28. With only ~2.2 years of history (one full holiday cycle), this is a useful caveat on how much confidence to place in any single fixed-window comparison, but it doesn't change which channels the models agree are working — see full detail in `reports/RESULTS.md`.

## Limitations & caveats

- **Media spend is simulated** (CPM-derived), not real platform billing data — every dollar-ROI figure here describes what a model concludes from this simulated pattern, not a validated real-world return.
- **~2.2 years of history, one full holiday cycle observed twice** — not enough to reliably separate genuine media effects from "the same channels happen to get more budget right before the one big seasonal event we have two examples of."
- **No experimental or geo-holdout spend variation** exists in the data, so no model here can fully separate correlation from a causal media effect. This is also why the Bayesian model's Hill-saturation curve shape is only weakly identified (r-hat up to ~1.4) even though its predictive accuracy is solid.
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
