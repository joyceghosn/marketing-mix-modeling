# Results — Marketing Mix Modeling

This is the distilled write-up of the full analysis in [`notebooks/`](../notebooks). It replaces the original presentation deck, whose headline claims (Ridge R²=0.677 called "the best model so far" on slides 19 *and* 23, next to an unremarked LightGBM slide at R²=0.888; a Bayesian model with negative R² presented alongside real ROI figures; four different, contradictory verdicts on Affiliate) did not hold up under a consistent re-test. The original deck (`Presentation (2).pptx`) is kept locally for reference but is not part of this repo — every number below was re-derived from scratch on one shared, time-based holdout.

## Data

- **3,051 weekly rows** are not one time series — they're **panel data**: 27 divisions × 113 weeks each (2018-01-06 to 2020-02-29). One division ("Z") was mislabeled and actually contains two distinct divisions; it's split into Z1/Z2 (see `notebooks/00`).
- **Media spend is simulated**, derived from `impressions / 1000 * assumed CPM` per channel — not real platform billing data. Every ROI number below describes what these models conclude from that simulated spend pattern, not a validated real-world return.
- Media channels: Google, Email, Facebook, Affiliate. Affiliate is a small channel: **~1% of total spend** ($78K of $8.3M over the training period).

## Methodology

All models share: adstock (geometric decay, computed within each division), Hill saturation (gamma calibrated from training weeks only), a `log1p(sales)` target where relevant, and — critically — **the same time-based holdout**: the last 10 of 113 weeks, held out identically for every model, never trained on. See `notebooks/00` for why this matters (every original notebook either pooled all divisions into one sorted-by-date series or used a row-order split that was actually splitting by division, not time).

Models: Ridge & Lasso (linear, with division fixed effects), LightGBM and XGBoost (spend-only and with-`sales_lag1` variants, to isolate leakage), and a from-scratch-debugged LightweightMMM Bayesian geo-hierarchical model.

## Results — same holdout, same metrics, every model

| Model | R² | RMSE | MAPE |
|---|---:|---:|---:|
| Naive (division training-mean) | 0.885 | 57,232 | 21.9% |
| XGBoost (+ sales_lag1) | 0.880 | 58,519 | 19.8% |
| LightGBM (+ sales_lag1) | 0.857 | 63,762 | 21.0% |
| **Lasso** | **0.856** | **64,036** | **24.2%** |
| Ridge | 0.832 | 69,233 | 22.9% |
| XGBoost (spend-only) | 0.830 | 69,464 | 24.4% |
| LightGBM (spend-only) | 0.810 | 73,553 | 25.0% |
| LightweightMMM (Bayesian, geo, hill_adstock) | 0.750 | 84,355 | 28.5% |

*Rows marked "+ sales_lag1" use last week's actual sales as a feature — informative for a short-term nowcast, not usable for a forward media-budget decision (see notebooks/02-03). They're reported for comparison, not used for the ROI conclusions below.*

**The naive baseline — "assume each division continues at its historical average" — beats every model on this holdout**, including the ones given privileged access to last week's sales. This is a real finding, not a discarded failure: notebook 05 shows the fixed 10-week holdout sits entirely inside a sharp post-holiday demand trough (total sales roughly quadruple for two weeks around Black Friday, then crash through February). Sliding the holdout window shows the naive baseline's apparent edge is holdout-dependent — it collapses to R²=0.60 once the window reaches back far enough to include the holiday spike itself, where a model with real media/seasonal signal pulls ahead. With only ~2.2 years of history (one full holiday cycle), no model here can be said to reliably outperform a division-average baseline in general — only within specific holdout windows.

## The Bayesian model — from R² = −221 to a working, converged model

The original LightweightMMM attempts (see `notebooks/99`, the preserved trial-and-error history) never got a positive R², down to −221 in the worst run. Debugging turned up five stacked bugs: the panel was pooled into one series instead of using the library's own geo-hierarchical mode; media was scaled to `[0,1]` by max while the target was left unscaled; there was no train/test split at all; media priors were a flat, uninformative 0.5 for every channel regardless of budget size; and a real JAX/library incompatibility (`jnp.where` keyword arguments made positional-only) crashed every fit outright once the first four were fixed. All five are fixed in `notebooks/04`, with a small documented compatibility patch (`mmm/lmmm_compat.py`) in place of chasing the library's own broken, Windows-incompatible dependency pins.

The corrected model converges cleanly on its pooled, channel-level parameters (r-hat ≈ 1.0–1.1) and gets a genuine, positive out-of-sample R² of 0.75. Its Hill-saturation curve *shape* parameters and per-division decomposition remain weakly identified (r-hat up to ~1.4) — a known identifiability issue when spend never varies enough to show visible diminishing returns, worsened by the panel's ~20x between-division scale gap. **Predictive performance from this model is trustworthy; its precise per-channel ROI point estimates should be read as directional.**

## Key insights (reconciled across models)

- **Facebook and Email are the most reliably positive channels.** Positive ROI in every counterfactual re-estimation (Ridge, Lasso, LightGBM, XGBoost) and the top two ranked channels in three of four. This is the one part of the original slides' story that survives scrutiny.
- **Google is a real, positive, but never top-ranked driver.** Consistently positive across every model including the Bayesian one — it's just also the largest budget line (~65% of spend), so its ROI-per-dollar is unremarkable even though its absolute contribution is large.
- **Affiliate's ROI cannot be reliably estimated from this data — and that is the finding, not a gap.** Ridge says +1,055 ROI; LightGBM says −72; XGBoost says −1,299; Lasso's cross-validated penalty shrinks it to exactly **zero**; the Bayesian model gives it the widest relative uncertainty of the four channels. With only ~1% of total budget, there is barely enough spend variation to identify any effect. The actionable recommendation is to either fund a real, sustained Affiliate test big enough to generate identifiable signal, or stop citing any single model's Affiliate number as decision-grade.

## Limitations & what a real deployment would need

- Media spend is simulated (CPM-derived), not real platform billing — treat all dollar-ROI figures as illustrative of the modeling approach, not validated returns.
- ~2.2 years of history covers only two holiday cycles; which weeks land in a fixed holdout window measurably changes which model looks best (see above and `notebooks/05`).
- No experimental or geo-holdout spend variation exists in this data, so no model here can fully separate correlation from a causal media effect — this is also why the Bayesian model's saturation-curve shape is only weakly identified.
- `organic_views`/`paid_views` are website-traffic-style controls that may themselves be partly caused by media spend; they measurably improved holdout fit and are kept, but some of the credit they absorb may belong to media.
- A real deployment would want actual spend data, 3-4+ years of history, and either a geo lift-test or a documented, isolated budget change — especially before trusting any Affiliate-specific number.
