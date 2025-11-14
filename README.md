# Marketing Mix Modeling (MMM) – Sales & Media Effectiveness

##  Project Overview

This project builds and compares several **Marketing Mix Models (MMM)** to understand how different media channels (Google, Facebook, Email, Affiliate) and organic activity contribute to weekly sales.

The project combines:

- A **Bayesian MMM** using **LightweightMMM (Google)**  
- A **regularized linear MMM** using **Ridge/Lasso** with Adstock & Saturation  
- A **non-linear benchmark model** using **LightGBM**

The goal is to:
1. Quantify the incremental impact of each media channel on sales  
2. Correct for carryover effects (adstock) and saturation  
3. Compare models in terms of accuracy, interpretability, and business insight  
4. Provide guidance on **which channels drive sales and how stable they are over time**

---

## Dataset

- File: `cleaned_with_spend.csv`  
- Size: 3,051 rows × 17 columns  
- Granularity: **Weekly** observations (per division)  
- Main columns:
  - `calendar_week` – week timestamp  
  - `sales` – target variable  
  - Media **impressions**:
    - `google_impressions`, `email_impressions`, `facebook_impressions`, `affiliate_impressions`
  - Media **spend**:
    - `spend_google`, `spend_email`, `spend_facebook`, `spend_affiliate`
  - Other controls:
    - `paid_views`, `organic_views`, `overall_views`, `year`, `month`, `season`

---

## Project Structure

- `lightweightMMMpart(1).ipynb`  
  *Data loading, preprocessing, scaling, and first LightweightMMM setup.*

- `lightweightMMMpart(2).ipynb`  
  *Fitting Bayesian MMM with LightweightMMM, extracting channel contributions, running rolling-window analysis, and analyzing stability of channel effects over time.*

- `Ridgelasso.ipynb`  
  *Classical MMM using Ridge & Lasso with: adstock transformation, Hill-type saturation, Fourier seasonality, interaction terms, and detailed coefficient/ROI interpretation.*

- `lightMGBM.ipynb`  
  *Non-linear benchmark using LightGBM to compare performance with the regression-based MMM.*

- `cleaned_with_spend.csv`  
  *Final cleaned dataset used across all models.*

You can follow the project in this logical order:

1. **Data & Bayesian MMM** → `lightweightMMMpart(1).ipynb` → `lightweightMMMpart(2).ipynb`  
2. **Custom Ridge/Lasso MMM** → `Ridgelasso.ipynb`  
3. **Tree-based benchmark** → `lightMGBM.ipynb`  

---

## Techniques Used

### Data Preparation & Feature Engineering
- Time-indexing by `calendar_week`
- Handling divisions and aggregating relevant metrics
- Train/validation/test splits with **time-series aware** (no shuffle) splitting
- Feature scaling / normalization (e.g. 0–1 range)
- Creation of **seasonality features**:
  - Fourier terms (sin/cos) and/or month/season variables

### MMM Transformations
- **Adstock transformation** to model carry-over effects  
  - Tuning the decay parameter θ (how long the effect of spend persists)
- **Hill / saturation function** to capture diminishing returns of media spend
- Testing different combinations of θ and γ to balance fit & interpretability

### Modeling
- **Bayesian MMM (LightweightMMM)**
  - Uses **Google’s LightweightMMM** library
  - Includes adstocked media channels + control variables
  - Generates posterior distributions for channel contributions
  - Rolling-window evaluation to see contribution stability over time

- **Ridge & Lasso Regression**
  - `RidgeCV`, `LassoCV` with cross-validated regularization
  - Models run on transformed media variables (adstocked + saturated)
  - Addition of interaction terms and seasonality
  - Evaluation on hold-out test set

- **LightGBM**
  - Gradient boosting trees as a non-linear benchmark
  - Used to check whether more complex interactions significantly improve performance vs MMM with hand-crafted transformations

### Evaluation & Diagnostics
- Metrics:
  - **R²**, **RMSE**, and sometimes **MAPE**
- Visual diagnostics:
  - Actual vs predicted sales curves
  - Residual analysis
  - Channel contribution plots over time
- Stability analysis:
  - Coefficient of variation (CV) of contributions per channel across rolling windows  
  - Identifies which channels are stable vs volatile

---

## Key Results & Insights

### 1. Ridge MMM with Adstock, Saturation & Seasonality
From the optimized Ridge model with interaction terms (see `Ridgelasso.ipynb`):

- **R² ≈ 0.68**
- **RMSE ≈ 71k** on the test set  
- The predicted sales curve closely tracks the actual sales, indicating a **good fit without extreme overfitting**.

**Channel insights (example interpretation from the final Ridge model):**

- `spend_facebook`  
  → Largest positive coefficient → **strongest driver of incremental sales**.  
- `spend_google`  
  → Strong positive contributor with slightly lower impact than Facebook.  
- `spend_email`  
  → Positive and meaningful contributor; performs well but behind Google/Facebook.  
- `spend_affiliate`  
  → Negative or weak impact in multiple specifications → potential **inefficient spend / cannibalization**.  
- `organic_views`  
  → Positive association with sales, capturing brand or baseline demand.

**Seasonality & interactions:**

- Fourier and interaction terms improved the model’s explanatory power compared to a pure linear baseline.  
- Interactions between media and seasonality captured periods where channels became temporarily more effective.

### 2. Bayesian MMM (LightweightMMM)

Using `lightweightMMMpart(1).ipynb` and `lightweightMMMpart(2).ipynb`:

- LightweightMMM produced **posterior distributions of weekly contributions** for each channel.  
- **Rolling-window analysis** showed:

  - **Affiliate**: most **stable** and consistently positive contribution (low coefficient of variation).  
  - **Facebook**: highly **volatile** channel with spikes in contribution → suggests campaign-driven bursts of effectiveness.  
  - **Email**: fluctuating impact, potentially tied to seasonal or campaign-based sends.  
  - **Google**: moderate variability; contribution increased for some periods then declined, possibly indicating **channel fatigue or strategy change**.

This approach gave a **probabilistic view** of media performance instead of single point estimates, useful for **risk-aware decision making**.

### 3. LightGBM Benchmark

- LightGBM served as a **non-linear reference model**.  
- It generally achieved strong R² while automatically learning interactions and non-linearities.  
- However, compared with Ridge MMM and LightweightMMM, its **interpretability is lower**, and channel-level ROI is harder to extract clearly.

---

##  Business Takeaways

- Facebook and Google tend to be the **key incremental drivers** of sales, but Facebook’s effect is **spiky** and campaign-dependent.  
- Email performs solidly and may be a **cost-effective, supportive channel**, especially when coordinated with other media.  
- Affiliate spend shows signs of **low or negative ROI**, suggesting that budget could be re-allocated or the program redesigned.  
- Seasonality and interactions matter: campaigns perform differently across months and seasons, and models that include these effects are more realistic.

---

##  Challenges & Lessons Learned

- **Environment & dependencies**  
  - Getting the correct versions of `numpy`, `scipy`, `jax`, `jaxlib`, and `lightweight-mmm` to work together required several iterations.
  - Balancing Colab vs local environment and avoiding version conflicts was a non-trivial part of the project.

- **Adstock & Saturation tuning**  
  - There is no single “correct” θ or γ.  
  - A grid search and intuition about realistic media carry-over were both required.

- **Trade-off: accuracy vs interpretability**  
  - LightGBM can capture complex patterns but is harder to interpret for media budgeting.  
  - Ridge MMM and LightweightMMM give clearer channel-level insights, even when their raw accuracy is similar.

- **Time-series leakage**  
  - Special care was taken to **avoid shuffling** when splitting train/test, to respect temporal ordering and avoid data leakage.

---

## Tools & Libraries

- **Python**  
- **pandas**, **numpy**, **scipy**  
- **scikit-learn** – Ridge/Lasso, metrics, train/test split  
- **LightGBM** – gradient boosting model  
- **LightweightMMM** – Bayesian Marketing Mix Modeling  
- **matplotlib**, **seaborn**  – visualization  
- **Google Colab** – experimentation environment

---

## Skills Demonstrated

- Marketing Mix Modeling (MMM)
- Bayesian modelling & probabilistic interpretation (LightweightMMM)
- Regularized regression (Ridge/Lasso) with feature engineering
- Time-series aware train/test splitting and evaluation
- Adstock & saturation transformation design and tuning
- Seasonality modelling (Fourier terms, calendar features)
- Rolling-window diagnostics and contribution stability analysis
- Model comparison: accuracy vs interpretability
- Python, data analysis, and visualization best practices

---

##  About the Author

I’m a Master’s student in Data Science, passionate about **analytics, predictive modeling, and data-driven marketing decisions**.  
This project is part of my ongoing work to connect **statistical modeling** with **real business impact**.
