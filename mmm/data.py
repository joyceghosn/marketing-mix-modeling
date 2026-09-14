"""Data loading and cleaning for the MMM project.

Key fix vs. the original notebooks: this is PANEL data (27 divisions x 113
weeks), not one 3,051-week series. All feature engineering below is done
per-division so adstock/lags never bleed across division boundaries, and the
mislabeled "Z" division (which is actually two distinct series sharing one
label) is split into Z1/Z2.
"""
import numpy as np
import pandas as pd

MEDIA_CHANNELS = ["spend_google", "spend_email", "spend_facebook", "spend_affiliate"]
CONTROL_VARS = ["organic_views", "paid_views"]
TARGET = "sales"
N_HOLDOUT_WEEKS = 10  # last 10 of 113 weeks reserved, never trained on
DEFAULT_DATA_PATH = "../data/processed/cleaned_with_spend.csv"  # relative to notebooks/


def load_raw(path=DEFAULT_DATA_PATH):
    # NOTE: do not re-sort here. The source file is already grouped by
    # division-then-week, and fix_z_division() relies on that original row
    # order (the two mislabeled "Z" sub-series are contiguous blocks that
    # would otherwise interleave once sorted by calendar_week alone).
    df = pd.read_csv(path, parse_dates=["calendar_week"])
    return df


def fix_z_division(df):
    """Split the mislabeled 'Z' division (226 rows = two distinct 113-week
    series sharing one label) into Z1 and Z2. Must run before any resort of
    the frame by calendar_week."""
    df = df.copy()
    z_mask = df["division"] == "Z"
    z_idx = df.index[z_mask].to_numpy()
    assert len(z_idx) == 226, f"expected 226 Z rows, got {len(z_idx)}"
    block1, block2 = z_idx[:113], z_idx[113:]
    # sanity: both blocks cover the same 113 calendar weeks
    w1 = df.loc[block1, "calendar_week"].to_numpy()
    w2 = df.loc[block2, "calendar_week"].to_numpy()
    assert (w1 == w2).all(), "Z blocks do not share the same week sequence"
    df.loc[block1, "division"] = "Z1"
    df.loc[block2, "division"] = "Z2"
    return df


def load_panel(path=DEFAULT_DATA_PATH):
    df = load_raw(path)
    df = fix_z_division(df)
    df = df.sort_values(["division", "calendar_week"]).reset_index(drop=True)
    n_div = df["division"].nunique()
    n_week = df["calendar_week"].nunique()
    assert len(df) == n_div * n_week, "panel is not balanced after cleaning"
    return df


def apply_adstock_grouped(df, cols, theta, group_col="division"):
    """Geometric-decay adstock, reset at each group (division) boundary."""
    out = pd.DataFrame(index=df.index)
    for col in cols:
        def _adstock(s):
            vals = s.to_numpy(dtype=float)
            res = np.empty_like(vals)
            prev = 0.0
            for i, v in enumerate(vals):
                prev = v + theta * prev
                res[i] = prev
            return pd.Series(res, index=s.index)
        out[col] = df.groupby(group_col, sort=False)[col].apply(_adstock).droplevel(0)
    out = out.reindex(df.index)
    return out


def hill_saturation(x, gamma, alpha=1.0):
    x = np.clip(x, 0, None)
    return (x ** alpha) / (x ** alpha + gamma ** alpha)


def add_seasonality(df, week_col="calendar_week", period=52.0):
    df = df.copy()
    # absolute week-of-year based seasonality, shared across divisions since
    # all divisions observe the *same* calendar weeks
    woy = df[week_col].dt.isocalendar().week.astype(float)
    df["sin_52"] = np.sin(2 * np.pi * woy / period)
    df["cos_52"] = np.cos(2 * np.pi * woy / period)
    return df


def time_split_weeks(df, n_holdout=N_HOLDOUT_WEEKS, week_col="calendar_week"):
    weeks = np.sort(df[week_col].unique())
    train_weeks = weeks[:-n_holdout]
    test_weeks = weeks[-n_holdout:]
    return train_weeks, test_weeks


def build_transformed_features(df, theta, gamma_frac, train_weeks, week_col="calendar_week"):
    """Adstock + Hill-saturate media channels; gamma is calibrated from
    TRAIN-ONLY max so no test-period information leaks into the transform."""
    adstocked = apply_adstock_grouped(df, MEDIA_CHANNELS, theta)
    train_mask = df[week_col].isin(train_weeks)
    feat = pd.DataFrame(index=df.index)
    for col in MEDIA_CHANNELS:
        gamma = adstocked.loc[train_mask, col].max() * gamma_frac
        feat[col + "_sat"] = hill_saturation(adstocked[col], gamma)
    return feat


def month_dummies(df, week_col="calendar_week"):
    """Month fixed effects. Chosen over a single sin/cos(52) harmonic after
    CV comparison: the sharp Nov-Dec holiday spike / Jan-Feb crash in this
    data is too abrupt for a smooth low-order Fourier term to capture."""
    month = df[week_col].dt.month
    return pd.get_dummies(month, prefix="m", drop_first=True).astype(float)


def log_controls(df):
    """log1p of the control variables. organic_views/paid_views are extremely
    right-skewed (min=1, max>500,000) -- log1p them before scaling, which CV
    showed generalizes much better on the holdout than using them raw."""
    return np.log1p(df[CONTROL_VARS]).rename(columns=lambda c: c + "_log")


def division_dummies(df):
    return pd.get_dummies(df["division"], prefix="div", drop_first=True).astype(float)
