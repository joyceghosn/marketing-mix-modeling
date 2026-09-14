import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def walk_forward_folds(train_weeks, n_folds=3, min_train_weeks=52, val_weeks=10):
    """Rolling-origin folds within the training period only (never touches
    the true holdout). Fold k trains on weeks[:cut] and validates on the
    val_weeks right after cut."""
    weeks = np.sort(np.asarray(train_weeks))
    folds = []
    total = len(weeks)
    last_cut = total - val_weeks
    first_cut = min_train_weeks
    if n_folds == 1:
        cuts = [last_cut]
    else:
        cuts = np.linspace(first_cut, last_cut, n_folds).astype(int)
    for cut in cuts:
        tr = weeks[:cut]
        va = weeks[cut:cut + val_weeks]
        if len(va) == 0:
            continue
        folds.append((tr, va))
    return folds
