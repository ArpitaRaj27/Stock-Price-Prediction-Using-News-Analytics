"""Decay curve fitting and half-life estimation.

Model: CAR(tau) = A * exp(-lambda * tau) + C,  for tau >= 0
  - A:      initial displacement (the "shock" size)
  - lambda: decay rate (1/days)
  - C:      asymptote (the permanent price impact)

Half-life (days until half the initial displacement is absorbed):
    t_half = ln(2) / lambda

Bootstrap confidence intervals come from resampling events (not observations
within an event) — this accounts for event-level randomness, which is the
relevant source of uncertainty for our research question.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src import config

log = logging.getLogger(__name__)


def decay_model(tau: np.ndarray, A: float, lam: float, C: float) -> np.ndarray:
    return A * np.exp(-lam * tau) + C


def fit_decay(
    tau: np.ndarray,
    car: np.ndarray,
    p0: tuple[float, float, float] | None = None,
) -> dict:
    """Fit the decay model. Returns dict with params, half_life, residuals."""
    # Only fit post-event (tau >= 0)
    mask = tau >= 0
    x, y = tau[mask], car[mask]

    if len(x) < 4 or np.all(np.isnan(y)):
        return {"A": np.nan, "lambda": np.nan, "C": np.nan, "half_life": np.nan, "rss": np.nan}

    # Initial guesses
    if p0 is None:
        # Assume starting displacement ~= CAR at tau=0, asymptote ~= final, lambda ~= 0.3
        A0 = float(y[0]) if not np.isnan(y[0]) else 0.01
        C0 = float(np.nanmean(y[-3:]))
        p0 = (A0, 0.3, C0)

    try:
        popt, _ = curve_fit(decay_model, x, y, p0=p0, maxfev=5000)
        A, lam, C = popt
        # Reject degenerate fits
        if lam <= 0 or lam > 10:
            return {"A": A, "lambda": lam, "C": C, "half_life": np.nan, "rss": np.nan}
        half = float(np.log(2) / lam)
        resid = y - decay_model(x, *popt)
        rss = float(np.nansum(resid ** 2))
        return {"A": float(A), "lambda": float(lam), "C": float(C), "half_life": half, "rss": rss}
    except (RuntimeError, ValueError) as e:
        log.warning(f"Decay fit failed: {e}")
        return {"A": np.nan, "lambda": np.nan, "C": np.nan, "half_life": np.nan, "rss": np.nan}


def bootstrap_decay(
    cars: pd.DataFrame,
    group_value: str | None = None,
    group_col: str | None = None,
    n_iter: int = config.BOOTSTRAP_ITERATIONS,
    seed: int = config.BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap the decay fit by resampling events with replacement.

    Returns dict with point estimate and 95% CI for each parameter.
    """
    rng = np.random.default_rng(seed)

    if group_col and group_value:
        subset = cars[cars[group_col] == group_value]
    else:
        subset = cars

    # Unique events in this subset
    event_keys = subset[["ticker", "event_date"]].drop_duplicates()
    n_events = len(event_keys)

    if n_events < 10:
        log.warning(f"Only {n_events} events — bootstrap may be unstable.")

    results = []
    for _ in range(n_iter):
        # Resample events with replacement
        idx = rng.integers(0, n_events, n_events)
        sampled_events = event_keys.iloc[idx]
        # Merge to get all CAR rows for sampled events
        sampled = sampled_events.merge(subset, on=["ticker", "event_date"], how="left")
        curve = sampled.groupby("tau")["CAR"].mean().reset_index()
        fit = fit_decay(curve["tau"].values, curve["CAR"].values)
        if not np.isnan(fit["half_life"]):
            results.append(fit)

    if not results:
        return {"point": None, "ci_lower": None, "ci_upper": None}

    df = pd.DataFrame(results)

    # Point estimate = fit on full data
    full_curve = subset.groupby("tau")["CAR"].mean().reset_index()
    point_fit = fit_decay(full_curve["tau"].values, full_curve["CAR"].values)

    return {
        "point": point_fit,
        "half_life_ci": (float(df["half_life"].quantile(0.025)),
                         float(df["half_life"].quantile(0.975))),
        "lambda_ci":    (float(df["lambda"].quantile(0.025)),
                         float(df["lambda"].quantile(0.975))),
        "A_ci":         (float(df["A"].quantile(0.025)),
                         float(df["A"].quantile(0.975))),
        "n_bootstrap":  len(results),
        "n_events":     n_events,
    }


def fit_by_group(cars: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Fit decay separately for each level of a grouping column."""
    rows = []
    for grp, sub in cars.groupby(group_col):
        curve = sub.groupby("tau")["CAR"].mean().reset_index()
        fit = fit_decay(curve["tau"].values, curve["CAR"].values)
        n_events = sub[["ticker", "event_date"]].drop_duplicates().shape[0]
        rows.append({group_col: grp, "n_events": n_events, **fit})
    return pd.DataFrame(rows).sort_values("half_life")

