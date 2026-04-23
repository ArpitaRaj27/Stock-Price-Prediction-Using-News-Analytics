"""Plotting helpers for decay curves, comparisons, and diagnostics."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.decay import decay_model

# Consistent style
plt.rcParams.update({
    "figure.figsize": (10, 5.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})

SECTOR_COLORS = {
    "Technology": "#2563eb",
    "Financials": "#059669",
    "Healthcare": "#dc2626",
    "Energy": "#ea580c",
    "Consumer Discretionary": "#7c3aed",
    "Utilities": "#0891b2",
}


def plot_mean_car(
    mean_curve: pd.DataFrame,
    direction: str = "negative",
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot mean CAR with ±1 SEM ribbon."""
    curve = mean_curve.copy()
    fig, ax = plt.subplots()
    ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="Event (τ=0)")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.plot(curve["tau"], curve["mean_CAR"], lw=2.2, label="Mean CAR")
    ax.fill_between(
        curve["tau"],
        curve["mean_CAR"] - curve["sem_CAR"],
        curve["mean_CAR"] + curve["sem_CAR"],
        alpha=0.25, label="±1 SEM",
    )
    ax.set_xlabel("Days relative to event (τ)")
    ax.set_ylabel("Cumulative Abnormal Return")
    ax.set_title(title or f"Mean CAR — {direction} events (n={curve['n'].iloc[0]})")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_decay_by_sector(
    cars: pd.DataFrame,
    direction: str = "negative",
    save_path: str | None = None,
):
    """Overlay fitted decay curves for each sector, with half-lives in legend."""
    from src.decay import fit_decay

    sub = cars[cars["direction"] == direction]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="red", linestyle="--", alpha=0.4)

    for sector, grp in sub.groupby("sector"):
        curve = grp.groupby("tau")["CAR"].mean().reset_index()
        color = SECTOR_COLORS.get(sector, "black")
        ax.plot(curve["tau"], curve["CAR"], "o-", color=color, alpha=0.5, markersize=3)

        # Fit post-event only
        fit = fit_decay(curve["tau"].values, curve["CAR"].values)
        if not np.isnan(fit["half_life"]):
            tau_fit = np.linspace(0, curve["tau"].max(), 100)
            car_fit = decay_model(tau_fit, fit["A"], fit["lambda"], fit["C"])
            ax.plot(tau_fit, car_fit, color=color, lw=2.2,
                    label=f"{sector} (t½={fit['half_life']:.1f}d)")
        else:
            ax.plot([], [], color=color, lw=2.2, label=f"{sector} (no fit)")

    ax.set_xlabel("Days relative to event (τ)")
    ax.set_ylabel("Cumulative Abnormal Return")
    ax.set_title(f"Sentiment Decay by Sector — {direction} events")
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_decay_by_tier(
    cars: pd.DataFrame,
    direction: str = "negative",
    save_path: str | None = None,
):
    """Compare decay speed across news source tiers."""
    from src.decay import fit_decay

    sub = cars[(cars["direction"] == direction) & (cars["source_tier"] != "unknown")]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="red", linestyle="--", alpha=0.4)

    tier_colors = {"tier_1": "#1e40af", "tier_2": "#d97706", "tier_3": "#7f1d1d"}

    for tier, grp in sub.groupby("source_tier"):
        curve = grp.groupby("tau")["CAR"].mean().reset_index()
        color = tier_colors.get(tier, "gray")
        ax.plot(curve["tau"], curve["CAR"], "o-", color=color, alpha=0.4, markersize=3)

        fit = fit_decay(curve["tau"].values, curve["CAR"].values)
        if not np.isnan(fit["half_life"]):
            tau_fit = np.linspace(0, curve["tau"].max(), 100)
            ax.plot(tau_fit, decay_model(tau_fit, fit["A"], fit["lambda"], fit["C"]),
                    color=color, lw=2.2,
                    label=f"{tier} (t½={fit['half_life']:.1f}d, n={len(grp.drop_duplicates(['ticker','event_date']))})")

    ax.set_xlabel("Days relative to event (τ)")
    ax.set_ylabel("Cumulative Abnormal Return")
    ax.set_title(f"Sentiment Decay by News Source Tier — {direction} events")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_halflife_comparison(halflife_df: pd.DataFrame, group_col: str, save_path: str | None = None):
    """Horizontal bar chart of half-lives."""
    df = halflife_df.dropna(subset=["half_life"]).sort_values("half_life")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df[group_col], df["half_life"], color="#2563eb", alpha=0.8)
    ax.set_xlabel("Half-life (days)")
    ax.set_title(f"Price Impact Half-Life by {group_col.title()}")
    for i, (v, n) in enumerate(zip(df["half_life"], df["n_events"])):
        ax.text(v + 0.05, i, f"{v:.1f}d (n={n})", va="center", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
