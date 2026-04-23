"""End-to-end pipeline. Run as: python -m src.pipeline --demo

Stages:
  1. Load prices + news
  2. Score sentiment
  3. Identify events
  4. Compute CARs
  5. Fit decay, save figures
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src import config, data_loader, sentiment, events, event_study, decay, viz


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main():
    parser = argparse.ArgumentParser(description="Sentiment decay pipeline")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data (no API keys)")
    parser.add_argument("--skip-fetch", action="store_true", help="Reuse cached processed data")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("pipeline")
    log.info(f"Running in {'DEMO' if args.demo else 'REAL'} mode")

    prices_path = config.PROCESSED_DIR / "prices.parquet"
    news_path   = config.PROCESSED_DIR / "news_scored.parquet"
    events_path = config.PROCESSED_DIR / "events.parquet"
    cars_path   = config.PROCESSED_DIR / "cars.parquet"

    # -------- 1. Prices + News --------
    if args.skip_fetch and prices_path.exists():
        log.info(f"Loading cached prices from {prices_path}")
        prices = pd.read_parquet(prices_path)
    else:
        log.info("Loading prices...")
        prices = data_loader.load_prices(demo=args.demo)
        prices.to_parquet(prices_path, index=False)
        log.info(f"Saved {len(prices):,} price rows -> {prices_path}")

    if args.skip_fetch and news_path.exists():
        log.info(f"Loading cached scored news from {news_path}")
        news_scored = pd.read_parquet(news_path)
    else:
        log.info("Loading news...")
        news = data_loader.load_news(demo=args.demo)
        log.info(f"{len(news):,} raw articles")

        log.info("Scoring sentiment...")
        news_scored = sentiment.score_news(news, demo=args.demo)
        news_scored.to_parquet(news_path, index=False)
        log.info(f"Saved scored news -> {news_path}")

    # -------- 2. Events --------
    log.info("Aggregating news daily...")
    daily = events.aggregate_news_daily(news_scored)

    log.info("Identifying events...")
    evts = events.identify_events(daily)
    evts.to_parquet(events_path, index=False)
    log.info(f"Identified {len(evts):,} events -> {events_path}")
    log.info(f"  By direction: {evts['direction'].value_counts().to_dict()}")
    log.info(f"  By sector:    {evts['sector'].value_counts().to_dict()}")

    # -------- 3. CARs --------
    log.info("Computing cumulative abnormal returns (this is the slow step)...")
    cars = event_study.compute_all_cars(prices, evts)
    cars.to_parquet(cars_path, index=False)
    log.info(f"Computed CARs for {cars[['ticker','event_date']].drop_duplicates().shape[0]:,} events")

    # -------- 4. Fit + plots --------
    log.info("Fitting decay curves and generating figures...")
    for direction in ["negative", "positive"]:
        sub = cars[cars["direction"] == direction]
        if sub.empty:
            continue

        # Mean CAR curve
        mean_curve = event_study.mean_car_curve(sub)
        viz.plot_mean_car(
            mean_curve, direction=direction,
            save_path=config.FIGURES_DIR / f"mean_car_{direction}.png",
        )

        # By sector
        viz.plot_decay_by_sector(
            sub.copy().assign(direction=direction), direction=direction,
            save_path=config.FIGURES_DIR / f"decay_by_sector_{direction}.png",
        )

        # By tier
        viz.plot_decay_by_tier(
            sub.copy().assign(direction=direction), direction=direction,
            save_path=config.FIGURES_DIR / f"decay_by_tier_{direction}.png",
        )

        # Half-life table per sector
        hl_sector = decay.fit_by_group(sub, "sector")
        hl_sector.to_csv(config.PROCESSED_DIR / f"halflife_sector_{direction}.csv", index=False)
        viz.plot_halflife_comparison(
            hl_sector, "sector",
            save_path=config.FIGURES_DIR / f"halflife_sector_{direction}.png",
        )

    log.info("Pipeline complete.")
    log.info(f"Figures saved to: {config.FIGURES_DIR}")
    log.info(f"Processed data:   {config.PROCESSED_DIR}")
    log.info("Run the dashboard:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
