"""Interactive dashboard for exploring sentiment decay results.

Run: streamlit run dashboard/app.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, decay  # noqa: E402

st.set_page_config(
    page_title="News Sentiment Decay",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Data loading ----------------
@st.cache_data
def load_all():
    cars_path   = config.PROCESSED_DIR / "cars.parquet"
    events_path = config.PROCESSED_DIR / "events.parquet"
    if not cars_path.exists() or not events_path.exists():
        return None, None
    return pd.read_parquet(cars_path), pd.read_parquet(events_path)

cars, events = load_all()

st.title("📉 News Sentiment Decay in US Equity Markets")
st.caption("How quickly does the market absorb financial news? A sector-level event study.")

if cars is None:
    st.error(
        "Processed data not found. Run the pipeline first:\n\n"
        "```bash\npython -m src.pipeline --demo\n```"
    )
    st.stop()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
direction = st.sidebar.radio("Event direction", ["negative", "positive"], index=0)
sectors_available = sorted(cars["sector"].dropna().unique())
sectors_selected = st.sidebar.multiselect(
    "Sectors", sectors_available, default=sectors_available,
)
tiers_available = [t for t in ["tier_1", "tier_2", "tier_3"] if t in cars["source_tier"].unique()]
tiers_selected = st.sidebar.multiselect("News source tiers", tiers_available, default=tiers_available)

filtered = cars[
    (cars["direction"] == direction)
    & (cars["sector"].isin(sectors_selected))
    & (cars["source_tier"].isin(tiers_selected))
]

