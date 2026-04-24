# Methodology

## Research question

> After a significant news event, how quickly is the sentiment reflected in the stock price — and does this speed vary by sector, market cap, and news source?

## Conceptual framework

The Efficient Market Hypothesis (Fama, 1970) predicts that prices instantly absorb all public information. In practice, post-event drift is well-documented (Bernard & Thomas, 1989) — markets digest news *gradually*, not instantaneously. This project measures the speed of that digestion.

## Step 1 — Universe selection

30 US equities, 5 per sector across 6 sectors (Technology, Financials, Healthcare, Energy, Consumer Discretionary, Utilities). Selection criteria:

- Liquid (high daily volume → cleaner price data)
- Sufficient news coverage in the 2-year window
- Mix of market caps for cross-sectional comparison

See `src/config.py::UNIVERSE` for the full list.

## Step 2 — Data sources

| Source   | What    | Why                                   |
|----------|---------|---------------------------------------|
| yfinance | Prices  | Free, well-maintained, adjusted close |
| Finnhub  | News    | Free tier, has timestamps + sources   |
| FRED     | Macro (optional) | Risk-free rate, controls       |

## Step 3 — Sentiment scoring

We use **FinBERT** (Araci, 2019), a BERT model fine-tuned on financial text. For each headline we compute three probabilities (positive, negative, neutral) and a compound score = positive − negative ∈ [-1, 1].

**Why FinBERT, not VADER?** VADER is tuned for general social-media text and stumbles on financial idioms. FinBERT was trained on Reuters TRC2 and the Financial PhraseBank, giving it the right vocabulary for our task.

**Limitations**: FinBERT can mislabel sarcasm and ambiguous headlines. Empirical validation on 50 random samples is recommended (see notebook 02).

## Step 4 — Event identification

A (ticker, date) pair is an "event" if EITHER:

- **Sentiment spike**: max |compound sentiment| across articles ≥ 0.7
- **Volume spike**: article count > mean + 3σ (30-day rolling window)

Both definitions are imperfect but capture different signals. The union is more inclusive than either alone.

Events are classified as `positive` or `negative` based on the dominant article's sentiment sign.

## Step 5 — Cumulative abnormal returns (CAR)

Using the **market model** (MacKinlay, 1997):

1. **Estimation window** (τ ∈ [-60, -11]): regress ticker daily return on SPY daily return → (α, β)
2. **Event window** (τ ∈ [-5, +10]): predicted normal return = α + β · R_market
3. **Abnormal return** AR_τ = R_actual,τ − R_predicted,τ
4. **Cumulative abnormal return** CAR(τ) = Σ_{s=0}^{τ} AR_s

**Critical**: estimation window ends 11 trading days before the event to prevent leakage. If the event itself influences the β estimate, the abnormal return is biased toward zero.

## Step 6 — Decay model

We fit:

```
CAR(τ) = A · exp(-λ · τ) + C,    for τ ≥ 0
```

- **A** = initial shock magnitude (CAR at τ=0 minus asymptote)
- **λ** = decay rate (1/days)
- **C** = permanent price impact (asymptote)
- **Half-life** t½ = ln(2)/λ

This is the simplest form that captures both rapid initial absorption and a long-run permanent component. Alternatives (power law, biexponential) didn't materially improve fit on validation samples.

## Step 7 — Inference

**Bootstrap**: resample events (not observations within events) with replacement N=1,000 times, refit decay, take 2.5th and 97.5th percentiles for 95% CIs. Resampling at the event level — not the day level — gives valid inference because events are the independent unit.

**Comparisons**: differences between sectors / tiers are reported as half-life differences with bootstrap CIs. Non-overlapping 95% CIs are treated as evidence of meaningful difference.

## Cross-references

- MacKinlay, A.C. (1997). *Event Studies in Economics and Finance.* Journal of Economic Literature 35(1).
- Bernard, V.L. & Thomas, J.K. (1989). *Post-Earnings-Announcement Drift.* Journal of Accounting Research 27.
- Fama, E.F. (1970). *Efficient Capital Markets.* Journal of Finance 25(2).
- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063.

