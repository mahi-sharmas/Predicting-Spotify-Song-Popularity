# Spotify Popularity — A Leakage Case Study

### What happens when 83 copies of the same song sit on both sides of your train/test split

---

## Summary

This project set out to predict Spotify track popularity from audio features. An initial Random Forest reached **R² = 0.655**, which looked like a solid result.

It wasn't. The dataset contains one row per song, per country, per day — so each track appears an average of **83 times** with identical audio features and near-identical popularity. A random `train_test_split` scattered those copies across train and test, letting tree models memorise song identity instead of learning anything about audio.

Re-splitting by track ID collapses the result:

| Model | Random row split | Grouped by song |
|---|---:|---:|
| Linear Regression | 0.055 | **0.030** |
| Decision Tree | 0.493 | **−1.014** |
| Random Forest | 0.638 | **−0.171** |
| *Predict the mean* | — | −0.001 |

Under a correct split, **audio features explain roughly 3% of the variance in popularity**, and both tree models perform worse than predicting the average for every track.

---

## The dataset

[Top Spotify Songs in 73 Countries (Daily Updated)](https://www.kaggle.com/datasets/asaniczka/top-spotify-songs-in-73-countries-daily-updated) — 2,110,316 rows, 25 columns.

The collection process matters here. The source scraper pulls 74 Spotify Top-50 playlists (global plus 73 countries) and appends them to a master file **every day**. That's ~3,700 new rows daily, accumulating into millions of rows that describe only **24,964 distinct tracks**.

| Metric | Value |
|---|---:|
| Rows (after dropping nulls) | 2,080,593 |
| Unique tracks | 24,964 |
| Mean rows per track | 83.3 |
| Most-repeated track | 16,301 rows |
| Mean within-track popularity σ | 6.32 |
| Overall popularity σ | 15.75 |

That last pair is the crux: popularity is close to a per-song constant. Recognising the song is nearly equivalent to knowing the answer.

---

## Method

**Features** — the 13 Spotify audio attributes only: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `time_signature`.

Deliberately excluded:
- `country` — geography, not audio, and popularity varies systematically by market
- `daily_rank`, `daily_movement`, `weekly_movement` — chart position is *downstream* of popularity, so including it is circular

**Splits** — identical models trained twice, changing only how the data is divided:
1. `train_test_split(test_size=0.2)` — the original, leaky approach
2. `GroupShuffleSplit(groups=spotify_id)` — every copy of a track lands entirely in train or entirely in test

Train/test track overlap under the grouped split is verified to be exactly **0**.

**Models** — Linear Regression, Decision Tree (`max_depth=15`), Random Forest (`n_estimators=50, max_depth=15`), each fit on a 20,000-row training subsample for runtime.

---

## Why the trees go negative

A negative R² means the model does worse than a horizontal line at the training mean.

Each track has a unique combination of 13 continuous audio values — effectively a fingerprint. Under the random split, a deep tree partitions the feature space finely enough to isolate individual songs, reads off the popularity it memorised, and finds the same songs waiting in the test set. Linear regression can't do this, which is why it barely moved between the two splits (0.055 → 0.030) while the forest fell off a cliff (0.638 → −0.171).

The gap between a linear model and a tree model is itself a useful leakage diagnostic. When trees dramatically outperform linear models on tabular data with no obvious interactions, it's worth asking whether they're learning structure or learning row identity.

---

## Limitations

1. **This is a null result, not a proof of impossibility.** Audio features alone don't predict popularity in this dataset. Artist-level features, release timing, playlist placement, and marketing spend are all absent and are plausibly where the real signal lives.

2. **Popularity is itself a moving target.** Spotify's popularity score is time-varying and partly recency-weighted, so a static per-track label is an approximation.

3. **The training subsample is 20,000 rows.** Larger samples were tested and did not change the conclusion, but the reported figures come from the subsample for reproducibility.

4. **Only charting songs are in the dataset.** Every track here reached a national Top 50, so popularity is range-restricted at the high end — which attenuates any correlation that might exist across the full catalogue.

---

## Reproducing

1. Download the dataset from the Kaggle link above (`archive.zip`)
2. Place it beside the notebook
3. Run `Spotify_Leakage_Analysis.ipynb` top to bottom

Runtime is a few minutes, dominated by the CSV read.

Note that the dataset updates daily, so row counts will exceed those quoted here. The ratio of rows to unique tracks — the thing that actually matters — is stable.

---

## Takeaway

When rows aren't independent — repeated entities, time series, multiple records per subject — a random split silently inflates results. The split has to be grouped on the true unit of observation.

Here that unit is the **song**, not the chart row.

---

## Author

**Mahi Sharma**
B.Tech Computer Science (Data Science), Manipal University Jaipur
GitHub: [@mahi-sharmas](https://github.com/mahi-sharmas)
