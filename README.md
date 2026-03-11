# Predicting Spotify Song Popularity

A regression analysis predicting song popularity scores using audio features from 2.1M+ Spotify chart entries. Compares Linear Regression, Decision Tree, and Random Forest, achieving an R² of 0.65 with the tuned Random Forest model.

## Problem Statement

What makes a song popular on Spotify? This project uses audio features (danceability, energy, tempo, etc.) and metadata to predict a song's popularity score. Understanding these drivers can help artists, labels, and playlist curators make data-informed decisions about music production and promotion.

## Dataset

- **Source:** Kaggle — Universal Top Spotify Songs
- **Records:** 2,110,316 chart entries
- **Features (25):** `spotify_id`, `name`, `artists`, `daily_rank`, `country`, `popularity`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `time_signature`, and more
- **Target:** `popularity` (integer score 0–100)

## Approach

1. **Data Cleaning** — Dropped rows with missing values, removed identifier columns (`name`, `artists`, `id`)
2. **Feature Engineering** — Label-encoded `country`, selected 14 audio + metadata features
3. **Subsampling** — Used 20,000 training samples for efficient hyperparameter search
4. **Baseline Models** — Linear Regression, Decision Tree (max_depth=15), Random Forest (50 estimators)
5. **Hyperparameter Tuning** — RandomizedSearchCV (10 iterations, 3-fold CV) on Random Forest

## Key Results

| Model | MSE | R² Score | Training Time |
|---|---|---|---|
| Linear Regression | 232.96 | 0.06 | 0.1s |
| Decision Tree (depth=15) | 142.95 | 0.42 | 0.4s |
| Random Forest (50 trees) | 93.05 | 0.62 | 10.5s |
| **Random Forest (Tuned)** | **85.34** | **0.65** | — |

- **Best Model:** Random Forest — `n_estimators=200`, `max_depth=30`, `min_samples_split=5`, `max_features=sqrt`
- **Best CV Score (MSE):** 91.05
- **Key Insight:** Audio features alone explain ~65% of popularity variance; linear models fail (R² = 0.06), confirming non-linear relationships

## Tech Stack

- **Language:** Python
- **Libraries:** Scikit-learn, Pandas, NumPy, SciPy, Matplotlib
- **Techniques:** Random Forest Regression, Randomized Search CV, Label Encoding, train-test split

## Project Structure

```
├── Spotify_Prediction.ipynb   # Full analysis and modeling notebook
├── spotify.jpeg               # Visualization
└── README.md
```

## How to Run

```bash
jupyter notebook Spotify_Prediction.ipynb
```

## Author

**Mahi Sharma**
B.Tech CSE (Data Science) — Manipal University Jaipur
[GitHub](https://github.com/mahi-sharmas)
