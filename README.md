## Predicting Spotify Song Popularity

A regression analysis on 2.1 million Spotify tracks to predict song popularity scores from audio features, comparing Linear Regression, Decision Tree, and Random Forest with hyperparameter tuning.

### Highlights

- Analyzed 2.1M+ Spotify chart entries with 14 audio features to predict popularity on a 0–100 scale
- Achieved R² = 0.655 and MSE = 85.34 with a tuned Random Forest — a 10× improvement over Linear Regression (R² = 0.06)
- Used RandomizedSearchCV with 3-fold cross-validation to optimize Random Forest hyperparameters across two tuning rounds
- Demonstrated that tree-based models significantly outperform linear models for capturing non-linear audio feature relationships

### Problem Statement

Predicting a song's popularity before release can help artists, record labels, and streaming platforms make data-driven decisions about marketing, playlist placement, and release strategy. This project builds a regression model that predicts Spotify popularity scores (0–100) from audio attributes like danceability, energy, and tempo — investigating which features matter most and how well popularity can be predicted from audio characteristics alone.

### Dataset

- **Source:** [Universal Top Spotify Songs](https://www.kaggle.com/datasets) (Kaggle)
- **Size:** 2,110,316 chart entries × 25 columns
- **Target:** `popularity` (continuous integer, 0–100 scale)
- **Audio features used (14):** danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, time_signature, country (label-encoded)
- **Missing values handled:** country (28,908), album_name (822), album_release_date (659), name (30), artists (29) — all rows with nulls dropped during preprocessing
- **Snapshot period:** Data from Spotify charts as of Jun 2025

### Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)
![NumPy](https://img.shields.io/badge/NumPy-Compute-yellow)
![SciPy](https://img.shields.io/badge/SciPy-Stats-red)

### Methodology

1. **Data Loading & Inspection** — Imported 2.1M Spotify tracks, inspected data types (25 columns), checked null distributions across all features
2. **Preprocessing** — Dropped rows with missing values, removed non-predictive columns (name, artists, id), encoded the categorical `country` feature using LabelEncoder
3. **Feature Selection** — Selected 14 features: 13 audio attributes + encoded country as predictors for the `popularity` target
4. **Train-Test Split** — 80/20 split with `random_state=42`, created a 20,000-sample training subset for efficient baseline model comparison
5. **Baseline Models** — Trained and timed Linear Regression (0.1s), Decision Tree with `max_depth=15` (0.4s), and Random Forest with 50 estimators and `max_depth=15` (10.5s) on the subset
6. **Hyperparameter Tuning (Round 1)** — RandomizedSearchCV (10 iterations, 3-fold CV) on Random Forest: best parameters `n_estimators=200, max_depth=30, min_samples_split=5, max_features='sqrt'` with best CV MSE of 91.05
7. **Hyperparameter Tuning (Round 2)** — Second RandomizedSearchCV with different parameter distributions: best parameters `n_estimators=137, max_depth=20, min_samples_split=3` with best CV MSE of 94.44
8. **Final Evaluation** — Best estimator from Round 2 evaluated on full test set for final MSE and R²

### Key Results

| Model | MSE | R² Score | Training Time |
|---|---|---|---|
| Linear Regression | 232.96 | 0.06 | 0.1s |
| Decision Tree (depth=15) | 142.95 | 0.42 | 0.4s |
| Random Forest (50 trees, depth=15) | 93.05 | 0.62 | 10.5s |
| **Random Forest (Tuned — 137 trees, depth=20)** | **85.34** | **0.655** | — |

**Best hyperparameters:** `n_estimators=137, max_depth=20, min_samples_split=3`

The tuned Random Forest explains 65.5% of the variance in popularity scores, reducing prediction error by 63% compared to the linear baseline. The massive gap between Linear Regression (R² = 0.06) and tree-based models confirms strong non-linear relationships between audio features and popularity.

### How to Run

```bash
git clone https://github.com/mahi-sharmas/Predicting-Spotify-Song-Popularity.git
cd Predicting-Spotify-Song-Popularity
pip install -r requirements.txt
jupyter notebook Spotify_Prediction.ipynb
```

**Note:** The dataset must be downloaded from [Kaggle](https://www.kaggle.com/datasets) and placed as `archive.zip` in the project directory.

### Project Structure

```
Predicting-Spotify-Song-Popularity/
├── Spotify_Prediction.ipynb    # Full pipeline — data loading, preprocessing, model comparison, tuning
├── spotify.jpeg                # Project visualization
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

### Future Improvements

- Incorporate artist popularity, genre metadata, and release timing as additional features to push R² beyond 0.65
- Add visualizations — feature importance plots, actual vs. predicted scatter plots, and residual analysis
- Experiment with gradient boosting models (XGBoost, LightGBM) which often outperform Random Forest on large tabular datasets

### Author

**Mahi Sharma** — B.Tech CSE (Data Science), Manipal University Jaipur (2023–2027)

GitHub: [github.com/mahi-sharmas](https://github.com/mahi-sharmas) | Email: mahi.sh4rma7@gmail.com

*Project completed: Jul 2025*
