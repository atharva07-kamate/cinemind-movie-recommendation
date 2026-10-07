# Recommendation System for Digital Platforms

CineMind is a local, explainable movie discovery application built for a Computer Engineering major project. It uses the MovieLens latest-small CSV data to analyze rating behavior, recommend unseen movies, find genre-similar titles, and report offline evaluation metrics.

## Problem statement

People can struggle to find relevant content in large catalogs. This project explores how observed user–movie ratings and movie genres can help rank films a user may enjoy.

## Objectives and features

- Load MovieLens data from the local `data/` folder; no recommendation feature depends on an external movie API.
- Provide a responsive dashboard with catalog and interaction statistics, Bayesian popularity picks, and featured catalog entries.
- Search all titles case-insensitively, inspect movie details, rating distributions, and genre-similar titles.
- Recommend unseen movies using user-based collaborative filtering, genre-based content filtering, popularity, or a configurable hybrid.
- Visualize rating, genre, popular-title, and user-activity statistics.
- Evaluate a deterministic 80/20 holdout using RMSE, MAE, Precision@10, and Recall@10.
- Explain the processing and recommendation pipeline for project demonstrations.

## Technologies

Python 3, Flask, Pandas, NumPy, scikit-learn, HTML, CSS, JavaScript, Chart.js, and Font Awesome.

## Dataset

The included files come from [MovieLens latest-small](https://grouplens.org/datasets/movielens/), supplied by GroupLens Research. This snapshot contains 9,742 movies, 100,836 ratings and 610 users (data generated in 2018). The original dataset README/license is preserved as `data/MOVIELENS_README.txt`; please follow its attribution and redistribution terms. `movies.csv` and `ratings.csv` are required. `links.csv` and `tags.csv` are loaded when present. The app does not fetch posters, and instead generates stylized local placeholders.

## System architecture

`app.py` loads and validates the CSV files once at startup, constructs the model once, and serves Jinja pages and JSON endpoints. `models/recommender.py` holds precomputed rating/genre representations and scoring logic. `models/evaluation.py` computes the holdout metrics. `utils/data_loader.py` handles local CSV loading and cleaning. Templates, styles, and browser interactions live in `templates/` and `static/`.

## Recommendation methodology

- **Popularity:** Bayesian weighted rating blends each title's mean with the dataset-wide mean. The shrinkage count is based on the 75th percentile of rating counts (at least 10), so a high mean from very few ratings is less influential.
- **Content-based:** genre strings become TF-IDF features. Cosine similarity compares genre vectors; user profiles are weighted by ratings of movies rated at least 4 stars.
- **Collaborative:** ratings form a user–movie matrix. Cosine similarity between user rows finds neighbors; their ratings are aggregated to score unseen titles.
- **Hybrid:** min-max normalized collaborative, content, and popularity scores are combined. Default weights are 0.45, 0.40, and 0.15 in `models/recommender.py`. “For You” emphasizes collaborative score (0.65/0.20/0.15).

Recommendations exclude movies the selected user has already rated. Similar-movie recommendations use genre cosine similarity. All recommendation cards include a reason derived from the scoring signals.

## Evaluation methodology

`models/evaluation.py` uses a reproducible random 80/20 rating split (seed 42). RMSE and MAE use a transparent baseline averaging the training global, user, and movie means. Precision@K and Recall@K rank unseen movies with cosine user-based collaborative filtering on the training portion; ratings of 4 stars or more in the held-out portion are relevant. Results are macro-averaged across users with at least one relevant held-out rating. These are offline estimates, not a guarantee of production performance. Since ratings are randomly split, they can mix earlier and later activity; a temporal split can be a future improvement.

## Project structure

```text
app.py
requirements.txt
README.md
PROJECT_EXPLANATION.md
data/                       # MovieLens CSV files and source README
models/                     # recommender and evaluation logic
utils/                      # data loading and validation
templates/                  # Flask/Jinja pages
templates/partials/         # reusable movie cards
static/css/style.css
static/js/app.js
tests/test_recommender.py
```

## Installation and running

Python 3.10 or later is recommended.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. Keep the MovieLens CSV files in `data/`. Run the included checks with `python -m unittest discover -s tests`.

## Deploy to Render

The repository includes `render.yaml`, `.python-version`, and a Linux-only Gunicorn dependency for Render. Create a GitHub repository, commit and push this project (including `data/`), then in Render choose **New → Blueprint**, connect that repository, review the `cinemind` web service, and deploy it. The Blueprint sets Python runtime, Singapore region, the Free plan, build command `pip install -r requirements.txt`, start command `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`, and health check `/about`. There are no secrets or manually supplied environment variables; Render supplies `PORT` to the server. A standard **New → Web Service** also works with those same settings if you prefer to enter them manually.

## Example usage

1. Open Overview to see dataset summaries and popular movies.
2. Use Discover to search titles such as “Toy Story”. Open a result to see rating distribution and similar genres.
3. Open Recommendations, choose a style, enter a MovieLens user ID (1–610), and generate 5, 10, 15, or 20 results. Similar Movies requires a catalog title.
4. Review Analytics and Evaluation for dataset charts and measured holdout metrics.

JSON endpoints include `/api/search?q=toy`, `/api/recommendations?user_id=1&type=hybrid&limit=10`, `/api/similar/1`, `/api/movie/1`, `/api/analytics`, and `/api/evaluation`.

## Screenshots

_Add screenshots of the dashboard, recommendations, analytics, and evaluation pages here._

## Future scope

Temporal and per-user evaluation, item-based collaborative filtering, implicit-feedback support, a lightweight persistent preference profile, accessibility refinements, and optional licensed poster integration.

## Limitations

MovieLens latest-small is a 2018 development snapshot without demographic data. User IDs are anonymous dataset IDs, not accounts. Genre metadata is sparse, so content similarity cannot capture plot or cast. Cold-start users and new titles have limited signals. Offline random-split metrics may overestimate future performance. Poster art is intentionally not fetched.

## Team / members

- Student/member name: _Add name_
- Guide/faculty: _Add name_
- Department/institute: _Add details_

## Acknowledgment

Harper, F. M., and Konstan, J. A. (2015). “The MovieLens Datasets: History and Context.” *ACM Transactions on Interactive Intelligent Systems*, 5(4), Article 19. See the included dataset README for full citation and terms.
