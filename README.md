# CineMind — Movie Recommendation System for Digital Platforms

CineMind is a local-first movie discovery and recommendation web application for a Computer Engineering major project. It uses real MovieLens latest-small rating records and movie genres. It runs on a student laptop or on Render and does not need an account database or a paid movie API.

## Problem and objective

Large movie catalogs can make it difficult to discover relevant films. CineMind analyzes user–movie rating interactions and genre metadata to rank unseen movies, explain recommendation signals, provide movie similarity and catalog browsing, and evaluate recommendation quality with a reproducible offline holdout.

## Features

- **Home dashboard:** actual catalog, user, rating, and average-rating statistics; Bayesian popularity, genre collections, highest-average movies with at least 20 ratings, and data-derived hidden gems.
- **For You:** select a MovieLens user ID and request 10, 20, or 30 personalized titles. Filter the unseen catalog by genre or minimum dataset average, sort by relevance/rating/popularity, and read an explanation tied to the applied model signal.
- **My Taste:** see how many movies a selected user rated, their average rating, frequently rated genres, highest-average genres, and highest-rated titles. Profiles with fewer than five ratings are marked as early/insufficient for a reliable summary. Genre activity counts can overlap for multi-genre titles.
- **Genres:** explore the exact labels present in the dataset, including paginated genre listings and horizontal homepage rows. Each genre link opens its filtered listing.
- **Explore Movies:** case-insensitive title search, genre and minimum-average filters, sorting by popularity/rating/title/release year, incremental API-backed filtering, and pagination.
- **Movie details:** release year, MovieLens average and rating count, half-star rating distribution, and genre-based similar movies.
- **Analytics:** dataset totals and charts for genre-tagged movie counts, rating distribution, top-rated-count movies, average rating vs. count, average by genre, monthly rating activity, and selected-user genre activity.
- **Evaluation and About:** actual holdout metrics, method explanations, pipeline, and limitations.

Genre movie counts and genre-linked rating counts overlap because a movie may have several genre labels. User history is described as rating history; a rating alone does not prove that the person watched the movie. Movie posters are local CSS placeholders, so core features do not depend on external images.

## Technology

Python 3.12, Flask, Pandas, NumPy, scikit-learn, HTML5, CSS3, JavaScript, Chart.js, and Font Awesome. MovieLens CSV files are the source of truth. On pages that use charts, a small local bar-list fallback appears if Chart.js is unavailable; recommendation routes remain local.

## Dataset

This project includes MovieLens latest-small: 9,742 movies, 100,836 ratings, and 610 users. The snapshot’s original README and use terms are preserved in `data/MOVIELENS_README.txt`. `movies.csv` and `ratings.csv` are required; `links.csv` and `tags.csv` are optional. No movie, rating, user preference, chart value, or evaluation metric is fabricated. See the [GroupLens MovieLens page](https://grouplens.org/datasets/movielens/) and the included dataset terms for citation and redistribution details.

## Recommendation methods

- **Popularity:** a Bayesian weighted mean blends each movie’s average with the global average. The rating-count shrinkage uses a data-derived percentile (with a floor), reducing the influence of very small samples.
- **Content-based similarity:** genre strings become TF-IDF features; cosine similarity compares vectors. The For You user profile uses genres from titles that user rated at least 4 stars.
- **Collaborative filtering:** the user–movie matrix has users as rows, movies as columns, and observed ratings as values. Cosine similarity compares user rating patterns; ratings from similar users contribute to unseen-title predictions.
- **Hybrid:** normalized collaborative, content, and popularity signals combine at configurable weights of 0.45, 0.40, and 0.15. The taste-led For You route weights them 0.65, 0.20, and 0.15. Movies already rated by the user are excluded from personalized results.

Recommendation reasons are based on the signals available for that result. When a new profile has limited collaborative or genre signals, the weighted popularity signal provides a fallback. Recommendation counts are unique movie IDs.

## Analytics and evaluation

Dataset-level analytic payloads are computed from the loaded CSVs and cached in memory for the process. Genre-tagged movie counts count distinct movies per genre; average genre rating is calculated from the actual rating records for movies in that genre. Monthly activity uses the Unix timestamps supplied by `ratings.csv`, converted to UTC months. Scatter data are sampled deterministically to keep the browser payload modest.

`models/evaluation.py` splits ratings into a reproducible random 80/20 train/test partition (seed 42). RMSE and MAE compare held-out ratings with a transparent baseline averaging training global, user, and movie means. Precision@K and Recall@K use user-based cosine collaborative filtering to rank unseen movies; test ratings of at least 4 stars count as relevant. Results are macro-averaged over users with at least one relevant held-out title. These are offline estimates, not a guarantee of future performance; a temporal split would better model future ratings.

## Project structure

```text
app.py
requirements.txt
README.md
PROJECT_EXPLANATION.md
data/                       # MovieLens CSV files and source README
models/                     # recommendation and evaluation logic
utils/                      # data loading and validation
templates/                  # Flask/Jinja pages
templates/partials/         # reusable movie cards
static/css/                 # application and responsive styling
static/js/app.js            # search, recommendation, and chart interactions
tests/test_recommender.py
```

## Run locally

Python 3.10 or newer is recommended. From the repository directory:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>. Keep the MovieLens CSV files in `data/`. Run the automated checks with:

```powershell
python -m unittest discover -s tests
```

## Deploy on Render

The repository includes a `render.yaml` Blueprint and `.python-version`. Push the application and its permitted MovieLens files to GitHub, then choose **New → Blueprint** in Render and connect the repository. The Blueprint installs `requirements.txt`, starts Gunicorn, binds to Render’s `PORT`, and checks `/about`. It specifies the free plan and Singapore region. The app uses relative paths from its own directory and needs no secret keys or external movie API.

Alternatively create a Render Web Service with build command `pip install -r requirements.txt`, start command `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`, and health path `/about`.

## Main routes and API

Pages: `/`, `/for-you`, `/my-taste`, `/genres`, `/search`, `/movie/<movie_id>`, `/recommendations` (advanced lab), `/analytics`, `/evaluation`, and `/about`.

JSON examples: `/api/search?q=toy`, `/api/movies?q=matrix&genre=Sci-Fi&sort=rating&page=1`, `/api/recommendations?user_id=1&type=hybrid&limit=10`, `/api/users/1/taste`, `/api/similar/1`, `/api/movie/1`, `/api/analytics?user_id=1`, `/api/evaluation`.

## Limitations and future work

MovieLens latest-small is a 2018 snapshot and has no demographic or explicit viewing data. Genre features are coarse; they do not encode plot, cast, or mood. New users and movies have less collaborative evidence. Random-split metrics may overestimate performance on future ratings. Future work could compare temporal evaluation, item-based or matrix-factorization models, and optional licensed poster sources.

## Project team

- Student/member: Atharva Kamate
- Guide/faculty: _Add name_
- Department/institute: _Add details_

## Acknowledgment

Harper, F. M., and Konstan, J. A. (2015). “The MovieLens Datasets: History and Context.” *ACM Transactions on Interactive Intelligent Systems*, 5(4), Article 19. Follow the citation and use terms in `data/MOVIELENS_README.txt`.
