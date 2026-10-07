# Project Explanation — Recommendation System for Digital Platforms

## Problem statement

Large digital catalogs can make it difficult for users to discover movies that match their interests. A recommendation system learns from past ratings and item descriptions to rank useful choices.

## Objective

Build a local web application that analyzes real MovieLens interactions, generates explainable personalized and similar-movie recommendations, and measures recommendation quality with a reproducible offline evaluation.

## Dataset

MovieLens latest-small contains movie titles and genres, user ratings, tags, and external IDs. The supplied snapshot has 9,742 movies, 100,836 ratings, and 610 users. It is stored in `data/`; no movie catalog is manually invented. See `data/MOVIELENS_README.txt` for the source citation and license.

## Data preprocessing

The loader removes duplicate movie rows and duplicate user/movie ratings, converts identifiers and ratings to numeric values, drops malformed records, accepts ratings from 0.5 to 5, fills missing genres, and ignores ratings whose movie ID is not present. `movies.csv` and `ratings.csv` are required; links and tags are optional.

## Algorithms

### Popularity-based

Popularity uses a Bayesian weighted average: a movie mean is blended with the global mean in proportion to its rating count. This reduces the impact of a high average based on very few ratings.

### Content-based

Movie genres are tokenized into TF-IDF features. TF-IDF represents each genre in a way that gives less weight to common terms. Cosine similarity compares vector directions and returns larger values for titles with more similar genre combinations. A user's content profile combines genres from movies rated at least 4 stars.

### Collaborative filtering

The user–item matrix has users as rows, movies as columns, and rating values in cells. Cosine similarity compares user rating patterns. Ratings from similar users are combined to score titles the active user has not rated. This can find interests that genre overlap alone would miss.

### Hybrid

Hybrid ranking normalizes the collaborative, content, and popularity signals to comparable ranges and combines them with configurable weights (0.45, 0.40, and 0.15 by default). “For You” uses a larger collaborative weight. Already-rated movies are filtered out.

## System workflow

1. Flask loads the local CSV files once when `app.py` starts.
2. The loader validates records and the model builds genre features, rating matrices, user similarities, and popularity statistics.
3. The browser requests recommendations or search results from JSON endpoints.
4. The Flask model returns ranked movies with relevant reasons.
5. Analytics and the evaluation page display measurements computed from the same dataset.

## Evaluation

A fixed-seed 80/20 split separates training and held-out ratings. RMSE and MAE compare held-out ratings to a baseline averaged from training global, user, and movie means. For Precision@K and Recall@K, user-based collaborative filtering ranks unseen movies and held-out ratings >=4 are considered relevant. Precision is relevant recommendations divided by K. Recall is relevant recommendations divided by all held-out relevant movies for that user. Values are averaged across eligible users. The evaluation is an offline estimate; a chronological split would better model future recommendations.

## Advantages

- Uses a real and commonly recognized dataset.
- Runs locally on CPU without paid APIs or a database.
- Separates data, model, routes, templates, and presentation.
- Combines explainable methods and reports genuine holdout metrics.
- Provides a responsive dashboard, movie search, details, analytics, and model notes.

## Limitations

- The data are a 2018 snapshot and do not include user demographics.
- A new user or movie has little collaborative evidence.
- Genres are coarse and do not describe plot, cast, or mood.
- Random holdout evaluation may leak time patterns and does not guarantee real-world results.
- MovieLens user IDs are dataset identifiers rather than accounts or identities.

## Future scope

Add temporal evaluation, compare item-based and matrix-factorization models, support implicit signals, allow local user preference profiles, improve accessibility, and optionally integrate licensed movie artwork.

## Possible viva questions and answers

**Q: What is a user–item matrix?**  
A: A table with users on one axis, movies on the other, and observed rating values in the cells. Empty cells represent movies the user has not rated.

**Q: Why use cosine similarity?**  
A: It compares the direction of two vectors, so similarity reflects rating/genre patterns rather than only their magnitude. It is efficient and widely used for sparse preference comparisons.

**Q: What does TF-IDF do here?**  
A: It transforms genre tokens into weighted numeric features. Common genre tokens receive lower distinguishing weight than less common ones.

**Q: What is collaborative filtering?**  
A: It uses patterns of ratings from users with similar behavior to estimate which unseen movies may be relevant.

**Q: Why use a popularity prior?**  
A: It provides a useful fallback and stabilizes average ratings for movies with few observations.

**Q: How does the hybrid model work?**  
A: It normalizes and combines collaborative, content, and popularity scores using explicit configurable weights.

**Q: Define Precision@K and Recall@K.**  
A: Precision@K is the share of the first K recommendations that are relevant. Recall@K is the share of all relevant held-out movies recovered in those first K.

**Q: What does RMSE indicate?**  
A: It measures typical rating prediction error while penalizing large errors more strongly. Smaller values indicate closer predictions on the test split.

**Q: What is a limitation of the evaluation split?**  
A: A random split does not preserve time order; a chronological split would better test how a model predicts future user activity.
