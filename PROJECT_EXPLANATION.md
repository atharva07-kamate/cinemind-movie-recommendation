# Project Explanation — CineMind

## Problem statement

People can have difficulty finding films that match their interests in a large catalog. A recommendation system can use known user ratings and movie information to rank items that may be relevant.

## Objective

CineMind is a local-first web application that analyzes real MovieLens rating interactions and genre metadata. It provides personalized recommendations, movie similarity, taste summaries, genre discovery, analytics, and offline model evaluation.

## Dataset

MovieLens latest-small includes 9,742 movies, 100,836 ratings, and 610 anonymous dataset users. The data are kept in the `data/` folder. The project does not invent movies, ratings, statistics, user preferences, or metrics. The original data README and terms are stored in `data/MOVIELENS_README.txt`.

The ratings represent ratings in the dataset; they do not prove a user watched the movie. User IDs shown in CineMind are MovieLens identifiers and are not CineMind accounts.

## Data preprocessing

`utils/data_loader.py`:

1. Loads `movies.csv` and `ratings.csv` from a path relative to the project.
2. Removes duplicate movie IDs and duplicate user/movie ratings.
3. Converts IDs and ratings to numeric values and removes malformed values.
4. Keeps ratings within MovieLens’s 0.5–5.0 scale and removes references to unknown movie IDs.
5. Replaces missing genre data safely and loads links/tags only when present.

The Flask app loads the dataset and creates the recommender once when its process starts. Expensive per-movie similarity matrices are not recomputed on every request. Dataset-level analytic summaries are cached in the model instance.

## Recommendation algorithms

### Popularity-based recommendation

The model calculates each movie’s mean rating and rating count. Its weighted popularity score blends the movie average with the global average. Movies with few ratings are pulled toward the global mean so a one-rating film does not automatically top the ranking.

### Content-based recommendation

Genre text is represented using TF-IDF. TF-IDF converts genre tokens into numeric features and reduces the relative weight of common tokens. Cosine similarity compares feature-vector direction and measures genre overlap. Similar Movies uses these movie features. A personalized genre profile uses movies rated at least 4 stars by that user.

### Collaborative filtering

The user–item matrix has users in rows, movies in columns, and known rating values in its cells. Cosine similarity compares users’ rating patterns. Ratings from similar users help score movies the selected user has not rated. This is collaborative filtering: recommendation comes from other users’ behavior, not genre overlap alone.

### Hybrid and For You

Collaborative, content, and popularity scores are normalized and combined. The configurable hybrid weights are 0.45, 0.40, and 0.15. The taste-led For You mode uses 0.65, 0.20, and 0.15. Personalized results exclude already-rated movies and each result has a reason derived from the signals available to it. When a profile has limited preference data, the popularity signal provides a fallback.

## My Taste

My Taste reads only the selected MovieLens user’s actual rating records. It reports the number of rated movies and their average rating, counts rated-movie interactions by genre, compares the user’s average score within each genre, and lists their highest-rated movies.

These are two different observations: **frequency** means how many rated titles are associated with a genre; **genre average** means the mean rating this profile assigned to titles in that genre. A movie with several genre labels contributes to each. A profile with fewer than five ratings receives an explicit low-data message.

## Genre discovery and browsing

Genre options are derived from the actual labels in `movies.csv`; “(no genres listed)” is handled as missing genre data. Home-page rows and genre pages rank movies using dataset ratings and weighted popularity. Highest-rated selections require a minimum rating count, while Hidden Gems require at least five ratings, an average of at least 3.8, and a rating count below a dataset-based cutoff.

Explore Movies provides title search, genre and minimum-average filters, sorting, and pagination. Movie details show dataset average/count, release year when present in the title, rating distribution, and genre-similar movies.

## Analytics

All chart inputs are calculated from loaded MovieLens data:

- Genre distribution counts unique catalog movies for each genre. Counts overlap for movies with multiple genres.
- User rating distribution counts each half-star rating value.
- Top 10 movies use their real rating counts.
- Average rating versus rating count uses a deterministic, bounded sample of rated movies.
- Average rating by genre uses rating records associated with movies in that genre; multi-genre films contribute to each one.
- Monthly activity uses Unix timestamps from `ratings.csv`, converted to UTC months.
- The optional user taste chart counts that selected user’s rated-movie interactions by genre.

If the Chart.js CDN is unavailable, the page displays compact data bars rather than blank charts. Recommendations and analytics inputs still come from local data.

## System workflow

1. Flask loads and cleans the local CSV files once.
2. scikit-learn builds genre TF-IDF features and cosine user similarity; Pandas/NumPy build the rating matrix and data summaries.
3. Flask routes validate user/movie IDs and call the model.
4. Jinja renders pages; JavaScript requests bounded JSON search/recommendation results and updates chart or filter views.
5. The evaluation module computes offline quality metrics from the rating data.

## Evaluation

`models/evaluation.py` uses a reproducible random 80/20 split with seed 42. For RMSE and MAE, it predicts from the training global, user, and movie averages. For Precision@K and Recall@K, a user-based cosine collaborative filter ranks unseen movies using training ratings. A held-out rating of 4 stars or more is relevant.

Precision@K = relevant items in the first K results ÷ K. Recall@K = relevant held-out items in the first K results ÷ all held-out relevant items for that user. The ranking metrics are macro-averaged across users who have at least one relevant held-out item. RMSE penalizes larger errors more; MAE is the mean absolute rating difference. A random split does not preserve time order, so it is an offline estimate, not proof of future performance.

## Advantages

- Uses a documented public movie-rating dataset with local CSV files.
- Combines content similarity, collaborative filtering, and confidence-adjusted popularity.
- Provides explanations, user taste analysis, and several movie discovery paths.
- Runs on CPU and does not need a separate database or paid API.
- Separates model, loader, Flask routes, templates, static assets, and documentation.

## Limitations

- MovieLens latest-small is a 2018 snapshot without demographic data or explicit view events.
- Genre tags cannot describe plot, cast, mood, or nuanced preference.
- New users and titles have limited collaborative evidence.
- Offline random-split metrics can differ from future user behavior.
- Poster imagery is represented by local stylized placeholders.

## Future scope

Compare temporal and item-based evaluation, try matrix-factorization models, add privacy-conscious local profile persistence, improve accessibility and keyboard testing, and consider optional licensed poster sources.

## Possible viva questions and answers

**Q: What is a user–item matrix?**  
A: A table with users on one axis, movies on the other, and known ratings in cells. A blank cell means there is no rating record for that user/movie pair.

**Q: Why use cosine similarity?**  
A: It compares the direction of feature or rating vectors, making it a practical way to compare patterns even when their absolute sizes differ.

**Q: What does TF-IDF do here?**  
A: It converts genre tokens into weighted numeric features. Common tokens receive less distinguishing weight than rarer ones.

**Q: What is collaborative filtering?**  
A: It uses patterns from users with similar rating behavior to score movies another user has not rated.

**Q: Why use Bayesian popularity?**
A: It uses the number of ratings as confidence. Sparse movie averages are pulled toward the overall mean so tiny samples have less influence.

**Q: How does the My Taste feature distinguish frequency from preference?**
A: Frequency counts rated-movie interactions attached to a genre; preference-by-score compares the average rating that user assigned to movies in that genre.

**Q: How does hybrid scoring work?**
A: It normalizes collaborative, content, and popularity signals and combines them with explicit weights.

**Q: Define Precision@K and Recall@K.**
A: Precision@K is the fraction of the top K results that are relevant. Recall@K is the fraction of all relevant held-out items recovered in those results.

**Q: What is a limitation of the evaluation?**
A: The random split does not preserve chronology. A time-based split would better represent predicting future ratings.
