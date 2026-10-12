"""Precomputed MovieLens recommendation models and data summaries."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Adjust these values to change the balance of the hybrid recommender.
HYBRID_WEIGHTS = {"collaborative": 0.45, "content": 0.40, "popularity": 0.15}
GENRE_SECTIONS = ("Action", "Romance", "Thriller", "Comedy", "Sci-Fi")
GENRE_FILTERS = (
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)


class MovieRecommender:
    """Build reusable MovieLens features once, then serve fast in-memory queries."""

    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        self.movies = movies.copy().reset_index(drop=True)
        self.ratings = ratings.copy().reset_index(drop=True)
        self.movie_idx = pd.Series(self.movies.index, index=self.movies.movieId)
        self.movie_stats = self.ratings.groupby("movieId").rating.agg(average="mean", count="count")
        self.global_mean = float(self.ratings.rating.mean()) if not self.ratings.empty else 0.0

        # Bayesian shrinkage prevents tiny samples from dominating popularity.
        if not self.movie_stats.empty:
            prior_count = max(10, int(self.movie_stats["count"].quantile(0.75)))
            counts = self.movie_stats["count"]
            self.movie_stats["popularity"] = (
                counts / (counts + prior_count) * self.movie_stats["average"]
                + prior_count / (counts + prior_count) * self.global_mean
            )
            self.movie_stats["pop_norm"] = self._normalize(self.movie_stats["popularity"])
        else:
            self.movie_stats["popularity"] = pd.Series(dtype=float)
            self.movie_stats["pop_norm"] = pd.Series(dtype=float)

        self.movies["year"] = self.movies.title.str.extract(r"\((\d{4})\)\s*$", expand=False)
        self.movies["year"] = pd.to_numeric(self.movies["year"], errors="coerce")
        self.movies["genre_list"] = self.movies.genres.fillna("").map(
            lambda value: [g for g in str(value).split("|") if g and g != "(no genres listed)"]
        )
        self.genres = sorted({genre for values in self.movies.genre_list for genre in values})
        self.genre_filter_options = list(dict.fromkeys([g for g in GENRE_FILTERS if g in self.genres] + self.genres))

        self.tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
        self.genre_matrix = self.tfidf.fit_transform(self.movies.genres.fillna(""))
        self.user_item = self.ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        self.user_item = self.user_item.reindex(columns=self.movies.movieId, fill_value=0)
        # Cosine nearest neighbors compare each movie's rating pattern across users.
        self.user_similarity = cosine_similarity(self.user_item.values).astype(np.float32)
        self.movie_to_pos = {int(mid): i for i, mid in enumerate(self.movies.movieId)}

        # Multi-genre films intentionally contribute to each matching genre.
        self.movie_genres = (
            self.movies[["movieId", "genre_list"]]
            .explode("genre_list")
            .rename(columns={"genre_list": "genre"})
        )
        self.movie_genres = self.movie_genres[self.movie_genres.genre.notna() & self.movie_genres.genre.ne("")]
        self.rating_genres = self.ratings[["userId", "movieId", "rating"]].merge(
            self.movie_genres, on="movieId", how="inner"
        )
        self._analytics_snapshot = self._build_analytics_snapshot()

    @staticmethod
    def _normalize(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        lo, hi = np.nanmin(values), np.nanmax(values)
        return (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)

    def search(self, query: str, limit: int = 30):
        q = str(query or "").strip().casefold()
        if not q:
            return []
        hits = self.movies[self.movies.title.str.casefold().str.contains(q, regex=False, na=False)].head(max(1, min(limit, 100)))
        return [self.movie_card(row.movieId) for row in hits.itertuples()]

    def movie_card(self, movie_id: int):
        row = self.movies[self.movies.movieId == int(movie_id)]
        if row.empty:
            return None
        movie = row.iloc[0]
        stat = self.movie_stats.loc[int(movie_id)] if int(movie_id) in self.movie_stats.index else None
        return {
            "movieId": int(movie_id),
            "title": str(movie.title),
            "genres": list(movie.genre_list),
            "year": int(movie.year) if pd.notna(movie.year) else None,
            "average": round(float(stat.average), 2) if stat is not None else None,
            "count": int(stat["count"]) if stat is not None else 0,
            "popularity": round(float(stat.popularity), 3) if stat is not None else None,
        }

    def browse_movies(
        self, query="", genre="", min_rating=0.0, sort="popularity", page=1, per_page=24
    ):
        """Filter and paginate titles without sending the full catalog to the browser."""
        frame = self.movies.copy()
        q = str(query or "").strip().casefold()
        genre = str(genre or "").strip()
        if q:
            frame = frame[frame.title.str.casefold().str.contains(q, regex=False, na=False)]
        if genre:
            frame = frame[frame.genre_list.map(lambda values: genre in values)]
        stats = self.movie_stats.reindex(frame.movieId)
        frame = frame.assign(
            average=stats.average.to_numpy(),
            rating_count=stats["count"].fillna(0).to_numpy(),
            popularity=stats.popularity.fillna(0).to_numpy(),
        )
        frame = frame[frame.average.fillna(0) >= float(min_rating)]
        sort = sort if sort in {"relevance", "title", "rating", "popularity", "year"} else "popularity"
        if sort in {"relevance", "title"}:
            frame = frame.sort_values("title", key=lambda col: col.str.casefold())
        elif sort == "rating":
            frame = frame.sort_values(["average", "rating_count", "title"], ascending=[False, False, True])
        elif sort == "year":
            frame = frame.sort_values(["year", "title"], ascending=[False, True], na_position="last")
        else:
            frame = frame.sort_values(["rating_count", "popularity", "title"], ascending=[False, False, True])

        per_page = max(1, min(int(per_page), 60))
        total = len(frame)
        pages = max(1, int(np.ceil(total / per_page)))
        page = max(1, min(int(page), pages))
        chunk = frame.iloc[(page - 1) * per_page : page * per_page]
        items = []
        for row in chunk.itertuples():
            item = self.movie_card(row.movieId)
            items.append(item)
        return {"items": items, "total": total, "page": page, "pages": pages, "per_page": per_page}

    def similar(self, movie_id: int, limit: int = 10):
        mid = int(movie_id)
        if mid not in self.movie_to_pos:
            return []
        pos = self.movie_to_pos[mid]
        # Compare one sparse vector against the catalog instead of storing a
        # large dense movie-by-movie similarity matrix in memory.
        scores = cosine_similarity(self.genre_matrix[pos], self.genre_matrix).ravel()
        order = np.argsort(-scores)
        results = []
        seen = {mid}
        for index in order:
            candidate = int(self.movies.iloc[index].movieId)
            if candidate in seen:
                continue
            seen.add(candidate)
            results.append(dict(self.movie_card(candidate), score=round(float(scores[index]), 3), reason="Shares genre features with the selected movie"))
            if len(results) >= max(1, min(int(limit), 30)):
                break
        return results

    def _user_preferences(self, user_id: int):
        user_id = int(user_id)
        if user_id not in self.user_item.index:
            return None
        rated = self.user_item.loc[user_id]
        liked = rated[rated >= 4.0]
        candidates = self.movies[~self.movies.movieId.isin(rated[rated > 0].index)]

        user_pos = self.user_item.index.get_loc(user_id)
        user_sims = self.user_similarity[user_pos].copy()
        user_sims[user_pos] = 0
        neighbor_positions = np.argsort(-user_sims)[:50]
        neighbors = self.user_item.iloc[neighbor_positions]
        weights = user_sims[neighbor_positions]
        denom = neighbors.gt(0).T @ weights
        numerator = neighbors.clip(lower=0).T @ weights
        raw = numerator / np.maximum(denom, 1e-9)
        candidate_ids = set(candidates.movieId.astype(int))
        collab = {
            int(mid): float(raw.loc[mid])
            for mid in raw.index
            if int(mid) in candidate_ids and denom.loc[mid] > 0
        }

        content = {}
        if not liked.empty:
            liked_ids = [int(mid) for mid in liked.index if mid in self.movie_to_pos]
            liked_positions = [self.movie_to_pos[mid] for mid in liked_ids]
            if liked_positions:
                liked_weights = liked.reindex(liked_ids).to_numpy()
                profile = np.average(self.genre_matrix[liked_positions].toarray(), axis=0, weights=np.maximum(liked_weights, 0))
                similarities = cosine_similarity(profile.reshape(1, -1), self.genre_matrix).ravel()
                content = {int(self.movies.iloc[i].movieId): float(similarities[i]) for i in range(len(similarities))}
        return candidates, collab, content, liked

    def recommend(self, kind="hybrid", user_id=None, movie_id=None, limit=10):
        limit = max(1, min(int(limit), len(self.movies)))
        kind = str(kind).lower()
        if kind == "popular":
            ids = self.movie_stats.sort_values("popularity", ascending=False).index[:limit]
            return [dict(self.movie_card(mid), score=round(float(self.movie_stats.loc[mid].popularity), 3), reason="Highly rated with Bayesian rating-count adjustment") for mid in ids]
        if kind == "similar":
            return self.similar(movie_id, limit) if movie_id else []
        if kind not in {"for_you", "hybrid"} or user_id is None or int(user_id) not in self.user_item.index:
            return []

        preferences = self._user_preferences(int(user_id))
        if preferences is None:
            return []
        candidates, collab, content, liked = preferences
        if candidates.empty:
            return []
        candidate_ids = candidates.movieId.to_numpy(dtype=int)
        collab_values = self._normalize(np.array([collab.get(mid, self.global_mean) for mid in candidate_ids]))
        content_values = self._normalize(np.array([content.get(mid, 0.0) for mid in candidate_ids]))
        popularity_values = np.array([self.movie_stats.loc[mid, "pop_norm"] if mid in self.movie_stats.index else 0 for mid in candidate_ids])
        weights = {"collaborative": 0.65, "content": 0.20, "popularity": 0.15} if kind == "for_you" else HYBRID_WEIGHTS
        scores = weights["collaborative"] * collab_values + weights["content"] * content_values + weights["popularity"] * popularity_values
        order = np.argsort(-scores)
        rated_genres = sorted({g for mid in liked.index if mid in self.movie_to_pos for g in self.movies.iloc[self.movie_to_pos[mid]].genre_list})
        genre_text = ", ".join(rated_genres[:2])

        output, seen = [], set()
        for index in order:
            mid = int(candidate_ids[index])
            if mid in seen:
                continue
            seen.add(mid)
            reason = (
                "Popular with users who share your rating patterns" if collab.get(mid, 0) > 0
                else f"Matches genres from movies you rated highly: {genre_text}" if content.get(mid, 0) > 0 and genre_text
                else "Highly rated by MovieLens users"
            )
            output.append(dict(self.movie_card(mid), score=round(float(scores[index]), 3), reason=reason))
            if len(output) >= limit:
                break
        return output

    def user_taste(self, user_id: int, min_ratings: int = 5):
        """Summarize rated-movie history by genre frequency and rating average."""
        user_id = int(user_id)
        if user_id not in self.user_item.index:
            return None
        history = self.ratings[self.ratings.userId == user_id]
        genre_rows = self.rating_genres[self.rating_genres.userId == user_id]
        genre_summary = (
            genre_rows.groupby("genre").rating.agg(rated_movie_count="count", average_rating="mean")
            .sort_values(["rated_movie_count", "average_rating"], ascending=[False, False])
        )
        frequency = [
            {"genre": genre, "count": int(row.rated_movie_count), "average": round(float(row.average_rating), 2)}
            for genre, row in genre_summary.iterrows()
        ]
        highly_rated = sorted(frequency, key=lambda item: (-item["average"], -item["count"], item["genre"]))
        preferred = []
        movie_lookup = self.movies.set_index("movieId")
        for rating in history.sort_values(["rating", "movieId"], ascending=[False, True]).head(8).itertuples():
            movie = movie_lookup.loc[rating.movieId]
            preferred.append({"movieId": int(rating.movieId), "title": str(movie.title), "rating": float(rating.rating), "genres": list(movie.genre_list)})

        enough_data = len(history) >= min_ratings
        frequent_names = ", ".join(item["genre"] for item in frequency[:3])
        high_names = ", ".join(item["genre"] for item in highly_rated[:3])
        if not enough_data:
            summary = f"Only {len(history)} rated movie(s) are available. More ratings are needed for a reliable taste summary."
        elif frequency:
            summary = f"This profile rates {frequent_names} movies most often. Its highest average ratings are for {high_names}."
        else:
            summary = "This profile has too few genre-tagged ratings for a reliable summary."

        return {
            "user_id": user_id,
            "rated_movie_count": int(len(history)),
            "average_rating": round(float(history.rating.mean()), 2) if not history.empty else None,
            "genre_frequency": frequency,
            "highest_rated_genres": highly_rated[:8],
            "preferred_movies": preferred,
            "genre_distribution": {item["genre"]: item["count"] for item in frequency},
            "enough_data": enough_data,
            "summary": summary,
        }

    def details(self, movie_id: int):
        card = self.movie_card(movie_id)
        if card is None:
            return None
        ratings = self.ratings[self.ratings.movieId == int(movie_id)]
        bins = np.arange(0.5, 5.1, 0.5)
        distribution = ratings.rating.value_counts().reindex(bins, fill_value=0)
        card["distribution"] = {str(value): int(distribution.loc[value]) for value in bins}
        card["similar"] = self.similar(movie_id, 6)
        return card

    def analytics(self, user_id=None):
        """Return cached dataset charts plus an optional user's real taste profile."""
        data = dict(self._analytics_snapshot)
        data["selected_taste"] = self.user_taste(user_id) if user_id in self.user_item.index else None
        return data

    def _build_analytics_snapshot(self):
        ratings = self.ratings
        rating_dist = ratings.rating.value_counts().reindex(np.arange(0.5, 5.1, 0.5), fill_value=0).sort_index()
        top_ids = self.movie_stats.sort_values(["count", "average"], ascending=[False, False]).head(10).index
        genre_counts = self.movie_genres.groupby("genre").movieId.nunique().sort_values(ascending=False)
        genre_rating = self.rating_genres.groupby("genre").rating.agg(["mean", "count"]).sort_values("mean", ascending=False)

        rated_movies = self.movie_stats.join(self.movies.set_index("movieId")[["title"]], how="left").sort_values("count")
        if len(rated_movies) > 500:
            positions = np.linspace(0, len(rated_movies) - 1, 500, dtype=int)
            rated_movies = rated_movies.iloc[positions]
        scatter = [
            {"title": str(row.title), "average": round(float(row.average), 3), "count": int(row.count)}
            for row in rated_movies.itertuples()
        ]
        timestamp_column = ratings.get("timestamp")
        if timestamp_column is not None:
            timestamps = pd.to_datetime(timestamp_column, unit="s", errors="coerce", utc=True)
            monthly = timestamps.dt.strftime("%Y-%m").value_counts().sort_index()
            activity = {str(month): int(count) for month, count in monthly.items() if month != "NaT"}
        else:
            activity = {}

        genre_rating_counts = self.rating_genres.groupby("genre").rating.count().sort_values(ascending=False)
        best_genre = str(genre_rating_counts.index[0]) if not genre_rating_counts.empty else "—"
        best_genre_count = int(genre_rating_counts.iloc[0]) if not genre_rating_counts.empty else 0
        return {
            "users": int(ratings.userId.nunique()),
            "movies": int(self.movies.movieId.nunique()),
            "ratings": int(len(ratings)),
            "average": round(float(ratings.rating.mean()), 3) if not ratings.empty else 0,
            "most_popular_genre": best_genre,
            "most_popular_genre_count": best_genre_count,
            "rating_distribution": {str(key): int(value) for key, value in rating_dist.items()},
            "top_rated": [dict(self.movie_card(mid), score=int(self.movie_stats.loc[mid, "count"])) for mid in top_ids],
            "genres": {str(key): int(value) for key, value in genre_counts.items()},
            "genre_ratings": {str(key): {"average": round(float(row["mean"]), 3), "count": int(row["count"])} for key, row in genre_rating.iterrows()},
            "scatter_movies": scatter,
            "rating_activity": activity,
            "user_activity": ratings.groupby("userId").size().describe().round(2).to_dict(),
            "selected_taste": None,
        }

    def hidden_gems(self, limit=8):
        """Find well-rated titles with enough, but not blockbuster-scale, votes."""
        if self.movie_stats.empty:
            return []
        cap = max(20, int(self.movie_stats["count"].quantile(0.75)))
        eligible = self.movie_stats[(self.movie_stats["count"] >= 5) & (self.movie_stats["count"] <= cap) & (self.movie_stats.average >= 3.8)]
        eligible = eligible.sort_values(["popularity", "count"], ascending=[False, False]).head(limit)
        return [dict(self.movie_card(mid), score=round(float(row.popularity), 3), reason="Strong ratings with a smaller rating count") for mid, row in eligible.iterrows()]

    def highest_rated(self, limit=8, minimum_votes=20):
        """High-mean titles with a modest minimum sample-size safeguard."""
        eligible = self.movie_stats[self.movie_stats["count"] >= minimum_votes]
        eligible = eligible.sort_values(["average", "count"], ascending=[False, False]).head(limit)
        return [dict(self.movie_card(mid), score=round(float(row.average), 3), reason=f"Average rating from {int(row['count'])} MovieLens ratings") for mid, row in eligible.iterrows()]
