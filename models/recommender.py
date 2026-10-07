"""Precomputed MovieLens recommendation models for the Flask application."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Adjust these values to change the balance of the hybrid recommender.
HYBRID_WEIGHTS = {"collaborative": 0.45, "content": 0.40, "popularity": 0.15}


class MovieRecommender:
    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        self.movies, self.ratings = movies.copy(), ratings.copy()
        self.movie_idx = pd.Series(self.movies.index, index=self.movies.movieId)
        self.movie_stats = self.ratings.groupby("movieId").rating.agg(average="mean", count="count")
        self.global_mean = float(self.ratings.rating.mean())
        # Bayesian shrinkage avoids placing a movie with only a couple of ratings at the top.
        m = max(10, int(self.movie_stats["count"].quantile(.75)))
        self.movie_stats["popularity"] = ((self.movie_stats["count"] / (self.movie_stats["count"] + m)) * self.movie_stats["average"] + (m / (self.movie_stats["count"] + m)) * self.global_mean)
        self.movie_stats["pop_norm"] = self._normalize(self.movie_stats["popularity"])
        self.tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
        self.genre_matrix = self.tfidf.fit_transform(self.movies.genres.fillna(""))
        self.user_item = self.ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        self.user_item = self.user_item.reindex(columns=self.movies.movieId, fill_value=0)
        # Cosine nearest neighbors compare each movie's rating pattern across users.
        self.user_similarity = cosine_similarity(self.user_item.values).astype(np.float32)
        self.movie_to_pos = {int(mid): i for i, mid in enumerate(self.movies.movieId)}

    @staticmethod
    def _normalize(values):
        values = np.asarray(values, dtype=float)
        lo, hi = np.nanmin(values), np.nanmax(values)
        return (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)

    def search(self, query: str, limit: int = 30):
        q = str(query or "").strip().casefold()
        if not q: return []
        hits = self.movies[self.movies.title.str.casefold().str.contains(q, regex=False, na=False)].head(max(1, min(limit, 100)))
        return [self.movie_card(row.movieId) for row in hits.itertuples()]

    def movie_card(self, movie_id: int):
        row = self.movies[self.movies.movieId == int(movie_id)]
        if row.empty: return None
        movie = row.iloc[0]
        stat = self.movie_stats.loc[int(movie_id)] if int(movie_id) in self.movie_stats.index else None
        return {"movieId": int(movie_id), "title": str(movie.title), "genres": [g for g in str(movie.genres).split("|") if g],
                "average": round(float(stat.average), 2) if stat is not None else None, "count": int(stat["count"]) if stat is not None else 0,
                "popularity": round(float(stat.popularity), 3) if stat is not None else None}

    def similar(self, movie_id: int, limit: int = 10):
        mid = int(movie_id)
        if mid not in self.movie_to_pos: return []
        pos = self.movie_to_pos[mid]
        # Compute one sparse-vector similarity row on demand; a full movie-by-movie
        # matrix would need hundreds of MB and is unnecessary for a single query.
        scores = cosine_similarity(self.genre_matrix[pos], self.genre_matrix).ravel()
        order = np.argsort(-scores)
        return [dict(self.movie_card(self.movies.iloc[i].movieId), score=round(float(scores[i]), 3), reason="Shares genre features with the selected movie") for i in order if i != pos][:limit]

    def _user_preferences(self, user_id: int):
        if int(user_id) not in self.user_item.index: return None
        rated = self.user_item.loc[int(user_id)]
        liked = rated[rated >= 4.0]
        candidates = self.movies[~self.movies.movieId.isin(rated[rated > 0].index)]
        collab, content = {}, {}
        # Similar users' positive ratings vote for unseen movies.
        user_vec = rated.values.reshape(1, -1)
        user_sims = self.user_similarity[self.user_item.index.get_loc(int(user_id))].copy()
        user_sims[self.user_item.index.get_loc(int(user_id))] = 0
        neighbor_positions = np.argsort(-user_sims)[:50]
        neighbors = self.user_item.iloc[neighbor_positions]
        weights = user_sims[neighbor_positions]
        denom = (neighbors.gt(0).T @ weights)
        numerator = (neighbors.clip(lower=0).T @ weights)
        raw = numerator / np.maximum(denom, 1e-9)
        collab = {int(mid): float(raw.loc[mid]) for mid in raw.index if mid in candidates.movieId.values and denom.loc[mid] > 0}
        # Genre profiles are a rating-weighted sum of liked movie vectors.
        if not liked.empty:
            liked_positions = [self.movie_to_pos[mid] for mid in liked.index if mid in self.movie_to_pos]
            if liked_positions:
                liked_weights = liked.loc[[mid for mid in liked.index if mid in self.movie_to_pos]].values
                weights2 = np.maximum(liked_weights, 0)
                profile = np.average(self.genre_matrix[liked_positions].toarray(), axis=0, weights=weights2)
                sims = cosine_similarity(profile.reshape(1, -1), self.genre_matrix).ravel()
                content = {int(self.movies.iloc[i].movieId): float(sims[i]) for i in range(len(sims))}
        return candidates, collab, content, liked

    def recommend(self, kind="hybrid", user_id=None, movie_id=None, limit=10):
        limit = max(1, min(int(limit), 20))
        kind = str(kind).lower()
        if kind == "popular":
            ids = self.movie_stats.sort_values("popularity", ascending=False).index.tolist()
            return [dict(self.movie_card(mid), score=round(float(self.movie_stats.loc[mid].popularity), 3), reason="Highly rated with Bayesian rating-count adjustment") for mid in ids[:limit]]
        if kind == "similar": return self.similar(movie_id, limit) if movie_id else []
        if user_id is None or int(user_id) not in self.user_item.index: return []
        result = self._user_preferences(int(user_id))
        candidates, collab, content, liked = result
        if candidates.empty: return []
        c_norm = self._normalize(np.array([collab.get(int(mid), self.global_mean) for mid in candidates.movieId]))
        g_norm = self._normalize(np.array([content.get(int(mid), 0) for mid in candidates.movieId]))
        p_norm = np.array([self.movie_stats.loc[mid, "pop_norm"] if mid in self.movie_stats.index else 0 for mid in candidates.movieId])
        if kind == "for_you": weights = {"collaborative": .65, "content": .20, "popularity": .15}
        else: weights = HYBRID_WEIGHTS
        scores = weights["collaborative"] * c_norm + weights["content"] * g_norm + weights["popularity"] * p_norm
        order = np.argsort(-scores)[:limit]
        output = []
        for i in order:
            mid = int(candidates.iloc[i].movieId)
            reason = "Popular with users who share your rating patterns" if collab.get(mid, 0) else "Matches genres from movies you rated highly" if content.get(mid, 0) else "Highly rated by MovieLens users"
            output.append(dict(self.movie_card(mid), score=round(float(scores[i]), 3), reason=reason))
        return output

    def details(self, movie_id: int):
        card = self.movie_card(movie_id)
        if card is None: return None
        r = self.ratings[self.ratings.movieId == int(movie_id)]
        bins = np.arange(.5, 5.1, .5)
        distribution = r.rating.value_counts().reindex(bins, fill_value=0)
        card["distribution"] = {str(v): int(distribution.loc[v]) for v in bins}
        card["similar"] = self.similar(movie_id, 6)
        return card

    def analytics(self):
        ratings = self.ratings
        rating_dist = ratings.rating.value_counts().sort_index()
        top_ids = self.movie_stats.sort_values("count", ascending=False).head(10).index
        genre_counts = {}
        for genre_list in self.movies.genres:
            for genre in str(genre_list).split("|"):
                if genre and genre != "(no genres listed)": genre_counts[genre] = genre_counts.get(genre, 0) + 1
        return {"users": int(ratings.userId.nunique()), "movies": int(self.movies.movieId.nunique()), "ratings": int(len(ratings)), "average": round(float(ratings.rating.mean()), 3),
                "rating_distribution": {str(k): int(v) for k, v in rating_dist.items()},
                "top_rated": [dict(self.movie_card(mid), score=int(self.movie_stats.loc[mid, "count"])) for mid in top_ids],
                "genres": dict(sorted(genre_counts.items(), key=lambda x: -x[1])),
                "user_activity": ratings.groupby("userId").size().describe().round(2).to_dict()}
