"""Deterministic holdout evaluation; reported metrics are computed from CSV ratings."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def evaluate(ratings: pd.DataFrame, k: int = 10, seed: int = 42) -> dict:
    train, test = train_test_split(ratings, test_size=.20, random_state=seed)
    global_mean = float(train.rating.mean())
    user_means = train.groupby("userId").rating.mean()
    movie_means = train.groupby("movieId").rating.mean()
    # A transparent user/movie mean baseline predicts ratings for the RMSE and MAE.
    actual = test.rating.to_numpy()
    predicted = np.array([np.mean([global_mean, user_means.get(u, global_mean), movie_means.get(m, global_mean)]) for u, m in zip(test.userId, test.movieId)])
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))

    # Rank unseen movies with user-based cosine CF on the training split; assess
    # against held-out relevant ratings (>= 4 stars) for users with test positives.
    matrix = train.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    ids = matrix.columns.to_numpy()
    user_vectors = matrix.to_numpy(dtype=np.float32)
    similarity = cosine_similarity(user_vectors)
    np.fill_diagonal(similarity, 0)
    recommendation_scores = similarity @ user_vectors
    relevant = test[test.rating >= 4.0].groupby("userId").movieId.apply(set)
    precisions, recalls = [], []
    for uid, truths in relevant.items():
        if uid not in matrix.index: continue
        pos = matrix.index.get_loc(uid)
        u = matrix.loc[uid].to_numpy()
        scores = recommendation_scores[pos].copy()
        scores[u > 0] = -np.inf
        chosen = set(ids[np.argsort(-scores)[:k]])
        hits = len(chosen & truths)
        precisions.append(hits / k)
        recalls.append(hits / len(truths))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "precision_at_k": round(float(np.mean(precisions)) if precisions else 0, 4),
            "recall_at_k": round(float(np.mean(recalls)) if recalls else 0, 4), "k": k, "train_ratings": int(len(train)), "test_ratings": int(len(test)),
            "users_evaluated": len(precisions), "relevance_threshold": 4, "seed": seed}
