"""Load and clean local MovieLens CSV files."""
from pathlib import Path
import pandas as pd


def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    required = [root / "movies.csv", root / "ratings.csv"]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required dataset file(s) in {root}: {', '.join(missing)}")
    movies = pd.read_csv(root / "movies.csv").drop_duplicates("movieId").dropna(subset=["movieId", "title"])
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    movies = movies.dropna(subset=["movieId"])
    movies["movieId"] = movies["movieId"].astype(int)
    movies["genres"] = movies["genres"].fillna("").replace("(no genres listed)", "")
    ratings = pd.read_csv(root / "ratings.csv").drop_duplicates(["userId", "movieId"])
    for col in ["userId", "movieId", "rating"]:
        ratings[col] = pd.to_numeric(ratings[col], errors="coerce")
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings = ratings[ratings.rating.between(0.5, 5.0)]
    ratings[["userId", "movieId"]] = ratings[["userId", "movieId"]].astype(int)
    ratings = ratings[ratings.movieId.isin(movies.movieId)]
    links = pd.read_csv(root / "links.csv") if (root / "links.csv").exists() else pd.DataFrame()
    tags = pd.read_csv(root / "tags.csv") if (root / "tags.csv").exists() else pd.DataFrame()
    return movies.reset_index(drop=True), ratings.reset_index(drop=True), links, tags
