"""Basic behavior checks against the bundled MovieLens CSV dataset."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_loader import load_movielens
from models.recommender import MovieRecommender
from models.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]


class RecommenderChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies, cls.ratings, _, _ = load_movielens(ROOT / "data")
        cls.engine = MovieRecommender(cls.movies, cls.ratings)

    def test_dataset_loading(self):
        self.assertGreater(len(self.movies), 9000)
        self.assertGreater(len(self.ratings), 100000)
        self.assertTrue(self.movies.movieId.is_unique)

    def test_movie_search(self):
        self.assertTrue(any("Toy Story" in x["title"] for x in self.engine.search("toy story")))
        self.assertEqual(self.engine.search(""), [])

    def test_similar_movies_and_invalid_id(self):
        self.assertEqual(len(self.engine.similar(1, 5)), 5)
        self.assertEqual(self.engine.similar(-5), [])

    def test_user_recommendations_and_invalid_user(self):
        result = self.engine.recommend("hybrid", user_id=1, limit=5)
        self.assertEqual(len(result), 5)
        already_rated = set(self.ratings[self.ratings.userId == 1].movieId)
        self.assertFalse(any(x["movieId"] in already_rated for x in result))
        self.assertEqual(self.engine.recommend("hybrid", user_id=999999), [])

    def test_recommendation_count_and_popularity(self):
        self.assertEqual(len(self.engine.recommend("popular", limit=7)), 7)

    def test_evaluation_metrics_are_computed(self):
        result = evaluate(self.ratings, k=5)
        self.assertTrue(0 <= result["precision_at_k"] <= 1)
        self.assertTrue(0 <= result["recall_at_k"] <= 1)
        self.assertGreater(result["rmse"], 0)
        self.assertGreater(result["mae"], 0)


if __name__ == "__main__":
    unittest.main()
