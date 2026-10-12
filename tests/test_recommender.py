"""Basic behavior checks against the bundled MovieLens CSV dataset."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_loader import load_movielens
from models.recommender import MovieRecommender
from models.evaluation import evaluate
from models.recommender import GENRE_FILTERS
from app import app as flask_app, engine as app_engine

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

    def test_user_taste_uses_real_rating_history(self):
        taste = self.engine.user_taste(1)
        user_ratings = self.ratings[self.ratings.userId == 1]
        self.assertEqual(taste["rated_movie_count"], len(user_ratings))
        self.assertAlmostEqual(taste["average_rating"], round(user_ratings.rating.mean(), 2))
        self.assertTrue(taste["genre_frequency"])
        self.assertEqual(self.engine.user_taste(999999), None)

    def test_genre_filters_match_actual_catalog_labels(self):
        actual = set(self.engine.genres)
        self.assertTrue(set(GENRE_FILTERS).issubset(actual))
        for genre in GENRE_FILTERS:
            page = self.engine.browse_movies(genre=genre, sort="title", per_page=6)
            self.assertGreater(page["total"], 0, genre)
            self.assertTrue(all(genre in movie["genres"] for movie in page["items"]), genre)
        self.assertEqual(self.engine.browse_movies(genre="Not A MovieLens Genre")["total"], 0)

    def test_filtered_movie_browse_and_pagination(self):
        first = self.engine.browse_movies(query="toy", sort="rating", per_page=3)
        self.assertEqual(len(first["items"]), 3)
        self.assertGreater(first["pages"], 1)
        second = self.engine.browse_movies(query="toy", sort="rating", per_page=3, page=2)
        self.assertFalse({x["movieId"] for x in first["items"]} & {x["movieId"] for x in second["items"]})
        high_rated = self.engine.browse_movies(min_rating=4.5, per_page=10)
        self.assertTrue(all(movie["average"] is not None and movie["average"] >= 4.5 for movie in high_rated["items"]))

    def test_for_you_returns_unique_unseen_movies_up_to_30(self):
        results = self.engine.recommend("for_you", user_id=1, limit=30)
        rated_ids = set(self.ratings[self.ratings.userId == 1].movieId)
        result_ids = [movie["movieId"] for movie in results]
        self.assertEqual(len(results), 30)
        self.assertEqual(len(result_ids), len(set(result_ids)))
        self.assertFalse(set(result_ids) & rated_ids)
        self.assertTrue(all(movie.get("reason") for movie in results))

    def test_flask_routes_and_json_endpoints(self):
        client = flask_app.test_client()
        for path in ["/", "/for-you", "/my-taste", "/genres", "/search", "/recommendations", "/analytics", "/evaluation", "/about", "/movie/1"]:
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 200)
        self.assertEqual(client.get("/movie/99999999").status_code, 404)
        self.assertEqual(client.get("/api/search?q=").get_json(), [])
        self.assertEqual(client.get("/api/recommendations?type=for_you&user_id=99999").status_code, 400)
        self.assertEqual(client.get("/api/users/99999/taste").status_code, 404)
        taste_page = client.get("/my-taste?user_id=1")
        self.assertEqual(taste_page.status_code, 200)
        self.assertIn(b"Genre preference details", taste_page.data)
        for_you = client.get("/for-you?user_id=1&limit=10&genre=Comedy&sort=rating")
        self.assertEqual(for_you.status_code, 200)
        self.assertIn(b"Recommendations for user 1", for_you.data)
        self.assertEqual(client.get("/genres?genre=Film-Noir&page=1").status_code, 200)
        self.assertEqual(client.get("/search?q=toy&genre=Animation&sort=rating&min_rating=3").status_code, 200)
        self.assertEqual(client.get("/for-you?user_id=99999").status_code, 200)
        self.assertIn(b"not in the MovieLens dataset", client.get("/for-you?user_id=99999").data)
        self.assertEqual(client.get("/api/recommendations?type=popular&limit=30").get_json().__len__(), 30)
        browse_api = client.get("/api/movies?genre=Sci-Fi&min_rating=4.0&sort=rating&page=1").get_json()
        self.assertLessEqual(len(browse_api["items"]), 24)
        self.assertTrue(all("Sci-Fi" in movie["genres"] and movie["average"] >= 4 for movie in browse_api["items"]))

    def test_analytics_are_actual_and_bounded(self):
        stats = app_engine.analytics(user_id=1)
        self.assertEqual(stats["ratings"], len(app_engine.ratings))
        self.assertAlmostEqual(stats["average"], round(app_engine.ratings.rating.mean(), 3))
        self.assertEqual(len(stats["rating_distribution"]), 10)
        self.assertTrue(stats["genres"])
        self.assertTrue(stats["rating_activity"])
        self.assertLessEqual(len(stats["scatter_movies"]), 500)
        self.assertIsNotNone(stats["selected_taste"])
        client = flask_app.test_client()
        analytics = client.get("/api/analytics?user_id=1").get_json()
        self.assertIn("genre_ratings", analytics)
        self.assertIn("selected_taste", analytics)
        page = client.get("/analytics?user_id=1").data
        for chart_id in [b"genreCountsChart", b"ratingChart", b"popularChart", b"movieScatterChart", b"genreRatingChart", b"activityChart", b"userTasteChart"]:
            self.assertIn(chart_id, page)
        self.assertIn(b"window.analyticsData", page)


if __name__ == "__main__":
    unittest.main()
