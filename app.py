"""Flask entry point for the MovieLens recommendation system."""
from pathlib import Path
import os
from flask import Flask, jsonify, render_template, request
from utils.data_loader import load_movielens
from models.recommender import MovieRecommender
from models.evaluation import evaluate

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
movies, ratings, links, tags = load_movielens(BASE_DIR / "data")
engine = MovieRecommender(movies, ratings)
evaluation = evaluate(ratings)


@app.get("/")
def home():
    stats = engine.analytics()
    popular = engine.recommend("popular", limit=8)
    featured = [engine.movie_card(mid) for mid in movies.movieId.tail(8)]
    return render_template("index.html", stats=stats, popular=popular, featured=featured)


@app.get("/recommendations")
def recommendations_page():
    return render_template("recommendations.html", user_ids=engine.user_item.index.tolist())


@app.get("/search")
def search_page(): return render_template("search.html")


@app.get("/movie/<int:movie_id>")
def movie_page(movie_id):
    detail = engine.details(movie_id)
    if detail is None: return render_template("not_found.html"), 404
    return render_template("movie.html", movie=detail)


@app.get("/analytics")
def analytics_page(): return render_template("analytics.html", stats=engine.analytics())


@app.get("/evaluation")
def evaluation_page(): return render_template("evaluation.html", result=evaluation)


@app.get("/about")
def about_page(): return render_template("about.html")


@app.get("/api/search")
def api_search():
    try: limit = max(1, min(100, int(request.args.get("limit", 30))))
    except (TypeError, ValueError): limit = 30
    return jsonify(engine.search(request.args.get("q", ""), limit))


@app.get("/api/recommendations")
def api_recommendations():
    try:
        kind = request.args.get("type", "hybrid")
        limit = max(1, min(20, int(request.args.get("limit", 10))))
        user_id = request.args.get("user_id", type=int)
        movie_id = request.args.get("movie_id", type=int)
        if kind in ("for_you", "hybrid") and user_id not in engine.user_item.index:
            return jsonify({"error": "Choose a valid MovieLens user ID."}), 400
        if kind == "similar" and (movie_id is None or movie_id not in engine.movie_to_pos):
            return jsonify({"error": "Choose a valid movie to find similar titles."}), 400
        return jsonify(engine.recommend(kind, user_id, movie_id, limit))
    except (TypeError, ValueError): return jsonify({"error": "Invalid recommendation settings."}), 400


@app.get("/api/similar/<int:movie_id>")
def api_similar(movie_id):
    if movie_id not in engine.movie_to_pos: return jsonify({"error": "Movie not found."}), 404
    return jsonify(engine.similar(movie_id))


@app.get("/api/movie/<int:movie_id>")
def api_movie(movie_id):
    detail = engine.details(movie_id)
    return (jsonify(detail), 200) if detail else (jsonify({"error": "Movie not found."}), 404)


@app.get("/api/analytics")
def api_analytics(): return jsonify(engine.analytics())


@app.get("/api/evaluation")
def api_evaluation(): return jsonify(evaluation)


if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
    )
