"""Flask entry point for CineMind's MovieLens recommendation platform."""
from pathlib import Path
import os

from flask import Flask, jsonify, render_template, request

from models.evaluation import evaluate
from models.recommender import GENRE_SECTIONS, MovieRecommender
from utils.data_loader import load_movielens


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
movies, ratings, links, tags = load_movielens(BASE_DIR / "data")
engine = MovieRecommender(movies, ratings)
evaluation = evaluate(ratings)


def _selected_user(raw_value):
    """Return a valid MovieLens ID, or None for an absent/invalid selection."""
    try:
        user_id = int(raw_value)
    except (TypeError, ValueError):
        return None
    return user_id if user_id in engine.user_item.index else None


def _positive_page(value):
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _minimum_rating(value):
    try:
        return max(0.0, min(5.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


@app.get("/")
def home():
    stats = engine.analytics()
    selected_user = _selected_user(request.args.get("user_id"))
    featured = [engine.movie_card(mid) for mid in movies.movieId.tail(8)]
    genre_rows = [
        {"genre": genre, "movies": engine.browse_movies(genre=genre, sort="popularity", per_page=8)["items"]}
        for genre in GENRE_SECTIONS if genre in engine.genres
    ]
    personalized = engine.recommend("for_you", selected_user, limit=8) if selected_user is not None else []
    raw_user = request.args.get("user_id", "").strip()
    profile_error = "" if not raw_user or selected_user is not None else "That user ID is not in the MovieLens dataset."
    return render_template(
        "index.html", stats=stats, popular=engine.recommend("popular", limit=8),
        featured=featured, genre_rows=genre_rows, hidden_gems=engine.hidden_gems(), user_ids=engine.user_item.index.tolist(),
        highest_rated=engine.highest_rated(), selected_user=selected_user, personalized=personalized, profile_error=profile_error,
    )


@app.get("/recommendations")
def recommendations_page():
    return render_template("recommendations.html", user_ids=engine.user_item.index.tolist())


@app.get("/for-you")
def for_you_page():
    raw_user = request.args.get("user_id", "").strip()
    selected_user = _selected_user(raw_user) if raw_user else None
    error = "" if not raw_user or selected_user is not None else "That user ID is not in the MovieLens dataset. Choose an ID from 1 to 610."
    try:
        limit = int(request.args.get("limit", "10"))
        if limit not in (10, 20, 30):
            limit = 10
    except ValueError:
        limit = 10
    genre = request.args.get("genre", "")
    minimum = _minimum_rating(request.args.get("min_rating", "0"))
    sort = request.args.get("sort", "relevance")
    results = []
    if selected_user is not None:
        # Rank the full unseen catalog before applying filters, then return only
        # the requested 10/20/30 cards to the page.
        results = engine.recommend("for_you", selected_user, limit=len(movies))
        if genre:
            results = [item for item in results if genre in item["genres"]]
        results = [item for item in results if (item["average"] or 0) >= minimum]
        if sort == "rating":
            results.sort(key=lambda item: (item["average"] or 0, item["count"]), reverse=True)
        elif sort == "popularity":
            results.sort(key=lambda item: (item["count"], item["popularity"] or 0), reverse=True)
        else:
            sort = "relevance"
            results.sort(key=lambda item: item["score"], reverse=True)
        results = results[:limit]
    return render_template(
        "for_you.html", user_id=raw_user, selected_user=selected_user, error=error,
        recommendations=results, limit=limit, genre=genre, min_rating=minimum,
        sort=sort, genres=engine.genre_filter_options, user_ids=engine.user_item.index.tolist(),
    )


@app.get("/my-taste")
def my_taste_page():
    raw_user = request.args.get("user_id", "").strip()
    selected_user = _selected_user(raw_user) if raw_user else None
    error = "" if not raw_user or selected_user is not None else "That user ID is not in the MovieLens dataset. Choose an ID from 1 to 610."
    taste = engine.user_taste(selected_user) if selected_user is not None else None
    return render_template("my_taste.html", user_id=raw_user, selected_user=selected_user, error=error, taste=taste, user_ids=engine.user_item.index.tolist())


@app.get("/genres")
def genres_page():
    genre = request.args.get("genre", "").strip()
    valid_genre = genre if genre in engine.genres else ""
    result = engine.browse_movies(genre=valid_genre, sort="popularity", page=_positive_page(request.args.get("page", "1")), per_page=24) if valid_genre else None
    return render_template(
        "genres.html", genres=engine.genres, selected_genre=valid_genre,
        movies=result["items"] if result else [], total=result["total"] if result else 0,
        page=result["page"] if result else 1, pages=result["pages"] if result else 1,
    )


@app.get("/search")
def search_page():
    filters = {
        "query": request.args.get("q", ""), "genre": request.args.get("genre", ""),
        "min_rating": _minimum_rating(request.args.get("min_rating", "0")),
        "sort": request.args.get("sort", "popularity"), "page": _positive_page(request.args.get("page", "1")),
    }
    result = engine.browse_movies(**filters, per_page=24)
    return render_template("search.html", result=result, filters=filters, genres=engine.genre_filter_options)


@app.get("/movie/<int:movie_id>")
def movie_page(movie_id):
    detail = engine.details(movie_id)
    if detail is None:
        return render_template("not_found.html"), 404
    return render_template("movie.html", movie=detail)


@app.get("/analytics")
def analytics_page():
    raw_user = request.args.get("user_id", "").strip()
    selected_user = _selected_user(raw_user) if raw_user else None
    error = "" if not raw_user or selected_user is not None else "Choose a valid MovieLens user ID."
    stats = engine.analytics(selected_user)
    return render_template("analytics.html", stats=stats, user_ids=engine.user_item.index.tolist(), selected_user=selected_user, raw_user=raw_user, error=error)


@app.get("/evaluation")
def evaluation_page():
    return render_template("evaluation.html", result=evaluation)


@app.get("/about")
def about_page():
    return render_template("about.html")


@app.get("/api/search")
def api_search():
    try:
        limit = max(1, min(100, int(request.args.get("limit", 30))))
    except (TypeError, ValueError):
        limit = 30
    return jsonify(engine.search(request.args.get("q", ""), limit))


@app.get("/api/movies")
def api_movies():
    result = engine.browse_movies(
        query=request.args.get("q", ""), genre=request.args.get("genre", ""),
        min_rating=_minimum_rating(request.args.get("min_rating", "0")),
        sort=request.args.get("sort", "popularity"), page=_positive_page(request.args.get("page", "1")),
        per_page=24,
    )
    return jsonify(result)


@app.get("/api/recommendations")
def api_recommendations():
    try:
        kind = request.args.get("type", "hybrid")
        if kind not in {"for_you", "hybrid", "popular", "similar"}:
            return jsonify({"error": "Choose a supported recommendation type."}), 400
        limit = max(1, min(30, int(request.args.get("limit", 10))))
        user_id = request.args.get("user_id", type=int)
        movie_id = request.args.get("movie_id", type=int)
        if kind in ("for_you", "hybrid") and user_id not in engine.user_item.index:
            return jsonify({"error": "Choose a valid MovieLens user ID."}), 400
        if kind == "similar" and (movie_id is None or movie_id not in engine.movie_to_pos):
            return jsonify({"error": "Choose a valid movie to find similar titles."}), 400
        return jsonify(engine.recommend(kind, user_id, movie_id, limit))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid recommendation settings."}), 400


@app.get("/api/users/<int:user_id>/taste")
def api_user_taste(user_id):
    taste = engine.user_taste(user_id)
    return (jsonify(taste), 200) if taste is not None else (jsonify({"error": "User ID not found."}), 404)


@app.get("/api/similar/<int:movie_id>")
def api_similar(movie_id):
    if movie_id not in engine.movie_to_pos:
        return jsonify({"error": "Movie not found."}), 404
    return jsonify(engine.similar(movie_id))


@app.get("/api/movie/<int:movie_id>")
def api_movie(movie_id):
    detail = engine.details(movie_id)
    return (jsonify(detail), 200) if detail else (jsonify({"error": "Movie not found."}), 404)


@app.get("/api/analytics")
def api_analytics():
    raw_user = request.args.get("user_id")
    if raw_user and _selected_user(raw_user) is None:
        return jsonify({"error": "Choose a valid MovieLens user ID."}), 400
    return jsonify(engine.analytics(_selected_user(raw_user) if raw_user else None))


@app.get("/api/evaluation")
def api_evaluation():
    return jsonify(evaluation)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
