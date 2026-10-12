const $ = (selector, root = document) => root.querySelector(selector);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[character]));

async function getJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "The request could not be completed.");
  return payload;
}

function movieCard(movie, index = 0) {
  const genres = Array.isArray(movie.genres) ? movie.genres : [];
  const year = movie.year || (movie.title || "").match(/\((\d{4})\)\s*$/)?.[1] || "YEAR N/A";
  const mode = $("#recType")?.value;
  const scoreLabel = mode === "similar" ? "Genre similarity" : mode === "popular" ? "Popularity score" : "Recommendation score";
  const score = movie.score == null ? "" : `<div class="recommendation-reason">${scoreLabel} · ${Number(movie.score).toFixed(3)}</div>`;
  const reason = movie.reason ? `<div class="recommendation-reason">${escapeHtml(movie.reason)}</div>` : "";
  return `<a class="movie-card" href="/movie/${Number(movie.movieId)}">
    <div class="poster poster-${index % 6}"><div class="poster-grain"></div><span class="poster-kicker">CINEMIND · ${escapeHtml(genres[0] || "CINEMA")}</span><i class="fa-solid fa-film poster-icon"></i><div class="poster-title">${escapeHtml((movie.title || "").split(" (")[0].slice(0, 32))}</div><div class="poster-year">${escapeHtml(year)}</div></div>
    <div class="card-info"><strong>${escapeHtml(movie.title)}</strong><div class="genre-line">${escapeHtml(genres.slice(0, 3).join(" · ") || "Uncategorized")}</div>
    <div class="card-meta"><span><i class="fa-solid fa-star"></i> ${movie.average ?? "—"}</span><span>${Number(movie.count || 0).toLocaleString()} ratings</span></div>${score}${reason}</div></a>`;
}

const chartInstances = new Map();
const chartPalette = ["#9182ee", "#72c9dc", "#72d2ae", "#e6c372", "#d387ad", "#6878b7", "#80a9dc", "#a78ed0", "#588f8d", "#d48b61", "#7187b8", "#ab7fb1"];

function showFallback(canvas, rows) {
  if (!canvas) return;
  canvas.hidden = true;
  const old = canvas.parentElement.querySelector(".chart-fallback");
  old?.remove();
  const wrapper = document.createElement("div");
  wrapper.className = "chart-fallback";
  const maxValue = Math.max(1, ...rows.map((row) => Number(row.value) || 0));
  if (!rows.length) {
    wrapper.innerHTML = '<p class="subhead">No dataset records are available for this chart.</p>';
  } else {
    wrapper.innerHTML = rows.slice(0, 16).map((row, index) => {
      const value = Number(row.value) || 0;
      const width = Math.max(1, value / maxValue * 100);
      return `<div class="fallback-row" title="${escapeHtml(row.label)}: ${value.toLocaleString()}"><span class="fallback-label">${escapeHtml(row.label)}</span><span class="fallback-track"><i class="fallback-fill" style="width:${width}%;background:${chartPalette[index % chartPalette.length]}"></i></span><span class="fallback-value">${value.toLocaleString()}</span></div>`;
    }).join("");
  }
  canvas.insertAdjacentElement("afterend", wrapper);
}

function drawChart(canvasId, config, fallbackRows = []) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  chartInstances.get(canvasId)?.destroy?.();
  chartInstances.delete(canvasId);
  canvas.hidden = false;
  canvas.parentElement.querySelector(".chart-fallback")?.remove();
  if (!window.Chart) {
    showFallback(canvas, fallbackRows);
    return;
  }
  const customOptions = config.options || {};
  const scales = customOptions.scales && Object.fromEntries(Object.entries(customOptions.scales).map(([axis, settings]) => [axis, {
    ticks: { color: "#8794aa", ...(settings.ticks || {}) },
    grid: { color: "#263149", ...(settings.grid || {}) },
    ...settings,
  }]));
  chartInstances.set(canvasId, new Chart(canvas, {
    type: config.type,
    data: config.data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 220 },
      ...customOptions,
      plugins: { legend: { labels: { color: "#aeb9cc", boxWidth: 10, padding: 12 } }, ...(customOptions.plugins || {}) },
      ...(scales ? { scales } : {}),
    },
  }));
}

function renderAnalytics(data) {
  if (!data) return;
  const genreEntries = Object.entries(data.genres || {});
  const ratingEntries = Object.entries(data.rating_distribution || {});
  const topMovies = data.top_rated || [];
  const genreRatings = Object.entries(data.genre_ratings || {});
  const activity = Object.entries(data.rating_activity || {});

  drawChart("genreCountsChart", {
    type: "doughnut",
    data: { labels: genreEntries.map(([genre]) => genre), datasets: [{ data: genreEntries.map(([, count]) => count), backgroundColor: chartPalette, borderWidth: 0 }] },
    options: { cutout: "62%", plugins: { legend: { position: "bottom", labels: { boxWidth: 8, padding: 8, font: { size: 9 } } } } },
  }, genreEntries.map(([label, value]) => ({ label, value })));

  drawChart("ratingChart", {
    type: "bar",
    data: { labels: ratingEntries.map(([rating]) => rating), datasets: [{ label: "Rating count", data: ratingEntries.map(([, count]) => count), backgroundColor: "#9182ee", borderRadius: 5 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "Rating (stars)" } }, y: { beginAtZero: true } } },
  }, ratingEntries.map(([label, value]) => ({ label: `${label} stars`, value })));

  drawChart("popularChart", {
    type: "bar",
    data: { labels: topMovies.map((movie) => movie.title), datasets: [{ label: "Ratings", data: topMovies.map((movie) => movie.count), backgroundColor: "#72c9ae", borderRadius: 4 }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true } } },
  }, topMovies.map((movie) => ({ label: movie.title, value: movie.count })));

  const scatter = data.scatter_movies || [];
  drawChart("movieScatterChart", {
    type: "scatter",
    data: { datasets: [{ label: "Movies", data: scatter.map((movie) => ({ x: movie.count, y: movie.average, title: movie.title })), backgroundColor: "#9888f0", pointRadius: 3 }] },
    options: { plugins: { legend: { display: false }, tooltip: { callbacks: { label: (context) => `${context.raw.title}: ${context.raw.y.toFixed(2)} / 5 · ${context.raw.x.toLocaleString()} ratings` } } }, scales: { x: { type: "linear", title: { display: true, text: "Rating count" } }, y: { min: 0, max: 5, title: { display: true, text: "Average rating (stars)" } } } },
  }, scatter.slice(0, 12).map((movie) => ({ label: `${movie.title} · avg ${movie.average}`, value: movie.count })));

  drawChart("genreRatingChart", {
    type: "bar",
    data: { labels: genreRatings.map(([genre]) => genre), datasets: [{ label: "Average rating", data: genreRatings.map(([, value]) => value.average), backgroundColor: "#80a9dc", borderRadius: 4 }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { min: 0, max: 5, title: { display: true, text: "Average rating (stars)" } } } },
  }, genreRatings.map(([label, item]) => ({ label: `${label} · ${item.count} ratings`, value: item.average })));

  drawChart("activityChart", {
    type: "bar",
    data: { labels: activity.map(([month]) => month), datasets: [{ label: "Ratings", data: activity.map(([, count]) => count), backgroundColor: "#72c9dc", borderRadius: 3 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { ticks: { maxTicksLimit: 12, maxRotation: 0 } }, y: { beginAtZero: true, title: { display: true, text: "Rating records" } } } },
  }, activity.map(([label, value]) => ({ label, value })));

  const taste = data.selected_taste;
  const tasteRows = Object.entries(taste?.genre_distribution || {}).map(([label, value]) => ({ label, value }));
  drawChart("userTasteChart", {
    type: "doughnut",
    data: { labels: tasteRows.map((item) => item.label), datasets: [{ data: tasteRows.map((item) => item.value), backgroundColor: chartPalette, borderWidth: 0 }] },
    options: { cutout: "60%", plugins: { legend: { position: "bottom", labels: { boxWidth: 8, padding: 8, font: { size: 9 } } } } },
  }, tasteRows);
}

renderAnalytics(window.analyticsData);
if (window.userTasteData) {
  const rows = Object.entries(window.userTasteData).map(([label, value]) => ({ label, value }));
  drawChart("userTasteChart", {
    type: "doughnut",
    data: { labels: rows.map((item) => item.label), datasets: [{ data: rows.map((item) => item.value), backgroundColor: chartPalette, borderWidth: 0 }] },
    options: { cutout: "60%", plugins: { legend: { position: "bottom", labels: { boxWidth: 8, padding: 8, font: { size: 9 } } } } },
  }, rows);
}

const recommendationType = $("#recType");
if (recommendationType) {
  const userField = $("#userField");
  const movieField = $("#movieField");
  const movieInput = $("#movieQuery");
  const movieList = $("#movieList");
  const results = $("#recommendResults");
  let selectedCount = 10;
  let movieTimer;

  const updateRecommendationFields = () => {
    userField.hidden = ["similar", "popular"].includes(recommendationType.value);
    movieField.hidden = recommendationType.value !== "similar";
  };
  recommendationType.addEventListener("change", updateRecommendationFields);
  updateRecommendationFields();

  movieInput?.addEventListener("input", () => {
    clearTimeout(movieTimer);
    const query = movieInput.value.trim();
    if (query.length < 2) { movieList.innerHTML = ""; return; }
    movieTimer = setTimeout(async () => {
      try {
        const matches = await getJson(`/api/search?q=${encodeURIComponent(query)}&limit=12`);
        movieList.innerHTML = matches.map((movie) => `<option value="${escapeHtml(movie.title)}" data-id="${movie.movieId}"></option>`).join("");
      } catch (_error) { movieList.innerHTML = ""; }
    }, 180);
  });

  document.querySelectorAll(".count-option").forEach((button) => button.addEventListener("click", () => {
    $(".count-option.selected")?.classList.remove("selected");
    button.classList.add("selected");
    selectedCount = Number(button.dataset.count);
  }));

  $("#generate").addEventListener("click", async () => {
    results.innerHTML = '<div class="empty-state"><div class="empty-icon"><i class="fa-solid fa-circle-notch fa-spin"></i></div><strong>Finding your next watch…</strong><p>Scoring unseen candidates from real MovieLens data.</p></div>';
    const params = new URLSearchParams({ type: recommendationType.value, limit: selectedCount });
    if (!userField.hidden && !["similar", "popular"].includes(recommendationType.value)) params.set("user_id", $("#userId").value);
    if (recommendationType.value === "similar") {
      const option = [...movieList.options].find((item) => item.value === movieInput.value);
      if (!option) {
        results.innerHTML = '<div class="error-banner" role="alert">Search for a movie and select a result before generating similar movies.</div>';
        return;
      }
      params.set("movie_id", option.dataset.id);
    }
    try {
      const recommendations = await getJson(`/api/recommendations?${params}`);
      if (!recommendations.length) {
        results.innerHTML = '<div class="empty-state"><strong>No recommendations found</strong><p>Try another profile or starting movie.</p></div>';
        return;
      }
      results.innerHTML = recommendations.map(movieCard).join("");
      $("#resultsTitle").textContent = ({ hybrid: "Hybrid recommendations", for_you: "Picked for you", similar: "Similar movies", popular: "Popular movies" })[recommendationType.value];
      $("#resultCount").textContent = `${recommendations.length} movies`;
    } catch (error) {
      results.innerHTML = `<div class="error-banner" role="alert">${escapeHtml(error.message)}</div>`;
    }
  });
}

const exploreForm = $("#exploreForm");
if (exploreForm) {
  const searchInput = $("#searchInput");
  const resultBox = $("#searchResults");
  const status = $("#searchStatus");
  const title = $("#searchTitle");
  const pager = $("#explorePagination");
  const pageInput = $("#explorePage");
  let searchTimer;

  function currentQuery() {
    return new URLSearchParams(new FormData(exploreForm));
  }
  function updatePagination(data, params) {
    pager.innerHTML = `<span>${data.total.toLocaleString()} results · page ${data.page} of ${data.pages}</span>`;
    if (data.page > 1) {
      const previous = new URLSearchParams(params); previous.set("page", String(data.page - 1));
      pager.insertAdjacentHTML("afterbegin", `<a class="button secondary" data-page="${data.page - 1}" href="/search?${previous}">Previous</a>`);
    }
    if (data.page < data.pages) {
      const next = new URLSearchParams(params); next.set("page", String(data.page + 1));
      pager.insertAdjacentHTML("beforeend", `<a class="button secondary" data-page="${data.page + 1}" href="/search?${next}">Next</a>`);
    }
  }
  async function loadMovies(pushHistory = true) {
    const params = currentQuery();
    status.textContent = "Updating results…";
    resultBox.setAttribute("aria-busy", "true");
    try {
      const data = await getJson(`/api/movies?${params}`);
      resultBox.innerHTML = data.items.length ? data.items.map(movieCard).join("") : '<div class="empty-state"><strong>No movies match those filters</strong><p>Try another title, a different genre, or a lower rating threshold.</p></div>';
      title.textContent = `${data.total.toLocaleString()} matching movies`;
      status.textContent = `Page ${data.page} of ${data.pages}`;
      pageInput.value = data.page;
      updatePagination(data, params);
      if (pushHistory) history.pushState({}, "", `/search?${params}`);
      else history.replaceState({}, "", `/search?${params}`);
    } catch (error) {
      status.textContent = "";
      resultBox.innerHTML = `<div class="error-banner" role="alert">${escapeHtml(error.message)}</div>`;
    } finally {
      resultBox.setAttribute("aria-busy", "false");
    }
  }

  exploreForm.addEventListener("submit", (event) => { event.preventDefault(); pageInput.value = "1"; loadMovies(); });
  exploreForm.querySelectorAll("select").forEach((select) => select.addEventListener("change", () => { pageInput.value = "1"; loadMovies(); }));
  searchInput.addEventListener("input", () => {
    clearTimeout(searchTimer);
    pageInput.value = "1";
    searchTimer = setTimeout(() => loadMovies(false), 240);
  });
  pager.addEventListener("click", (event) => {
    const link = event.target.closest("a[data-page]");
    if (!link) return;
    event.preventDefault();
    pageInput.value = link.dataset.page;
    loadMovies();
  });
  window.addEventListener("popstate", () => window.location.reload());
}

if (window.ratingDistribution && $("#detailChart")) {
  const entries = Object.entries(window.ratingDistribution);
  drawChart("detailChart", {
    type: "bar",
    data: { labels: entries.map(([rating]) => rating), datasets: [{ label: "Rating count", data: entries.map(([, count]) => count), backgroundColor: "#9888f0", borderRadius: 4 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "Rating (stars)" } }, y: { beginAtZero: true, title: { display: true, text: "Number of ratings" } } } },
  }, entries.map(([label, value]) => ({ label: `${label} stars`, value })));
}

const backLink = $("#movieBack");
if (backLink && document.referrer.startsWith(window.location.origin) && history.length > 1) {
  backLink.addEventListener("click", (event) => { event.preventDefault(); history.back(); });
}
