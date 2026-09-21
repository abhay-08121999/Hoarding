/*
 * ui.js — rendering and interaction.
 *
 * State lives in one object, every view is a pure render from it, and any
 * change calls render(). Watchlist and taste ratings persist to
 * localStorage so a reload doesn't lose your picks.
 */
(function () {
  "use strict";

  const E = window.Engine;
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) =>
    Array.prototype.slice.call((root || document).querySelectorAll(sel));

  /* ------------------------------------------------ poster colour keys */
  /* Deterministic two-tone field per primary genre. The dataset ships no
     images, so colour plus large type has to do the work a poster would. */
  const GENRE_PAINT = {
    Action: ["#B8321F", "#7E1E13"],
    Adventure: ["#C97014", "#8A470B"],
    Animation: ["#2B7FB8", "#1A5480"],
    Biography: ["#5C6B8A", "#3A4560"],
    Comedy: ["#D99414", "#A2660A"],
    Crime: ["#3E4750", "#23292F"],
    Documentary: ["#4A7A6B", "#2E5046"],
    Drama: ["#7A3F5D", "#4E2740"],
    Family: ["#C9832B", "#96591A"],
    Fantasy: ["#6B4C9A", "#42306A"],
    History: ["#8A6A3C", "#5A4325"],
    Horror: ["#2A2230", "#15111B"],
    Music: ["#A83E6E", "#6E2447"],
    Musical: ["#A83E6E", "#6E2447"],
    Mystery: ["#34506B", "#1E3145"],
    Romance: ["#C0505F", "#84303C"],
    "Sci-Fi": ["#1F6F7A", "#124750"],
    Sport: ["#3F7A42", "#255028"],
    Thriller: ["#5A3550", "#351E31"],
    War: ["#6A5A3A", "#423722"],
    Western: ["#A35F2A", "#6E3D18"],
  };
  const PAINT_FALLBACK = ["#44586A", "#2A3743"];

  function paintFor(movie) {
    return GENRE_PAINT[movie.primary_genre] || PAINT_FALLBACK;
  }

  /* --------------------------------------------------------- persistence */
  const STORE_KEY = "hoarding.state.v1";

  function loadStore() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (err) {
      return {}; // private-mode or quota — the app still works, just amnesiac
    }
  }

  function saveStore() {
    try {
      localStorage.setItem(
        STORE_KEY,
        JSON.stringify({
          watchlist: state.watchlist,
          liked: state.liked,
          disliked: state.disliked,
          theme: state.theme,
          account: state.account,
        })
      );
      syncPreferences();
    } catch (err) {
      /* ignore — persistence is a convenience, not a requirement */
    }
  }

  const saved = loadStore();

  /* --------------------------------------------------------------- state */
  const state = {
    tab: "discover",
    query: "",
    matched: [],
    filters: {
      genres: [], languages: [], moods: [], countries: [], ageBands: [],
      minRating: null, maxRuntime: null, yearRange: null, awardedOnly: false,
    },
    sort: "relevance",
    seedId: null,
    lambda: 0.75,
    mode: "closest",
    account: saved.account || null,
    watchlist: saved.watchlist || [],
    liked: saved.liked || [],
    disliked: saved.disliked || [],
    theme: saved.theme || "dark",
    compareA: null,
    compareB: null,
    railOpen: false,
    showAll: false,   // Discover: force the flat grid instead of genre shelves
    modalId: null,
  };

  const T = E.STATS.totals;
  state.filters.yearRange = [T.yearMin, T.yearMax];
  state.filters.minRating = T.ratingMin;
  state.filters.maxRuntime = T.runtimeMax;

  async function syncPreferences() {
    if (!state.account?.token) return;
    try {
      await fetch("/api/preferences", {
        method: "PUT",
        headers: { "Content-Type": "application/json", Authorization: "Bearer " + state.account.token },
        body: JSON.stringify({ watchlist: state.watchlist, liked: state.liked, disliked: state.disliked, recent: [] }),
      });
    } catch (err) { /* local-first when opened directly from file:// */ }
  }

  async function accountRequest(path, payload) {
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Account request failed");
    return data;
  }

  function applyAccount(data) {
    state.account = { token: data.token, user: data.user };
    state.watchlist = data.preferences.watchlist || [];
    state.liked = data.preferences.liked || [];
    state.disliked = data.preferences.disliked || [];
    saveStore();
    render();
  }

  function openAccount() {
    const drawer = $("#account-drawer");
    drawer.hidden = false;
    $("#account-status").textContent = state.account ? "Signed in as " + state.account.user.username : "Local mode — sign in to sync across devices.";
    $("#account-title").textContent = state.account ? "Your royal collection" : "Sync your collection";
  }

  /* ------------------------------------------------------------ helpers */
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function toast(message) {
    const el = $("#toast");
    el.textContent = message;
    el.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { el.hidden = true; }, 2200);
  }

  const inList = (list, id) => list.indexOf(id) !== -1;

  function toggleIn(list, id) {
    const i = list.indexOf(id);
    if (i === -1) list.push(id);
    else list.splice(i, 1);
    return i === -1;
  }

  /* ------------------------------------------------- real posters (TMDB) */
  /* Poster URLs already resolved this session. Cards are re-rendered on every
     state change, so remembering them stops the painted placeholder flashing
     back in each time you press Like. */
  const posterUrls = new Map();
  let posterObserver = null;

  function loadPoster(node) {
    const m = E.byId(node.dataset.id);
    if (!m || !window.TMDB) return;
    TMDB.get(m, false).then((meta) => {
      if (!meta || !meta.matched || !meta.poster) return;
      const img = node.querySelector(".poster__img");
      if (!img) return;
      const url = TMDB.imageUrl(meta.poster, "w342");
      img.onload = () => {
        posterUrls.set(m.movie_id, url);
        node.classList.add("has-img");
      };
      img.onerror = () => { posterUrls.delete(m.movie_id); node.classList.remove("has-img"); };
      img.srcset =
        TMDB.imageUrl(meta.poster, "w185") + " 185w, " + url + " 342w, " +
        TMDB.imageUrl(meta.poster, "w500") + " 500w";
      img.sizes = "(max-width: 560px) 45vw, 220px";
      img.src = url;
    });
  }

  /* Fetch posters only for cards that scroll near the viewport. */
  function hydratePosters(root) {
    if (!window.TMDB || !TMDB.configured()) return;
    const nodes = $$(".poster[data-id]:not([data-h]):not(.has-img)", root || document);
    if (!nodes.length) return;
    if (!("IntersectionObserver" in window)) {
      nodes.forEach((n) => { n.dataset.h = "1"; loadPoster(n); });
      return;
    }
    if (!posterObserver) {
      posterObserver = new IntersectionObserver(
        (entries) => entries.forEach((en) => {
          if (!en.isIntersecting) return;
          posterObserver.unobserve(en.target);
          loadPoster(en.target);
        }),
        { rootMargin: "320px 0px" }
      );
    }
    nodes.forEach((n) => { n.dataset.h = "1"; posterObserver.observe(n); });
  }

  /* ------------------------------------------------------ card renderer */
  function cardHTML(movie, extra) {
    extra = extra || {};
    const [a, b] = paintFor(movie);
    const posterUrl = posterUrls.get(movie.movie_id);
    const saved = inList(state.watchlist, movie.movie_id);
    const reasons = (extra.why || [])
      .map(
        (r) =>
          `<span class="reason reason--${esc(r.kind)}">${esc(r.label)}</span>`
      )
      .join("");

    return `
      <article class="card" data-id="${esc(movie.movie_id)}"
               style="--poster-a:${a};--poster-b:${b}">
        <button class="poster${posterUrl ? " has-img" : ""}" data-act="open"
                data-id="${esc(movie.movie_id)}"
                aria-label="Open details for ${esc(movie.title)}">
          <img class="poster__img" alt=""${posterUrl ? ` src="${esc(posterUrl)}"` : ""}>
          <span class="poster__lang">${esc(movie.language)}</span>
          <span class="poster__score">${movie.rating.toFixed(1)}</span>
          <span class="poster__title">${esc(movie.title)}</span>
        </button>
        <div class="card__body">
          <div class="card__meta">
            <span>${esc(movie.year)}</span>
            <span>${esc(movie.runtime)} min</span>
            <span>${esc(movie.age_rating)}</span>
          </div>
          <div class="card__genres">${esc(movie.genres.join(", "))}</div>
          <p class="card__desc">${esc(movie.description)}</p>
          ${reasons ? `<div class="reasons">${reasons}</div>` : ""}
          <div class="card__foot">
            <button class="mini-btn" data-act="watch" data-id="${esc(movie.movie_id)}"
                    data-on="${saved}" title="Add to watchlist">
              ${saved ? "Saved" : "Save"}
            </button>
            <button class="mini-btn mini-btn--similar" data-act="seed" data-id="${esc(movie.movie_id)}"
                    title="Find similar films">Similar</button>
            <button class="mini-btn" data-act="like" data-id="${esc(movie.movie_id)}"
                    data-on="${inList(state.liked, movie.movie_id)}"
                    title="More like this">Like</button>
            <button class="mini-btn mini-btn--down" data-act="dislike"
                    data-id="${esc(movie.movie_id)}"
                    data-on="${inList(state.disliked, movie.movie_id)}"
                    title="Less like this">Not for me</button>
            ${
              extra.score != null
                ? `<span class="score-tag">${extra.score.toFixed(2)}</span>`
                : ""
            }
          </div>
        </div>
      </article>`;
  }

  function renderGrid(target, items, emptyTitle, emptyNote) {
    if (!items.length) {
      target.innerHTML = `
        <div class="empty">
          <h3>${esc(emptyTitle)}</h3>
          <p>${esc(emptyNote)}</p>
        </div>`;
      return;
    }
    target.innerHTML =
      '<div class="grid">' +
      items
        .map((it) =>
          it.movie ? cardHTML(it.movie, it) : cardHTML(it, {})
        )
        .join("") +
      "</div>";
  }

  /* ------------------------------------------------------------ filters */
  function buildRail() {
    const rail = $("#rail");
    const facet = (title, key, entries, limit) => {
      const shown = limit ? entries.slice(0, limit) : entries;
      return `
        <div class="rail__group">
          <div class="rail__head">
            <span class="rail__title">${esc(title)}</span>
            <button class="rail__reset" data-clear="${key}">Clear</button>
          </div>
          <div class="checks">
            ${shown
              .map(
                ([name, count]) => `
              <label class="check">
                <input type="checkbox" data-facet="${key}" value="${esc(name)}">
                <span>${esc(name)}</span>
                <span class="check__count">${count}</span>
              </label>`
              )
              .join("")}
          </div>
        </div>`;
    };

    rail.innerHTML = `
      <div class="rail__group">
        <div class="rail__head">
          <span class="rail__title">Refine</span>
          <button class="rail__reset" data-clear="all">Reset all</button>
        </div>
        <div class="slider-row">
          <label for="f-rating">Minimum rating
            <output id="out-rating">${T.ratingMin}</output>
          </label>
          <input type="range" id="f-rating" min="${T.ratingMin}"
                 max="${T.ratingMax}" step="0.1" value="${T.ratingMin}">
        </div>
        <div class="slider-row">
          <label for="f-runtime">Maximum length
            <output id="out-runtime">${T.runtimeMax} min</output>
          </label>
          <input type="range" id="f-runtime" min="${T.runtimeMin}"
                 max="${T.runtimeMax}" step="5" value="${T.runtimeMax}">
        </div>
        <div class="slider-row">
          <label for="f-year">Released after
            <output id="out-year">${T.yearMin}</output>
          </label>
          <input type="range" id="f-year" min="${T.yearMin}"
                 max="${T.yearMax}" step="1" value="${T.yearMin}">
        </div>
        <label class="check">
          <input type="checkbox" id="f-awarded">
          <span>Award winners only</span>
        </label>
      </div>
      ${facet("Genre", "genres", E.STATS.genres)}
      ${facet("Mood", "moods", E.STATS.moods)}
      ${facet("Language", "languages", E.STATS.languages)}
      ${facet("Country", "countries", E.STATS.countries, 10)}
    `;

    rail.addEventListener("change", (ev) => {
      const box = ev.target;
      if (box.dataset.facet) {
        const list = state.filters[box.dataset.facet];
        const i = list.indexOf(box.value);
        if (box.checked && i === -1) list.push(box.value);
        if (!box.checked && i !== -1) list.splice(i, 1);
        render();
      }
      if (box.id === "f-awarded") {
        state.filters.awardedOnly = box.checked;
        render();
      }
    });

    rail.addEventListener("input", (ev) => {
      const el = ev.target;
      if (el.id === "f-rating") {
        state.filters.minRating = parseFloat(el.value);
        $("#out-rating").textContent = el.value;
        render();
      }
      if (el.id === "f-runtime") {
        state.filters.maxRuntime = parseInt(el.value, 10);
        $("#out-runtime").textContent = el.value + " min";
        render();
      }
      if (el.id === "f-year") {
        state.filters.yearRange = [parseInt(el.value, 10), T.yearMax];
        $("#out-year").textContent = el.value;
        render();
      }
    });

    rail.addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-clear]");
      if (!btn) return;
      const key = btn.dataset.clear;
      if (key === "all") resetFilters();
      else {
        state.filters[key] = [];
        $$(`[data-facet="${key}"]`, rail).forEach((b) => (b.checked = false));
      }
      render();
    });
  }

  function resetFilters() {
    state.filters.genres = [];
    state.filters.languages = [];
    state.filters.moods = [];
    state.filters.countries = [];
    state.filters.ageBands = [];
    state.filters.awardedOnly = false;
    state.filters.minRating = T.ratingMin;
    state.filters.maxRuntime = T.runtimeMax;
    state.filters.yearRange = [T.yearMin, T.yearMax];
    $$("[data-facet]").forEach((b) => (b.checked = false));
    const awarded = $("#f-awarded");
    if (awarded) awarded.checked = false;
    const r = $("#f-rating"), rt = $("#f-runtime"), yr = $("#f-year");
    if (r) { r.value = T.ratingMin; $("#out-rating").textContent = T.ratingMin; }
    if (rt) { rt.value = T.runtimeMax; $("#out-runtime").textContent = T.runtimeMax + " min"; }
    if (yr) { yr.value = T.yearMin; $("#out-year").textContent = T.yearMin; }
  }

  function activeFilterCount() {
    const f = state.filters;
    let n = f.genres.length + f.languages.length + f.moods.length +
      f.countries.length;
    if (f.awardedOnly) n++;
    if (f.minRating > T.ratingMin) n++;
    if (f.maxRuntime < T.runtimeMax) n++;
    if (f.yearRange[0] > T.yearMin) n++;
    return n;
  }

  /* ------------------------------------------------------- sort helpers */
  const SORTS = {
    relevance: null,
    rating: (a, b) => b.rating - a.rating,
    newest: (a, b) => b.year - a.year,
    oldest: (a, b) => a.year - b.year,
    title: (a, b) => a.title.localeCompare(b.title),
    shortest: (a, b) => a.runtime - b.runtime,
  };

  /* ------------------------------------------------------------- shelves */
  /* A horizontally scrolling row of cards: the "recommended in <genre>" unit
     used on Discover, Similar films and For you. */
  function shelfHTML(o) {
    const cards = o.items
      .map((it) => cardHTML(it.movie, { why: it.why, score: o.showScore ? it.score : null }))
      .join("");
    return `
      <section class="shelf" aria-label="${esc(o.title)}">
        <div class="shelf__head">
          <h3 class="shelf__title">${o.titleHTML || esc(o.title)}</h3>
          ${o.note ? `<span class="shelf__note">${esc(o.note)}</span>` : ""}
          ${o.genre
            ? `<button class="shelf__all" data-act="see-genre" data-genre="${esc(o.genre)}">
                 See all ${esc(o.total || "")} &rarr;</button>`
            : ""}
        </div>
        <div class="shelf__wrap">
          <button class="shelf__nav shelf__nav--prev" data-act="shelf-prev"
                  aria-label="Scroll left" tabindex="-1">&lsaquo;</button>
          <div class="shelf__track">${cards}</div>
          <button class="shelf__nav shelf__nav--next" data-act="shelf-next"
                  aria-label="Scroll right" tabindex="-1">&rsaquo;</button>
        </div>
      </section>`;
  }

  /* ---------------------------------------------------------- genre bar */
  function buildGenreBar() {
    $("#genre-bar").innerHTML =
      `<button class="chip chip--genre" data-genre="" aria-pressed="true">All genres</button>` +
      E.STATS.genres
        .map(
          ([g, n]) =>
            `<button class="chip chip--genre" data-genre="${esc(g)}" aria-pressed="false">
               ${esc(g)} <span class="chip__n">${n}</span></button>`
        )
        .join("");
  }

  /* Keep every control that shows filter state in step with state.filters,
     so the genre chips, mood chips and side rail can never disagree. */
  function syncFacetInputs() {
    $$("[data-facet]").forEach((b) => {
      b.checked = state.filters[b.dataset.facet].indexOf(b.value) !== -1;
    });
    $$("#mood-chips .chip").forEach((c) =>
      c.setAttribute("aria-pressed", String(state.filters.moods.indexOf(c.dataset.mood) !== -1)));
    $$("#genre-bar .chip").forEach((c) => {
      const g = c.dataset.genre;
      const on = g === "" ? !state.filters.genres.length : state.filters.genres.indexOf(g) !== -1;
      c.setAttribute("aria-pressed", String(on));
    });
  }

  /* ------------------------------------------------------------ discover */
  function isBrowseMode() {
    return !state.query.trim() && activeFilterCount() === 0 &&
      state.sort === "relevance" && !state.showAll;
  }

  function emptySearchHTML(q) {
    const sugg = E.suggest(q, 4);
    const genres = E.STATS.genres.slice(0, 7);
    return `
      <div class="empty">
        <h3>No films match &ldquo;${esc(q)}&rdquo;</h3>
        <p>Check the spelling, or search by a genre, a director, an actor or a mood.</p>
        ${sugg.length
          ? `<div class="empty__row"><span>Did you mean</span>${sugg
              .map((m) => `<button class="tag tag--link" data-act="search-term"
                            data-term="${esc(m.title)}">${esc(m.title)}</button>`)
              .join("")}</div>`
          : ""}
        <div class="empty__row"><span>Try a genre</span>${genres
          .map(([g]) => `<button class="tag tag--link" data-act="search-term"
                          data-term="${esc(g)}">${esc(g)}</button>`)
          .join("")}</div>
      </div>`;
  }

  function renderDiscover() {
    const out = $("#discover-results");
    const q = state.query.trim();
    const note = $("#matched-note");
    const toggle = $("#view-toggle");
    const nFilters = activeFilterCount();
    let items;

    /* ---- 1. a typed search */
    if (q) {
      const hits = E.search(q, { filters: state.filters, topN: 500 });
      state.matched = hits.length ? hits[0].matched || [] : [];
      items = hits.map((h) => ({ movie: h.movie, why: h.why }));
      if (state.sort !== "relevance")
        items.sort((x, y) => SORTS[state.sort](x.movie, y.movie));

      const exact = hits.length && E.normalise(hits[0].movie.title) === E.normalise(q);
      if (exact) {
        note.innerHTML =
          `Showing <b>${esc(hits[0].movie.title)}</b> &middot; ` +
          `<button class="link-btn" data-act="seed" data-id="${esc(hits[0].movie.movie_id)}">` +
          `Find films like this &rarr;</button>`;
      } else if (state.matched.length) {
        note.innerHTML = "Reading that as " + state.matched.map((m) => `<b>${esc(m)}</b>`).join(", ");
      } else if (hits.length) {
        note.textContent = "Matching on titles, names and wording.";
      } else {
        note.textContent = "";
      }

      toggle.hidden = true;
      $("#discover-count").innerHTML =
        `<b>${items.length}</b> film${items.length === 1 ? "" : "s"} for &ldquo;${esc(q)}&rdquo;` +
        (nFilters ? ` &middot; ${nFilters} filter${nFilters === 1 ? "" : "s"} on` : "");
      if (!items.length) {
        out.innerHTML = nFilters
          ? `<div class="empty"><h3>Nothing matches with these filters</h3>
               <p>&ldquo;${esc(q)}&rdquo; has results, but your filters exclude them. Try Reset all.</p></div>`
          : emptySearchHTML(q);
        return;
      }
      renderGrid(out, items, "", "");
      return;
    }

    state.matched = [];
    note.textContent = "";

    /* ---- 2. landing view: recommended shelves, one per genre */
    if (isBrowseMode()) {
      const shelves = E.genreShelves({
        likedIds: state.liked, dislikedIds: state.disliked,
        perShelf: 12, filters: state.filters,
      });
      const personal = state.liked.length > 0;
      $("#discover-count").innerHTML =
        `<b>${shelves.length}</b> genres &middot; ` +
        (personal ? "ranked for your taste" : "top picks in each");
      toggle.hidden = false;
      toggle.textContent = "Show all " + E.MOVIES.length + " films";

      out.innerHTML =
        `<p class="shelf-intro">${
          personal
            ? `Every shelf is ranked by the ${state.liked.length} film${state.liked.length === 1 ? "" : "s"} you liked.`
            : "The best-rated films in every genre. Hit <b>Like</b> on a few and these shelves start adapting to you."
        }</p>` +
        shelves
          .map((sh) =>
            shelfHTML({
              title: (personal ? "Picked for you in " : "Top in ") + sh.genre,
              titleHTML: (personal ? "Picked for you in " : "Top in ") + `<em>${esc(sh.genre)}</em>`,
              note: sh.total + " films",
              genre: sh.genre, total: sh.total, items: sh.items,
            }))
          .join("");
      return;
    }

    /* ---- 3. a filtered / sorted grid */
    const pool = E.filter(state.filters);
    let cmp = SORTS[state.sort];
    if (!cmp) {
      // "Best match" with no query: the person's taste if we have one, else rating.
      const taste = E.rankByTaste(state.liked, state.disliked, pool);
      cmp = taste
        ? (a, b) => taste.get(b.movie_id) - taste.get(a.movie_id)
        : SORTS.rating;
    }
    items = pool.slice().sort(cmp).map((m) => ({ movie: m }));

    toggle.hidden = nFilters > 0;
    toggle.textContent = "Back to genre shelves";

    const genreLabel = state.filters.genres.length
      ? state.filters.genres.join(" + ") + " &middot; "
      : "";
    $("#discover-count").innerHTML =
      `${genreLabel}<b>${items.length}</b> film${items.length === 1 ? "" : "s"}` +
      (nFilters ? ` &middot; ${nFilters} filter${nFilters === 1 ? "" : "s"} on` : "");

    renderGrid(
      out, items,
      "Nothing matches that yet",
      "Try loosening a filter, or search for a mood like \u201cmind bending\u201d or a name like \u201cKurosawa\u201d."
    );
  }

  /* ----------------------------------------------------------- recommend */
  function renderRecommend() {
    const out = $("#recommend-results");
    const seed = state.seedId ? E.byId(state.seedId) : null;

    if (!seed) {
      out.innerHTML = `
        <div class="empty">
          <h3>Pick a film to start from</h3>
          <p>Choose any title above and you'll get the closest matches in the
             catalogue, each labelled with what it has in common.</p>
        </div>`;
      $("#seed-summary").innerHTML = "";
      return;
    }

    const recs = E.recommendMode(state.mode, {
      seedId: seed.movie_id,
      likedIds: state.liked,
      dislikedIds: state.disliked,
      topN: 18,
      lambda: state.lambda,
      filters: state.filters,
    });

    const [a, b] = paintFor(seed);
    $("#seed-summary").innerHTML = `
      <div class="seed-panel" style="--poster-a:${a};--poster-b:${b}">
        <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start">
          <div style="flex:1;min-width:240px">
            <h3 style="font-family:var(--font-display);font-size:1.8rem">
              ${esc(seed.title)}
            </h3>
            <div class="card__meta" style="margin-top:4px">
              <span>${esc(seed.year)}</span>
              <span>${esc(seed.language)}</span>
              <span>${esc(seed.runtime)} min</span>
              <span>Rated ${seed.rating.toFixed(1)}</span>
            </div>
            <p style="font-size:0.86rem;color:var(--text-dim);margin-top:8px">
              ${esc(seed.description)}
            </p>
            <div class="tag-row" style="margin-top:8px">
              ${seed.keyword_list
                .slice(0, 8)
                .map((k) => `<span class="tag">${esc(k)}</span>`)
                .join("")}
            </div>
          </div>
        </div>
      </div>`;

    renderGrid(
      out,
      recs,
      "No matches survive those filters",
      "The seed film has close relatives, but your filters exclude them all."
    );

    /* Same spirit, different genre: what the seed's DNA looks like elsewhere. */
    const cross = E.crossGenreShelves(seed.movie_id, { filters: state.filters });
    if (cross.length) {
      out.insertAdjacentHTML(
        "beforeend",
        `<div class="section__head section__head--split">
           <h3 class="section__title">Same spirit, different genre</h3>
           <span class="section__note">If you like ${esc(seed.title)}, these are the closest films
             filed under other genres.</span>
         </div>` +
        cross.map((sh) =>
          shelfHTML({
            title: sh.genre + " picks",
            titleHTML: `<em>${esc(sh.genre)}</em> picks`,
            note: "close to " + seed.title,
            items: sh.items,
          })).join("")
      );
    }
  }

  /* -------------------------------------------------------------- for you */
  function renderForYou() {
    const out = $("#foryou-results");
    const aside = $("#taste-readout");

    if (!state.liked.length) {
      out.innerHTML = `
        <div class="empty">
          <h3>Rate a few films and this fills up</h3>
          <p>Hit “Like” on anything you enjoyed. Two or three is enough to
             build a profile; “Not for me” sharpens it further.</p>
        </div>`;
      aside.innerHTML = `
        <h4 style="font-family:var(--font-display);font-size:1.3rem">
          Your taste profile
        </h4>
        <p style="font-size:0.8rem;color:var(--text-faint)">
          Nothing rated yet.
        </p>`;
      return;
    }

    const recs = E.forProfile(state.liked, state.disliked, {
      topN: 18,
      lambda: state.lambda,
      filters: state.filters,
    });

    const themes = E.profileThemes(state.liked, state.disliked, 8);
    const maxW = themes.length ? themes[0].weight : 1;

    aside.innerHTML = `
      <h4 style="font-family:var(--font-display);font-size:1.3rem;margin-bottom:10px">
        Your taste profile
      </h4>
      <p style="font-size:0.78rem;color:var(--text-faint);margin-bottom:14px">
        Built from ${state.liked.length} liked and
        ${state.disliked.length} rejected film${state.disliked.length === 1 ? "" : "s"}.
      </p>
      ${themes
        .map(
          (t) => `
        <div class="theme-bar">
          <div class="theme-bar__label">
            <span>${esc(t.label)}</span>
            <span>${t.kind}</span>
          </div>
          <div class="theme-bar__track">
            <div class="theme-bar__fill" style="width:${(t.weight / maxW) * 100}%"></div>
          </div>
        </div>`
        )
        .join("")}
      <div class="rated-list">
        ${state.liked
          .map((id) => {
            const m = E.byId(id);
            return m
              ? `<div class="rated">
                   <span class="rated__title">${esc(m.title)}</span>
                   <button class="rated__x" data-act="unrate" data-id="${esc(id)}"
                           aria-label="Remove ${esc(m.title)}">&times;</button>
                 </div>`
              : "";
          })
          .join("")}
        ${state.disliked
          .map((id) => {
            const m = E.byId(id);
            return m
              ? `<div class="rated rated--down">
                   <span class="rated__title">${esc(m.title)}</span>
                   <button class="rated__x" data-act="unrate" data-id="${esc(id)}"
                           aria-label="Remove ${esc(m.title)}">&times;</button>
                 </div>`
              : "";
          })
          .join("")}
      </div>
      <button class="btn btn--ghost" style="margin-top:12px;width:100%"
              data-act="clear-taste">Clear ratings</button>`;

    renderGrid(
      out,
      recs,
      "Nothing left to suggest",
      "You've rated most of what matches. Loosen a filter for more."
    );

    const beyond = E.beyondYourGenres(state.liked, state.disliked, { topN: 10, filters: state.filters });
    if (beyond.length) {
      out.insertAdjacentHTML(
        "beforeend",
        `<div class="section__head section__head--split">
           <h3 class="section__title">Outside your usual genres</h3>
           <span class="section__note">Genres none of your liked films belong to &mdash; ranked by
             how well they still fit your taste.</span>
         </div>` +
        shelfHTML({ title: "Try something different", items: beyond })
      );
    }
  }

  /* ------------------------------------------------------------ explore */
  function barChart(entries, max, colour) {
    return `<div class="bar-row">${entries
      .map(
        ([label, value]) => `
      <div class="bar">
        <span class="bar__label">${esc(label)}</span>
        <span class="bar__track">
          <span class="bar__fill" style="width:${(value / max) * 100}%;
                background:${colour}"></span>
        </span>
        <span class="bar__val">${value}</span>
      </div>`
      )
      .join("")}</div>`;
  }

  /* Hand-rolled SVG line chart — avoids pulling in a charting library and
     keeps the whole app dependency-free. */
  function lineChart(points, opts) {
    opts = opts || {};
    const w = 320, h = 140, padL = 34, padB = 26, padT = 10, padR = 8;
    const ys = points.map((p) => p[1]);
    const lo = opts.min != null ? opts.min : Math.min.apply(null, ys);
    const hi = opts.max != null ? opts.max : Math.max.apply(null, ys);
    const range = hi - lo || 1;
    const stepX = (w - padL - padR) / Math.max(points.length - 1, 1);

    const coords = points.map((p, i) => {
      const x = padL + i * stepX;
      const y = padT + (1 - (p[1] - lo) / range) * (h - padT - padB);
      return [x, y];
    });

    const path = coords
      .map((c, i) => (i ? "L" : "M") + c[0].toFixed(1) + " " + c[1].toFixed(1))
      .join(" ");
    const area =
      path + ` L${coords[coords.length - 1][0].toFixed(1)} ${h - padB}` +
      ` L${coords[0][0].toFixed(1)} ${h - padB} Z`;

    return `
      <div class="chart">
        <svg viewBox="0 0 ${w} ${h}" role="img"
             aria-label="${esc(opts.label || "trend chart")}">
          <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}"
                stroke="currentColor" stroke-opacity="0.22"/>
          <path d="${area}" fill="${opts.colour || "#f0a32b"}" fill-opacity="0.14"/>
          <path d="${path}" fill="none" stroke="${opts.colour || "#f0a32b"}"
                stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
          ${coords
            .map(
              (c, i) =>
                `<circle cx="${c[0].toFixed(1)}" cy="${c[1].toFixed(1)}" r="2.6"
                         fill="${opts.colour || "#f0a32b"}">
                   <title>${esc(points[i][0])}: ${esc(points[i][1])}</title>
                 </circle>`
            )
            .join("")}
          ${points
            .map((p, i) =>
              i % Math.ceil(points.length / 6) === 0
                ? `<text x="${coords[i][0].toFixed(1)}" y="${h - padB + 14}"
                         font-size="9" text-anchor="middle">${esc(p[0])}</text>`
                : ""
            )
            .join("")}
          <text x="${padL - 6}" y="${padT + 6}" font-size="9"
                text-anchor="end">${hi}</text>
          <text x="${padL - 6}" y="${h - padB}" font-size="9"
                text-anchor="end">${lo}</text>
        </svg>
      </div>`;
  }

  function renderExplore() {
    const s = E.STATS;
    const maxGenre = s.genres[0][1];
    const maxLang = s.languages[0][1];
    const maxDir = s.topDirectors[0][1];

    const decadePoints = s.decades.map(([d, n]) => [d, n]);
    const ratingPoints = s.ratingHistogram.map(([r, n]) => [r, n]);

    const genreRating = Object.entries(s.genreRating)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12);

    $("#explore-charts").innerHTML = `
      <div class="chart-card">
        <h3>Genres</h3>
        <p>A film counts once for each genre it carries.</p>
        ${barChart(s.genres.slice(0, 12), maxGenre, "var(--marigold)")}
      </div>
      <div class="chart-card">
        <h3>Languages</h3>
        <p>Original language of each film.</p>
        ${barChart(s.languages.slice(0, 12), maxLang, "var(--jade)")}
      </div>
      <div class="chart-card">
        <h3>Films by decade</h3>
        <p>How the catalogue spreads across time.</p>
        ${lineChart(decadePoints, { colour: "#e0452f", min: 0, label: "films per decade" })}
      </div>
      <div class="chart-card">
        <h3>Rating spread</h3>
        <p>Number of films at each IMDb score.</p>
        ${lineChart(ratingPoints, { colour: "#f0a32b", min: 0, label: "rating distribution" })}
      </div>
      <div class="chart-card">
        <h3>Average rating by genre</h3>
        <p>Genres with at least three films.</p>
        ${barChart(genreRating, 10, "var(--violet)")}
      </div>
      <div class="chart-card">
        <h3>Most represented directors</h3>
        <p>Click a name to see their films.</p>
        ${barChart(s.topDirectors.slice(0, 12), maxDir, "var(--vermilion)")}
      </div>`;

    const cols = E.collections();
    $("#collections").innerHTML = cols
      .map(
        (c) => `
      <section class="section">
        <div class="section__head">
          <h3 class="section__title">${esc(c.name)}</h3>
          <span class="section__note">${esc(c.note)}</span>
        </div>
        <div class="grid grid--tight">
          ${c.items.slice(0, 6).map((m) => cardHTML(m, {})).join("")}
        </div>
      </section>`
      )
      .join("");
  }

  /* ------------------------------------------------------------ compare */
  function renderCompare() {
    const box = $("#compare-body");
    const a = state.compareA ? E.byId(state.compareA) : null;
    const b = state.compareB ? E.byId(state.compareB) : null;

    if (!a || !b) {
      box.innerHTML = `
        <div class="empty">
          <h3>Choose two films</h3>
          <p>Pick a title on each side to see how much they actually share,
             broken down by signal.</p>
        </div>`;
      return;
    }

    const c = E.compare(a.movie_id, b.movie_id);
    const sigColour = {
      text: "var(--marigold)", categorical: "var(--vermilion)",
      people: "var(--jade)", numeric: "var(--violet)",
    };
    const sigLabel = {
      text: "Themes", categorical: "Genre & mood",
      people: "Cast & crew", numeric: "Era & length",
    };

    box.innerHTML = `
      <div class="compare-grid">
        <div class="compare-col" style="--poster-a:${paintFor(a)[0]}">
          <h3 style="font-family:var(--font-display);font-size:1.5rem">${esc(a.title)}</h3>
          <div class="card__meta"><span>${esc(a.year)}</span><span>${esc(a.language)}</span>
            <span>Rated ${a.rating.toFixed(1)}</span></div>
          <p style="font-size:0.82rem;color:var(--text-dim)">${esc(a.description)}</p>
        </div>
        <div class="compare-mid">
          <div class="compare-score">${(c.overall * 100).toFixed(0)}</div>
          <div style="font-size:0.74rem;color:var(--text-faint);margin-bottom:14px">
            similarity out of 100
          </div>
          <div class="signals">
            ${Object.keys(sigLabel)
              .map(
                (k) => `
              <div class="signal__row">
                <span class="signal__name">${sigLabel[k]}</span>
                <span class="signal__track">
                  <span class="signal__fill" style="width:${(c.signals[k] * 100).toFixed(0)}%;
                        background:${sigColour[k]}"></span>
                </span>
                <span class="signal__val">${(c.signals[k] * 100).toFixed(0)}</span>
              </div>`
              )
              .join("")}
          </div>
          ${
            c.shared.length
              ? `<div class="reasons" style="justify-content:center;margin-top:14px">
                   ${c.shared
                     .map(
                       (r) =>
                         `<span class="reason reason--${esc(r.kind)}">${esc(r.label)}</span>`
                     )
                     .join("")}
                 </div>`
              : `<p style="font-size:0.76rem;color:var(--text-faint);margin-top:14px">
                   Nothing substantial in common.
                 </p>`
          }
        </div>
        <div class="compare-col">
          <h3 style="font-family:var(--font-display);font-size:1.5rem">${esc(b.title)}</h3>
          <div class="card__meta"><span>${esc(b.year)}</span><span>${esc(b.language)}</span>
            <span>Rated ${b.rating.toFixed(1)}</span></div>
          <p style="font-size:0.82rem;color:var(--text-dim)">${esc(b.description)}</p>
        </div>
      </div>`;
  }

  /* ---------------------------------------------------------- watchlist */
  function renderWatchlist() {
    const out = $("#watchlist-results");
    const items = state.watchlist.map((id) => E.byId(id)).filter(Boolean);
    $("#watchlist-count-label").textContent =
      items.length + (items.length === 1 ? " film saved" : " films saved");

    if (!items.length) {
      out.innerHTML = `
        <div class="empty">
          <h3>Your watchlist is empty</h3>
          <p>Hit “Save” on any film and it'll wait for you here. The list
             survives a reload.</p>
        </div>`;
      return;
    }

    const total = items.reduce((sum, m) => sum + (m.runtime || 0), 0);
    $("#watchlist-time").textContent =
      Math.floor(total / 60) + "h " + (total % 60) + "m of viewing";

    renderGrid(out, items, "", "");
  }

  /* --------------------------------------------------------------- modal */
  function openModal(id) {
    const m = E.byId(id);
    if (!m) return;
    const [a, b] = paintFor(m);
    const similar = E.similarTo(id, { topN: 6, lambda: 0.7 });

    $("#modal").style.setProperty("--poster-a", a);
    $("#modal").style.setProperty("--poster-b", b);

    $("#modal-content").innerHTML = `
      <div class="modal__banner" style="--poster-a:${a};--poster-b:${b}">
        <button class="modal__close" data-act="close-modal"
                aria-label="Close">&times;</button>
        <div class="modal__banner-inner">
          <h2 class="modal__title">${esc(m.title)}</h2>
          <div class="modal__sub">
            ${esc(m.year)} · ${esc(m.language)} · ${esc(m.genres.join(", "))}
          </div>
        </div>
      </div>
      <div class="modal__body">
        <div class="media-row" id="media-row"></div>
        <div class="trailer-box" id="trailer-box" hidden></div>
        <div class="facts">
          <div><div class="fact__label">Rating</div>
               <div class="fact__value">${m.rating.toFixed(1)} / 10</div></div>
          <div><div class="fact__label">Runtime</div>
               <div class="fact__value">${esc(m.runtime)} min</div></div>
          <div><div class="fact__label">Certificate</div>
               <div class="fact__value">${esc(m.age_rating)}</div></div>
          <div><div class="fact__label">Country</div>
               <div class="fact__value">${esc(m.country)}</div></div>
          <div><div class="fact__label">Style</div>
               <div class="fact__value">${esc(m.sub_genre)}</div></div>
        </div>

        <div class="modal__section">
          <h4>Synopsis</h4>
          <p>${esc(m.description)}</p>
        </div>

        <div class="modal__section">
          <h4>Director</h4>
          <div class="tag-row">
            ${m.directors
              .map(
                (d) =>
                  `<button class="tag tag--link" data-act="search-term"
                           data-term="${esc(d)}">${esc(d)}</button>`
              )
              .join("")}
          </div>
        </div>

        <div class="modal__section">
          <h4>Cast</h4>
          <div class="tag-row">
            ${m.cast_list
              .map(
                (c) =>
                  `<button class="tag tag--link" data-act="search-term"
                           data-term="${esc(c)}">${esc(c)}</button>`
              )
              .join("")}
          </div>
        </div>

        <div class="modal__section">
          <h4>Themes</h4>
          <div class="tag-row">
            ${m.keyword_list
              .map(
                (k) =>
                  `<button class="tag tag--link" data-act="search-term"
                           data-term="${esc(k)}">${esc(k)}</button>`
              )
              .join("")}
            ${m.moods
              .map((x) => `<span class="reason reason--mood">${esc(x)}</span>`)
              .join("")}
          </div>
        </div>

        <div class="modal__section">
          <h4>Awards</h4>
          <p>${esc(m.awards)}</p>
        </div>

        <div class="modal__actions">
          <button class="btn" data-act="seed" data-id="${esc(m.movie_id)}">
            Find films like this
          </button>
          <button class="btn btn--ghost" data-act="watch" data-id="${esc(m.movie_id)}">
            ${inList(state.watchlist, m.movie_id) ? "Remove from watchlist" : "Save to watchlist"}
          </button>
          <button class="btn btn--ghost" data-act="compare-a" data-id="${esc(m.movie_id)}">
            Compare
          </button>
        </div>

        <div class="modal__section" style="margin-top:22px">
          <h4>Closest in the catalogue</h4>
          <div class="grid grid--tight">
            ${similar.map((r) => cardHTML(r.movie, r)).join("")}
          </div>
        </div>
      </div>`;

    state.modalId = id;
    $("#scrim").hidden = false;
    document.body.style.overflow = "hidden";
    $("#modal-content").scrollTop = 0;
    loadMedia(m);
    hydratePosters($("#modal"));
  }

  /* Poster, backdrop and trailer for the open film. Always leaves the person
     a working trailer link, and says plainly why TMDB data is missing. */
  const YT_ID = /^[\w-]{6,20}$/;

  function renderMedia(m, meta) {
    const row = $("#media-row");
    if (!row || state.modalId !== m.movie_id) return;

    const ytSearch = TMDB.youtubeSearchUrl(m);
    const searchBtn =
      `<a class="btn btn--ghost" href="${esc(ytSearch)}" target="_blank" rel="noopener">` +
      `&#9654; Find trailer on YouTube</a>`;

    if (meta && meta.matched) {
      const banner = $(".modal__banner");
      if (banner && meta.backdrop) {
        banner.style.backgroundImage =
          `linear-gradient(180deg, rgba(8,10,20,.30) 0%, rgba(8,10,20,.92) 100%), ` +
          `url("${TMDB.imageUrl(meta.backdrop, "w780")}")`;
        banner.style.backgroundSize = "cover";
        banner.style.backgroundPosition = "center 25%";
        banner.classList.add("has-backdrop");
      }
      const poster = meta.poster
        ? `<img class="media__poster" src="${esc(TMDB.imageUrl(meta.poster, "w342"))}"
                alt="Poster for ${esc(m.title)}">`
        : "";
      const trailer = meta.trailer && YT_ID.test(meta.trailer.key) ? meta.trailer : null;
      const actions = trailer
        ? `<button class="btn" data-act="play-trailer" data-key="${esc(trailer.key)}">
             &#9654; Watch trailer</button>
           <a class="btn btn--ghost" target="_blank" rel="noopener"
              href="https://www.youtube.com/watch?v=${esc(trailer.key)}">Open on YouTube</a>`
        : searchBtn;
      const vote = meta.vote ? Number(meta.vote).toFixed(1) : null;
      row.innerHTML = `
        ${poster}
        <div class="media__info">
          ${meta.tagline ? `<p class="media__tagline">&ldquo;${esc(meta.tagline)}&rdquo;</p>` : ""}
          ${meta.overview ? `<p class="media__overview">${esc(meta.overview)}</p>` : ""}
          ${vote ? `<div class="media__stats">TMDB ${vote} / 10${
            meta.votes ? " &middot; " + Number(meta.votes).toLocaleString() + " votes" : ""}</div>` : ""}
          <div class="media__actions">${actions}</div>
          ${!trailer && meta.trailer === null
            ? `<p class="media__note">TMDB has no trailer listed for this film.</p>` : ""}
        </div>`;
      return;
    }

    let reason;
    if (!TMDB.configured()) {
      reason = "Posters and trailers are off. Add your TMDB key in web/js/config.js " +
        "(or set TMDB_API_KEY before running python app.py).";
    } else if (meta && meta.error) {
      reason = meta.error;
    } else {
      reason = "TMDB has no film that matches this title and year.";
    }
    row.innerHTML = `
      <div class="media__info">
        <div class="media__actions">${searchBtn}</div>
        <p class="media__note">${esc(reason)}</p>
      </div>`;
  }

  function loadMedia(m) {
    if (!window.TMDB) return;
    if (!TMDB.configured()) { renderMedia(m, null); return; }
    const row = $("#media-row");
    if (row) row.innerHTML = `<div class="media__info"><p class="media__note">Loading poster and trailer…</p></div>`;
    TMDB.get(m, true).then((meta) => renderMedia(m, meta));
  }

  function playTrailer(key) {
    if (!YT_ID.test(key)) return;
    // YouTube refuses to play embeds on pages opened from disk (no referrer),
    // so a double-clicked index.html opens the trailer in a new tab instead.
    if (location.protocol === "file:") {
      window.open("https://www.youtube.com/watch?v=" + key, "_blank", "noopener");
      return;
    }
    const box = $("#trailer-box");
    box.hidden = false;
    box.innerHTML =
      `<iframe src="https://www.youtube-nocookie.com/embed/${key}?autoplay=1&rel=0"
               title="Trailer" allowfullscreen
               allow="autoplay; encrypted-media; picture-in-picture; fullscreen"
               referrerpolicy="strict-origin-when-cross-origin"></iframe>`;
    box.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function closeModal() {
    const box = $("#trailer-box");
    if (box) { box.innerHTML = ""; box.hidden = true; } // stops playback
    state.modalId = null;
    $("#scrim").hidden = true;
    document.body.style.overflow = "";
  }

  /* -------------------------------------------------------------- render */
  function render() {
    $$(".tab").forEach((t) =>
      t.setAttribute("aria-selected", String(t.dataset.tab === state.tab))
    );
    $$(".panel").forEach((p) => (p.hidden = p.dataset.panel !== state.tab));

    $("#watch-count").textContent = state.watchlist.length;
    $("#watch-count").style.visibility = state.watchlist.length ? "visible" : "hidden";

    syncFacetInputs();

    if (state.tab === "discover") renderDiscover();
    if (state.tab === "recommend") renderRecommend();
    if (state.tab === "foryou") renderForYou();
    if (state.tab === "explore") renderExplore();
    if (state.tab === "compare") renderCompare();
    if (state.tab === "watchlist") renderWatchlist();

    hydratePosters();
  }

  /* ------------------------------------------------------------- events */
  function handleAction(ev) {
    const btn = ev.target.closest("[data-act]");
    if (!btn) return;
    const act = btn.dataset.act;
    const id = btn.dataset.id;

    if (act === "open") { openModal(id); return; }
    if (act === "close-modal") { closeModal(); return; }

    if (act === "watch") {
      const added = toggleIn(state.watchlist, id);
      saveStore();
      toast(added ? "Saved to watchlist" : "Removed from watchlist");
      // Update buttons where they are, so an open modal (and a playing
      // trailer) isn't torn down and rebuilt.
      $$('[data-act="watch"]').forEach((b) => {
        if (b.dataset.id !== id) return;
        if (b.closest(".modal__actions"))
          b.textContent = added ? "Remove from watchlist" : "Save to watchlist";
        else {
          b.textContent = added ? "Saved" : "Save";
          b.dataset.on = String(added);
        }
      });
      render();
      return;
    }

    if (act === "like") {
      const di = state.disliked.indexOf(id);
      if (di !== -1) state.disliked.splice(di, 1);
      const added = toggleIn(state.liked, id);
      saveStore();
      toast(added ? "Added to your taste profile" : "Removed from profile");
      render();
      return;
    }

    if (act === "dislike") {
      const li = state.liked.indexOf(id);
      if (li !== -1) state.liked.splice(li, 1);
      const added = toggleIn(state.disliked, id);
      saveStore();
      toast(added ? "Noted — you'll see fewer like that" : "Removed from profile");
      render();
      return;
    }

    if (act === "unrate") {
      [state.liked, state.disliked].forEach((list) => {
        const i = list.indexOf(id);
        if (i !== -1) list.splice(i, 1);
      });
      saveStore();
      render();
      return;
    }

    if (act === "clear-taste") {
      state.liked = [];
      state.disliked = [];
      saveStore();
      toast("Ratings cleared");
      render();
      return;
    }

    if (act === "seed") {
      state.seedId = id;
      state.tab = "recommend";
      closeModal();
      $("#seed-input").value = E.byId(id).title;
      render();
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }

    if (act === "compare-a") {
      if (!state.compareA || state.compareB) {
        state.compareA = id;
        state.compareB = null;
        toast("First film chosen — pick a second to compare");
      } else {
        state.compareB = id;
        state.tab = "compare";
        closeModal();
      }
      $("#cmp-a").value = state.compareA ? E.byId(state.compareA).title : "";
      $("#cmp-b").value = state.compareB ? E.byId(state.compareB).title : "";
      render();
      return;
    }

    if (act === "search-term") {
      state.query = btn.dataset.term;
      state.tab = "discover";
      $("#search").value = state.query;
      closeModal();
      render();
      $("#search").focus({ preventScroll: true });
      $("#search").select();
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }

    if (act === "see-genre") {
      state.filters.genres = [btn.dataset.genre];
      state.query = "";
      $("#search").value = "";
      state.showAll = false;
      state.tab = "discover";
      render();
      window.scrollTo({ top: $(".workbench").offsetTop - 90, behavior: "smooth" });
      return;
    }

    if (act === "toggle-view") {
      if (isBrowseMode()) state.showAll = true;
      else { state.showAll = false; state.sort = "relevance"; $("#sort").value = "relevance"; }
      render();
      return;
    }

    if (act === "shelf-prev" || act === "shelf-next") {
      const track = btn.closest(".shelf__wrap").querySelector(".shelf__track");
      const dir = act === "shelf-next" ? 1 : -1;
      track.scrollBy({ left: dir * track.clientWidth * 0.85, behavior: "smooth" });
      return;
    }

    if (act === "play-trailer") { playTrailer(btn.dataset.key); return; }

    if (act === "account") {
      openAccount();
      return;
    }

    if (act === "surprise") {
      const m = E.randomMovie(state.filters);
      if (m) openModal(m.movie_id);
      else toast("No films match the current filters");
      return;
    }
  }

  function resolveTitle(value) {
    const norm = E.normalise(value);
    if (!norm) return null;
    let exact = null, partial = null;
    for (const m of E.MOVIES) {
      const t = E.normalise(m.title);
      if (t === norm) { exact = m; break; }
      if (!partial && t.indexOf(norm) !== -1) partial = m;
    }
    return exact || partial;
  }

  function wire() {
    document.addEventListener("click", handleAction);

    $$(".tab").forEach((t) =>
      t.addEventListener("click", () => {
        state.tab = t.dataset.tab;
        render();
      })
    );

    let debounce;
    $("#search").addEventListener("input", (ev) => {
      clearTimeout(debounce);
      const value = ev.target.value;
      debounce = setTimeout(() => {
        state.query = value;
        render();
      }, 130);
    });

    $("#search-clear").addEventListener("click", () => {
      state.query = "";
      $("#search").value = "";
      render();
      $("#search").focus();
    });

    $("#sort").addEventListener("change", (ev) => {
      state.sort = ev.target.value;
      render();
    });

    $("#genre-bar").addEventListener("click", (ev) => {
      const chip = ev.target.closest(".chip");
      if (!chip) return;
      const g = chip.dataset.genre;
      if (g === "") state.filters.genres = [];
      else {
        const i = state.filters.genres.indexOf(g);
        if (i === -1) state.filters.genres.push(g);
        else state.filters.genres.splice(i, 1);
      }
      state.showAll = false;
      render();
    });

    $("#mood-chips").addEventListener("click", (ev) => {
      const chip = ev.target.closest(".chip");
      if (!chip) return;
      const mood = chip.dataset.mood;
      const i = state.filters.moods.indexOf(mood);
      if (i === -1) state.filters.moods.push(mood);
      else state.filters.moods.splice(i, 1);
      chip.setAttribute("aria-pressed", String(i === -1));
      $$(`[data-facet="moods"]`).forEach((box) => {
        box.checked = state.filters.moods.indexOf(box.value) !== -1;
      });
      render();
    });

    $("#seed-input").addEventListener("change", (ev) => {
      const m = resolveTitle(ev.target.value);
      if (m) { state.seedId = m.movie_id; render(); }
      else if (ev.target.value.trim()) toast("No film by that name");
    });

    $("#lambda").addEventListener("input", (ev) => {
      state.lambda = parseFloat(ev.target.value);
      $("#out-lambda").textContent =
        state.lambda >= 0.9 ? "closest matches"
        : state.lambda >= 0.6 ? "balanced"
        : "wider variety";
      render();
    });

    $("#recommend-mode").addEventListener("change", (ev) => {
      state.mode = ev.target.value;
      render();
    });

    let registerMode = false;
    $("#account-switch").addEventListener("click", () => {
      registerMode = !registerMode;
      $("#account-title").textContent = registerMode ? "Join the royal collection" : "Sync your collection";
      $("#account-submit").textContent = registerMode ? "Create account" : "Sign in";
      $("#account-switch").textContent = registerMode ? "Already have an account? Sign in" : "Need an account? Create one";
    });
    $("#account-close").addEventListener("click", () => { $("#account-drawer").hidden = true; });
    $("#account-form").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const status = $("#account-status");
      status.textContent = "Opening the private box office…";
      try {
        const data = await accountRequest(registerMode ? "/api/auth/register" : "/api/auth/login", {
          username: $("#account-username").value,
          password: $("#account-password").value,
        });
        applyAccount(data);
        status.textContent = "Signed in as " + data.user.username + ". Your collection is synced.";
        toast("Royal collection synced");
      } catch (err) { status.textContent = err.message + " (run with python app.py to enable sync)"; }
    });

    ["cmp-a", "cmp-b"].forEach((idName) => {
      $("#" + idName).addEventListener("change", (ev) => {
        const m = resolveTitle(ev.target.value);
        if (!m) { if (ev.target.value.trim()) toast("No film by that name"); return; }
        if (idName === "cmp-a") state.compareA = m.movie_id;
        else state.compareB = m.movie_id;
        render();
      });
    });

    $("#theme-toggle").addEventListener("click", () => {
      state.theme = state.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = state.theme;
      saveStore();
    });

    $("#rail-toggle").addEventListener("click", () => {
      state.railOpen = !state.railOpen;
      $("#rail").dataset.open = String(state.railOpen);
    });

    $("#scrim").addEventListener("click", (ev) => {
      if (ev.target.id === "scrim") closeModal();
    });

    $("#export-watchlist").addEventListener("click", () => {
      const rows = state.watchlist
        .map((id) => E.byId(id))
        .filter(Boolean)
        .map((m) =>
          [m.title, m.year, m.language, m.rating, m.genres.join("/")]
            .map((v) => `"${String(v).replace(/"/g, '""')}"`)
            .join(",")
        );
      if (!rows.length) { toast("Nothing to export yet"); return; }
      const csv = ["title,year,language,rating,genres"].concat(rows).join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "watchlist.csv";
      a.click();
      URL.revokeObjectURL(a.href);
      toast("Watchlist downloaded");
    });

    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") { closeModal(); return; }
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(ev.target.tagName);
      if (typing) return;
      if (ev.key === "/") { ev.preventDefault(); $("#search").focus(); }
      if (ev.key === "r") handleAction({ target: $("#surprise-btn") });
    });
  }

  /* ---------------------------------------------------------------- init */
  function init() {
    document.documentElement.dataset.theme = state.theme;
    document.body.classList.add("is-ready");
    $("#recommend-mode").value = state.mode;

    $("#stat-films").textContent = T.movies;
    $("#stat-languages").textContent = T.languages;
    $("#stat-genres").textContent = T.genres;
    $("#stat-years").textContent = T.yearMin + "–" + T.yearMax;

    $("#mood-chips").innerHTML = E.STATS.moods
      .map(
        ([mood, n]) =>
          `<button class="chip" data-mood="${esc(mood)}" aria-pressed="false">
             ${esc(mood.replace(/-/g, " "))} <span style="opacity:.6">${n}</span>
           </button>`
      )
      .join("");

    const options = E.MOVIES.slice()
      .sort((a, b) => a.title.localeCompare(b.title))
      .map((m) => `<option value="${esc(m.title)}">`)
      .join("");
    $("#film-list").innerHTML = options;

    $("#colophon-meta").textContent =
      `${T.movies} films · ${E.META.vocabSize} indexed terms · built ${E.META.generated}`;

    const m = E.META.metrics || {};
    if (m["genre_precision@k"] != null) {
      $("#colophon-metrics").textContent =
        `genre precision@10 ${(m["genre_precision@k"] * 100).toFixed(1)}% · ` +
        `director recall ${(m["director_recall@k"] * 100).toFixed(1)}% · ` +
        `coverage ${(m.catalogue_coverage * 100).toFixed(0)}%`;
    }

    buildGenreBar();
    buildRail();
    wire();
    render();
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();