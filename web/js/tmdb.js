/*
 * tmdb.js — posters, backdrops and trailers from TMDB.
 *
 * Lookup order for a film:
 *   1. the Flask proxy  (/api/metadata/<id>)   — key stays on the server
 *   2. TMDB directly, using APP_CONFIG.TMDB_API_KEY
 *   3. nothing — the painted poster stays, the app keeps working
 *
 * Results are cached in localStorage so each film is looked up once, and
 * requests run through a small queue so a page of 60 cards doesn't hammer
 * the API. Every failure carries a readable reason (`TMDB.status()`), because
 * "it silently doesn't work" is the hardest thing to debug.
 */
(function (global) {
  "use strict";

  const cfg = Object.assign(
    {
      TMDB_API_KEY: "",
      TMDB_API_BASE: "https://api.themoviedb.org/3",
      TMDB_IMAGE_BASE: "https://image.tmdb.org/t/p/",
      USE_SERVER_PROXY: true,
    },
    global.APP_CONFIG || {}
  );
  const KEY = String(cfg.TMDB_API_KEY || "").trim();
  const IMG = cfg.TMDB_IMAGE_BASE.replace(/\/?$/, "/");

  const CACHE_KEY = "hoarding.tmdb.v2";
  const HIT_TTL = 30 * 24 * 3600 * 1000;
  const MISS_TTL = 6 * 3600 * 1000;

  /* ------------------------------------------------------------ state */
  let proxy = cfg.USE_SERVER_PROXY && location.protocol !== "file:" ? "unknown" : "off";
  let lastError = null;
  const memory = new Map();       // movie_id -> meta
  const inflight = new Map();     // key -> Promise
  let disk = readDisk();
  let dirty = false;

  function readDisk() {
    try { return JSON.parse(localStorage.getItem(CACHE_KEY)) || {}; }
    catch (e) { return {}; }
  }
  function flushDisk() {
    if (!dirty) return;
    dirty = false;
    try { localStorage.setItem(CACHE_KEY, JSON.stringify(disk)); }
    catch (e) { disk = {}; /* quota — carry on without persistence */ }
  }
  let flushTimer;
  function scheduleFlush() {
    dirty = true;
    clearTimeout(flushTimer);
    flushTimer = setTimeout(flushDisk, 400);
  }

  function fresh(entry) {
    if (!entry) return false;
    const ttl = entry.matched === false ? MISS_TTL : HIT_TTL;
    return Date.now() - (entry.ts || 0) < ttl;
  }

  /* -------------------------------------------------------- run queue */
  const MAX_PARALLEL = 4;
  let active = 0;
  const waiting = [];
  function enqueue(task) {
    return new Promise((resolve, reject) => {
      waiting.push({ task, resolve, reject });
      pump();
    });
  }
  function pump() {
    while (active < MAX_PARALLEL && waiting.length) {
      const job = waiting.shift();
      active += 1;
      job.task().then(job.resolve, job.reject).finally(() => {
        active -= 1;
        pump();
      });
    }
  }

  /* ----------------------------------------------------- direct TMDB */
  const isBearer = KEY.indexOf("eyJ") === 0 || KEY.length > 40;

  async function tmdb(path, params) {
    const url = new URL(cfg.TMDB_API_BASE.replace(/\/$/, "") + path);
    Object.keys(params || {}).forEach((k) => {
      if (params[k] != null && params[k] !== "") url.searchParams.set(k, params[k]);
    });
    const headers = { Accept: "application/json" };
    if (isBearer) headers.Authorization = "Bearer " + KEY;
    else url.searchParams.set("api_key", KEY);

    let res;
    try {
      res = await fetch(url.toString(), { headers });
    } catch (e) {
      throw new Error("Couldn't reach TMDB (network or blocked request)");
    }
    if (res.status === 401)
      throw new Error("TMDB rejected the key (401) — check APP_CONFIG.TMDB_API_KEY");
    if (res.status === 429) throw new Error("TMDB rate limit hit (429) — try again shortly");
    if (!res.ok) throw new Error("TMDB error " + res.status);
    return res.json();
  }

  const norm = (s) =>
    String(s || "").normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
      .toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

  /* Pick the right hit. Title alone isn't enough (two different films are
     called "Drishyam"), so the release year carries most of the weight. */
  function pickBest(results, movie) {
    const want = norm(movie.title);
    const wantTok = new Set(want.split(" "));
    let best = null, bestScore = 0;
    (results || []).slice(0, 10).forEach((r) => {
      const titles = [norm(r.title), norm(r.original_title)];
      let t = 0;
      if (titles.indexOf(want) !== -1) t = 3;
      else if (titles.some((x) => x && (x.indexOf(want) !== -1 || want.indexOf(x) !== -1))) t = 1.5;
      else {
        const tt = new Set(titles[0].split(" "));
        let overlap = 0;
        wantTok.forEach((w) => { if (tt.has(w)) overlap += 1; });
        if (overlap / Math.max(wantTok.size, tt.size) >= 0.6) t = 1;
      }
      if (!t) return;

      let y = 0;
      const ry = parseInt(String(r.release_date || "").slice(0, 4), 10);
      if (ry && movie.year) {
        const d = Math.abs(ry - movie.year);
        y = d === 0 ? 3 : d === 1 ? 1.5 : d <= 3 ? -0.5 : -2;
      }
      const pop = Math.min(Math.log10((r.vote_count || 0) + 1), 4) * 0.3;
      const s = t + y + pop + (r.poster_path ? 0.3 : 0);
      if (s > bestScore) { bestScore = s; best = r; }
    });
    return bestScore >= 2 ? best : null;
  }

  function pickTrailer(videos) {
    const list = ((videos && videos.results) || []).filter(
      (v) => v.site === "YouTube" && v.key
    );
    if (!list.length) return null;
    const rank = (v) => {
      const type = { Trailer: 0, Teaser: 2, Clip: 3 }[v.type];
      let r = type == null ? 4 : type;
      if (v.type === "Trailer" && v.official) r -= 0.5;
      if (v.iso_639_1 === "en") r -= 0.2;
      return r;
    };
    list.sort((a, b) =>
      rank(a) - rank(b) || String(b.published_at).localeCompare(String(a.published_at)));
    const v = list[0];
    return { key: v.key, name: v.name || "Trailer", type: v.type || "" };
  }

  async function direct(movie, full) {
    const search = async (withYear) =>
      tmdb("/search/movie", {
        query: movie.title, year: withYear ? movie.year : "",
        include_adult: "false", language: "en-US",
      });
    let data = await search(true);
    let hit = pickBest(data.results, movie);
    if (!hit) { data = await search(false); hit = pickBest(data.results, movie); }
    if (!hit) return { matched: false, source: "tmdb" };

    const out = {
      matched: true, source: "tmdb", tmdbId: hit.id,
      poster: hit.poster_path || null, backdrop: hit.backdrop_path || null,
      overview: hit.overview || "", release: hit.release_date || "",
      vote: hit.vote_average || null, votes: hit.vote_count || 0,
      tagline: "", trailer: undefined, // undefined = not fetched yet
    };
    if (full) await addDetails(out, hit);
    return out;
  }

  async function addDetails(out, hit) {
    const langs = ["en", "null"];
    if (hit && hit.original_language && langs.indexOf(hit.original_language) === -1)
      langs.push(hit.original_language);
    const d = await tmdb("/movie/" + out.tmdbId, {
      append_to_response: "videos", language: "en-US",
      include_video_language: langs.join(","),
    });
    out.tagline = d.tagline || "";
    out.overview = d.overview || out.overview;
    out.runtime = d.runtime || null;
    out.trailer = pickTrailer(d.videos);
    if (d.poster_path) out.poster = d.poster_path;
    if (d.backdrop_path) out.backdrop = d.backdrop_path;
  }

  /* ------------------------------------------------ Flask proxy path */
  async function viaProxy(movie, full) {
    if (proxy === "off") return null;
    let res;
    try {
      res = await fetch(
        "/api/metadata/" + encodeURIComponent(movie.movie_id) + (full ? "" : "?lite=1")
      );
    } catch (e) { proxy = "off"; return null; }
    if (!res.ok) { proxy = "off"; return null; }

    let j;
    try { j = await res.json(); } catch (e) { proxy = "off"; return null; }
    proxy = "ok";
    if (j.configured === false) { proxy = "off"; return null; } // server has no key
    if (j.error) { lastError = j.error; return { error: j.error }; }
    if (!j.matched) return { matched: false, source: "tmdb" };
    return {
      matched: true, source: "tmdb", tmdbId: j.tmdb_id,
      poster: j.poster, backdrop: j.backdrop, overview: j.overview || "",
      release: j.release_date || "", vote: j.vote_average, votes: j.vote_count || 0,
      tagline: j.tagline || "", runtime: j.runtime || null,
      trailer: full ? j.trailer || null : j.trailer === undefined ? undefined : j.trailer,
    };
  }

  /* --------------------------------------------------------- public */
  function configured() { return proxy !== "off" || !!KEY; }

  /**
   * Metadata for one film. `full` also fetches the trailer (an extra
   * request), so cards call it with false and the detail modal with true.
   * Never rejects: returns { matched:false, error } on failure.
   */
  function get(movie, full) {
    if (!configured()) return Promise.resolve({ matched: false, disabled: true });

    const id = movie.movie_id;
    const cached = memory.get(id) || (fresh(disk[id]) ? disk[id] : null);
    if (cached && (!full || cached.trailer !== undefined || cached.matched === false)) {
      memory.set(id, cached);
      return Promise.resolve(cached);
    }

    const flightKey = id + (full ? ":full" : ":lite");
    if (inflight.has(flightKey)) return inflight.get(flightKey);

    const p = enqueue(async () => {
      let out = null;
      try {
        out = await viaProxy(movie, full);
        if (out && out.error) out = null; // fall through to direct if we can
        if (!out && KEY) out = await direct(movie, full);
        if (!out) {
          lastError = lastError ||
            "No TMDB key found — add it to web/js/config.js or set TMDB_API_KEY for python app.py";
          return { matched: false, error: lastError };
        }
        lastError = null;
      } catch (e) {
        lastError = e.message;
        return { matched: false, error: e.message };
      }
      // Keep a trailer we already have when a lite lookup comes back later.
      const prev = memory.get(id);
      if (prev && prev.trailer && out.trailer === undefined) out.trailer = prev.trailer;
      out.ts = Date.now();
      memory.set(id, out);
      disk[id] = out;
      scheduleFlush();
      return out;
    }).finally(() => inflight.delete(flightKey));

    inflight.set(flightKey, p);
    return p;
  }

  function imageUrl(path, size) {
    return path ? IMG + (size || "w342") + path : "";
  }

  function youtubeSearchUrl(movie) {
    return "https://www.youtube.com/results?search_query=" +
      encodeURIComponent(movie.title + " " + movie.year + " official trailer");
  }

  global.TMDB = {
    get, imageUrl, youtubeSearchUrl, configured,
    status: () => ({ mode: proxy === "ok" ? "server" : KEY ? "browser" : "off", lastError }),
    clearCache() { memory.clear(); disk = {}; dirty = true; flushDisk(); },
  };
})(window);