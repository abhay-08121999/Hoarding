/*
 * engine.js — the recommender, running in the browser.
 *
 * This is a faithful port of src/recommender.py. It re-implements the same
 * four-signal hybrid score, the same Rocchio taste profile and the same MMR
 * diversification, reading the TF-IDF vectors and IDF weights that build.py
 * exported into data.js.
 *
 * Running it client-side means the whole app works from a file:// open with
 * no Python process alive, and every slider moves results instantly instead
 * of waiting on a round trip.
 */
(function (global) {
  "use strict";

  const DB = global.MOVIE_DB;
  if (!DB) throw new Error("data.js must load before engine.js");

  const MOVIES = DB.movies;
  const LEX = DB.lexicon;
  const STATS = DB.stats;

  /* Weights mirror config.SCORE_WEIGHTS. Exposed so the UI can retune them
     live and the user can watch the ranking react. */
  const WEIGHTS = Object.assign(
    { text: 0.58, categorical: 0.22, people: 0.12, numeric: 0.08 },
    DB.meta.scoreWeights || {}
  );
  const POPULARITY_PRIOR = 0.03;

  /* ---------------------------------------------------------- indexing */
  const byId = new Map();
  MOVIES.forEach((m, i) => {
    m._i = i;
    byId.set(m.movie_id, m);

    // Sparse vector as a Map for O(1) term lookup during cosine.
    m._vec = new Map(m.vec);

    // Pre-built lowercase tag sets for the categorical / people signals.
    m._cats = new Set(
      []
        .concat(m.genres.map((g) => "g:" + g.toLowerCase()))
        .concat(m.sub_genres.map((s) => "s:" + s.toLowerCase()))
        .concat(["l:" + String(m.language).toLowerCase()])
        .concat(["c:" + String(m.country).toLowerCase()])
        .concat(m.moods.map((x) => "m:" + x))
        .concat(["e:" + String(m.era).toLowerCase()])
    );
    m._people = new Set(
      []
        .concat(m.directors.map((d) => "d:" + d.toLowerCase()))
        .concat(m.cast_list.map((a) => "a:" + a.toLowerCase()))
    );
    m._search = (
      m.title +
      " " +
      m.director +
      " " +
      m.cast +
      " " +
      m.genres.join(" ") +
      " " +
      m.keyword_list.join(" ")
    ).toLowerCase();
  });
  // Word-level token sets (filled in below, once normalise() exists) so
  // search can match whole words instead of raw substrings.

  const T = STATS.totals;
  const span = (lo, hi) => (hi - lo > 0 ? hi - lo : 1);
  MOVIES.forEach((m) => {
    m._num = [
      (m.rating - T.ratingMin) / span(T.ratingMin, T.ratingMax),
      (m.year - T.yearMin) / span(T.yearMin, T.yearMax),
      (m.runtime - T.runtimeMin) / span(T.runtimeMin, T.runtimeMax),
    ];
    m._quality = (m.rating - T.ratingMin) / span(T.ratingMin, T.ratingMax);
  });

  /* ------------------------------------------------------------- text */
  function normalise(text) {
    return String(text || "")
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, " ")
      .trim();
  }

  MOVIES.forEach((m) => {
    m._titleNorm = normalise(m.title);
    m._titleTok = new Set(m._titleNorm.split(" ").filter(Boolean));
    m._peopleNames = m.directors.concat(m.cast_list).map(normalise);
    m._peopleTok = new Set(
      m._peopleNames.join(" ").split(" ").filter(Boolean)
    );
  });

  /*
   * Fallback suffix stripper.
   *
   * Every word that appears in the catalogue is already in LEX.stems, mapped
   * straight to its vocabulary index, so the exact Porter stems from Python
   * are used for anything the corpus knows. This only handles the long tail —
   * a user typing "dreaming" when the corpus only ever said "dreams" — by
   * reducing to a form that may hit LEX.stemIndex.
   */
  function lightStem(word) {
    let w = word;
    if (w.length <= 3) return w;
    if (/ies$/.test(w)) return w.slice(0, -3) + "i";
    if (/(sses|shes|ches|xes)$/.test(w)) return w.slice(0, -2);
    if (/[^s]s$/.test(w)) w = w.slice(0, -1);
    if (/ing$/.test(w) && w.length > 5) w = w.slice(0, -3);
    else if (/ed$/.test(w) && w.length > 4) w = w.slice(0, -2);
    if (/ly$/.test(w) && w.length > 4) w = w.slice(0, -2);
    return w;
  }

  function applySynonyms(text) {
    let out = text;
    const terms = Object.keys(LEX.synonyms).sort((a, b) => b.length - a.length);
    terms.forEach((t) => {
      out = out.replace(
        new RegExp("\\b" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\b", "g"),
        LEX.synonyms[t]
      );
    });
    return out;
  }

  /*
   * Greedy longest-phrase matching, mirroring query_engine.QueryExpander.
   *
   * Without this the atomic tokens the catalogue is indexed with
   * (kw_mind_bending, genre_sci_fi) are unreachable from typed text, and a
   * search for "mind bending space movies" degrades into matching the bare
   * word "space" anywhere it appears.
   */
  function expandQuery(query) {
    const words = applySynonyms(normalise(query)).split(/\s+/).filter(Boolean);
    const termIdx = [];
    const matched = [];
    let i = 0;

    while (i < words.length) {
      let hit = false;
      const upper = Math.min(4, words.length - i);
      for (let span = upper; span >= 1; span--) {
        const phrase = words.slice(i, i + span).join(" ");
        if (LEX.phrases[phrase]) {
          LEX.phrases[phrase].forEach((idx) => {
            termIdx.push(idx, idx, idx); // repeat: concepts outweigh prose
          });
          matched.push(phrase);
          i += span;
          hit = true;
          break;
        }
      }
      if (!hit) {
        const w = words[i];
        if (LEX.stems[w] !== undefined) termIdx.push(LEX.stems[w]);
        else {
          const s = lightStem(w);
          if (LEX.stemIndex[s] !== undefined) termIdx.push(LEX.stemIndex[s]);
        }
        i += 1;
      }
    }
    return { termIdx, matched };
  }

  /* Build an L2-normalised TF-IDF query vector (sublinear tf, as in Python). */
  function queryVector(termIdx) {
    const counts = new Map();
    termIdx.forEach((t) => counts.set(t, (counts.get(t) || 0) + 1));
    const vec = new Map();
    let norm = 0;
    counts.forEach((tf, t) => {
      const w = (1 + Math.log(tf)) * (LEX.idf[t] || 1);
      vec.set(t, w);
      norm += w * w;
    });
    norm = Math.sqrt(norm) || 1;
    vec.forEach((v, t) => vec.set(t, v / norm));
    return vec;
  }

  /* --------------------------------------------------------- similarity */
  function cosine(vecA, vecB) {
    // Iterate the smaller map; both are already L2-normalised.
    let a = vecA, b = vecB;
    if (a.size > b.size) { a = vecB; b = vecA; }
    let dot = 0;
    a.forEach((w, t) => {
      const o = b.get(t);
      if (o !== undefined) dot += w * o;
    });
    return dot;
  }

  function setCosine(a, b) {
    if (!a.size || !b.size) return 0;
    let shared = 0;
    const small = a.size <= b.size ? a : b;
    const large = a.size <= b.size ? b : a;
    small.forEach((v) => { if (large.has(v)) shared += 1; });
    return shared / Math.sqrt(a.size * b.size);
  }

  const NUM_TOLERANCE = [0.15, 0.3, 0.4];
  function numericSim(a, b) {
    let total = 0;
    for (let d = 0; d < 3; d++) {
      total += Math.exp(-Math.abs(a._num[d] - b._num[d]) / NUM_TOLERANCE[d]);
    }
    return total / 3;
  }

  function signals(seed, other) {
    return {
      text: cosine(seed._vec, other._vec),
      categorical: setCosine(seed._cats, other._cats),
      people: setCosine(seed._people, other._people),
      numeric: numericSim(seed, other),
    };
  }

  function blend(sig, movie, weights) {
    const w = weights || WEIGHTS;
    return (
      w.text * sig.text +
      w.categorical * sig.categorical +
      w.people * sig.people +
      w.numeric * sig.numeric +
      POPULARITY_PRIOR * movie._quality
    );
  }

  /* ------------------------------------------------------- explanations */
  const PRETTY_PREFIX = {
    person_: "person", kw_: "theme", genre_: "genre", sub_: "style",
    lang_: "language", country_: "place", era_: "era", mood_: "mood",
  };

  function prettyTerm(token) {
    for (const p in PRETTY_PREFIX) {
      if (token.indexOf(p) === 0) {
        const body = token.slice(p.length).replace(/_/g, " ");
        return {
          label: p === "person_" ? titleCase(body) : body,
          kind: PRETTY_PREFIX[p],
        };
      }
    }
    return null; // bare stemmed prose word — too mangled to show
  }

  function titleCase(s) {
    return s.replace(/\b\w/g, (c) => c.toUpperCase());
  }

  /*
   * Why were these two matched? Take the element-wise minimum of the two
   * TF-IDF vectors (the mass they share) and report the heaviest terms.
   */
  function explain(a, b, limit) {
    limit = limit || 4;
    const shared = [];
    const small = a._vec.size <= b._vec.size ? a._vec : b._vec;
    const large = a._vec.size <= b._vec.size ? b._vec : a._vec;
    small.forEach((w, t) => {
      const o = large.get(t);
      if (o !== undefined) shared.push([t, Math.min(w, o)]);
    });
    shared.sort((x, y) => y[1] - x[1]);

    const out = [];
    const seen = new Set();
    for (const [t] of shared) {
      const pretty = prettyTerm(LEX.vocab[t]);
      if (!pretty || seen.has(pretty.label.toLowerCase())) continue;
      seen.add(pretty.label.toLowerCase());
      out.push(pretty);
      if (out.length >= limit) break;
    }
    return out;
  }

  /* ------------------------------------------------------------ filters */
  function passes(m, f) {
    if (!f) return true;
    if (f.genres && f.genres.length &&
        !m.genres.some((g) => f.genres.indexOf(g) !== -1)) return false;
    if (f.languages && f.languages.length &&
        f.languages.indexOf(m.language) === -1) return false;
    if (f.moods && f.moods.length &&
        !m.moods.some((x) => f.moods.indexOf(x) !== -1)) return false;
    if (f.countries && f.countries.length &&
        f.countries.indexOf(m.country) === -1) return false;
    if (f.ageBands && f.ageBands.length &&
        f.ageBands.indexOf(m.age_band) === -1) return false;
    if (f.minRating != null && m.rating < f.minRating) return false;
    if (f.maxRuntime != null && m.runtime > f.maxRuntime) return false;
    if (f.yearRange && (m.year < f.yearRange[0] || m.year > f.yearRange[1]))
      return false;
    if (f.awardedOnly && !m.has_awards) return false;
    return true;
  }

  function filter(f) {
    return MOVIES.filter((m) => passes(m, f));
  }

  /* --------------------------------------------------------------- MMR */
  /*
   * Maximal Marginal Relevance. Greedily take the candidate that is most
   * relevant while least like what's already chosen. Without it, "films like
   * The Godfather" returns Part II plus five more mafia dramas.
   */
  function mmr(scored, lambda, topN) {
    if (lambda >= 0.999 || !scored.length) return scored.slice(0, topN);
    const pool = scored.slice(0, Math.max(topN * 5, 40));
    const picked = [];
    while (pool.length && picked.length < topN) {
      let bestI = 0;
      if (picked.length) {
        let bestVal = -Infinity;
        for (let i = 0; i < pool.length; i++) {
          let maxSim = 0;
          for (let j = 0; j < picked.length; j++) {
            const s = cosine(pool[i].movie._vec, picked[j].movie._vec);
            if (s > maxSim) maxSim = s;
          }
          const val = lambda * pool[i].score - (1 - lambda) * maxSim;
          if (val > bestVal) { bestVal = val; bestI = i; }
        }
      }
      picked.push(pool[bestI]);
      pool.splice(bestI, 1);
    }
    return picked;
  }

  /* ----------------------------------------------------------- public API */

  /** Films similar to one seed film. */
  function similarTo(seedId, opts) {
    opts = opts || {};
    const seed = byId.get(seedId);
    if (!seed) return [];
    const topN = opts.topN || 12;
    const lambda = opts.lambda == null ? 0.75 : opts.lambda;

    const scored = [];
    for (const m of MOVIES) {
      if (m.movie_id === seedId) continue;
      if (!passes(m, opts.filters)) continue;
      const sig = signals(seed, m);
      scored.push({ movie: m, score: blend(sig, m, opts.weights), sig });
    }
    scored.sort((a, b) => b.score - a.score);

    return mmr(scored, lambda, topN).map((r) => ({
      movie: r.movie,
      score: r.score,
      signals: r.sig,
      why: explain(seed, r.movie),
    }));
  }

  /**
   * Rocchio taste profile: centroid of liked films, pushed away from the
   * centroid of disliked ones. Negative feedback is weighted lower because
   * a thumbs-down is a noisier signal than a thumbs-up.
   */
  function tasteVector(likedIds, dislikedIds, alpha, beta) {
    alpha = alpha == null ? 1.0 : alpha;
    beta = beta == null ? 0.45 : beta;
    const acc = new Map();

    const addAll = (ids, sign, weight) => {
      if (!ids.length) return;
      ids.forEach((id) => {
        const m = byId.get(id);
        if (!m) return;
        m._vec.forEach((w, t) => {
          acc.set(t, (acc.get(t) || 0) + (sign * weight * w) / ids.length);
        });
      });
    };

    addAll(likedIds, +1, alpha);
    addAll(dislikedIds || [], -1, beta);

    let norm = 0;
    acc.forEach((v, t) => {
      if (v <= 0) acc.delete(t);
      else norm += v * v;
    });
    norm = Math.sqrt(norm) || 1;
    acc.forEach((v, t) => acc.set(t, v / norm));
    return acc;
  }

  /* A reusable scorer for one taste profile, so forProfile, the genre shelves
     and the "beyond your usual genres" shelf all rank the same way. */
  function makeProfile(likedIds, dislikedIds) {
    dislikedIds = dislikedIds || [];
    const profile = tasteVector(likedIds, dislikedIds);
    const liked = likedIds.map((id) => byId.get(id)).filter(Boolean);
    const rated = new Set(likedIds.concat(dislikedIds));
    const n = liked.length || 1;

    function score(m, weights) {
      let cat = 0, ppl = 0, num = 0;
      liked.forEach((s) => {
        cat += setCosine(s._cats, m._cats);
        ppl += setCosine(s._people, m._people);
        num += numericSim(s, m);
      });
      const sig = {
        text: cosine(profile, m._vec),
        categorical: cat / n,
        people: ppl / n,
        numeric: num / n,
      };
      return { score: blend(sig, m, weights), sig };
    }

    function because(m) {
      let best = liked[0], bestSim = -1;
      liked.forEach((s) => {
        const sim = cosine(s._vec, m._vec);
        if (sim > bestSim) { bestSim = sim; best = s; }
      });
      return best;
    }

    return { profile, liked, rated, score, because };
  }

  function decorate(ctx, r) {
    const best = ctx.because(r.movie);
    return {
      movie: r.movie,
      score: r.score,
      signals: r.sig,
      because: best,
      why: best ? explain(best, r.movie) : [],
    };
  }

  /** Recommendations against a multi-film taste profile. */
  function forProfile(likedIds, dislikedIds, opts) {
    opts = opts || {};
    dislikedIds = dislikedIds || [];
    const topN = opts.topN || 12;
    const lambda = opts.lambda == null ? 0.7 : opts.lambda;
    if (!likedIds.length) return [];

    const ctx = makeProfile(likedIds, dislikedIds);
    const scored = [];
    for (const m of MOVIES) {
      if (ctx.rated.has(m.movie_id)) continue;
      if (!passes(m, opts.filters)) continue;
      if (opts.genresNotIn && m.genres.some((g) => opts.genresNotIn.has(g)))
        continue;
      const r = ctx.score(m, opts.weights);
      scored.push({ movie: m, score: r.score, sig: r.sig });
    }
    scored.sort((a, b) => b.score - a.score);
    return mmr(scored, lambda, topN).map((r) => decorate(ctx, r));
  }

  /** Top themes in a taste profile, for the "your taste" readout. */
  function profileThemes(likedIds, dislikedIds, limit) {
    limit = limit || 10;
    if (!likedIds.length) return [];
    const vec = tasteVector(likedIds, dislikedIds);
    const entries = [];
    vec.forEach((w, t) => entries.push([t, w]));
    entries.sort((a, b) => b[1] - a[1]);
    const out = [];
    const seen = new Set();
    for (const [t, w] of entries) {
      const p = prettyTerm(LEX.vocab[t]);
      if (!p || seen.has(p.label.toLowerCase())) continue;
      seen.add(p.label.toLowerCase());
      out.push({ label: p.label, kind: p.kind, weight: w });
      if (out.length >= limit) break;
    }
    return out;
  }

  /* ================================================================ search */
  /*
   * Search reads a query on four channels and only returns films that hit at
   * least one of them for real:
   *
   *   1. meaning   — the query expanded onto the corpus's atomic tokens
   *                  (TF-IDF cosine), as in the Python engine
   *   2. title     — exact, whole-word, prefix and typo-tolerant matching
   *   3. people    — director / cast names, whole-word or typo-tolerant
   *   4. entities  — genre, language, country, year and decade words
   *                  ("drama", "romantic", "bollywood", "korean", "90s")
   *
   * The old version added a small popularity prior *before* its relevance
   * cut-off, so every high-rated film cleared the bar: a nonsense query
   * still returned 60 results, and genre words such as "drama" (dropped from
   * the vocabulary for appearing in >55% of films) matched nothing at all.
   * The prior is now a tie-breaker applied after the relevance floor.
   */
  const SEARCH_FLOOR = 0.1;
  const STOP = new Set(
    ("a an and the of in on at for to by from with about like movie movies " +
     "film films some any me my show find want watch something").split(" ")
  );

  const GENRE_ALIASES = {
    "scifi": "Sci-Fi", "science fiction": "Sci-Fi", "sf": "Sci-Fi",
    "romantic": "Romance", "romcom": "Romance", "love": null,
    "animated": "Animation", "cartoon": "Animation", "cartoons": "Animation",
    "anime": "Animation", "scary": "Horror", "spooky": "Horror",
    "funny": "Comedy", "hilarious": "Comedy", "comedies": "Comedy",
    "biopic": "Biography", "biopics": "Biography",
    "historical": "History", "sports": "Sport", "docs": "Documentary",
    "documentaries": "Documentary", "musicals": "Musical", "kids": "Family",
    "spy": null, "gangster": "Crime", "detective": "Mystery",
  };
  const LANGUAGE_ALIASES = {
    bollywood: "Hindi", kollywood: "Tamil", tollywood: "Telugu",
    mollywood: "Malayalam", sandalwood: "Kannada", chinese: "Mandarin",
  };
  const COUNTRY_ALIASES = {
    american: "USA", america: "USA", usa: "USA", hollywood: "USA",
    british: "UK", england: "UK", english: null, indian: "India",
    korean: "South Korea", korea: "South Korea", japanese: "Japan",
    french: "France", italian: "Italy", german: "Germany",
    spanish: "Spain", brazilian: "Brazil", mexican: "Mexico",
    australian: "Australia", taiwanese: "Taiwan",
  };

  // word/phrase -> { genres, languages, countries }, built from the catalogue
  // so it can never point at a value that isn't there.
  const ENTITY_WORDS = new Map();
  function addEntity(word, kind, value) {
    if (!word || !value) return;
    const key = normalise(word);
    if (!key) return;
    if (!ENTITY_WORDS.has(key)) ENTITY_WORDS.set(key, []);
    const list = ENTITY_WORDS.get(key);
    if (!list.some((e) => e.kind === kind && e.value === value))
      list.push({ kind, value });
  }
  STATS.genres.forEach(([g]) => {
    addEntity(g, "genre", g);
    addEntity(g + "s", "genre", g);
  });
  Object.keys(GENRE_ALIASES).forEach((w) => {
    const g = GENRE_ALIASES[w];
    if (g && STATS.genres.some((e) => e[0] === g)) addEntity(w, "genre", g);
  });
  STATS.languages.forEach(([l]) => addEntity(l, "language", l));
  Object.keys(LANGUAGE_ALIASES).forEach((w) => {
    const l = LANGUAGE_ALIASES[w];
    if (STATS.languages.some((e) => e[0] === l)) addEntity(w, "language", l);
  });
  STATS.countries.forEach(([c]) => addEntity(c, "country", c));
  Object.keys(COUNTRY_ALIASES).forEach((w) => {
    const c = COUNTRY_ALIASES[w];
    if (c && STATS.countries.some((e) => e[0] === c)) addEntity(w, "country", c);
  });
  // "korean"/"japanese"/"french"… are both a language and a nationality.
  ["korean", "japanese", "french", "italian", "spanish", "german"].forEach(
    (w) => {
      const lang = w.charAt(0).toUpperCase() + w.slice(1);
      if (STATS.languages.some((e) => e[0] === lang))
        addEntity(w, "language", lang);
    }
  );

  const ENTITY_WEIGHT = {
    genre: 0.3, language: 0.32, country: 0.22, year: 0.32, decade: 0.28,
  };

  function yearEntity(word) {
    let m = /^(19|20)\d\d$/.exec(word);
    if (m) {
      const y = parseInt(word, 10);
      return { kind: "year", value: word, test: (f) => f.year === y };
    }
    m = /^((?:19|20)?)(\d)0s$/.exec(word);
    if (m) {
      const d = parseInt(m[2], 10);
      let start;
      if (m[1]) start = parseInt(m[1] + m[2] + "0", 10);
      else start = d >= 3 ? 1900 + d * 10 : 2000 + d * 10;
      return {
        kind: "decade", value: start + "s",
        test: (f) => f.year >= start && f.year < start + 10,
      };
    }
    return null;
  }

  function entityTest(e) {
    if (e.test) return e.test;
    if (e.kind === "genre") return (f) => f.genres.indexOf(e.value) !== -1;
    if (e.kind === "language") return (f) => f.language === e.value;
    return (f) => f.country === e.value;
  }

  /* Bounded Levenshtein distance; returns max+1 as soon as it must exceed. */
  function lev(a, b, max) {
    if (a === b) return 0;
    if (Math.abs(a.length - b.length) > max) return max + 1;
    let prev = [];
    for (let j = 0; j <= b.length; j++) prev.push(j);
    for (let i = 1; i <= a.length; i++) {
      const cur = [i];
      let rowMin = i;
      for (let j = 1; j <= b.length; j++) {
        const cost = a.charCodeAt(i - 1) === b.charCodeAt(j - 1) ? 0 : 1;
        const v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
        cur.push(v);
        if (v < rowMin) rowMin = v;
      }
      if (rowMin > max) return max + 1;
      prev = cur;
    }
    return prev[b.length];
  }

  /* How well does one query word match any word in a token set?
     1 exact · 0.6 prefix (typing "incep") · 0.5 typo ("interstellr"). */
  function tokenStrength(q, tokens) {
    if (tokens.has(q)) return 1;
    let best = 0;
    if (q.length >= 3) {
      tokens.forEach((t) => {
        if (best < 0.6 && t.length > q.length && t.indexOf(q) === 0) best = 0.6;
      });
    }
    if (best < 0.5 && q.length >= 5) {
      const max = q.length >= 8 ? 2 : 1;
      tokens.forEach((t) => {
        if (best < 0.5 && lev(q, t, max) <= max) best = 0.5;
      });
    }
    return best;
  }

  /* Mean strength across the query's words, or 0 when fewer than 60% of the
     words matched at all (so "shah rukh khan" doesn't drag in every Khan). */
  function coverage(words, tokens) {
    if (!words.length) return 0;
    let hit = 0, sum = 0;
    for (const w of words) {
      const s = tokenStrength(w, tokens);
      if (s > 0) { hit += 1; sum += s; }
    }
    if (hit / words.length < 0.6) return 0;
    return sum / words.length;
  }

  function parseQuery(query) {
    const qNorm = normalise(query);
    const raw = qNorm.split(" ").filter(Boolean);
    const content = raw.filter((w) => !STOP.has(w));
    const words = content.length ? content : raw;

    // Entities: try two-word phrases first ("science fiction", "south korea").
    const entities = [];
    const consumed = new Set();
    let i = 0;
    while (i < words.length) {
      let took = 0;
      if (i + 1 < words.length) {
        const two = words[i] + " " + words[i + 1];
        if (ENTITY_WORDS.has(two)) {
          ENTITY_WORDS.get(two).forEach((e) => entities.push(Object.assign({}, e)));
          consumed.add(i); consumed.add(i + 1);
          took = 2;
        }
      }
      if (!took) {
        const one = words[i];
        if (ENTITY_WORDS.has(one)) {
          ENTITY_WORDS.get(one).forEach((e) => entities.push(Object.assign({}, e)));
          consumed.add(i);
          took = 1;
        } else {
          const ye = yearEntity(one);
          if (ye) { entities.push(ye); consumed.add(i); took = 1; }
        }
      }
      i += took || 1;
    }
    entities.forEach((e) => { e.test = entityTest(e); });

    // Words not explained as entities are what title / people matching uses.
    const cover = words.filter((_, idx) => !consumed.has(idx));
    return { qNorm, words, cover, entities };
  }

  /** Free-text search over meaning, titles, people and facets. */
  function search(query, opts) {
    opts = opts || {};
    const topN = opts.topN || 40;
    if (!String(query).trim()) return [];

    const parsed = parseQuery(query);
    const { qNorm, cover, entities } = parsed;
    const { termIdx, matched } = expandQuery(query);
    const qVec = queryVector(termIdx);
    const padded = " " + qNorm + " ";

    // Entity labels shown as "Reading that as …". Genre entities that are
    // also nationality words (korean → language + country) show once.
    const entityLabels = [];
    entities.forEach((e) => {
      if (entityLabels.indexOf(e.value) === -1) entityLabels.push(e.value);
    });
    const labels = [];
    matched.concat(entityLabels).forEach((l) => {
      if (!labels.some((x) => x.toLowerCase() === String(l).toLowerCase()))
        labels.push(l);
    });

    const scored = [];
    for (const m of MOVIES) {
      if (!passes(m, opts.filters)) continue;

      const text = qVec.size ? cosine(qVec, m._vec) : 0;

      // -- title
      let titleScore = 0;
      if (qNorm) {
        if (m._titleNorm === qNorm) titleScore = 1.0;
        else if (!cover.length) titleScore = 0; // "war" means the genre, not "Infinity War"
        else if (
          padded.indexOf(" " + m._titleNorm + " ") !== -1 ||
          (" " + m._titleNorm + " ").indexOf(padded) !== -1
        ) titleScore = 0.6;
        else titleScore = 0.55 * coverage(cover, m._titleTok);
      }

      // -- people
      let personScore = 0, personHit = null;
      if (cover.length) {
        for (const full of m._peopleNames) {
          if (full && padded.indexOf(" " + full + " ") !== -1) {
            personScore = 0.6; personHit = full; break;
          }
        }
        if (!personScore) {
          const c = coverage(cover, m._peopleTok);
          if (c) {
            personScore = 0.5 * c;
            personHit = m.directors.concat(m.cast_list).find((p) => {
              const toks = normalise(p).split(" ");
              return cover.some((w) => toks.some((t) => t === w || t.indexOf(w) === 0));
            }) || null;
          }
        }
      }

      // -- entities
      let entityScore = 0;
      const hits = [];
      entities.forEach((e) => {
        if (e.test(m)) {
          entityScore += ENTITY_WEIGHT[e.kind] || 0.25;
          hits.push(e);
        }
      });

      const relevance = text + titleScore + personScore + entityScore;
      if (relevance < SEARCH_FLOOR) continue;

      scored.push({
        movie: m,
        score: relevance + POPULARITY_PRIOR * m._quality,
        matched: labels,
        _text: text, _person: personHit, _hits: hits, _title: titleScore,
      });
    }
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, topN);

    // Reason chips: *why* each hit was returned.
    top.forEach((r) => {
      const why = [];
      const seen = new Set();
      const push = (kind, label) => {
        const k = String(label).toLowerCase();
        if (seen.has(k)) return;
        seen.add(k);
        why.push({ kind, label });
      };
      if (r._title >= 0.55) push("theme", "title match");
      if (r._person) push("person", titleCase(r._person));
      r._hits.forEach((e) =>
        push(e.kind === "genre" ? "genre" : "style", e.value));
      if (r._text > 0) {
        const shared = [];
        qVec.forEach((w, t) => {
          const o = r.movie._vec.get(t);
          if (o !== undefined) shared.push([t, Math.min(w, o)]);
        });
        shared.sort((a, b) => b[1] - a[1]);
        for (const [t] of shared) {
          const p = prettyTerm(LEX.vocab[t]);
          if (p) push(p.kind, p.label);
          if (why.length >= 4) break;
        }
      }
      r.why = why.slice(0, 4);
    });
    return top;
  }

  /** "Did you mean…" — closest titles when a search finds nothing. */
  function suggest(query, limit) {
    limit = limit || 4;
    const q = normalise(query);
    if (!q) return [];
    const maxD = Math.max(2, Math.ceil(q.length * 0.4));
    const out = [];
    MOVIES.forEach((m) => {
      const t = m._titleNorm;
      let s = 0;
      const d = lev(q, t, maxD);
      if (d <= maxD) s = 1 - d / Math.max(q.length, t.length);
      if (t.indexOf(q) === 0 || q.indexOf(t) === 0) s = Math.max(s, 0.7);
      q.split(" ").forEach((w) => {
        if (w.length >= 4 && tokenStrength(w, m._titleTok) > 0) s = Math.max(s, 0.55);
      });
      if (s >= 0.5) out.push([s, m]);
    });
    out.sort((a, b) => b[0] - a[0]);
    return out.slice(0, limit).map((e) => e[1]);
  }

  /** Side-by-side comparison of two films. */
  function compare(idA, idB) {
    const a = byId.get(idA), b = byId.get(idB);
    if (!a || !b) return null;
    const sig = signals(a, b);
    return {
      a, b, signals: sig,
      overall: blend(sig, b, WEIGHTS),
      shared: explain(a, b, 8),
      sharedGenres: a.genres.filter((g) => b.genres.indexOf(g) !== -1),
      sharedPeople: a.directors
        .concat(a.cast_list)
        .filter((p) => b.directors.concat(b.cast_list).indexOf(p) !== -1),
    };
  }

  /* Auto-generated browse collections. */
  function collections() {
    const out = [];
    const sorted = (arr, key) => arr.slice().sort(key);

    out.push({
      id: "top-rated",
      name: "Highest rated",
      note: "The catalogue's best-reviewed films",
      items: sorted(MOVIES, (a, b) => b.rating - a.rating).slice(0, 18),
    });

    // "Hidden gems": strong ratings but no major awards listed.
    out.push({
      id: "hidden-gems",
      name: "Hidden gems",
      note: "Highly rated, but no major awards on record",
      items: sorted(
        MOVIES.filter((m) => !m.has_awards && m.rating >= 7.8),
        (a, b) => b.rating - a.rating
      ).slice(0, 18),
    });

    out.push({
      id: "short",
      name: "Under 100 minutes",
      note: "When you don't have all evening",
      items: sorted(
        MOVIES.filter((m) => m.runtime <= 100),
        (a, b) => b.rating - a.rating
      ).slice(0, 18),
    });

    out.push({
      id: "epics",
      name: "Long haul",
      note: "Films that take their time",
      items: sorted(
        MOVIES.filter((m) => m.runtime >= 165),
        (a, b) => b.rating - a.rating
      ).slice(0, 18),
    });

    out.push({
      id: "family",
      name: "Watch with anyone",
      note: "Rated for all ages",
      items: sorted(
        MOVIES.filter((m) => m.age_band === "All ages"),
        (a, b) => b.rating - a.rating
      ).slice(0, 18),
    });

    // One collection per prolific director.
    const dirCount = new Map();
    MOVIES.forEach((m) =>
      m.directors.forEach((d) => dirCount.set(d, (dirCount.get(d) || 0) + 1))
    );
    Array.from(dirCount.entries())
      .filter((e) => e[1] >= 3)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .forEach(([d, n]) => {
        out.push({
          id: "dir-" + normalise(d).replace(/\s+/g, "-"),
          name: d,
          note: n + " films in the catalogue",
          items: sorted(
            MOVIES.filter((m) => m.directors.indexOf(d) !== -1),
            (a, b) => b.rating - a.rating
          ),
        });
      });

    // One per language with enough films to browse.
    STATS.languages
      .filter((e) => e[1] >= 5)
      .forEach(([lang, n]) => {
        out.push({
          id: "lang-" + normalise(lang).replace(/\s+/g, "-"),
          name: lang + " cinema",
          note: n + " films",
          items: sorted(
            MOVIES.filter((m) => m.language === lang),
            (a, b) => b.rating - a.rating
          ).slice(0, 18),
        });
      });

    return out;
  }

  /** Curated recommendation modes used by the premium recommendations panel. */
  function recommendMode(mode, opts) {
    opts = opts || {};
    const topN = opts.topN || 18;
    const rated = new Set((opts.likedIds || []).concat(opts.dislikedIds || []));
    const seedResults = opts.seedId ? similarTo(opts.seedId, { topN: 80, lambda: opts.lambda, filters: opts.filters }) : [];
    const scoreRows = (predicate, scorer) => MOVIES.filter((m) => !rated.has(m.movie_id) && predicate(m) && passes(m, opts.filters))
      .map((m) => ({ movie: m, score: scorer(m), why: [] }))
      .sort((a, b) => b.score - a.score).slice(0, topN);

    if (mode === "closest") return seedResults.slice(0, topN);
    if (mode === "hidden") return scoreRows((m) => !m.has_awards && m.rating >= 7.5, (m) => m.rating / 10 + (m.runtime < 130 ? .03 : 0));
    if (mode === "comfort") return scoreRows((m) => m.moods.includes("feel-good") || m.genres.some((g) => ["Comedy", "Family", "Romance", "Musical"].includes(g)), (m) => m.rating / 10 + (m.moods.includes("feel-good") ? .12 : 0));
    if (mode === "royal") return scoreRows(() => true, (m) => m.rating / 10 + (m.has_awards ? .12 : 0));
    if (mode === "short") return scoreRows((m) => m.runtime <= 120, (m) => m.rating / 10 + (120 - m.runtime) / 1000);
    if (mode === "challenge") {
      const liked = (opts.likedIds || []).map((id) => byId.get(id)).filter(Boolean);
      return scoreRows(() => true, (m) => {
        if (!liked.length) return m.rating / 10;
        const familiarity = Math.max(...liked.map((s) => cosine(s._vec, m._vec)));
        return m.rating / 10 + (1 - familiarity) * .25;
      });
    }
    return seedResults.slice(0, topN);
  }

  /**
   * Genre shelves for the Discover landing view: the best films in each
   * genre, one shelf per genre.
   *
   * With no history a shelf is ranked by rating plus an awards nudge. Once
   * the person has liked films, every shelf is ranked by *their* taste
   * profile instead, so "Comedy" means comedies they would enjoy. A film is
   * shown on at most two shelves so Drama and Crime don't repeat the same
   * ten titles.
   */
  function genreShelves(opts) {
    opts = opts || {};
    const perShelf = opts.perShelf || 12;
    const maxShelves = opts.maxShelves || 12;
    const minFilms = opts.minFilms || 5;
    const likedIds = opts.likedIds || [];
    const dislikedIds = opts.dislikedIds || [];
    const ctx = likedIds.length ? makeProfile(likedIds, dislikedIds) : null;
    const skip = new Set(dislikedIds);
    const used = new Map();

    const genres = STATS.genres.filter((g) => g[1] >= minFilms).slice(0, maxShelves);
    const shelves = [];

    genres.forEach(([genre, total]) => {
      const pool = MOVIES.filter(
        (m) => m.genres.indexOf(genre) !== -1 && !skip.has(m.movie_id) &&
               passes(m, opts.filters)
      );
      if (!pool.length) return;

      const ranked = pool
        .map((m) => {
          if (ctx && !ctx.rated.has(m.movie_id)) {
            const r = ctx.score(m);
            return { movie: m, score: r.score, ctx: true };
          }
          if (ctx && ctx.rated.has(m.movie_id)) return null; // already liked
          return {
            movie: m,
            score: m.rating / 10 + (m.has_awards ? 0.04 : 0) + m._quality * 0.02,
          };
        })
        .filter(Boolean)
        .sort((a, b) => b.score - a.score);

      const items = [];
      for (const r of ranked) {
        if ((used.get(r.movie.movie_id) || 0) >= 2) continue;
        items.push({
          movie: r.movie,
          score: r.score,
          why: ctx ? explain(ctx.because(r.movie), r.movie, 3) : [],
        });
        if (items.length >= perShelf) break;
      }
      if (!items.length) return;
      items.forEach((it) =>
        used.set(it.movie.movie_id, (used.get(it.movie.movie_id) || 0) + 1));
      shelves.push({ genre, total: pool.length, personalised: !!ctx, items });
    });
    return shelves;
  }

  /**
   * "Same spirit, different genre": take the films most similar to a seed,
   * drop those from the seed's own primary genre, and group what is left by
   * primary genre. A Sci-Fi heist film gets a Mystery shelf, a Drama shelf,
   * a Crime shelf — the same DNA in a genre it wasn't filed under.
   */
  function crossGenreShelves(seedId, opts) {
    opts = opts || {};
    const seed = byId.get(seedId);
    if (!seed) return [];
    const perShelf = opts.perShelf || 6;
    const maxShelves = opts.maxShelves || 5;
    const seedGenres = new Set(seed.genres);

    const pool = similarTo(seedId, {
      topN: 120, lambda: 1, filters: opts.filters,
    }).filter((r) => r.movie.primary_genre !== seed.primary_genre);

    const groups = new Map();
    pool.forEach((r) => {
      const g = r.movie.primary_genre || r.movie.genres[0];
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(r);
    });

    const shelves = [];
    groups.forEach((items, genre) => {
      if (items.length < 2) return;
      shelves.push({
        genre,
        best: items[0].score,
        // A shelf of genres the seed already carries is less of a surprise.
        fresh: !seedGenres.has(genre),
        items: items.slice(0, perShelf),
      });
    });
    shelves.sort((a, b) => (b.fresh - a.fresh) || (b.best - a.best));
    return shelves.slice(0, maxShelves);
  }

  /** Taste-ranked films from genres the person hasn't liked yet. */
  function beyondYourGenres(likedIds, dislikedIds, opts) {
    opts = opts || {};
    if (!likedIds.length) return [];
    const liked = likedIds.map((id) => byId.get(id)).filter(Boolean);
    const known = new Set();
    liked.forEach((m) => m.genres.forEach((g) => known.add(g)));
    return forProfile(likedIds, dislikedIds, {
      topN: opts.topN || 8, lambda: 0.7, filters: opts.filters,
      genresNotIn: known,
    });
  }

  /** Rank the whole catalogue by taste (used when sorting by "Best match"). */
  function rankByTaste(likedIds, dislikedIds, pool) {
    if (!likedIds.length) return null;
    const ctx = makeProfile(likedIds, dislikedIds);
    const out = new Map();
    pool.forEach((m) => {
      if (ctx.rated.has(m.movie_id)) { out.set(m.movie_id, -1); return; }
      out.set(m.movie_id, ctx.score(m).score);
    });
    return out;
  }

  function randomMovie(filters) {
    const pool = filter(filters);
    if (!pool.length) return null;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  global.Engine = {
    MOVIES, STATS, META: DB.meta, WEIGHTS,
    byId: (id) => byId.get(id),
    similarTo, forProfile, profileThemes, tasteVector, recommendMode,
    genreShelves, crossGenreShelves, beyondYourGenres, rankByTaste,
    search, suggest, compare, collections, filter, randomMovie,
    explain, signals, blend, normalise, expandQuery, prettyTerm, titleCase,
  };
})(window);