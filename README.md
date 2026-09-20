# Hoarding — a content-based movie recommender

A movie recommender built from four spreadsheets, with a from-scratch NLP
pipeline (Python) and a fully interactive browser app (HTML/CSS/JS) that
runs the same ranking math client-side.

Catalogue: **184 unique films** merged from four differently-formatted
source files, spanning 1942–2024, 14 languages, 21 genres.

## Quick start

### Option A — just open it (no install)
```
web/index.html
```
Double-click it. The web app ships with a pre-built `web/js/data.js` and
runs entirely in the browser — no server, no Python needed.

### Option B — rebuild the model / run the Flask API
```bash
pip install -r requirements.txt
python build.py          # clean data, train, evaluate, re-export web/js/data.js
python app.py             # optional JSON API + serves the web app at :5000
python -m pytest tests/   # or: python tests/test_pipeline.py
```

## What it does

**Discover** — free-text NLP search ("mind bending space movies", "funny
bollywood college") plus a full filter rail (genre, language, mood, rating,
runtime, year, awards).

**Similar films** — pick a seed film, get its closest relatives with a
variety slider (Maximal Marginal Relevance) so you don't just get five
sequels.

**For you** — like/dislike a few films and a taste profile builds live,
shown as a bar chart of the themes driving it.

**Explore** — six charts (genre counts, languages, films-by-decade, rating
distribution, rating-by-genre, top directors) plus auto-generated shelves
("Hidden gems", "Under 100 minutes", per-director and per-language
collections).

**Compare** — two films side by side, broken into the four similarity
signals that produced the score.

**Watchlist** — persists to localStorage, exports to CSV.

Every recommendation is labelled with *why* it was suggested (shared
theme, director, genre, mood) — not just a bare score.

## Architecture

```
data/raw/*.xlsx                  4 source spreadsheets, 4 different column namings
      │
src/schema.py                    map every header variant onto one canonical schema
src/data_loader.py                clean, coerce types, merge cross-file duplicates,
                                   derive era/mood/runtime-band facets
      │
src/text_nlp.py                  normaliser, stopword filter, from-scratch Porter
                                   stemmer (no nltk), atomic entity/phrase tokens
src/features.py                  weighted token "soup" → TF-IDF (184×2414) +
                                   categorical + people + numeric matrices
src/query_engine.py               phrase-aware query expansion (typed text →
                                   the corpus's atomic tokens)
src/recommender.py                hybrid similarity, Rocchio taste profiles,
                                   MMR diversification, explanations
src/evaluate.py                   proxy-relevance metrics, ablation study
      │
src/export_web.py + build.py     serialise catalogue + TF-IDF vectors + lexicon
                                   into web/js/data.js
      │
web/js/engine.js                 the same hybrid scoring, ported to JS,
                                   verified to match Python's scores exactly
web/js/ui.js + index.html + style.css   the interactive app
      │
app.py                            optional Flask JSON API (not required to
                                   use the app — everything above works from
                                   a plain file:// open)
```

### The four data-quality problems this had to solve

1. **Four column namings for the same 15 fields.** `Movie Name` / `Title` /
   `title` / `Title` again but under `Category` instead of `Genre` for the
   fourth file. Solved with an alias table plus positional fallback.
2. **Three different multi-value separators** across cast/genre columns
   (`,`, `|`, `/`, and `&`). Solved with one regex covering all of them.
3. **Cross-file duplicates that are the same film** (3 Idiots, Dangal,
   Gladiator, Inception, Interstellar, The Dark Knight each appear in two
   files) **vs. same-titled films that are genuinely different** (two
   films called *Drishyam*: the 2013 Malayalam original and its 2015 Hindi
   remake). Solved by keying merges on (normalised title, year), not title
   alone, with field-level coalescing so a row missing its synopsis
   inherits one from its duplicate.
4. **Search that only matched literal words.** The catalogue is indexed
   with atomic tokens (`kw_mind_bending`, `genre_sci_fi`) for precision,
   which meant a naive query for "mind bending space movies" returned
   *Toy Story* first (it matched "space" in "space ranger" and nothing
   else). Fixed with a query-expansion layer that resolves typed phrases
   onto the corpus's atomic tokens before scoring — there's a regression
   test (`test_search_understands_themes_not_just_words`) pinning this.

## How recommendations are scored

Four independent similarity signals, blended:

| Signal | Weight | What it captures |
|---|---|---|
| Text (TF-IDF cosine) | 0.58 | Theme/plot/keyword similarity over the weighted soup |
| Categorical | 0.22 | Shared genre, subgenre, language, mood, era |
| People | 0.12 | Shared director / cast |
| Numeric | 0.08 | Closeness in rating, year, runtime |

Kept separate rather than concatenated into one vector so each
recommendation can report *how much* of the match came from each signal
(see the Compare tab), and so an ablation study can check each signal
earns its weight rather than being decoration.

## Evaluation

No user-interaction data exists in this catalogue (no ratings matrix, no
click history), so quality is measured with proxy-relevance metrics
computed over the whole catalogue — see `reports/evaluation.txt` for full
numbers, or regenerate with `python build.py`.

```
genre_precision@10      0.966   (96.6% of recs share ≥1 genre with the seed)
theme_overlap@10        0.043   (mean Jaccard over curated keyword sets)
director_recall@10      0.938   (recs surface a same-director sibling film)
intra_list_diversity    0.947   (1 − mean pairwise similarity within a list)
catalogue_coverage      1.000   (every film appears in someone's top-10)
```

The ablation study (`python build.py`, or `src/evaluate.ablation`) turns
each signal off in turn and re-measures — dropping the categorical signal
costs ~8 points of genre precision, dropping people costs ~10 points of
director recall, confirming the hybrid isn't padding.

## Tests

`tests/test_pipeline.py` — 24 tests, no pytest required to run them:
```bash
python tests/test_pipeline.py
```
Covers schema mapping across all four namings, the Porter stemmer against
its 37 canonical test cases, cross-file dedup (including the Drishyam
same-title-different-film case), recommendation sanity (a seed is never
recommended back to itself, its sequel ranks first, MMR increases
variety), and the search regression above.

## Project layout

```
build.py                  one-command pipeline: clean → features → evaluate → export
app.py                     optional Flask API + static file server
requirements.txt
data/raw/                  the four source workbooks
data/processed/            cleaned catalogue (csv + json)
reports/                   evaluation output
src/                        Python pipeline (see Architecture above)
web/                        the browser app — open web/index.html directly
tests/                      test suite
```
