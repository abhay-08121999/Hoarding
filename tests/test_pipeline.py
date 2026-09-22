"""
Test suite.

Run with:  python -m pytest tests/ -v
       or: python tests/test_pipeline.py     (no pytest needed)

Covers the parts most likely to break quietly: schema mapping across the
four differently-named workbooks, the stemmer, cross-file deduplication,
and whether recommendations are actually sensible rather than merely
non-empty.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import build_clean_frame
from src.features import build_feature_space
from src.recommender import MovieRecommender
from src.schema import map_columns, split_multi
from src.text_nlp import normalise, stem, tokenize

_CACHE = {}


def model():
    """Build once, reuse across tests."""
    if "rec" not in _CACHE:
        df = build_clean_frame()
        space = build_feature_space(df)
        _CACHE["df"] = df
        _CACHE["space"] = space
        _CACHE["rec"] = MovieRecommender(space)
    return _CACHE["rec"]


# ------------------------------------------------------------ schema
def test_schema_maps_all_four_namings():
    """Each workbook names its columns differently; all must map cleanly."""
    variants = [
        ["Movie ID", "Movie Name", "Genre", "Sub-Genre", "Language"],
        ["ID", "Title", "Genres", "Subgenre", "Original Language"],
        ["movie_id", "title", "genre", "sub_genre", "language"],
        ["MovieID", "Title", "Category", "Sub Genre", "Lang"],
    ]
    for headers in variants:
        mapped = map_columns(headers)
        assert set(mapped.values()) == {
            "source_id", "title", "genre", "sub_genre", "language"
        }, f"failed for {headers}"


def test_split_multi_handles_every_separator():
    assert split_multi("Action, Crime, Drama") == ["Action", "Crime", "Drama"]
    assert split_multi("Action|Crime|Drama") == ["Action", "Crime", "Drama"]
    assert split_multi("Crime/Drama") == ["Crime", "Drama"]
    assert split_multi("Roger Allers & Rob Minkoff") == [
        "Roger Allers", "Rob Minkoff"
    ]
    assert split_multi("") == []
    assert split_multi(None) == []


# --------------------------------------------------------------- nlp
def test_porter_stemmer():
    cases = {
        "caresses": "caress", "ponies": "poni", "cats": "cat",
        "plastered": "plaster", "motoring": "motor", "sing": "sing",
        "happy": "happi", "relational": "relat", "rationalize": "ration",
        "hopeful": "hope", "goodness": "good", "revival": "reviv",
        "adoption": "adopt", "effective": "effect", "roll": "roll",
    }
    for word, expected in cases.items():
        assert stem(word) == expected, f"{word} -> {stem(word)} != {expected}"


def test_normalise_folds_accents():
    assert normalise("Amélie  Poulain—Café") == "amelie poulain cafe"


def test_tokenize_removes_stopwords():
    tokens = tokenize("A thief who steals the secrets from a movie")
    assert "the" not in tokens and "who" not in tokens
    assert "movie" not in tokens  # domain stopword
    assert "thief" in tokens


# --------------------------------------------------------- data loading
def test_all_four_files_are_loaded():
    df = build_clean_frame()
    sources = {s for v in df["source_file"] for s in str(v).split(" + ")}
    assert len(sources) == 4, f"expected 4 source files, got {sources}"


def test_cross_file_duplicates_are_merged():
    """3 Idiots, Dangal, Gladiator etc. appear in two files each."""
    df = build_clean_frame()
    merged = df[df["merged_from"] > 1]
    assert len(merged) >= 4
    assert "3 Idiots" in set(merged["title"])


def test_distinct_films_sharing_a_title_are_kept_apart():
    """Two different films are both called Drishyam (2013 and 2015)."""
    df = build_clean_frame()
    drishyam = df[df["title"] == "Drishyam"]
    assert len(drishyam) == 2, "the two Drishyam films must not be merged"
    assert set(drishyam["year"]) == {2013, 2015}


def test_no_missing_critical_fields():
    df = build_clean_frame()
    for col in ["title", "description", "language", "genres"]:
        assert df[col].map(lambda v: len(v) > 0).all(), f"{col} has empties"


def test_ids_are_unique():
    df = build_clean_frame()
    assert df["movie_id"].is_unique


def test_mood_blockers_apply():
    """A crime film must never be tagged feel-good."""
    df = build_clean_frame()
    godfather = df[df["title"] == "The Godfather"].iloc[0]
    assert "feel-good" not in godfather["moods"]


# ------------------------------------------------------- recommendations
def test_recommendations_share_a_genre():
    """At least 4 of 5 recommendations should share a genre with the seed."""
    rec = model()
    for title in ["Inception", "The Godfather", "Toy Story", "3 Idiots"]:
        idx = rec.index_of(title)
        seed_genres = {g.lower() for g in rec.df.loc[idx, "genres"]}
        recs = rec.similar_to(title, top_n=5)
        shared = sum(
            1 for r in recs
            if seed_genres & {
                g.lower() for g in rec.df.loc[rec.index_of(r.movie_id), "genres"]
            }
        )
        assert shared >= 4, f"{title}: only {shared}/5 shared a genre"


def test_seed_is_never_recommended_back():
    rec = model()
    recs = rec.similar_to("Inception", top_n=10)
    assert all(r.title != "Inception" for r in recs)


def test_sequel_is_top_match():
    """The Godfather Part II is the obvious nearest neighbour."""
    rec = model()
    recs = rec.similar_to("The Godfather", top_n=3, mmr_lambda=1.0)
    assert recs[0].title == "The Godfather Part II"


def test_recommendations_carry_explanations():
    rec = model()
    recs = rec.similar_to("Inception", top_n=5)
    assert all(len(r.reasons) > 0 for r in recs)


def test_mmr_increases_variety():
    """Lower lambda should reduce how many recs share the seed's subgenre."""
    rec = model()
    tight = rec.similar_to("The Godfather", top_n=8, mmr_lambda=1.0)
    loose = rec.similar_to("The Godfather", top_n=8, mmr_lambda=0.4)
    sub = lambda rs: len({
        rec.df.loc[rec.index_of(r.movie_id), "sub_genre"] for r in rs
    })
    assert sub(loose) >= sub(tight)


def test_filters_are_respected():
    rec = model()
    recs = rec.similar_to(
        "Inception", top_n=10,
        filters={"languages": ["Hindi"], "min_rating": 7.5},
    )
    for r in recs:
        row = rec.df.loc[rec.index_of(r.movie_id)]
        assert row["language"] == "Hindi"
        assert row["rating"] >= 7.5


# -------------------------------------------------------------- search
def test_search_finds_exact_title():
    rec = model()
    assert rec.search("The Godfather", top_n=3)[0].title == "The Godfather"


def test_search_understands_themes_not_just_words():
    """The regression this was written for: 'mind bending space movies'
    must not return Toy Story because its synopsis says 'space ranger'."""
    rec = model()
    titles = [r.title for r in rec.search("mind bending space movies", top_n=5)]
    assert "Toy Story" not in titles
    assert any(t in titles for t in ["Inception", "Interstellar", "Tenet"])


def test_search_by_director_returns_their_films():
    rec = model()
    titles = [r.title for r in rec.search("christopher nolan", top_n=5)]
    assert len(titles) >= 3
    rec_dirs = [
        rec.df.loc[rec.index_of(r.movie_id), "director"]
        for r in rec.search("christopher nolan", top_n=5)
    ]
    assert sum("Nolan" in d for d in rec_dirs) >= 4


def test_search_synonyms_work():
    rec = model()
    titles = [r.title for r in rec.search("scary survival", top_n=5)]
    assert any(t in titles for t in ["A Quiet Place", "Alien", "Train to Busan"])


def test_search_nonsense_returns_nothing():
    """Regression: the popularity prior used to clear the relevance floor on
    its own, so 'xyz' returned 60 high-rated films."""
    rec = model()
    assert rec.search("xyz qwerty", top_n=50) == []


def test_search_understands_genre_words():
    """'drama' and 'romantic' were dropped from the TF-IDF vocabulary (too
    common) and matched nothing; they must resolve to the genre."""
    rec = model()
    for query, genre in [("drama", "Drama"), ("romantic", "Romance"),
                         ("animated", "Animation"), ("scary", "Horror")]:
        hits = rec.search(query, top_n=200)
        assert hits, query
        genres = [rec.df.loc[rec.index_of(r.movie_id), "genres"] for r in hits]
        assert all(genre in g for g in genres), query


def test_search_genre_word_is_not_a_title_substring():
    """'war' means the genre - it must not put Star Wars / Infinity War first."""
    rec = model()
    top = rec.search("war", top_n=5)
    genres = [rec.df.loc[rec.index_of(r.movie_id), "genres"] for r in top]
    assert all("War" in g for g in genres)


def test_search_language_and_decade_words():
    rec = model()
    hindi = rec.search("bollywood", top_n=10)
    assert hindi and all(
        rec.df.loc[rec.index_of(r.movie_id), "language"] == "Hindi" for r in hindi
    )
    nineties = rec.search("90s", top_n=30)
    assert nineties and all(
        1990 <= rec.df.loc[rec.index_of(r.movie_id), "year"] < 2000 for r in nineties
    )


# ------------------------------------------------------ catalogue growth
def _fake_tmdb(**over):
    d = {
        "id": 501, "title": "Sample Film", "release_date": "2019-05-03", "adult": False,
        "overview": "A long enough synopsis for the importer to keep this film.",
        "vote_average": 7.44, "vote_count": 900, "runtime": 118,
        "genres": [{"name": "Science Fiction"}, {"name": "TV Movie"}, {"name": "Drama"}],
        "original_language": "hi", "poster_path": "/abc.jpg",
        "production_countries": [{"name": "India"}, {"name": "United States of America"}],
        "credits": {"cast": [{"name": "B", "order": 1}, {"name": "A", "order": 0}],
                    "crew": [{"job": "Director", "name": "Dee"}, {"job": "Editor", "name": "X"}]},
        "keywords": {"keywords": [{"name": "independent film"}, {"name": "family"}]},
        "release_dates": {"results": [
            {"iso_3166_1": "US", "release_dates": [{"certification": "PG-13"}]},
            {"iso_3166_1": "IN", "release_dates": [{"certification": "UA"}]}]},
    }
    d.update(over)
    return d


def _importer():
    import importlib.util
    spec = importlib.util.spec_from_file_location("import_tmdb", ROOT / "tools" / "import_tmdb.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_importer_maps_tmdb_onto_workbook_schema():
    imp = _importer()
    row = imp.to_row(_fake_tmdb(), tags={"independent"})
    assert row["Genres"] == "Sci-Fi|Drama"          # renamed; TV Movie dropped
    assert row["Language"] == "Hindi" and row["Country"] == "India"
    assert row["Age Rating"] == "U/A"                # India's board for Indian films
    assert row["Cast"] == "A|B" and row["Director"] == "Dee"
    assert row["Rating"] == 7.4 and row["TMDB ID"] == 501
    assert row["Poster Path"] == "/abc.jpg" and row["Collections"] == "independent"
    assert imp.to_row(_fake_tmdb(production_countries=[{"name": "United States of America"}],
                                 original_language="en"))["Age Rating"] == "PG-13"


def test_importer_skips_unusable_records():
    imp = _importer()
    assert imp.to_row(_fake_tmdb(adult=True)) is None
    assert imp.to_row(_fake_tmdb(overview="short")) is None
    assert imp.to_row(_fake_tmdb(runtime=20)) is None            # a short film
    assert imp.to_row(_fake_tmdb(vote_average=0)) is None
    assert imp.to_row(_fake_tmdb(release_date="")) is None
    assert imp.to_row(_fake_tmdb(runtime=20, genres=[{"name": "Documentary"}])) is not None


def test_collections_are_derived_from_language_country_and_genre():
    from src.data_loader import derive_collections
    base = {"language": "", "country": "", "genres": [], "keyword_list": []}
    assert derive_collections({**base, "language": "Tamil", "country": "India"}) == [
        "Indian cinema", "Regional-language"]
    assert derive_collections({**base, "language": "Hindi", "country": "India"}) == ["Indian cinema"]
    assert derive_collections({**base, "language": "Korean", "country": "South Korea"}) == ["Korean cinema"]
    assert derive_collections({**base, "language": "French", "country": "France"}) == ["European cinema"]
    assert derive_collections({**base, "language": "Japanese", "genres": ["Animation"]}) == ["Anime"]
    assert derive_collections({**base, "genres": ["Documentary"]}) == ["Documentaries"]
    assert derive_collections(base, tags=["Independent"]) == ["Independent"]


def test_saved_ids_survive_importing_more_films():
    """Likes and watchlists are keyed by movie_id: adding films must not renumber."""
    import shutil, tempfile
    import pandas as pd
    from src import config
    from src.data_loader import build_clean_frame
    baseline = build_clean_frame()
    with tempfile.TemporaryDirectory() as tmp:
        for f in config.RAW_DIR.glob("*.xlsx"):
            shutil.copy(f, tmp)
        imp = _importer()
        rows = [imp.to_row(_fake_tmdb(id=900 + i, title=f"Zed Film {i}"), set()) for i in range(3)]
        # one imported row duplicates a hand-built film and must merge into it
        first = baseline.iloc[0]
        rows.append(imp.to_row(_fake_tmdb(id=999, title=first["title"],
                                          release_date=f"{int(first['year'])}-01-01"), set()))
        pd.DataFrame(rows).to_excel(Path(tmp) / "tmdb_import.xlsx", index=False)
        grown = build_clean_frame(Path(tmp))
    assert len(grown) == len(baseline) + 3
    old = dict(zip(zip(baseline["title"], baseline["year"]), baseline["movie_id"]))
    new = dict(zip(zip(grown["title"], grown["year"]), grown["movie_id"]))
    assert all(new[k] == v for k, v in old.items())
    assert sorted(grown[grown["movie_id"].str.startswith("t")]["movie_id"]) == ["t900", "t901", "t902"]
    merged = grown[grown["title"] == first["title"]].iloc[0]
    assert merged["tmdb_id"] == 999                      # picked up the TMDB link


# ------------------------------------------------------- taste profiles
def test_taste_profile_prefers_similar_films():
    rec = model()
    recs = rec.for_profile(["3 Idiots", "Dangal"], top_n=6)
    langs = [
        rec.df.loc[rec.index_of(r.movie_id), "language"] for r in recs
    ]
    assert langs.count("Hindi") >= 4


def test_rated_films_excluded_from_profile_results():
    rec = model()
    recs = rec.for_profile(["3 Idiots"], disliked=["Dangal"], top_n=10)
    titles = {r.title for r in recs}
    assert "3 Idiots" not in titles and "Dangal" not in titles


def test_negative_feedback_changes_results():
    rec = model()
    base = [r.title for r in rec.for_profile(["Inception"], top_n=8)]
    with_neg = [
        r.title for r in rec.for_profile(["Inception"], disliked=["Tenet"], top_n=8)
    ]
    assert base != with_neg


# --------------------------------------------------------------- runner
if __name__ == "__main__":
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    sys.exit(1 if failed else 0)
