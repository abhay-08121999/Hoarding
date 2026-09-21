"""
Natural-language query understanding.

The catalogue is indexed with *atomic* tokens — `kw_mind_bending`,
`genre_sci_fi`, `person_christopher_nolan` — because that is what makes
film-to-film similarity precise. But it breaks naive search: a person
typing "mind bending space movies" produces the loose words
`mind`, `bend`, `space`, none of which match `kw_mind_bending`. The query
then only matches the literal word "space" wherever it appears in prose,
which is how *Toy Story* ends up as the top hit for a Nolan-shaped query.

`QueryExpander` closes that gap. It scans the query for the longest
phrases that correspond to real atomic tokens in the vocabulary and emits
those tokens alongside the ordinary stemmed words.
"""
from __future__ import annotations

import re

import numpy as np

from .text_nlp import normalise, tokenize

# Atomic token prefixes that correspond to a human-typable phrase.
PHRASE_PREFIXES = (
    "kw_", "genre_", "sub_", "person_", "mood_", "lang_", "country_", "era_",
)

MAX_PHRASE_WORDS = 4

# Everyday phrasings that should map onto catalogue vocabulary.
SYNONYMS = {
    "scifi": "sci fi", "science fiction": "sci fi", "sci-fi": "sci fi",
    "rom com": "romance comedy", "romcom": "romance comedy",
    "funny": "comedy", "hilarious": "comedy", "laugh": "comedy",
    "scary": "horror", "creepy": "horror", "frightening": "horror",
    "sad": "tearjerker", "emotional": "tearjerker", "cry": "tearjerker",
    "twisty": "mind bending", "trippy": "mind bending",
    "confusing": "mind bending", "cerebral": "mind bending",
    "tense": "edge of seat", "gripping": "edge of seat",
    "nail biting": "edge of seat", "suspenseful": "edge of seat",
    "uplifting": "feel good", "wholesome": "feel good",
    "happy": "feel good", "light": "feel good",
    "cartoon": "animation", "anime": "animation",
    "bollywood": "hindi", "tollywood": "telugu", "kollywood": "tamil",
    "desi": "india", "indian": "india",
    "k drama": "korean", "kdrama": "korean",
    "gangster": "mafia", "mob": "mafia",
    "outer space": "space", "spaceship": "space travel",
    "superhero": "superhero", "comic book": "superhero",
    "based on a true story": "biography", "true story": "biography",
    "old": "classic", "vintage": "classic", "retro": "classic",
}


class QueryExpander:
    """Maps free text onto the atomic tokens actually present in the index."""

    def __init__(self, vocabulary):
        self.vocabulary = list(vocabulary)
        self.phrase_map: dict[str, list[str]] = {}

        for token in self.vocabulary:
            for prefix in PHRASE_PREFIXES:
                if token.startswith(prefix):
                    phrase = token[len(prefix):].replace("_", " ").strip()
                    if not phrase:
                        continue
                    self.phrase_map.setdefault(phrase, [])
                    if token not in self.phrase_map[phrase]:
                        self.phrase_map[phrase].append(token)
                    break

        self._max_words = min(
            MAX_PHRASE_WORDS,
            max((len(p.split()) for p in self.phrase_map), default=1),
        )

    # ------------------------------------------------------------ helpers
    def _apply_synonyms(self, text: str) -> str:
        out = text
        # Longest synonyms first so "science fiction" wins over "fiction".
        for term in sorted(SYNONYMS, key=len, reverse=True):
            out = re.sub(rf"\b{re.escape(term)}\b", SYNONYMS[term], out)
        return out

    def expand(self, query: str) -> tuple:
        """
        Return (expanded_token_string, matched_phrases).

        Greedy longest-match: scan left to right, and at each position try
        the longest phrase that resolves to a real atomic token before
        falling back to single stemmed words.
        """
        text = self._apply_synonyms(normalise(query))
        words = text.split()
        tokens: list[str] = []
        matched: list[str] = []

        i = 0
        while i < len(words):
            hit = False
            upper = min(self._max_words, len(words) - i)
            for span in range(upper, 0, -1):
                phrase = " ".join(words[i:i + span])
                if phrase in self.phrase_map:
                    atomic = self.phrase_map[phrase]
                    # Repeat matched concepts so they outweigh loose prose.
                    tokens.extend(atomic * 3)
                    matched.append(phrase)
                    i += span
                    hit = True
                    break
            if not hit:
                tokens.extend(tokenize(words[i]))
                i += 1

        return " ".join(tokens), matched


SEARCH_FLOOR = 0.10  # relevance a film needs from a *real* signal (not the prior)

_STOP = set(
    "a an and the of in on at for to by from with about like movie movies "
    "film films some any me my show find want watch something".split()
)
_GENRE_ALIASES = {
    "scifi": "Sci-Fi", "science fiction": "Sci-Fi", "romantic": "Romance",
    "animated": "Animation", "cartoon": "Animation", "cartoons": "Animation",
    "anime": "Animation", "scary": "Horror", "spooky": "Horror",
    "funny": "Comedy", "hilarious": "Comedy", "comedies": "Comedy",
    "biopic": "Biography", "historical": "History", "sports": "Sport",
    "documentaries": "Documentary", "musicals": "Musical", "gangster": "Crime",
    "detective": "Mystery",
}
_LANGUAGE_ALIASES = {
    "bollywood": "Hindi", "kollywood": "Tamil", "tollywood": "Telugu",
    "mollywood": "Malayalam", "sandalwood": "Kannada", "chinese": "Mandarin",
}
_COUNTRY_ALIASES = {
    "american": "USA", "america": "USA", "usa": "USA", "hollywood": "USA",
    "british": "UK", "england": "UK", "indian": "India", "korean": "South Korea",
    "korea": "South Korea", "japanese": "Japan", "french": "France",
    "italian": "Italy", "german": "Germany",
}
_ENTITY_WEIGHT = {"genre": 0.30, "language": 0.32, "country": 0.22, "year": 0.32, "decade": 0.28}


def _entity_index(df) -> dict:
    """word/phrase -> [(kind, value)], built from the catalogue itself."""
    index: dict = {}

    def add(word, kind, value):
        key = normalise(word)
        if key and value:
            index.setdefault(key, [])
            if (kind, value) not in index[key]:
                index[key].append((kind, value))

    genres = {g for gs in df["genres"] for g in gs}
    for g in genres:
        add(g, "genre", g)
        add(g + "s", "genre", g)
    for word, g in _GENRE_ALIASES.items():
        if g in genres:
            add(word, "genre", g)
    languages = set(df["language"].dropna())
    for lang in languages:
        add(lang, "language", lang)
    for word, lang in _LANGUAGE_ALIASES.items():
        if lang in languages:
            add(word, "language", lang)
    countries = set(df["country"].dropna())
    for c in countries:
        add(c, "country", c)
    for word, c in _COUNTRY_ALIASES.items():
        if c in countries:
            add(word, "country", c)
    return index


def _entity_hit(kind: str, value, row) -> bool:
    if kind == "genre":
        return value in row["genres"]
    if kind == "language":
        return row["language"] == value
    if kind == "country":
        return row["country"] == value
    year = row["year"]
    if year != year:
        return False
    if kind == "year":
        return int(year) == value
    return value <= int(year) < value + 10  # decade


def _parse_entities(words: list, index: dict) -> tuple:
    """Split query words into (entities, leftover words). Two-word phrases first."""
    entities, leftover, i = [], [], 0
    while i < len(words):
        two = " ".join(words[i:i + 2])
        if i + 1 < len(words) and two in index:
            entities += index[two]
            i += 2
            continue
        w = words[i]
        if w in index:
            entities += index[w]
        elif re.fullmatch(r"(19|20)\d\d", w):
            entities.append(("year", int(w)))
        elif re.fullmatch(r"(?:19|20)?\d0s", w):
            digits = w[:-1]
            start = int(digits) if len(digits) == 4 else (
                1900 + int(digits) if int(digits) >= 30 else 2000 + int(digits))
            entities.append(("decade", start))
        else:
            leftover.append(w)
        i += 1
    return entities, leftover


def search_scores(recommender, query: str) -> tuple:
    """
    Score every film against a free-text query.

    Four channels, each only counting when it actually matches: expanded-token
    cosine similarity (meaning), whole-word title matches, director/cast
    names, and genre / language / country / year words ("drama", "bollywood",
    "90s"). Returns (relevance, matched_labels). The popularity prior is *not*
    included here - the caller adds it after applying SEARCH_FLOOR, otherwise
    every well-rated film clears the bar and nonsense queries return results.
    """
    expander = recommender.query_expander
    expanded, matched = expander.expand(query)

    from sklearn.metrics.pairwise import cosine_similarity

    df = recommender.df
    if expanded.strip():
        q_vec = recommender.space.vectorizer.transform([expanded])
        scores = cosine_similarity(q_vec, recommender.space.tfidf).ravel()
    else:
        scores = np.zeros(len(df))

    q_norm = normalise(query)
    raw = q_norm.split()
    words = [w for w in raw if w not in _STOP] or raw
    index = getattr(recommender, "_entity_index", None)
    if index is None:
        index = recommender._entity_index = _entity_index(df)
    entities, cover = _parse_entities(words, index)
    padded = f" {q_norm} "

    for i, row in df.reset_index(drop=True).iterrows():
        t = normalise(row["title"])
        if q_norm and t == q_norm:
            scores[i] += 1.0
        elif cover and (f" {t} " in padded or padded in f" {t} "):
            scores[i] += 0.6  # whole-word only: "war" no longer matches "Star Wars"
        elif cover:
            t_tok = set(t.split())
            hit = [w for w in cover if w in t_tok]
            if len(hit) / len(cover) >= 0.6:
                scores[i] += 0.55 * len(hit) / len(cover)

        if cover:
            names = [normalise(p) for p in list(row["directors"]) + list(row["cast_list"])]
            if any(n and f" {n} " in padded for n in names):
                scores[i] += 0.6
            else:
                p_tok = set(" ".join(names).split())
                hit = [w for w in cover if w in p_tok]
                if len(hit) / len(cover) >= 0.6:
                    scores[i] += 0.5 * len(hit) / len(cover)

        for kind, value in entities:
            if _entity_hit(kind, value, row):
                scores[i] += _ENTITY_WEIGHT[kind]

    labels = list(matched)
    for kind, value in entities:
        label = f"{value}s" if kind == "decade" else str(value)
        if label.lower() not in (m.lower() for m in labels):
            labels.append(label)
    return scores, labels