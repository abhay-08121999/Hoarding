"""
Self-contained NLP toolkit.

Deliberately has no nltk/spacy dependency: everything the recommender needs
(normalising, tokenising, stopword removal, stemming, phrase handling) is
implemented here in plain Python. That keeps `pip install -r requirements`
small and makes it possible to port the exact same rules to JavaScript for
the browser build, so client-side search and server-side search agree.
"""
from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------- stopwords
# General English stopwords plus a film-domain list. Words like "film",
# "movie", "story" appear in almost every synopsis, so they add noise
# rather than signal and are stripped before vectorising.
BASE_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "cannot", "could", "did", "do", "does", "doing", "don", "down",
    "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "let", "me", "more", "most", "must", "my", "myself",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your",
    "yours", "yourself", "yourselves", "s", "t", "don", "shall", "may",
    "might", "upon", "onto", "via", "whose", "also", "however",
}

DOMAIN_STOPWORDS = {
    "film", "films", "movie", "movies", "story", "stories", "tale", "tales",
    "cinema", "picture", "feature", "min", "mins", "minute", "minutes",
    "one", "two", "three", "man", "woman", "young", "old", "new", "must",
    "becomes", "become", "find", "finds", "take", "takes", "make", "makes",
    "get", "gets", "goes", "go", "set", "sets",
}

STOPWORDS = BASE_STOPWORDS | DOMAIN_STOPWORDS

# Tokens shorter than this are dropped (after stemming).
MIN_TOKEN_LEN = 2

_WORD_RE = re.compile(r"[a-z0-9]+")
_VOWELS = set("aeiou")


# ------------------------------------------------------------ normalising
def normalise(text) -> str:
    """
    Lowercase, strip accents, and reduce anything non-alphanumeric to space.

    Accent folding matters here: the catalogue mixes `Amelie`/`Amélie` and
    `Ines`/`Inés` style spellings, and we want those to collide.
    """
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------- Porter stemmer
class PorterStemmer:
    """
    Compact implementation of the Porter (1980) stemming algorithm.

    Porter is a good fit here: it is purely rule-based, so it needs no
    downloaded corpora and it is short enough to port verbatim to JS,
    guaranteeing the browser and the Python build produce identical stems.
    """

    def _is_consonant(self, word: str, i: int) -> bool:
        ch = word[i]
        if ch in _VOWELS:
            return False
        if ch == "y":
            return i == 0 or not self._is_consonant(word, i - 1)
        return True

    def _measure(self, stem: str) -> int:
        """Count vowel-consonant sequences (the 'm' of the paper)."""
        n = 0
        i = 0
        length = len(stem)
        # skip leading consonants
        while i < length and self._is_consonant(stem, i):
            i += 1
        while i < length:
            while i < length and not self._is_consonant(stem, i):
                i += 1
            if i >= length:
                break
            n += 1
            while i < length and self._is_consonant(stem, i):
                i += 1
        return n

    def _has_vowel(self, stem: str) -> bool:
        return any(not self._is_consonant(stem, i) for i in range(len(stem)))

    def _double_consonant_suffix(self, word: str) -> bool:
        if len(word) < 2:
            return False
        if word[-1] != word[-2]:
            return False
        return self._is_consonant(word, len(word) - 1)

    def _cvc(self, word: str) -> bool:
        """consonant-vowel-consonant where the final isn't w, x or y."""
        if len(word) < 3:
            return False
        i = len(word) - 1
        if (
            not self._is_consonant(word, i)
            or self._is_consonant(word, i - 1)
            or not self._is_consonant(word, i - 2)
        ):
            return False
        return word[i] not in "wxy"

    def _step1a(self, word: str) -> str:
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word

    def _step1b(self, word: str) -> str:
        if word.endswith("eed"):
            if self._measure(word[:-3]) > 0:
                return word[:-1]
            return word
        flag = False
        if word.endswith("ed") and self._has_vowel(word[:-2]):
            word, flag = word[:-2], True
        elif word.endswith("ing") and self._has_vowel(word[:-3]):
            word, flag = word[:-3], True
        if flag:
            if word.endswith(("at", "bl", "iz")):
                return word + "e"
            if self._double_consonant_suffix(word) and word[-1] not in "lsz":
                return word[:-1]
            if self._measure(word) == 1 and self._cvc(word):
                return word + "e"
        return word

    def _step1c(self, word: str) -> str:
        if word.endswith("y") and self._has_vowel(word[:-1]):
            return word[:-1] + "i"
        return word

    _STEP2 = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
        ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
        ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
        ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
        ("biliti", "ble"),
    ]

    _STEP3 = [
        ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
        ("ical", "ic"), ("ful", ""), ("ness", ""),
    ]

    _STEP4 = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
        "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    ]

    def _step2(self, word: str) -> str:
        for suffix, repl in self._STEP2:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                return stem + repl if self._measure(stem) > 0 else word
        return word

    def _step3(self, word: str) -> str:
        for suffix, repl in self._STEP3:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                return stem + repl if self._measure(stem) > 0 else word
        return word

    def _step4(self, word: str) -> str:
        for suffix in self._STEP4:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 1:
                    return stem
                return word
        if word.endswith("ion"):
            stem = word[:-3]
            if self._measure(stem) > 1 and stem and stem[-1] in "st":
                return stem
        return word

    def _step5(self, word: str) -> str:
        if word.endswith("e"):
            stem = word[:-1]
            m = self._measure(stem)
            if m > 1 or (m == 1 and not self._cvc(stem)):
                word = stem
        if (
            word.endswith("ll")
            and self._measure(word) > 1
        ):
            word = word[:-1]
        return word

    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word
        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5(word)
        return word


_STEMMER = PorterStemmer()
_STEM_CACHE: dict[str, str] = {}


def stem(word: str) -> str:
    """Memoised stemming — the same few thousand tokens recur constantly."""
    cached = _STEM_CACHE.get(word)
    if cached is None:
        cached = _STEMMER.stem(word)
        _STEM_CACHE[word] = cached
    return cached


def stem_cache() -> dict[str, str]:
    """Expose the raw-token -> stem map so the web build can ship it."""
    return dict(_STEM_CACHE)


# ------------------------------------------------------------ tokenising
def tokenize(text, keep_stopwords: bool = False) -> list[str]:
    """Normalise -> split -> drop stopwords -> stem."""
    words = _WORD_RE.findall(normalise(text))
    out = []
    for w in words:
        if not keep_stopwords and w in STOPWORDS:
            continue
        s = stem(w)
        if len(s) < MIN_TOKEN_LEN:
            continue
        out.append(s)
    return out


def entity_token(name: str, prefix: str) -> str:
    """
    Collapse a person/place name into a single atomic token.

    `Christopher Nolan` -> `person_christopher_nolan`. Without this, the
    vectoriser would match any two films sharing the common first name
    "christopher", which is noise, not signal.
    """
    slug = normalise(name).replace(" ", "_")
    return f"{prefix}_{slug}" if slug else ""


def phrase_token(phrase: str) -> str:
    """
    Keep a multi-word keyword together as one token.

    `father daughter` -> `kw_father_daughter`, so the pairing is matched
    as a concept rather than as two unrelated words.
    """
    slug = normalise(phrase).replace(" ", "_")
    return f"kw_{slug}" if slug else ""


def analyzer(text: str) -> list[str]:
    """
    Analyzer handed to scikit-learn's TfidfVectorizer.

    The soup arrives pre-tokenised with atomic tokens already marked by a
    `person_` / `kw_` / `tag_` prefix, so those pass through untouched
    while ordinary prose words get stopword-filtered and stemmed.
    """
    out: list[str] = []
    for chunk in str(text).split():
        if "_" in chunk and chunk.split("_", 1)[0] in {
            "person", "kw", "tag", "genre", "sub", "lang", "country",
            "era", "mood", "cert", "award", "runtime",
        }:
            out.append(chunk)
        else:
            out.extend(tokenize(chunk))
    return out
