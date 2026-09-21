"""
Central configuration for the movie recommender.

Everything tunable lives here so experiments don't require touching logic.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
WEB_DIR = ROOT / "web"
WEB_DATA_JS = WEB_DIR / "js" / "data.js"
REPORTS_DIR = ROOT / "reports"

CLEAN_CSV = PROCESSED_DIR / "movies_clean.csv"
CLEAN_JSON = PROCESSED_DIR / "movies_clean.json"

# ------------------------------------------------- feature soup weights
# How many times each field is repeated in the "soup" before vectorising.
# Repetition is the simplest way to weight fields inside a single TF-IDF
# space: a term appearing in a x3 field gets three times the raw count.
FIELD_WEIGHTS = {
    "genre": 3,
    "sub_genre": 3,
    "keywords": 3,
    "director": 2,
    "cast": 2,
    "description": 1,
    "language": 1,
    "country": 1,
    "era": 1,
    "awards": 1,
    "mood": 2,
}

# ------------------------------------------------------ vectoriser setup
TFIDF = {
    "min_df": 1,
    "max_df": 0.55,        # drop terms in >55% of films (they carry no signal)
    "ngram_range": (1, 2),
    "sublinear_tf": True,  # 1 + log(tf); stops x3 repetition dominating
    "norm": "l2",
}

# ------------------------------------------------------- hybrid scoring
# Final similarity = weighted blend of four independent signals.
SCORE_WEIGHTS = {
    "text": 0.58,      # TF-IDF cosine over the feature soup
    "categorical": 0.22,  # Jaccard over genre / subgenre / language sets
    "people": 0.12,    # shared director & cast
    "numeric": 0.08,   # rating + era + runtime proximity
}

# Small bonus so that, among equally similar films, better-rated ones win.
POPULARITY_PRIOR = 0.03

# Maximal Marginal Relevance: 1.0 = pure relevance, 0.0 = pure diversity.
DEFAULT_MMR_LAMBDA = 0.75

# Neighbours precomputed per film for the static web export.
TOP_N_NEIGHBOURS = 24

# ----------------------------------------------------------- mood lexicon
# Hand-built mood definitions. `cues` are matched as whole words against a
# film's keywords / subgenre / genre / description. `blocks` lists genres
# that disqualify the mood outright — without them, ambiguous cues like
# "family" drag *The Godfather* into "feel-good".
MOOD_LEXICON = {
    "feel-good": {
        "cues": [
            "friendship", "heartwarming", "comedy", "joy", "uplifting",
            "coming of age", "musical", "holiday", "charming", "romance",
            "hope", "reunion", "celebration", "underdog",
        ],
        "blocks": ["Horror", "War", "Crime"],
    },
    "mind-bending": {
        "cues": [
            "mind bending", "dreams", "dream", "time travel", "time inversion",
            "reality", "memory", "identity", "twist", "puzzle", "subconscious",
            "paradox", "surreal", "psychological", "illusion", "simulation",
        ],
        "blocks": [],
    },
    "edge-of-seat": {
        "cues": [
            "thriller", "heist", "survival", "chase", "hostage", "suspense",
            "manhunt", "conspiracy", "assassin", "escape", "countdown",
            "cat and mouse", "investigation",
        ],
        "blocks": [],
    },
    "tearjerker": {
        "cues": [
            "loss", "grief", "illness", "sacrifice", "farewell", "tragedy",
            "separation", "memory loss", "terminal", "orphan", "widow",
            "father son", "father daughter", "mother son",
        ],
        "blocks": [],
    },
    "epic-scale": {
        "cues": [
            "war", "empire", "kingdom", "revolution", "battle", "epic",
            "space", "period", "historical", "colonial", "saga", "dynasty",
            "voyage", "expedition",
        ],
        "blocks": [],
    },
    "dark-and-gritty": {
        "cues": [
            "crime", "mafia", "gangster", "revenge", "corruption", "violence",
            "noir", "drugs", "prison", "murder", "underworld",
            "serial killer", "betrayal",
        ],
        "blocks": ["Animation", "Family"],
    },
    "social-lens": {
        "cues": [
            "caste", "class", "poverty", "inequality", "protest", "justice",
            "discrimination", "society", "politics", "labour", "activism",
            "women empowerment", "education", "racism", "slavery", "migrant",
        ],
        "blocks": [],
    },
    "wonder": {
        "cues": [
            "animation", "fantasy", "magic", "adventure", "spirits",
            "creature", "childhood", "imagination", "fairy", "myth",
            "discovery", "enchanted",
        ],
        "blocks": ["Crime"],
    },
}

# Era buckets used as a categorical feature and as a browse facet.
ERA_BUCKETS = [
    (0, 1969, "Classic"),
    (1970, 1989, "New Wave"),
    (1990, 1999, "The Nineties"),
    (2000, 2009, "2000s"),
    (2010, 2019, "2010s"),
    (2020, 9999, "Contemporary"),
]

RUNTIME_BUCKETS = [
    (0, 100, "Short"),
    (101, 135, "Standard"),
    (136, 165, "Long"),
    (166, 9999, "Epic"),
]
