"""TuneScout RAG: retrieval-augmented music recommendations.

Pipeline for one request (see diagrams/architecture.mmd):

    request
      -> input guardrail          (empty / too long / no letters)
      -> retrieve guide notes     (mood & genre guides, TF-IDF)
      -> retrieve songs           (request + matched guide notes, TF-IDF)
      -> relevance guardrail      (best song score below threshold -> "no match")
      -> prompt Gemini with ONLY the retrieved songs + notes
      -> output guardrail         (drop any song id that wasn't retrieved)
      -> answer with cited song ids
    If Gemini is unavailable or returns nothing usable, fall back to the
    Module 3 rule-based scorer (src/recommender.py).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .llm import Generator, LLMUnavailable
from .recommender import load_songs, recommend_songs

logger = logging.getLogger(__name__)

# ---- Tunable settings -------------------------------------------------------
MAX_REQUEST_CHARS = 300      # input guardrail
TOP_K_SONGS = 8              # songs passed to the LLM as context
TOP_K_GUIDES = 3             # guide chunks passed to the LLM as context
MIN_GUIDE_SCORE = 0.10       # guide chunks below this are ignored
MIN_SONG_SCORE = 0.08        # relevance guardrail: best song must reach this
DEFAULT_RESULTS = 3          # how many songs to recommend

# Words that appear in almost every music request and carry no meaning for
# matching (e.g. "songs" was matching the pop guide's "sing-along songs").
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | {
    "song", "songs", "music", "track", "tracks", "tune", "tunes", "playlist",
    "listen", "listening", "play", "want", "like", "kind", "stuff", "vibe", "vibes",
})


# ---- Specialization: few-shot "house style" for reasons -------------------
# The examples use requests and songs that are NOT in the evaluation set,
# so the measured improvement isn't just the model copying an answer.
STYLES = ("specialized", "baseline")
HYPE_WORDS = ("perfect", "perfectly", "amazing", "ideal", "ultimate", "awesome")
STYLE_RULES = """
House style for every reason (follow exactly):
- At most 20 words.
- Quote at least one exact number from the song: energy (e.g. 0.35) or tempo (e.g. 72 BPM).
- Echo a specific word or phrase from the listener's request.
- No hype words (perfect, perfectly, amazing, ideal, ultimate, awesome).

Examples of the house style:
Request: "upbeat songs for cleaning the house"
  {"id": 1, "reason": "Happy pop at 0.82 energy and 118 BPM keeps the cleaning momentum going."}
Request: "something dark for a late night walk"
  {"id": 20, "reason": "Dark trip-hop, slow 85 BPM and 0.30 energy, suits a quiet late night walk."}
"""


# ---- Data types -------------------------------------------------------------
@dataclass
class GuideChunk:
    kind: str     # "mood" or "genre"
    label: str    # exact label used in songs.csv, e.g. "chill"
    text: str


@dataclass
class Pick:
    song: Dict
    reason: str


@dataclass
class RecommendationResult:
    status: str                      # "ok" | "fallback" | "no_match" | "invalid_input"
    request: str
    picks: List[Pick] = field(default_factory=list)
    message: str = ""
    confidence: Optional[float] = None   # LLM's self-rated confidence (0-1)
    retrieval_score: float = 0.0         # best song similarity (0-1)
    retrieved_ids: List[int] = field(default_factory=list)
    guide_labels: List[str] = field(default_factory=list)
    dropped_ids: List[int] = field(default_factory=list)


# ---- Document building ------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase and fix tokens the default tokenizer would lose (e.g. 'r&b')."""
    text = text.lower()
    text = re.sub(r"\br\s*&\s*b\b", "rnb", text)
    text = re.sub(r"\blo-fi\b", "lofi", text)
    return text


def _energy_words(energy: float) -> str:
    if energy < 0.45:
        return "low energy, calm, soft, quiet"
    if energy < 0.7:
        return "medium energy, moderate"
    return "high energy, loud, powerful, upbeat"


def _tempo_words(bpm: float) -> str:
    if bpm < 85:
        return "slow tempo"
    if bpm <= 115:
        return "mid-tempo"
    return "fast tempo"


def song_to_document(song: Dict) -> str:
    """Turn one catalog row into a searchable text description."""
    parts = [
        f"{song['title']} by {song['artist']}.",
        f"Genre: {song['genre']}. Mood: {song['mood']}.",
        f"{_energy_words(song['energy'])} ({song['energy']:.2f}).",
        f"{_tempo_words(song['tempo_bpm'])} ({song['tempo_bpm']:.0f} BPM).",
    ]
    if song["valence"] >= 0.65:
        parts.append("Bright, positive feel.")
    elif song["valence"] <= 0.35:
        parts.append("Dark, gloomy feel.")
    if song["danceability"] >= 0.75:
        parts.append("Very danceable, groovy.")
    parts.append(
        "Acoustic, organic sound." if song["acousticness"] >= 0.5
        else "Electronic, produced sound."
    )
    return " ".join(parts)


def load_guides(knowledge_dir: Path) -> List[GuideChunk]:
    """Split each guide file into one chunk per '## label' section."""
    chunks: List[GuideChunk] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        kind = "mood" if "mood" in path.stem else "genre"
        sections = re.split(r"^## +", path.read_text(encoding="utf-8"), flags=re.M)
        for section in sections[1:]:  # sections[0] is the file intro
            label, _, body = section.partition("\n")
            chunks.append(GuideChunk(kind, label.strip(), body.strip()))
    return chunks


# ---- The RAG system ---------------------------------------------------------
class TuneScoutRAG:
    def __init__(
        self,
        songs_path: str = "data/songs.csv",
        knowledge_dir: str = "data/knowledge",
        generator: Optional[Generator] = None,
        use_guides: bool = True,
        style: str = "specialized",
    ):
        """
        use_guides: False = songs.csv is the only data source (the
            "before" baseline for the RAG-enhancement comparison).
        style: "specialized" = few-shot house style for reasons;
            "baseline" = plain instructions only.
        """
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}")
        self.songs = load_songs(songs_path)
        self.songs_by_id = {s["id"]: s for s in self.songs}
        self.guides = load_guides(Path(knowledge_dir)) if use_guides else []
        self.use_guides = use_guides
        self.style = style
        self.generator = generator

        if not self.songs:
            raise ValueError(f"No songs found in {songs_path}")

        self.song_docs = [song_to_document(s) for s in self.songs]
        guide_docs = [f"{g.label}. {g.text}" for g in self.guides]

        # One vocabulary shared by songs and guides, so guide words can
        # expand a request into words that appear in song documents.
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize, stop_words=STOP_WORDS, sublinear_tf=True
        )
        self.vectorizer.fit(self.song_docs + guide_docs)  # songs only if no guides
        self.song_matrix = self.vectorizer.transform(self.song_docs)
        self.guide_matrix = self.vectorizer.transform(guide_docs) if guide_docs else None

        logger.info(
            "Index built: %d songs, %d guide chunks, %d terms",
            len(self.songs), len(self.guides), len(self.vectorizer.vocabulary_),
        )

    # -- Guardrail 1: input ---------------------------------------------------
    @staticmethod
    def validate_request(request: str) -> Optional[str]:
        """Return an error message, or None if the request is usable."""
        if request is None or not request.strip():
            return "Please describe the music you want, e.g. 'calm songs for studying'."
        if len(request) > MAX_REQUEST_CHARS:
            return f"Please keep your request under {MAX_REQUEST_CHARS} characters."
        if not re.search(r"[a-zA-Z]", request):
            return "Please describe the music in words."
        return None

    # -- Retrieval ------------------------------------------------------------
    def retrieve_guides(self, request: str) -> List[Tuple[GuideChunk, float]]:
        if self.guide_matrix is None:
            return []
        scores = cosine_similarity(
            self.vectorizer.transform([request]), self.guide_matrix
        )[0]
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:TOP_K_GUIDES]
        return [(self.guides[i], float(s)) for i, s in ranked if s >= MIN_GUIDE_SCORE]

    def retrieve_songs(
        self, request: str, guides: Sequence[GuideChunk] = ()
    ) -> List[Tuple[Dict, float]]:
        # Query expansion: add matched guide labels + notes to the request,
        # so "mellow" also searches for "chill", "relaxed", "low energy", ...
        expanded = " ".join([request] + [f"{g.label} {g.text}" for g in guides])
        scores = cosine_similarity(
            self.vectorizer.transform([expanded]), self.song_matrix
        )[0]
        # Score the request on its own too, so a direct hit (e.g. a title)
        # isn't drowned out by the expansion.
        direct = cosine_similarity(
            self.vectorizer.transform([request]), self.song_matrix
        )[0]
        combined = [max(a, b) for a, b in zip(scores, direct)]
        ranked = sorted(
            enumerate(combined), key=lambda x: (-x[1], self.songs[x[0]]["id"])
        )[:TOP_K_SONGS]
        return [(self.songs[i], float(s)) for i, s in ranked]

    # -- Prompt ---------------------------------------------------------------
    def build_prompt(
        self,
        request: str,
        songs: Sequence[Tuple[Dict, float]],
        guides: Sequence[GuideChunk],
        k: int,
        extra: str = "",
    ) -> str:
        song_lines = "\n".join(
            f"- id {s['id']}: {song_to_document(s)}" for s, _ in songs
        )
        guide_lines = "\n".join(
            f"- [{g.kind}: {g.label}] {g.text}" for g in guides
        ) or "- (none matched)"
        return f"""You are TuneScout, a music recommender.

A listener asked: "{request}"

CANDIDATE SONGS (the ONLY songs you may recommend):
{song_lines}

MUSIC GUIDE NOTES (use these to interpret the request):
{guide_lines}

Rules:
1. Recommend up to {k} songs, chosen ONLY from the candidate list above, best match first.
2. Refer to songs by their id. Never invent songs, artists, or ids.
3. For each pick, give one short reason grounded in the song's listed features
   (genre, mood, energy, tempo, feel, sound) and how they fit the request.
4. If a candidate clearly does not fit the request, leave it out, even if that
   means fewer than {k} picks.
5. Rate your overall confidence from 0.0 to 1.0 that these picks fit the request.
{STYLE_RULES if self.style == "specialized" else ""}{extra}
Respond with JSON only, in exactly this shape:
{{"picks": [{{"id": <int>, "reason": "<string>"}}], "confidence": <float>, "note": "<one sentence summary for the listener>"}}
"""

    # -- Guardrail 3: output --------------------------------------------------
    def parse_and_validate(
        self, raw: str, allowed_ids: Sequence[int], k: int
    ) -> Tuple[List[Pick], Optional[float], str, List[int]]:
        """Parse the LLM's JSON and drop any pick that wasn't retrieved."""
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"LLM returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMUnavailable("LLM JSON was not an object")

        picks: List[Pick] = []
        dropped: List[int] = []
        seen = set()
        for item in data.get("picks") or []:
            try:
                song_id = int(item["id"])
            except (KeyError, TypeError, ValueError):
                logger.warning("Dropped malformed pick: %r", item)
                continue
            if song_id not in allowed_ids:
                dropped.append(song_id)
                continue
            if song_id in seen:
                continue
            seen.add(song_id)
            reason = str(item.get("reason", "")).strip() or "fits your request"
            picks.append(Pick(self.songs_by_id[song_id], reason))
            if len(picks) == k:
                break

        confidence = data.get("confidence")
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = None
        note = str(data.get("note", "")).strip()
        return picks, confidence, note, dropped

    # -- Fallback: Module 3 rule-based scorer ---------------------------------
    def fallback(
        self,
        result: RecommendationResult,
        songs: Sequence[Tuple[Dict, float]],
        guides: Sequence[GuideChunk],
        k: int,
        reason: str,
    ) -> RecommendationResult:
        """Turn retrieval results into a Module 3 profile and rank with score_song."""
        prefs: Dict = {}
        mood = next((g.label for g in guides if g.kind == "mood"), None)
        genre = next((g.label for g in guides if g.kind == "genre"), None)
        if mood:
            prefs["mood"] = mood
        if genre:
            prefs["genre"] = genre
        top = [s for s, _ in songs[:3]]
        prefs["energy"] = round(sum(s["energy"] for s in top) / len(top), 2)

        ranked = recommend_songs(prefs, self.songs, k=k)
        result.status = "fallback"
        result.picks = [Pick(song, why) for song, _, why in ranked]
        result.message = (
            f"AI unavailable ({reason}); used rule-based ranking with "
            f"profile {prefs}."
        )
        logger.warning("Fallback used: %s | profile=%s", reason, prefs)
        return result

    # -- Main entry point -----------------------------------------------------
    def recommend(self, request: str, k: int = DEFAULT_RESULTS) -> RecommendationResult:
        result = RecommendationResult(status="ok", request=request or "")

        error = self.validate_request(request)
        if error:
            logger.info("Rejected request %r: %s", request, error)
            result.status, result.message = "invalid_input", error
            return result
        request = request.strip()

        guide_hits = self.retrieve_guides(request)
        guides = [g for g, _ in guide_hits]
        songs = self.retrieve_songs(request, guides)
        result.guide_labels = [f"{g.kind}:{g.label}" for g in guides]
        result.retrieved_ids = [s["id"] for s, _ in songs]
        result.retrieval_score = round(songs[0][1], 3) if songs else 0.0
        logger.info(
            "Request %r | guides=%s | songs=%s",
            request,
            [(f"{g.kind}:{g.label}", round(s, 3)) for g, s in guide_hits],
            [(s["id"], round(sc, 3)) for s, sc in songs],
        )

        # Guardrail 2: relevance
        if result.retrieval_score < MIN_SONG_SCORE:
            result.status = "no_match"
            result.message = (
                "I couldn't find anything in the catalog that matches that request."
            )
            logger.info("No match (best score %.3f < %.2f)",
                        result.retrieval_score, MIN_SONG_SCORE)
            return result

        if self.generator is None:
            return self.fallback(result, songs, guides, k, "no API key")

        prompt = self.build_prompt(request, songs, guides, k)
        try:
            raw = self.generator(prompt)
            picks, confidence, note, dropped = self.parse_and_validate(
                raw, result.retrieved_ids, k
            )
        except LLMUnavailable as exc:
            logger.error("LLM failed: %s", exc)
            return self.fallback(result, songs, guides, k, str(exc))

        if dropped:
            logger.warning("Output guardrail dropped non-retrieved ids: %s", dropped)
        result.dropped_ids = dropped
        if not picks:
            return self.fallback(result, songs, guides, k, "LLM returned no valid picks")

        result.picks, result.confidence, result.message = picks, confidence, note
        logger.info(
            "Answer: ids=%s confidence=%s dropped=%s",
            [p.song["id"] for p in picks], confidence, dropped,
        )
        return result
