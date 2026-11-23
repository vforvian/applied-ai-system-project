"""Reliability tests for the TuneScout RAG pipeline.

Gemini is replaced by fake generators so these tests are free, offline and
deterministic. The fakes let us force failures (invented songs, broken JSON,
API errors) and check that every guardrail catches them.
"""

import json

import pytest

from src.llm import GeminiGenerator, LLMUnavailable
from src.rag import (
    MAX_REQUEST_CHARS,
    TuneScoutRAG,
    load_guides,
    normalize,
    song_to_document,
)


class FakeGenerator:
    """Stands in for Gemini: records prompts and returns a canned reply."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


@pytest.fixture(scope="module")
def rag():
    return TuneScoutRAG()


def with_generator(rag, generator):
    rag.generator = generator
    return rag


# ---- Knowledge base & documents ---------------------------------------------
def test_every_catalog_mood_and_genre_has_a_guide(rag):
    labels = {(g.kind, g.label) for g in rag.guides}
    for song in rag.songs:
        assert ("mood", song["mood"]) in labels, f"missing mood guide: {song['mood']}"
        assert ("genre", song["genre"]) in labels, f"missing genre guide: {song['genre']}"


def test_song_document_describes_features(rag):
    library_rain = rag.songs_by_id[4]
    doc = song_to_document(library_rain)
    for expected in ("Library Rain", "lofi", "chill", "low energy", "slow tempo", "Acoustic"):
        assert expected in doc


def test_guides_split_into_one_chunk_per_heading(tmp_path):
    (tmp_path / "moods.md").write_text("# intro\n\n## calm\nquiet\n\n## loud\nnoisy\n")
    chunks = load_guides(tmp_path)
    assert [(c.kind, c.label, c.text) for c in chunks] == [
        ("mood", "calm", "quiet"), ("mood", "loud", "noisy"),
    ]


def test_normalize_keeps_r_and_b():
    assert "rnb" in normalize("Some R&B please")
    assert "lofi" in normalize("lo-fi beats")


# ---- Guardrail 1: input -----------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", None, "1234 !!!", "x" * (MAX_REQUEST_CHARS + 1)])
def test_invalid_requests_are_rejected_without_calling_llm(rag, bad):
    fake = FakeGenerator(reply={"picks": []})
    result = with_generator(rag, fake).recommend(bad)
    assert result.status == "invalid_input"
    assert result.message
    assert fake.prompts == []


# ---- Retrieval quality (golden set) -----------------------------------------
from src.eval_cases import GOLDEN  # shared with src/evaluate.py


@pytest.mark.parametrize("request_text,expected_ids", GOLDEN)
def test_retrieval_finds_expected_song_in_top_3(rag, request_text, expected_ids):
    guides = [g for g, _ in rag.retrieve_guides(request_text)]
    top3 = {s["id"] for s, _ in rag.retrieve_songs(request_text, guides)[:3]}
    assert top3 & expected_ids, f"{request_text!r}: got {top3}, wanted one of {expected_ids}"


def test_similar_mood_is_found_via_guides(rag):
    """Module 3 bug: a 'chill' listener got no credit for the 'relaxed' song.
    With the mood guide, 'chill' expands to 'relaxed' and finds it."""
    guides = [g for g, _ in rag.retrieve_guides("chill")]
    assert "relaxed" in {g.label for g in guides}
    top5 = [s["id"] for s, _ in rag.retrieve_songs("chill", guides)[:5]]
    assert 7 in top5  # Coffee Shop Stories (relaxed jazz)


def test_retrieval_is_deterministic(rag):
    q = "upbeat happy pop for a road trip"
    first = [s["id"] for s, _ in rag.retrieve_songs(q)]
    second = [s["id"] for s, _ in rag.retrieve_songs(q)]
    assert first == second


# ---- Guardrail 2: relevance -------------------------------------------------
@pytest.mark.parametrize("request_text", ["Gregorian chant with bagpipes", "asdf qwerty"])
def test_unrelated_request_returns_no_match_without_calling_llm(rag, request_text):
    fake = FakeGenerator(reply={"picks": [{"id": 1, "reason": "x"}]})
    result = with_generator(rag, fake).recommend(request_text)
    assert result.status == "no_match"
    assert result.picks == []
    assert fake.prompts == []


# ---- Prompt grounding -------------------------------------------------------
def test_prompt_contains_only_retrieved_songs_and_guides(rag):
    fake = FakeGenerator(reply={"picks": [{"id": 4, "reason": "calm lofi"}], "confidence": 0.9})
    result = with_generator(rag, fake).recommend("mellow songs for studying")
    prompt = fake.prompts[0]
    for song_id in result.retrieved_ids:
        assert f"id {song_id}:" in prompt
    not_retrieved = set(rag.songs_by_id) - set(result.retrieved_ids)
    for song_id in not_retrieved:
        assert f"id {song_id}:" not in prompt
    assert "[mood: chill]" in prompt
    assert "ONLY songs you may recommend" in prompt


# ---- Happy path -------------------------------------------------------------
def test_valid_llm_answer_is_returned_with_reasons_and_confidence(rag):
    fake = FakeGenerator(reply={
        "picks": [{"id": 4, "reason": "soft, slow lofi"}, {"id": 9, "reason": "steady focus beat"}],
        "confidence": 0.85,
        "note": "Calm lofi for studying.",
    })
    result = with_generator(rag, fake).recommend("mellow songs for studying", k=3)
    assert result.status == "ok"
    assert [p.song["id"] for p in result.picks] == [4, 9]
    assert result.picks[0].reason == "soft, slow lofi"
    assert result.confidence == 0.85
    assert result.message == "Calm lofi for studying."


def test_code_fenced_json_is_accepted(rag):
    reply = '```json\n{"picks": [{"id": 4, "reason": "calm"}], "confidence": 0.7}\n```'
    result = with_generator(rag, FakeGenerator(reply=reply)).recommend("mellow studying")
    assert result.status == "ok"
    assert [p.song["id"] for p in result.picks] == [4]


# ---- Guardrail 3: output ----------------------------------------------------
def test_invented_song_ids_are_dropped(rag):
    fake = FakeGenerator(reply={
        "picks": [{"id": 999, "reason": "made up"}, {"id": 4, "reason": "real"}],
        "confidence": 0.9,
    })
    result = with_generator(rag, fake).recommend("mellow songs for studying")
    assert result.status == "ok"
    assert [p.song["id"] for p in result.picks] == [4]
    assert result.dropped_ids == [999]


def test_real_but_not_retrieved_song_is_dropped(rag):
    """Iron Tide (13, metal) exists, but isn't retrieved for a calm request."""
    fake = FakeGenerator(reply={"picks": [{"id": 13, "reason": "?"}, {"id": 4, "reason": "ok"}]})
    result = with_generator(rag, fake).recommend("mellow songs for studying")
    assert 13 not in result.retrieved_ids
    assert [p.song["id"] for p in result.picks] == [4]
    assert result.dropped_ids == [13]


def test_duplicates_removed_and_k_respected(rag):
    fake = FakeGenerator(reply={"picks": [
        {"id": 4, "reason": "a"}, {"id": 4, "reason": "dup"},
        {"id": 9, "reason": "b"}, {"id": 2, "reason": "c"},
    ]})
    result = with_generator(rag, fake).recommend("mellow songs for studying", k=2)
    assert [p.song["id"] for p in result.picks] == [4, 9]


def test_confidence_is_clamped_to_0_1(rag):
    fake = FakeGenerator(reply={"picks": [{"id": 4, "reason": "a"}], "confidence": 1.7})
    result = with_generator(rag, fake).recommend("mellow songs for studying")
    assert result.confidence == 1.0


# ---- Fallback ---------------------------------------------------------------
@pytest.mark.parametrize("fake", [
    FakeGenerator(reply={"picks": [{"id": 999, "reason": "all invented"}]}),
    FakeGenerator(reply="this is not json"),
    FakeGenerator(reply="[1, 2, 3]"),
    FakeGenerator(error=LLMUnavailable("network down")),
], ids=["all-invented", "bad-json", "json-not-object", "api-error"])
def test_llm_failures_fall_back_to_rule_based_ranking(rag, fake):
    result = with_generator(rag, fake).recommend("mellow songs for studying", k=3)
    assert result.status == "fallback"
    assert len(result.picks) == 3
    assert all(p.song["id"] in rag.songs_by_id for p in result.picks)
    assert "rule-based" in result.message


def test_no_api_key_uses_fallback_with_sensible_picks(rag):
    result = with_generator(rag, None).recommend("mellow songs for studying on a rainy day")
    assert result.status == "fallback"
    assert {p.song["genre"] for p in result.picks} <= {"lofi", "ambient", "jazz"}


def test_gemini_without_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable):
        GeminiGenerator()


# ---- Known limitations (documented, expected to fail) -----------------------
@pytest.mark.xfail(reason="TF-IDF ignores negation: 'NOT sad' still matches sad songs", strict=True)
def test_negation_is_understood(rag):
    q = "NOT sad, something happy"
    guides = [g for g, _ in rag.retrieve_guides(q)]
    top = rag.retrieve_songs(q, guides)[0][0]
    assert top["mood"] != "sad"


@pytest.mark.xfail(reason="Words missing from songs + guides get no match (e.g. 'asleep', 'funeral')", strict=True)
@pytest.mark.parametrize("request_text", ["something to fall asleep to", "music for a funeral"])
def test_reasonable_requests_outside_guide_vocabulary(rag, request_text):
    result = with_generator(rag, None).recommend(request_text)
    assert result.status != "no_match"


@pytest.mark.xfail(reason="Catalog has no real artists, so artist-similarity requests can't match", strict=True)
def test_artist_similarity_request(rag):
    result = with_generator(rag, None).recommend("songs like Taylor Swift")
    assert result.status != "no_match"
