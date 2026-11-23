"""Offline tests for the plan -> act -> check -> retry agent."""

import json

import pytest

from src.agent import TuneScoutAgent
from src.llm import LLMUnavailable
from src.rag import TuneScoutRAG


class ScriptedGenerator:
    """Returns one scripted reply per call (planner first, then recommender)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


@pytest.fixture(scope="module")
def rag():
    return TuneScoutRAG()


HAPPY_PLAN = {
    "search_query": "happy upbeat cheerful",
    "moods": ["happy"], "genres": [], "avoid_moods": ["sad", "melancholic"],
    "energy_min": 0.5, "energy_max": 1.0, "reasoning": "wants happy, not sad",
}


def steps(trace):
    return [s["step"] for s in trace]


def test_full_chain_records_every_step(rag):
    gen = ScriptedGenerator(HAPPY_PLAN, {"picks": [{"id": 1, "reason": "happy pop"}], "confidence": 0.9})
    result, trace = TuneScoutAgent(rag, gen).run("NOT sad, something happy")
    assert result.status == "ok"
    assert [p.song["id"] for p in result.picks] == [1]
    assert steps(trace) == [
        "request", "plan", "tool: search_catalog", "tool: find_by_label",
        "tool: apply_constraints", "recommend", "check", "answer",
    ]


def test_constraints_remove_avoided_moods_before_llm_sees_them(rag):
    gen = ScriptedGenerator(HAPPY_PLAN, {"picks": [{"id": 1, "reason": "x"}]})
    result, _ = TuneScoutAgent(rag, gen).run("NOT sad, something happy")
    candidate_moods = {rag.songs_by_id[i]["mood"] for i in result.retrieved_ids}
    assert not candidate_moods & {"sad", "melancholic"}
    assert "id 16:" not in gen.prompts[1]  # Last Bus Home (sad) never offered


def test_valid_pick_passes_check_without_retry(rag):
    plan = {**HAPPY_PLAN, "search_query": "calm sleep", "moods": ["chill"],
            "avoid_moods": [], "energy_min": 0.0, "energy_max": 0.4}
    gen = ScriptedGenerator(plan, {"picks": [{"id": 4, "reason": "calm"}]})
    result, trace = TuneScoutAgent(rag, gen).run("something to fall asleep to")
    checks = [s for s in trace if s["step"] == "check"]
    assert len(checks) == 1 and checks[0]["violations"] == "none"
    assert len(gen.prompts) == 2  # one plan call + one recommend call
    assert all(rag.songs_by_id[i]["energy"] <= 0.45 for i in result.retrieved_ids)


def test_rule_breaking_pick_is_retried_then_removed(rag, monkeypatch):
    agent = TuneScoutAgent(rag, None)
    # Make the constraint tool keep everything, so a rule-breaking song
    # reaches the LLM and the CHECK step has to catch it.
    monkeypatch.setattr(agent, "tool_apply_constraints", lambda songs, plan: (songs, []))
    # A (bad) search query that retrieves the sad song, so it reaches the LLM.
    plan = {**HAPPY_PLAN, "search_query": "Last Bus Home happy"}
    sad_song_id = 16  # Last Bus Home
    agent.generator = ScriptedGenerator(
        plan,
        {"picks": [{"id": sad_song_id, "reason": "x"}, {"id": 1, "reason": "y"}]},
        {"picks": [{"id": sad_song_id, "reason": "again"}, {"id": 10, "reason": "z"}]},
    )
    result, trace = agent.run("NOT sad, something happy")
    assert sad_song_id in result.retrieved_ids

    checks = [s for s in trace if s["step"] == "check"]
    assert len(checks) == 2                       # first check failed -> retried
    assert sad_song_id in checks[0]["violations"]
    assert "broke the plan" in agent.generator.prompts[2]   # feedback sent
    assert [p.song["id"] for p in result.picks] == [10]     # still-bad pick removed


def test_planner_cannot_invent_labels_or_bad_energy(rag):
    gen = ScriptedGenerator({
        "search_query": "x", "moods": ["happy", "hyperpop-core"], "genres": ["polka"],
        "avoid_moods": [], "energy_min": 1.7, "energy_max": -2, "reasoning": "",
    })
    plan = TuneScoutAgent(rag, gen).make_plan("anything")
    assert plan.moods == ["happy"] and plan.genres == []
    assert 0.0 <= plan.energy_min <= plan.energy_max <= 1.0


def test_planner_failure_falls_back_to_plain_rag(rag):
    gen = ScriptedGenerator(LLMUnavailable("503"), {"picks": [{"id": 4, "reason": "calm"}]})
    result, trace = TuneScoutAgent(rag, gen).run("mellow songs for studying")
    assert "failed" in trace[1]["result"]
    assert result.status in ("ok", "fallback")
    assert result.picks


def test_invalid_input_stops_before_planning(rag):
    gen = ScriptedGenerator()
    result, trace = TuneScoutAgent(rag, gen).run("   ")
    assert result.status == "invalid_input"
    assert gen.prompts == []
