"""TuneScout Agent: a plan -> act -> recommend -> check -> retry workflow.

Plain RAG searches with the listener's exact words, so it fails on negation
("NOT sad") and on words that aren't in the guides ("fall asleep"). The agent
adds a planning step and a self-check:

  1. PLAN     Gemini turns the request into a structured search plan:
              search terms, wanted moods/genres, moods to avoid, energy range.
  2. ACT      Python tools run the plan:
                - search_catalog(query)      TF-IDF retrieval (same as RAG)
                - find_by_label(moods, genres)  exact catalog lookup
                - apply_constraints(...)     drop songs that break the plan
  3. RECOMMEND Gemini picks songs from the surviving candidates only.
  4. CHECK    Python verifies every pick against the plan's constraints.
  5. RETRY    If a pick breaks a rule, Gemini gets the feedback and tries
              once more; anything still breaking a rule is removed.

Every step is recorded in a trace, so the reasoning can be inspected later.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .llm import Generator, LLMUnavailable
from .rag import TOP_K_SONGS, RecommendationResult, TuneScoutRAG

logger = logging.getLogger(__name__)

TRACE_PATH = Path("logs/agent_traces.jsonl")
ENERGY_TOLERANCE = 0.05   # small slack so 0.46 isn't rejected for "< 0.45"
MAX_RETRIES = 1


@dataclass
class Plan:
    search_query: str
    moods: List[str] = field(default_factory=list)
    genres: List[str] = field(default_factory=list)
    avoid_moods: List[str] = field(default_factory=list)
    energy_min: float = 0.0
    energy_max: float = 1.0
    reasoning: str = ""


class TuneScoutAgent:
    def __init__(self, rag: TuneScoutRAG, generator: Optional[Generator] = None):
        self.rag = rag
        self.generator = generator or rag.generator
        self.mood_labels = sorted({s["mood"] for s in rag.songs})
        self.genre_labels = sorted({s["genre"] for s in rag.songs})

    # ---- Step 1: PLAN --------------------------------------------------------
    def planner_prompt(self, request: str) -> str:
        return f"""You are the planning step of TuneScout, a music recommender.
Turn the listener's request into a search plan. Do NOT recommend songs yet.

Listener request: "{request}"

Catalog mood labels: {", ".join(self.mood_labels)}
Catalog genre labels: {", ".join(self.genre_labels)}

Instructions:
- search_query: rewrite the request as plain descriptive words a keyword search
  can match (moods, genres, energy, tempo, activities). Drop negated words.
- moods / genres: labels from the lists above that fit. Use [] if unsure.
- avoid_moods: labels the listener explicitly or clearly does NOT want
  (e.g. "not sad" -> ["sad", "melancholic"]).
- energy_min / energy_max: the acceptable energy range from 0.0 to 1.0
  (e.g. sleep -> 0.0 to 0.4, workout -> 0.75 to 1.0, no preference -> 0.0 to 1.0).
- reasoning: one or two sentences explaining your plan.

Respond with JSON only:
{{"search_query": "<string>", "moods": [<labels>], "genres": [<labels>], "avoid_moods": [<labels>], "energy_min": <float>, "energy_max": <float>, "reasoning": "<string>"}}
"""

    def make_plan(self, request: str) -> Plan:
        raw = self.generator(self.planner_prompt(request))
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"planner returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMUnavailable("planner JSON was not an object")

        def labels(key, allowed):
            # Keep only real catalog labels, so the plan can't invent new ones.
            return [x for x in data.get(key) or [] if isinstance(x, str) and x in allowed]

        def energy(key, default):
            try:
                return max(0.0, min(1.0, float(data.get(key, default))))
            except (TypeError, ValueError):
                return default

        lo, hi = energy("energy_min", 0.0), energy("energy_max", 1.0)
        if lo > hi:
            lo, hi = hi, lo
        return Plan(
            search_query=str(data.get("search_query") or request).strip(),
            moods=labels("moods", self.mood_labels),
            genres=labels("genres", self.genre_labels),
            avoid_moods=labels("avoid_moods", self.mood_labels),
            energy_min=lo,
            energy_max=hi,
            reasoning=str(data.get("reasoning", "")).strip(),
        )

    # ---- Step 2: ACT (tools) -------------------------------------------------
    def tool_search_catalog(self, query: str) -> Tuple[List[Tuple[Dict, float]], list]:
        guides = [g for g, _ in self.rag.retrieve_guides(query)]
        return self.rag.retrieve_songs(query, guides), guides

    def tool_find_by_label(self, moods: List[str], genres: List[str]) -> List[Dict]:
        return [s for s in self.rag.songs if s["mood"] in moods or s["genre"] in genres]

    @staticmethod
    def violations(song: Dict, plan: Plan) -> List[str]:
        problems = []
        if song["mood"] in plan.avoid_moods:
            problems.append(f"mood '{song['mood']}' is on the avoid list")
        if song["energy"] < plan.energy_min - ENERGY_TOLERANCE:
            problems.append(f"energy {song['energy']:.2f} below {plan.energy_min:.2f}")
        if song["energy"] > plan.energy_max + ENERGY_TOLERANCE:
            problems.append(f"energy {song['energy']:.2f} above {plan.energy_max:.2f}")
        return problems

    def tool_apply_constraints(self, songs: List[Dict], plan: Plan) -> Tuple[List[Dict], List[int]]:
        kept = [s for s in songs if not self.violations(s, plan)]
        removed = [s["id"] for s in songs if self.violations(s, plan)]
        return kept, removed

    # ---- Main loop -----------------------------------------------------------
    def run(self, request: str, k: int = 3) -> Tuple[RecommendationResult, List[Dict]]:
        trace: List[Dict] = [{"step": "request", "text": request}]
        result = RecommendationResult(status="ok", request=request or "")

        error = self.rag.validate_request(request)
        if error:
            trace.append({"step": "input_guardrail", "result": "rejected", "message": error})
            result.status, result.message = "invalid_input", error
            return self._finish(result, trace)
        request = request.strip()

        if self.generator is None:
            trace.append({"step": "plan", "result": "skipped: no LLM, using plain RAG"})
            return self._finish(self.rag.recommend(request, k), trace)

        # 1. PLAN
        try:
            plan = self.make_plan(request)
        except LLMUnavailable as exc:
            trace.append({"step": "plan", "result": f"failed ({exc}); using plain RAG"})
            return self._finish(self.rag.recommend(request, k), trace)
        trace.append({"step": "plan", **plan.__dict__})

        # 2. ACT
        search_hits, guides = self.tool_search_catalog(plan.search_query)
        trace.append({
            "step": "tool: search_catalog",
            "input": plan.search_query,
            "output": [(s["id"], s["title"], round(sc, 2)) for s, sc in search_hits],
            "guides": [f"{g.kind}:{g.label}" for g in guides],
        })
        label_hits = self.tool_find_by_label(plan.moods, plan.genres)
        trace.append({
            "step": "tool: find_by_label",
            "input": {"moods": plan.moods, "genres": plan.genres},
            "output": [(s["id"], s["title"]) for s in label_hits],
        })

        # Merge: relevant search hits first, then label matches.
        candidates, seen = [], set()
        for song in [s for s, sc in search_hits if sc > 0] + label_hits:
            if song["id"] not in seen:
                seen.add(song["id"])
                candidates.append(song)
        candidates, removed = self.tool_apply_constraints(candidates, plan)
        candidates = candidates[:TOP_K_SONGS]
        trace.append({
            "step": "tool: apply_constraints",
            "removed_ids": removed,
            "kept": [(s["id"], s["title"]) for s in candidates],
        })
        result.retrieved_ids = [s["id"] for s in candidates]
        result.guide_labels = [f"{g.kind}:{g.label}" for g in guides]
        result.retrieval_score = round(search_hits[0][1], 3) if search_hits else 0.0

        if not candidates:
            result.status = "no_match"
            result.message = "I couldn't find anything in the catalog that fits that plan."
            trace.append({"step": "decision", "result": "no candidates left -> no match"})
            return self._finish(result, trace)

        # 3. RECOMMEND  4. CHECK  5. RETRY
        plan_note = (
            f"\nThe planning step decided: {plan.reasoning}\n"
            f"Wanted moods: {plan.moods or 'any'}; avoid moods: {plan.avoid_moods or 'none'}; "
            f"energy between {plan.energy_min:.2f} and {plan.energy_max:.2f}.\n"
        )
        feedback = ""
        for attempt in range(1 + MAX_RETRIES):
            prompt = self.rag.build_prompt(
                request, [(s, 0.0) for s in candidates], guides, k, extra=plan_note + feedback
            )
            try:
                raw = self.generator(prompt)
                picks, confidence, note, dropped = self.rag.parse_and_validate(
                    raw, result.retrieved_ids, k
                )
            except LLMUnavailable as exc:
                trace.append({"step": "recommend", "attempt": attempt + 1, "result": f"failed ({exc})"})
                return self._finish(self.rag.fallback(result, [(s, 0.0) for s in candidates], guides, k, str(exc)), trace)

            trace.append({
                "step": "recommend",
                "attempt": attempt + 1,
                "picks": [(p.song["id"], p.song["title"]) for p in picks],
                "confidence": confidence,
                "dropped_ids": dropped,
            })
            bad = {p.song["id"]: self.violations(p.song, plan) for p in picks}
            bad = {i: v for i, v in bad.items() if v}
            trace.append({"step": "check", "attempt": attempt + 1,
                          "violations": bad or "none"})
            if not bad:
                break
            feedback = "\nYour previous answer broke the plan: " + "; ".join(
                f"song {i}: {', '.join(v)}" for i, v in bad.items()
            ) + ". Choose different songs.\n"
        else:
            # Retries used up: remove whatever still breaks the plan.
            picks = [p for p in picks if p.song["id"] not in bad]
            trace.append({"step": "decision", "result": f"removed rule-breaking ids {list(bad)}"})

        if not picks:
            return self._finish(self.rag.fallback(result, [(s, 0.0) for s in candidates], guides, k,
                                                  "agent found no valid picks"), trace)
        result.picks, result.confidence, result.message = picks, confidence, note
        result.dropped_ids = dropped
        trace.append({"step": "answer", "ids": [p.song["id"] for p in picks]})
        return self._finish(result, trace)

    def _finish(self, result: RecommendationResult, trace: List[Dict]):
        try:
            TRACE_PATH.parent.mkdir(exist_ok=True)
            with TRACE_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"request": result.request, "status": result.status,
                                    "trace": trace}, default=str) + "\n")
        except OSError as exc:
            logger.warning("Could not save agent trace: %s", exc)
        logger.info("Agent finished %r with status %s", result.request, result.status)
        return result, trace


def format_trace(trace: List[Dict]) -> str:
    """Render a trace as readable Markdown-friendly text."""
    lines = []
    for i, step in enumerate(trace, 1):
        name = step.get("step")
        body = {k: v for k, v in step.items() if k != "step"}
        lines.append(f"{i}. [{name}] " + json.dumps(body, default=str, ensure_ascii=False))
    return "\n".join(lines)
