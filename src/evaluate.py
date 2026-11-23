"""TuneScout evaluation harness.

Runs the system on predefined inputs (src/eval_cases.py) and prints a
pass/fail summary. Also writes a Markdown report and the agent's reasoning
traces to reports/.

    python -m src.evaluate             # everything (live Gemini if key is set)
    python -m src.evaluate --offline   # retrieval + guardrails only, no API calls
    python -m src.evaluate --only style    # just one live section
    GEMINI_MODEL=gemini-3.5-flash-lite python -m src.evaluate --only agent

Free-tier Gemini keys allow about 20 requests per model per day, so the live
sections can be run separately (and on different models). Requests that fall
back because the LLM was unavailable are NOT counted as results; they are
reported as skipped and the run is marked incomplete.

Sections:
  1. RAG enhancement   retrieval with songs only vs. songs + guides
  2. Guardrails        invalid input and no-match requests
  3. Specialization    baseline prompt vs. few-shot house style (live)
  4. Agent             plain RAG vs. plan/act/check agent on hard cases (live)
"""

import argparse
import logging
import re
import statistics
from pathlib import Path
from typing import Dict, List

from .eval_cases import GOLDEN, HARD_CASES, INVALID_INPUTS, NO_MATCH_INPUTS
from .rag import HYPE_WORDS, MIN_SONG_SCORE, STOP_WORDS, TuneScoutRAG, normalize

REPORT_DIR = Path("reports")


class Report:
    """Collects lines for both the terminal and the Markdown report."""

    def __init__(self):
        self.md: List[str] = []
        self.summary: List[str] = []
        self.incomplete = False

    def heading(self, text):
        print(f"\n=== {text} ===")
        self.md += ["", f"## {text}", ""]

    def table(self, header: List[str], rows: List[List]):
        widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
        for row in [header] + rows:
            print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
        self.md.append("| " + " | ".join(header) + " |")
        self.md.append("|" + "---|" * len(header))
        self.md += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]

    def line(self, text, summary=False):
        print(text)
        self.md += ["", text]
        if summary:
            self.summary.append(text)


def mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ---- 1. RAG enhancement --------------------------------------------------
def eval_retrieval(report: Report):
    report.heading("1. RAG enhancement: songs only vs. songs + guides")
    results = {}
    rows = []
    for use_guides in (False, True):
        rag = TuneScoutRAG(use_guides=use_guides)
        passes = 0
        for request, expected in GOLDEN:
            guides = [g for g, _ in rag.retrieve_guides(request)]
            hits = rag.retrieve_songs(request, guides)
            top3 = [s["id"] for s, _ in hits[:3]]
            # A "no match" (score below threshold) counts as a failure even
            # if tie-breaking happened to put a right song in the top 3.
            ok = hits[0][1] >= MIN_SONG_SCORE and bool(set(top3) & expected)
            passes += ok
            results.setdefault(request, {})[use_guides] = (ok, hits[0][1], top3, rag)
        results[use_guides] = passes
    for request, _ in GOLDEN:
        before, after = results[request][False], results[request][True]
        rows.append([
            f'"{request}"',
            f"{mark(before[0])} ({before[1]:.2f})",
            f"{mark(after[0])} ({after[1]:.2f})",
        ])
    report.table(["Request", "Songs only (best score)", "Songs + guides (best score)"], rows)
    report.line(
        f"Retrieval hit@3: songs only {results[False]}/{len(GOLDEN)} -> "
        f"songs + guides {results[True]}/{len(GOLDEN)}",
        summary=True,
    )


# ---- 2. Guardrails --------------------------------------------------------
def eval_guardrails(report: Report):
    report.heading("2. Guardrails")
    rag = TuneScoutRAG()  # no generator: guardrails run before any LLM call
    rows, passes = [], 0
    for text in INVALID_INPUTS:
        status = rag.recommend(text).status
        ok = status == "invalid_input"
        passes += ok
        rows.append([repr(text if len(text) < 20 else text[:12] + "..."), "invalid_input", status, mark(ok)])
    for text in NO_MATCH_INPUTS:
        status = rag.recommend(text).status
        ok = status == "no_match"
        passes += ok
        rows.append([repr(text), "no_match", status, mark(ok)])
    report.table(["Input", "Expected", "Got", "Result"], rows)
    total = len(INVALID_INPUTS) + len(NO_MATCH_INPUTS)
    report.line(f"Guardrails: {passes}/{total} passed", summary=True)


# ---- 3. Specialization ----------------------------------------------------
def style_metrics(request: str, reason: str) -> Dict[str, bool]:
    words = reason.split()
    request_terms = {
        w for w in re.findall(r"[a-z]+", normalize(request))
        if w not in STOP_WORDS and len(w) > 2
    }
    return {
        "words": len(words),
        "short": len(words) <= 20,
        "number": bool(re.search(r"\b0\.\d\d\b|\b\d{2,3}\s?BPM\b", reason)),
        "no_hype": not any(re.search(rf"\b{h}\b", reason.lower()) for h in HYPE_WORDS),
        "echo": any(t in reason.lower() for t in request_terms),
    }


def eval_style(report: Report, generator) -> None:
    report.heading("3. Specialization: baseline prompt vs. few-shot house style")
    summary_rows = []
    for style in ("baseline", "specialized"):
        rag = TuneScoutRAG(generator=generator, style=style)
        metrics, top1, confs, skipped = [], 0, [], 0
        for request, expected in GOLDEN:
            result = rag.recommend(request)
            if result.status != "ok":
                skipped += 1
                report.line(f"  SKIPPED ({style}) {request!r}: {result.status}")
                continue
            top1 += result.picks[0].song["id"] in expected
            if result.confidence is not None:
                confs.append(result.confidence)
            metrics += [style_metrics(request, p.reason) for p in result.picks]
        n = len(metrics) or 1
        answered = len(GOLDEN) - skipped
        report.incomplete |= skipped > 0
        summary_rows.append([
            style,
            f"{top1}/{answered}" + (f" ({skipped} skipped)" if skipped else ""),
            f"{statistics.mean(m['words'] for m in metrics):.1f}" if metrics else "-",
            f"{sum(m['short'] for m in metrics) / n:.0%}",
            f"{sum(m['number'] for m in metrics) / n:.0%}",
            f"{sum(m['no_hype'] for m in metrics) / n:.0%}",
            f"{sum(m['echo'] for m in metrics) / n:.0%}",
            f"{statistics.mean(confs):.2f}" if confs else "-",
        ])
    report.table(
        ["Prompt", "#1 pick correct", "Avg words/reason", "<=20 words",
         "Quotes a number", "No hype words", "Echoes request", "Avg confidence"],
        summary_rows,
    )
    b, s = summary_rows
    report.line(
        f"Style: reasons quoting a number {b[4]} -> {s[4]}, <=20 words {b[3]} -> {s[3]}, "
        f"no hype words {b[5]} -> {s[5]}; #1 pick correct {b[1]} -> {s[1]}",
        summary=True,
    )


# ---- 4. Agent ---------------------------------------------------------------
def eval_agent(report: Report, generator, agent_golden: bool = False) -> None:
    from .agent import TuneScoutAgent, format_trace

    report.heading("4. Agent: plain RAG vs. plan/act/check agent")
    rag = TuneScoutRAG(generator=generator)
    agent = TuneScoutAgent(rag)
    rows, rag_pass, agent_pass = [], 0, 0
    traces = ["# Agent Reasoning Traces", "",
              "Generated by `python -m src.evaluate`. One trace per hard-case request.", ""]

    def judge(result, rule):
        if result.status == "fallback":
            report.incomplete = True
            return None  # LLM unavailable: not a real result
        return result.status == "ok" and bool(result.picks) and all(rule(p.song) for p in result.picks)

    def show(ok):
        return "SKIPPED" if ok is None else mark(ok)

    for request, rule_text, rule in HARD_CASES:
        r1 = rag.recommend(request)
        r2, trace = agent.run(request)
        ok1, ok2 = judge(r1, rule), judge(r2, rule)
        rag_pass += bool(ok1)
        agent_pass += bool(ok2)
        titles = lambda r: ", ".join(p.song["title"] for p in r.picks) or r.status
        rows.append([f'"{request}"', rule_text,
                     f"{show(ok1)}: {titles(r1)}", f"{show(ok2)}: {titles(r2)}"])
        traces += [f"## \"{request}\"", "", f"Rule every pick must satisfy: {rule_text}  ",
                   f"Result: **{show(ok2)}**: {titles(r2)}", "", "```", format_trace(trace), "```", ""]

    report.table(["Hard request", "Rule", "Plain RAG", "Agent"], rows)
    report.line(
        f"Hard cases: plain RAG {rag_pass}/{len(HARD_CASES)} -> agent "
        f"{agent_pass}/{len(HARD_CASES)}",
        summary=True,
    )

    if agent_golden:  # optional: costs ~3 API calls per request
        golden_agent, answered = 0, 0
        for request, expected in GOLDEN:
            r2, _ = agent.run(request)
            if r2.status != "ok":
                report.incomplete = True
                continue
            answered += 1
            golden_agent += r2.picks[0].song["id"] in expected
        report.line(f"Agent on golden set: #1 pick correct {golden_agent}/{answered}", summary=True)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "agent_traces.md").write_text("\n".join(traces), encoding="utf-8")
    print(f"  traces saved to {REPORT_DIR / 'agent_traces.md'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TuneScout evaluation harness")
    parser.add_argument("--offline", action="store_true", help="skip live Gemini sections")
    parser.add_argument("--only", choices=["style", "agent"],
                        help="run only this live section (sections 1-2 always run)")
    parser.add_argument("--agent-golden", action="store_true",
                        help="also run the agent on the golden set (~24 extra API calls)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="[%(levelname)s] %(message)s")

    report = Report()
    eval_retrieval(report)
    eval_guardrails(report)

    generator = None
    if not args.offline:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        from .llm import make_default_generator
        generator = make_default_generator()
        if generator is None:
            report.line("No GEMINI_API_KEY: skipping live sections 3 and 4.", summary=True)
    if generator is not None:
        report.line(f"Live sections use model: {generator.model}", summary=True)
        if args.only in (None, "style"):
            eval_style(report, generator)
        if args.only in (None, "agent"):
            eval_agent(report, generator, args.agent_golden)
        if report.incomplete:
            report.line("WARNING: some requests fell back (LLM unavailable, e.g. quota); "
                        "they were skipped, so live results are incomplete.", summary=True)

    print("\n=== SUMMARY ===")
    for line in report.summary:
        print(" - " + line)
    REPORT_DIR.mkdir(exist_ok=True)
    name = "offline" if args.offline or generator is None else (args.only or "full")
    out = REPORT_DIR / f"eval_report_{name}.md"
    out.write_text(
        "# TuneScout Evaluation Report\n\nGenerated by `python -m src.evaluate"
        + (" --offline" if args.offline else "") + (f" --only {args.only}" if args.only else "")
        + "`.\n\n## Summary\n\n"
        + "\n".join(f"- {s}" for s in report.summary) + "\n" + "\n".join(report.md) + "\n",
        encoding="utf-8",
    )
    print(f"\nReport saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
