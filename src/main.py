"""TuneScout RAG command-line app.

Usage:
    python -m src.main "mellow songs for studying on a rainy day"
    python -m src.main                 # interactive mode
    python -m src.main --agent --trace "NOT sad, something happy"
    python -m src.main --legacy        # original Module 3 profile demo
"""

import argparse
import logging
import sys
from pathlib import Path

from .recommender import MAX_SCORE, load_songs, recommend_songs

LOG_PATH = Path("logs/tunescout.log")
WIDTH = 64

# ---- Module 3 demo profiles (kept for --legacy) -----------------------------
PROFILES = {
    "High-Energy Pop": {
        "genre": "pop", "mood": "happy", "energy": 0.85, "likes_acoustic": False,
    },
    "Chill Lofi": {
        "genre": "lofi", "mood": "chill", "energy": 0.35, "likes_acoustic": True,
    },
    "Deep Intense Rock": {
        "genre": "rock", "mood": "intense", "energy": 0.92, "likes_acoustic": False,
    },
}


def setup_logging() -> None:
    """Log everything to logs/tunescout.log; only errors to the terminal."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [file_handler, console]
    # The HTTP client logs every request at INFO; keep the log readable.
    for noisy in ("httpx", "httpx2", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def print_result(result) -> None:
    """Print a RecommendationResult as a readable terminal report."""
    print()
    print("=" * WIDTH)
    print(f' Request: "{result.request}"')
    print("-" * WIDTH)

    if result.status in ("invalid_input", "no_match"):
        print(f" ⚠  {result.message}")
        print("=" * WIDTH)
        return

    mode = "AI (Gemini + retrieval)" if result.status == "ok" else "Rule-based fallback"
    print(f" Mode: {mode}")
    if result.guide_labels:
        print(f" Matched guide notes: {', '.join(result.guide_labels)}")
    print(f" Retrieval score: {result.retrieval_score:.2f}", end="")
    if result.confidence is not None:
        print(f"   AI confidence: {result.confidence:.2f}", end="")
    print()
    if result.status == "fallback":
        print(f" Note: {result.message}")
    print("-" * WIDTH)

    for rank, pick in enumerate(result.picks, start=1):
        s = pick.song
        print(f"\n {rank}. {s['title']}  ({s['artist']})   [song id {s['id']}]")
        print(f"    {s['genre']} · {s['mood']} · energy {s['energy']:.2f}")
        print(f"    Why: {pick.reason}")

    if result.status == "ok" and result.message:
        print(f"\n {result.message}")
    if result.dropped_ids:
        print(f"\n (Removed {len(result.dropped_ids)} pick(s) not found in the catalog.)")
    print()
    print("=" * WIDTH)


def run_legacy() -> None:
    """The original Module 3 demo: fixed profiles, rule-based scoring."""
    songs = load_songs("data/songs.csv")
    for name, prefs in PROFILES.items():
        print(f"\n>>> {name}  {prefs}")
        for rank, (song, score, why) in enumerate(recommend_songs(prefs, songs, k=5), 1):
            print(f" {rank}. {song['title']:<22} {score:.2f}/{MAX_SCORE:.1f}  {why}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TuneScout RAG music recommender")
    parser.add_argument("request", nargs="*", help="what you want to listen to")
    parser.add_argument("-k", type=int, default=3, help="number of songs (1-5)")
    parser.add_argument("--agent", action="store_true",
                        help="use the plan/act/check agent instead of plain RAG")
    parser.add_argument("--trace", action="store_true",
                        help="with --agent: print the agent's reasoning trace")
    parser.add_argument("--legacy", action="store_true",
                        help="run the original Module 3 profile demo")
    args = parser.parse_args(argv)

    setup_logging()
    if args.legacy:
        run_legacy()
        return 0

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from .llm import make_default_generator
    from .rag import TuneScoutRAG

    k = max(1, min(5, args.k))
    try:
        rag = TuneScoutRAG(generator=make_default_generator())
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).error("Startup failed: %s", exc)
        print(f"Could not load the catalog: {exc}")
        return 1

    if rag.generator is None:
        print("(No GEMINI_API_KEY found: running in rule-based fallback mode.)")

    if args.agent:
        from .agent import TuneScoutAgent, format_trace
        agent = TuneScoutAgent(rag)

        def answer(request):
            result, trace = agent.run(request, k=k)
            if args.trace:
                print("\n--- agent trace ---")
                print(format_trace(trace))
            return result
    else:
        def answer(request):
            return rag.recommend(request, k=k)

    if args.request:
        print_result(answer(" ".join(args.request)))
        return 0

    print("TuneScout RAG: describe what you want to hear (blank line to quit).")
    while True:
        try:
            request = input("\n🎧 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not request:
            break
        print_result(answer(request))
    return 0


if __name__ == "__main__":
    sys.exit(main())
