"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

try:
    from .recommender import load_songs, recommend_songs, MAX_SCORE  # python -m src.main
except ImportError:
    from recommender import load_songs, recommend_songs, MAX_SCORE   # python src/main.py


# Core taste profiles: three distinct, realistic listeners.
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

# Adversarial / edge-case profiles: designed to probe weak spots in score_song.
ADVERSARIAL_PROFILES = {
    # Conflict: the only "sad" song is low-energy (0.33). Does mood or energy win?
    "Conflicted: Sad but Hyped": {"genre": "blues", "mood": "sad", "energy": 0.9},
    # Genre not in the catalog, but "pop" is a substring of "k-pop" -> partial credit.
    "Unknown Genre (k-pop)": {"genre": "k-pop", "mood": "happy", "energy": 0.8},
    # Out-of-range energy: every gap >= 0.5, so energy barely discriminates.
    "Out-of-Range Energy (1.5)": {"genre": "edm", "mood": "euphoric", "energy": 1.5},
    # String "False" is truthy -> bool("False") is True, so this user gets acoustic songs.
    "Stringly-Typed Acoustic": {"genre": "metal", "energy": 0.9, "likes_acoustic": "False"},
    # Empty profile: every song scores 0, so ranking falls back to song id order.
    "Empty Profile": {},
}


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    for group, profiles in (("CORE", PROFILES), ("ADVERSARIAL", ADVERSARIAL_PROFILES)):
        for name, user_prefs in profiles.items():
            print(f"\n\n>>> [{group}] {name}")
            recommendations = recommend_songs(user_prefs, songs, k=5)
            print_recommendations(user_prefs, recommendations)


WIDTH = 64


def print_recommendations(user_prefs: dict, recommendations: list) -> None:
    """Print the ranked recommendations as a readable terminal report."""
    profile = ", ".join(f"{key}={value}" for key, value in user_prefs.items())

    print()
    print("=" * WIDTH)
    print(f" Top {len(recommendations)} recommendations".center(WIDTH))
    print("=" * WIDTH)
    print(f" Profile: {profile}")
    print("-" * WIDTH)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        filled = round(score / MAX_SCORE * 20)
        bar = "#" * filled + "." * (20 - filled)
        print(f"\n {rank}. {song['title']}  ({song['artist']})")
        print(f"    Score: {score:.2f} / {MAX_SCORE:.1f}  [{bar}]")
        print(f"    {song['genre']} · {song['mood']} · energy {song['energy']:.2f}")
        print("    Why:")
        for reason in explanation.split("; "):
            print(f"      • {reason}")

    print()
    print("=" * WIDTH)


if __name__ == "__main__":
    main()
