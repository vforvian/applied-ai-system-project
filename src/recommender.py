import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Algorithm Recipe weights (see README "Algorithm Recipe").
# Weight shift: energy doubled (2.0 -> 4.0), genre halved (2.0 -> 1.0).
# Partial genre credit stays at half of a full genre match (1.0 -> 0.5).
GENRE_POINTS = 1.0
PARTIAL_GENRE_POINTS = 0.5
MOOD_POINTS = 1.5
ENERGY_POINTS = 4.0
ACOUSTIC_POINTS = 0.5
# Best possible score: exact genre + mood + perfect energy + acoustic fit = 7.0.
MAX_SCORE = GENRE_POINTS + MOOD_POINTS + ENERGY_POINTS + ACOUSTIC_POINTS
ACOUSTIC_THRESHOLD = 0.5  # acousticness >= this counts as "acoustic"


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv into dicts with numeric columns converted to numbers."""
    int_fields = ("id",)
    float_fields = (
        "energy", "tempo_bpm", "valence", "danceability", "acousticness",
    )

    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            song: Dict = {key: value.strip() for key, value in row.items()}
            for field in int_fields:
                song[field] = int(song[field])
            for field in float_fields:
                song[field] = float(song[field])
            songs.append(song)

    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against the user; return (score, list of reasons)."""
    score = 0.0
    reasons: List[str] = []

    # Genre: +1.0 exact, +0.5 partial (e.g. "indie pop" vs. "pop").
    user_genre = str(user_prefs.get("genre", "")).strip().lower()
    song_genre = str(song["genre"]).strip().lower()
    if user_genre:
        if song_genre == user_genre:
            score += GENRE_POINTS
            reasons.append(
                f"genre match: {song['genre']} (+{GENRE_POINTS:.1f})"
            )
        elif user_genre in song_genre or song_genre in user_genre:
            score += PARTIAL_GENRE_POINTS
            reasons.append(
                f"similar genre: {song['genre']} (+{PARTIAL_GENRE_POINTS:.1f})"
            )

    # Mood: +1.5 exact match.
    user_mood = str(user_prefs.get("mood", "")).strip().lower()
    if user_mood and str(song["mood"]).strip().lower() == user_mood:
        score += MOOD_POINTS
        reasons.append(f"mood match: {song['mood']} (+{MOOD_POINTS:.1f})")

    # Energy: up to +4.0, scaled by how close the song is to the target.
    if "energy" in user_prefs:
        gap = abs(float(song["energy"]) - float(user_prefs["energy"]))
        target = float(user_prefs["energy"])
        energy_points = ENERGY_POINTS * max(0.0, 1.0 - gap)
        score += energy_points
        reasons.append(
            f"energy {song['energy']:.2f} vs. target {target:.2f}"
            f" (+{energy_points:.2f})"
        )

    # Acoustic: +0.5 if the song's sound matches the user's preference.
    if "likes_acoustic" in user_prefs:
        is_acoustic = float(song["acousticness"]) >= ACOUSTIC_THRESHOLD
        if is_acoustic == bool(user_prefs["likes_acoustic"]):
            score += ACOUSTIC_POINTS
            label = "acoustic" if is_acoustic else "non-acoustic"
            reasons.append(
                f"{label} sound fits your taste (+{ACOUSTIC_POINTS:.1f})"
            )

    return round(score, 2), reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Rank all songs by score; return top k as (song, score, explanation)."""
    # Judge every song: one (song, score, reasons) tuple per song.
    scored = [(song, *score_song(user_prefs, song)) for song in songs]

    # Rank: highest score first; ties go to the lower song id.
    # sorted() returns a new list, so the caller's `songs` order is untouched.
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]["id"]))

    return [
        (song, score, "; ".join(reasons) or "no strong matches")
        for song, score, reasons in ranked[:k]
    ]
