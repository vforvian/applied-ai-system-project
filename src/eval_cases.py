"""Predefined inputs shared by the test suite and the evaluation harness."""

# (request, ids of songs that are a correct answer)
GOLDEN = [
    ("mellow songs for studying on a rainy day", {2, 4, 9}),   # lofi
    ("something relaxed and chill, kind of jazzy", {7}),       # Coffee Shop Stories
    ("songs to get pumped up at the gym", {3, 5}),             # Storm Runner, Gym Hero
    ("sad breakup songs", {16}),                               # Last Bus Home
    ("date night slow jams", {17}),                            # Velvet Hours
    ("beach vacation vibes", {19}),                            # Island Time Radio
    ("night drive through the city", {8}),                     # Night Drive Loop
    ("rnb", {17}),                                             # Velvet Hours (r&b)
]

# Requests that plain retrieval gets wrong. Each has a rule every pick must
# satisfy, so the harness can check whether a mode handles it.
SAD_MOODS = {"sad", "melancholic"}
HARD_CASES = [
    ("NOT sad, something happy",
     "no sad/melancholic picks",
     lambda song: song["mood"] not in SAD_MOODS),
    ("something to fall asleep to",
     "every pick is low energy (< 0.45)",
     lambda song: song["energy"] < 0.45),
    ("music for a funeral",
     "every pick is low energy and not upbeat (valence < 0.6)",
     lambda song: song["energy"] < 0.5 and song["valence"] < 0.6),
]

INVALID_INPUTS = ["", "   ", "1234 !!!", "x" * 301]
NO_MATCH_INPUTS = ["Gregorian chant with bagpipes", "asdf qwerty"]
