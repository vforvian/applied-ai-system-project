# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Real-world platforms like Spotify and YouTube blend two approaches. **Collaborative filtering** learns from the behavior of millions of users: likes, skips, replays, watch time, and which songs show up together on playlists. **Content-based filtering** looks at the songs themselves: audio features like tempo, energy, and valence, plus genre and mood tags. These systems first pull a few hundred candidates out of millions of items, then score and rank them, then adjust the final list for variety. My version is a small, purely content-based recommender, because the dataset has song attributes but no user behavior. It prioritizes **closeness to the user's taste, not extreme values**. Each song gets points for matching the user's favorite genre (+2.0, or +1.0 for a partial match like "indie pop" vs. "pop") and mood (+1.5), plus up to 2.0 points for how close its energy is to the user's target (`1 - |song.energy - target_energy|`) and up to 0.5 for matching their acoustic preference. A separate ranking step then sorts the scored songs and returns the top *k*, each with a short explanation of why it was chosen.

### Data Flow Plan

```
INPUT                          PROCESS (the loop)                 OUTPUT (the ranking)
─────                          ──────────────────                 ────────────────────
User prefs                     for each of the 20 songs:          sort by score, high → low
  genre, mood,          ──►      score_song(user, song)    ──►    (ties → lower song id)
  target energy,                   + genre / mood points          keep the top k
  likes_acoustic                   + energy closeness             return (song, score,
                                   + acoustic fit                         explanation)
data/songs.csv                     → (score, reasons)
  └ load_songs()
    (strings → numbers)
```

- **Input:** the user's taste profile, plus the catalog loaded from `data/songs.csv`. Numeric columns are converted from text to numbers once, when the file is loaded.
- **Process:** every song is judged on its own by `score_song`, using the recipe below. Each rule that adds points also adds a short reason (e.g. "mood match"), so the explanation always matches the score.
- **Output:** `recommend_songs` sorts all scored songs and returns the top *k*. The class version (`Recommender.recommend` / `explain_recommendation`) calls the same scoring logic, so both versions give the same results.

### Algorithm Recipe (final)

**Scoring rule (one song vs. one user):**

| Signal | Points | Rule |
|---|---|---|
| Genre match | **+2.0** | `song.genre == user.genre` |
| Partial genre match | **+1.0** | one genre contains the other (e.g. "indie pop" / "pop"), only if no exact match |
| Mood match | **+1.5** | `song.mood == user.mood` |
| Energy closeness | **0 to +2.0** | `2.0 * (1 - abs(song.energy - user.target_energy))` |
| Acoustic preference | **+0.5** | song is acoustic (`acousticness >= 0.5`) and the user likes acoustic, or song is not acoustic and the user doesn't |

Maximum possible score: **6.0**.

**Ranking rule (whole catalog):** score every song, sort from highest to lowest score (ties broken by song id), and return the top *k* with their reasons.

**Why these weights:**
- **Genre is the strongest single signal (2.0)**, but it's rare in this catalog: 20 songs cover 17 genres, so only lofi (3 songs) and pop / indie pop (3) ever produce more than one match. Genre picks out the first few results, and mood and energy order everything after them.
- **Mood (1.5) is close to genre, not half of it.** With the starter weight of +1.0, a "sad, low-energy metal" listener got *Iron Tide* (angry, 0.97 energy) as their #1 just because of the genre label. With 1.5, *Last Bus Home* (sad, 0.33 energy) ranks first, which fits the request for something sad and quiet.
- **Energy can be worth as much as a genre match (2.0)**, so it doesn't just break ties. A song 0.5 away from the target loses 1.0 point, which is enough to push a high-energy song below a calm one.
- **Acoustic (0.5) is a small nudge** that separates songs that are otherwise similar (e.g. *Library Rain* vs. *Midnight Coding*).
- Valence, tempo, and danceability are left out for now. They're candidates for the "add tempo or valence" experiment.

### Potential Biases

- **Genre can still win over a better fit.** A genre match (+2.0) can lift a song whose mood and energy are wrong. For a "happy pop" listener, *Gym Hero* (an intense workout track) ranks #3 mostly because it's labeled pop.
- **Exact-word matching misses similar moods.** "Chill", "relaxed", and "focused" count as completely different moods, so *Coffee Shop Stories* (relaxed jazz) gets no mood credit from a chill listener even though it fits.
- **The catalog favors some listeners.** Lofi and pop/indie pop have 3 songs each; the other 14 genres have just 1. Fans of those genres get one strong match and then a list sorted almost entirely by energy.
- **Very high- or low-energy songs are at a disadvantage.** Energy is scored by distance from the target, so mid-energy songs are never far from anyone. Songs at the extremes, like *Iron Tide* (0.97) or *Rain on Glass Harbor* (0.22), drop sharply for most users.
- **One genre and one mood per user.** The profile can't express "lofi *or* jazz" or a mood that changes with the time of day.
- **No variety check.** Nothing stops the top *k* from being near-copies of each other (e.g. two LoRoom lofi tracks).

Some prompts to answer:

- What features does each `Song` use in your system
  - For example: genre, mood, energy, tempo
- What information does your `UserProfile` store
- How does your `Recommender` compute a score for each song
- How do you choose which songs to recommend

You can include a simple diagram or bullet list if helpful.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

```
# e.g.:
# User profile: genre=indie, mood=chill, energy=low
# Recommendations:
#   1. ...
#   2. ...
#   3. ...
```

================================================================
                      Top 5 recommendations                     
================================================================
 Profile: genre=pop, mood=happy, energy=0.8
----------------------------------------------------------------

 1. Sunrise City  (Neon Echo)
    Score: 5.46 / 6.0  [##################..]
    pop · happy · energy 0.82
    Why:
      • genre match: pop (+2.0)
      • mood match: happy (+1.5)
      • energy 0.82 vs. target 0.80 (+1.96)

 2. Rooftop Lights  (Indigo Parade)
    Score: 4.42 / 6.0  [###############.....]
    indie pop · happy · energy 0.76
    Why:
      • similar genre: indie pop (+1.0)
      • mood match: happy (+1.5)
      • energy 0.76 vs. target 0.80 (+1.92)

 3. Gym Hero  (Max Pulse)
    Score: 3.74 / 6.0  [############........]
    pop · intense · energy 0.93
    Why:
      • genre match: pop (+2.0)
      • energy 0.93 vs. target 0.80 (+1.74)

 4. Concrete Crown  (Kilo Verse)
    Score: 1.96 / 6.0  [#######.............]
    hip-hop · confident · energy 0.78
    Why:
      • energy 0.78 vs. target 0.80 (+1.96)

 5. Fuego Lento  (Los Faroles)
    Score: 1.92 / 6.0  [######..............]
    latin · playful · energy 0.84
    Why:
      • energy 0.84 vs. target 0.80 (+1.92)

================================================================


>>> [CORE] High-Energy Pop

================================================================
                      Top 5 recommendations                     
================================================================
 Profile: genre=pop, mood=happy, energy=0.85, likes_acoustic=False
----------------------------------------------------------------

 1. Sunrise City  (Neon Echo)
    Score: 5.94 / 6.0  [####################]
    pop · happy · energy 0.82
    Why:
      • genre match: pop (+2.0)
      • mood match: happy (+1.5)
      • energy 0.82 vs. target 0.85 (+1.94)
      • non-acoustic sound fits your taste (+0.5)

 2. Rooftop Lights  (Indigo Parade)
    Score: 4.82 / 6.0  [################....]
    indie pop · happy · energy 0.76
    Why:
      • similar genre: indie pop (+1.0)
      • mood match: happy (+1.5)
      • energy 0.76 vs. target 0.85 (+1.82)
      • non-acoustic sound fits your taste (+0.5)

 3. Gym Hero  (Max Pulse)
    Score: 4.34 / 6.0  [##############......]
    pop · intense · energy 0.93
    Why:
      • genre match: pop (+2.0)
      • energy 0.93 vs. target 0.85 (+1.84)
      • non-acoustic sound fits your taste (+0.5)

 4. Storm Runner  (Voltline)
    Score: 2.38 / 6.0  [########............]
    rock · intense · energy 0.91
    Why:
      • energy 0.91 vs. target 0.85 (+1.88)
      • non-acoustic sound fits your taste (+0.5)

 5. Concrete Crown  (Kilo Verse)
    Score: 2.36 / 6.0  [########............]
    hip-hop · confident · energy 0.78
    Why:
      • energy 0.78 vs. target 0.85 (+1.86)
      • non-acoustic sound fits your taste (+0.5)

================================================================


>>> [CORE] Chill Lofi

================================================================
                      Top 5 recommendations                     
================================================================
 Profile: genre=lofi, mood=chill, energy=0.35, likes_acoustic=True
----------------------------------------------------------------

 1. Library Rain  (Paper Lanterns)
    Score: 6.00 / 6.0  [####################]
    lofi · chill · energy 0.35
    Why:
      • genre match: lofi (+2.0)
      • mood match: chill (+1.5)
      • energy 0.35 vs. target 0.35 (+2.00)
      • acoustic sound fits your taste (+0.5)

 2. Midnight Coding  (LoRoom)
    Score: 5.86 / 6.0  [####################]
    lofi · chill · energy 0.42
    Why:
      • genre match: lofi (+2.0)
      • mood match: chill (+1.5)
      • energy 0.42 vs. target 0.35 (+1.86)
      • acoustic sound fits your taste (+0.5)

 3. Focus Flow  (LoRoom)
    Score: 4.40 / 6.0  [###############.....]
    lofi · focused · energy 0.40
    Why:
      • genre match: lofi (+2.0)
      • energy 0.40 vs. target 0.35 (+1.90)
      • acoustic sound fits your taste (+0.5)

 4. Spacewalk Thoughts  (Orbit Bloom)
    Score: 3.86 / 6.0  [#############.......]
    ambient · chill · energy 0.28
    Why:
      • mood match: chill (+1.5)
      • energy 0.28 vs. target 0.35 (+1.86)
      • acoustic sound fits your taste (+0.5)

 5. Coffee Shop Stories  (Slow Stereo)
    Score: 2.46 / 6.0  [########............]
    jazz · relaxed · energy 0.37
    Why:
      • energy 0.37 vs. target 0.35 (+1.96)
      • acoustic sound fits your taste (+0.5)

================================================================


>>> [CORE] Deep Intense Rock

================================================================
                      Top 5 recommendations                     
================================================================
 Profile: genre=rock, mood=intense, energy=0.92, likes_acoustic=False
----------------------------------------------------------------

 1. Storm Runner  (Voltline)
    Score: 5.98 / 6.0  [####################]
    rock · intense · energy 0.91
    Why:
      • genre match: rock (+2.0)
      • mood match: intense (+1.5)
      • energy 0.91 vs. target 0.92 (+1.98)
      • non-acoustic sound fits your taste (+0.5)

 2. Gym Hero  (Max Pulse)
    Score: 3.98 / 6.0  [#############.......]
    pop · intense · energy 0.93
    Why:
      • mood match: intense (+1.5)
      • energy 0.93 vs. target 0.92 (+1.98)
      • non-acoustic sound fits your taste (+0.5)

 3. Neon Cathedral  (Pulse Theory)
    Score: 2.44 / 6.0  [########............]
    edm · euphoric · energy 0.95
    Why:
      • energy 0.95 vs. target 0.92 (+1.94)
      • non-acoustic sound fits your taste (+0.5)

 4. Iron Tide  (Black Meridian)
    Score: 2.40 / 6.0  [########............]
    metal · angry · energy 0.97
    Why:
      • energy 0.97 vs. target 0.92 (+1.90)
      • non-acoustic sound fits your taste (+0.5)

 5. Sunrise City  (Neon Echo)
    Score: 2.30 / 6.0  [########............]
    pop · happy · energy 0.82
    Why:
      • energy 0.82 vs. target 0.92 (+1.80)
      • non-acoustic sound fits your taste (+0.5)

================================================================


>>> [ADVERSARIAL] Conflicted: Sad but Hyped

================================================================
                      Top 5 recommendations                     
================================================================
 Profile: genre=blues, mood=sad, energy=0.9
----------------------------------------------------------------

 1. Last Bus Home  (Delta June)
    Score: 4.36 / 6.0  [###############.....]
    blues · sad · energy 0.33
    Why:
      • genre match: blues (+2.0)
      • mood match: sad (+1.5)
      • energy 0.33 vs. target 0.90 (+0.86)

 2. Storm Runner  (Voltline)
    Score: 1.98 / 6.0  [#######.............]
    rock · intense · energy 0.91
    Why:
      • energy 0.91 vs. target 0.90 (+1.98)

 3. Gym Hero  (Max Pulse)
    Score: 1.94 / 6.0  [######..............]
    pop · intense · energy 0.93
    Why:
      • energy 0.93 vs. target 0.90 (+1.94)

 4. Neon Cathedral  (Pulse Theory)
    Score: 1.90 / 6.0  [######..............]
    edm · euphoric · energy 0.95
    Why:
      • energy 0.95 vs. target 0.90 (+1.90)

 5. Fuego Lento  (Los Faroles)
    Score: 1.88 / 6.0  [######..............]
    latin · playful · energy 0.84
    Why:
      • energy 0.84 vs. target 0.90 (+1.88)

================================================================


>>> [ADVERSARIAL] Unknown Genre (k-pop)

================================================================
                      Top 5 recommendations                     
================================================================
 Profile: genre=k-pop, mood=happy, energy=0.8
----------------------------------------------------------------

 1. Sunrise City  (Neon Echo)
    Score: 4.46 / 6.0  [###############.....]
    pop · happy · energy 0.82
    Why:
      • similar genre: pop (+1.0)
      • mood match: happy (+1.5)
      • energy 0.82 vs. target 0.80 (+1.96)

 2. Rooftop Lights  (Indigo Parade)
    Score: 3.42 / 6.0  [###########.........]
    indie pop · happy · energy 0.76
    Why:
      • mood match: happy (+1.5)
      • energy 0.76 vs. target 0.80 (+1.92)

 3. Gym Hero  (Max Pulse)
    Score: 2.74 / 6.0  [#########...........]
    pop · intense · energy 0.93
    Why:
      • similar genre: pop (+1.0)
      • energy 0.93 vs. target 0.80 (+1.74)

 4. Concrete Crown  (Kilo Verse)
    Score: 1.96 / 6.0  [#######.............]
    hip-hop · confident · energy 0.78
    Why:
      • energy 0.78 vs. target 0.80 (+1.96)

 5. Fuego Lento  (Los Faroles)
    Score: 1.92 / 6.0  [######..............]
    latin · playful · energy 0.84
    Why:
      • energy 0.84 vs. target 0.80 (+1.92)

================================================================


>>> [ADVERSARIAL] Out-of-Range Energy (1.5)

================================================================
                      Top 5 recommendations                     
================================================================
 Profile: genre=edm, mood=euphoric, energy=1.5
----------------------------------------------------------------

 1. Neon Cathedral  (Pulse Theory)
    Score: 4.40 / 6.0  [###############.....]
    edm · euphoric · energy 0.95
    Why:
      • genre match: edm (+2.0)
      • mood match: euphoric (+1.5)
      • energy 0.95 vs. target 1.50 (+0.90)

 2. Iron Tide  (Black Meridian)
    Score: 0.94 / 6.0  [###.................]
    metal · angry · energy 0.97
    Why:
      • energy 0.97 vs. target 1.50 (+0.94)

 3. Gym Hero  (Max Pulse)
    Score: 0.86 / 6.0  [###.................]
    pop · intense · energy 0.93
    Why:
      • energy 0.93 vs. target 1.50 (+0.86)

 4. Storm Runner  (Voltline)
    Score: 0.82 / 6.0  [###.................]
    rock · intense · energy 0.91
    Why:
      • energy 0.91 vs. target 1.50 (+0.82)

 5. Fuego Lento  (Los Faroles)
    Score: 0.68 / 6.0  [##..................]
    latin · playful · energy 0.84
    Why:
      • energy 0.84 vs. target 1.50 (+0.68)

================================================================


>>> [ADVERSARIAL] Stringly-Typed Acoustic

================================================================
                      Top 5 recommendations                     
================================================================
 Profile: genre=metal, energy=0.9, likes_acoustic=False
----------------------------------------------------------------

 1. Iron Tide  (Black Meridian)
    Score: 3.86 / 6.0  [#############.......]
    metal · angry · energy 0.97
    Why:
      • genre match: metal (+2.0)
      • energy 0.97 vs. target 0.90 (+1.86)

 2. Fuego Lento  (Los Faroles)
    Score: 2.38 / 6.0  [########............]
    latin · playful · energy 0.84
    Why:
      • energy 0.84 vs. target 0.90 (+1.88)
      • acoustic sound fits your taste (+0.5)

 3. Storm Runner  (Voltline)
    Score: 1.98 / 6.0  [#######.............]
    rock · intense · energy 0.91
    Why:
      • energy 0.91 vs. target 0.90 (+1.98)

 4. Gym Hero  (Max Pulse)
    Score: 1.94 / 6.0  [######..............]
    pop · intense · energy 0.93
    Why:
      • energy 0.93 vs. target 0.90 (+1.94)

 5. Neon Cathedral  (Pulse Theory)
    Score: 1.90 / 6.0  [######..............]
    edm · euphoric · energy 0.95
    Why:
      • energy 0.95 vs. target 0.90 (+1.90)

================================================================


>>> [ADVERSARIAL] Empty Profile

================================================================
                      Top 5 recommendations                     
================================================================
 Profile: 
----------------------------------------------------------------

 1. Sunrise City  (Neon Echo)
    Score: 0.00 / 6.0  [....................]
    pop · happy · energy 0.82
    Why:
      • no strong matches

 2. Midnight Coding  (LoRoom)
    Score: 0.00 / 6.0  [....................]
    lofi · chill · energy 0.42
    Why:
      • no strong matches

 3. Storm Runner  (Voltline)
    Score: 0.00 / 6.0  [....................]
    rock · intense · energy 0.91
    Why:
      • no strong matches

 4. Library Rain  (Paper Lanterns)
    Score: 0.00 / 6.0  [....................]
    lofi · chill · energy 0.35
    Why:
      • no strong matches

 5. Gym Hero  (Max Pulse)
    Score: 0.00 / 6.0  [....................]
    pop · intense · energy 0.93
    Why:
      • no strong matches

================================================================

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



