# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**TuneScout 1.0**

---

## 2. Intended Use

**Goal:** TuneScout suggests 5 songs that match what you ask for. It doesn't predict what you'll play. It just scores how close each song is to your request.

**Intended use:** A classroom project for learning how recommenders score and rank songs.

**Not intended for:** Real users or real music apps. It only has 20 songs and doesn't learn from listening history.

---

## 3. How the Model Works

You give it a genre, a mood, an energy level (0 to 1), and whether you like acoustic songs. Every song gets points, up to 7:

- **Energy:** up to 4 points. The closer to your target, the more points.
- **Mood:** 1.5 points for an exact match.
- **Genre:** 1 point for a match, or 0.5 for a close one (like "indie pop" for pop).
- **Acoustic:** 0.5 points if the sound fits your taste.

It shows the 5 highest scores, with the reasons for each. I started with genre at 2 and energy at 2. Then I doubled energy and halved genre as an experiment.

---

## 4. Data

- 20 songs: 10 from the starter file, plus 10 I added.
- Each song has a genre, mood, energy, tempo, valence, danceability, and acousticness. Only the first three and acousticness are used.
- 17 genres and 16 moods, so most of them have just one song.
- The songs are made up, not real tracks.

---

## 5. Strengths

- Songs that match everything come first. Chill Lofi got Library Rain with a perfect 7.
- Opposite listeners get opposite lists. Pop and Lofi share no songs.
- Every pick shows why it was chosen.

---

## 6. Limitations and Bias

Energy now matters more than genre. So the system groups people by how energetic they want their music, not by what they like. In my tests, 19 of 38 genre fans got a top pick from a different genre. For example, every high-energy fan got the rock song Storm Runner, even lofi and jazz fans. Fans of rare or extreme styles, like metal, rarely see their songs.

---

## 7. Evaluation

I tested 3 normal profiles (High-Energy Pop, Chill Lofi, Deep Intense Rock) and 5 tricky ones (sad but high-energy, K-pop, energy 1.5, "False" typed as text, and an empty profile). Pop and Lofi got completely different lists, which makes sense. Pop and Rock got very similar lists, because both want loud music. Gym Hero showed up in 5 of the 8 lists. It's loud and labeled pop, so it only loses the mood points for a Happy Pop fan. The tricky profiles found bugs. For example, "False" was read as yes, and an empty profile still got 5 songs.

---

## 8. Future Work

1. Raise genre back to 2 points, so genre fans see their genre.
2. Give partial credit for similar moods, like "sad" and "melancholic."
3. Check the input. Reject energy above 1, read "False" as no, and warn when a profile is empty.

---

## 9. Personal Reflection  

My biggest learning moment came when I doubled the energy weight and every score went up. At first that looked like an improvement, but bad matches went up just as much as good ones, so a higher score didn't mean a better recommendation. AI tools helped me write code fast, come up with tricky test profiles, and find the "False" bug I would have missed. But I had to double-check them. I pushed for a real answer on whether the new weights were "more accurate," and I made sure the numbers in this card matched my terminal. It surprised me that a few simple point rules, plus a "Why" list, felt like a real app, even though Gym Hero showed up for almost everyone who wanted loud music. The system wasn't understanding taste, just adding up points. Next, I'd write down the songs each profile should get and test against that list, add credit for similar moods, and use tempo and danceability in the score.
