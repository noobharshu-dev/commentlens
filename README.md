# CommentLens

**Your comment section, read for you.**

A YouTube video can collect thousands of comments in a day. CommentLens reads them all, sorts them by what they mean, and gives the creator a clear view of what the audience loved, what confused them, and what they want next.

Paste a link. Click a group. Read the people who watched.

---

## A day in the creator's life

It is 9 a.m. The video went live last night. There are 4,200 comments, and the first 60 are all "first!" and emojis. Somewhere in the rest is the one comment that explains why the second half lost viewers, and another that asks for the part two everyone is waiting for.

Scrolling will not find them. CommentLens will.

It fetches the comments, has Google Gemini read each one, and shows the result as a row of groups with counts: 2,000 positive, 900 negative, 412 questions, 287 suggestions. Click **Negative** and every negative comment is on screen. Click **Questions** and the answers you owe your audience are in one list.

---

## What the creator sees

**The palette row.** Every group is one click away, each with its count. Click once to open it, click again to go back to all comments. Duplicates are marked, not deleted, so you can see when a hundred people said the same thing.

**The mood reading.** A single score from 0 to 100 with a plain label: mostly negative, mixed, or mostly positive. Next to it, a short written summary of what viewers liked, what they criticized, and what they asked for.

**The theme bars.** The most-mentioned topics in suggestions and criticism, such as "audio quality," "pacing," or "part 2 request." Click a theme to see the comments behind it.

**The timeline.** Positive and negative reactions by day. If the mood dropped on the day after a sponsor segment, you will see it.

**The detail drawer.** Open any comment to see its full text, every label it carries, a one-line summary, and a suggested reply you can copy with one click. A comment can be a question and negative at the same time, and it will appear in both groups.

**Ask the comments.** Type a question in plain words: "What do viewers want next?" or "Are there bugs?" The answer cites the comment IDs it used, so every claim can be checked.

**Five worth reading.** A short list of non-toxic comments that are praise or ideas, picked to cover different themes.

**The creator report.** One click produces a Markdown file with your stats, themes, quotes, and suggested replies, ready to share with an editor.

**Export.** Download everything, or only the group you are looking at. The CSV is saved so that Hindi and Bengali text opens correctly in Excel.

---

## How the reading works

1. **Gather.** The YouTube Data API returns comments and replies, up to the limit you choose.
2. **Tidy.** Empty comments are removed. Repeated comments are kept and counted.
3. **Read.** Gemini reads comments in batches of 40 and returns structured labels: sentiment, category, theme, language, priority, and a reply. Sarcasm counts as negative. Hinglish is understood.
4. **Check.** Every answer is checked against the expected format. Failed batches are retried, and anything still unread goes to an Unclassified group, so nothing is silently lost.
5. **Count.** Totals, percentages, the mood score, and the timeline are calculated on your machine.

Toxic comments are hidden by default. Turn on **Show toxic comments** only when you need to see them.

---

## Built with

| Part | Tool |
|---|---|
| Interface | Streamlit |
| Reading comments | Google Gemini API (`google-genai`) |
| Fetching comments | YouTube Data API v3 |
| Tables and charts | pandas, Plotly |
| Settings | python-dotenv |

---

## Run it in five minutes

```bash
git clone https://github.com/ashadul-20/commentlens.git
cd commentlens
pip install -r requirements.txt
copy .env.example .env
python -m streamlit run app.py
```

Add your keys to `.env`:

```
GEMINI_API_KEY=your_gemini_key
YOUTUBE_API_KEY=your_youtube_key
```

Open http://localhost:8501.

No keys yet? Click **Load sample video (demo)** on the first screen and explore the full dashboard with a built-in set of 150 multilingual comments.

**Where to get the keys**

- Gemini: Google AI Studio, aistudio.google.com
- YouTube: Google Cloud Console, enable *YouTube Data API v3*, then create an API key

---

## Project layout

```
app.py              the screen you see
core/youtube.py     reads links and fetches comments
core/cleaning.py    tidies comments and marks repeats
core/classifier.py  talks to Gemini and checks its answers
core/analysis.py    counts, groups, score, themes, timeline
core/creator.py     quotes, answers, and the creator report
core/export.py      CSV files for Excel
core/cache.py       remembers each analysed video
tests/              unit tests
```

Run the tests with `python -m pytest tests/test_all.py -v`.

---

## Your data

Comment text goes to Google Gemini so it can be read. Results are saved only in the local `cache/` folder on your computer. Your keys stay in `.env`, which Git never uploads.

---

## Roadmap

- **Compare two videos** side by side, sentiment against sentiment.
- **Reply drafts you can edit** before copying.
- **Alerts** when the mood of a new upload drops sharply.

---

## License

MIT. Use it, change it, and share it.

Built for a hackathon, for creators who deserve to hear their audience without losing a week to scrolling.
