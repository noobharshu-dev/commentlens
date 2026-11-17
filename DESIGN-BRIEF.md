# CommentLens: UI/UX design brief

Use the seven PNG images in this folder as the visual reference. Match them closely. Where this file and an image disagree, the image wins. The images use example data.

| Image | Screen |
|---|---|
| 01-landing.png | Start screen: headline, link field, comment slider, sample button, preview of results |
| 02-analyzing.png | Loading: step list, progress, mood bar filling as batches finish |
| 03-dashboard.png | Results: score, mood bar, palettes, summary, themes, timeline, table |
| 04-explore-drawer.png | Negative palette open, one comment selected, detail panel on the right |
| 05-ask-the-comments.png | Question answered with cited comments |
| 06-mobile.png | 390px wide layout |
| 07-design-system.png | Colors, type, components, rules |

## The one memorable idea: the mood bar
A single full-width bar split into Positive, Neutral, Mixed, and Negative. Each segment's width matches its comment count. It sits directly above the palette pills. On the loading screen it fills as batches finish. Everything else stays quiet so this bar is what people remember.

## Colors (light theme only)
| Token | Hex | Use |
|---|---|---|
| paper | #F5F7F6 | Page background |
| surface | #FFFFFF | Table, drawer, inputs |
| ink | #16211F | Text |
| muted | #667370 | Notes and labels |
| line | #E1E6E3 | 1px borders |
| accent | #0F766E | Active pills, primary buttons, progress |
| accent-tint | #E4F1EF | Selected row, hover |
| positive | #2F8A5B | Sentiment only |
| negative | #C2453D | Sentiment only |
| neutral | #A3ABA8 | Sentiment only |
| mixed | #D4A017 | Sentiment only |
| question | #3B6FD4 | Question label and dot |

## Type
- Inter for everything. Inter Display for titles, headings and big numbers.
- Landing title 60px/1.04 weight 650, letter-spacing -0.025em. Video title 34px. Mood score 84px, letter-spacing -0.04em. Section heading 20px weight 650. Body 14 to 15px. Notes 13px muted.
- Sentence case everywhere. No all-caps labels. Tabular numbers for every count (`font-variant-numeric: tabular-nums`).

## Layout rules
- Content width 1180px, left aligned. Sections 56px apart. Base unit 8px.
- Radius: pills fully round, inputs and buttons 8px, table and drawer 12px. 1px borders. No shadows, no gradients.
- Metrics are plain numbers on the page, not a row of identical cards. The table is the only boxed container, plus the chart.
- Dashboard order: video header, score and mood bar, palette pills, summary and theme bars, timeline, table.
- Selecting a table row opens a 470px detail panel on the right with a faint scrim over the table.
- Mobile: stack everything, palette pills scroll sideways, comments become a simple list with a bottom action bar.

## Palettes (the filter pills)
Order: All, Positive, Negative, Neutral, Questions, Suggestions, Bug reports, Praise, Duplicates, Replies, Spam, Top liked. Rude or toxic appears only when "Show toxic comments" is on. Unclassified appears only when its count is above 0.
- Label then count, for example `Negative 220`.
- Sentiment pills carry a small colored dot. The active pill is filled accent with white text and no dot.
- Clicking the active pill returns to All.
- Under the pills: "A comment can belong to more than one group, so counts can add up to more than the total."

## Motion
One moment: the mood bar fills while batches finish. Everything else only reacts to a click: pill fill, panel open, "Copied" for 2 seconds on the copy button. Respect `prefers-reduced-motion`.

## Copy rules
Plain verbs and sentence case: Analyze comments, Re-analyze, Download report, Copy reply. No emoji, no exclamation marks, no marketing words. Errors say what happened and what to do next.

## Do not
Dark mode, emoji, decorative icons, all-caps labels, purple or glow effects, gradients, drop shadows, identical rounded cards everywhere, mid-dot separated meta lines.

## Building this in Streamlit
- Set `.streamlit/config.toml` to the light theme with the accent, paper, surface and ink colors above, then add the rest in `styles.css`.
- Hide Streamlit's menu, header and footer with CSS.
- Palette pills: `st.pills` or `st.segmented_control`, styled with CSS to match the pill component.
- Mood bar: render with `st.markdown(..., unsafe_allow_html=True)` using flex segments sized by count. It is a visual, and the pills below it do the filtering.
- Detail panel: `st.dataframe(on_select="rerun")` then show the selected comment in a right-hand column or `st.dialog`.
- Charts: Plotly with transparent background, gridlines #E6EAE7, positive and negative colors above, no title inside the figure.

## Paste this into Antigravity
```
Redesign the CommentLens UI to match the reference images in /ui-kit (01 to 07) and the rules in /ui-kit/DESIGN-BRIEF.md.
Keep all existing functions and logic. Change only the layout, styles.css, .streamlit/config.toml and the UI code in app.py.
Light theme only. Use the exact colors, type sizes, radii and copy from the brief.
Build the signature mood bar (segments sized by count) above the palette pills, with the 0 to 100 score on its left.
Replace the row of identical metric cards with large plain numbers as in 03-dashboard.png.
Add the right-hand detail panel from 04-explore-drawer.png and the mobile layout from 06-mobile.png.
No emoji, no gradients, no shadows, no all-caps labels.
When done, run the app, compare each screen to its reference image, and list any differences.
```
