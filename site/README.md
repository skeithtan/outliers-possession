# The demo site

This folder is a static web page that shows how our engine labels the match. It tells the same
story as the technical document, in the same order:

1. The whole match on one ribbon.
2. At a glance: the plain-language summary, the five key numbers of the PDF's glance page (the
   learned models' gains over the rule engine, this match, the public window) and the compliance
   statement.
3. The approach, and why: the metric, the submitted engine step by step (clean tracking, the 50
   features, the network, the per-frame model, the blend, the Viterbi decoder, the fallback) on a
   pipeline diagram and synthetic panels, and each choice with the alternative we rejected.
4. Missing and noisy tracking: ball gaps, defects added in training, the still ball, the cleaning
   that the submitted engine keeps from the rule engine.
5. Results: the learned models' gains on matches they never saw (provider left out, PFF
   benchmark, sealed PFF), then the rule engine as the reference, with its details folded away.
   Then the label-free checks on this match, with the public window last, as a record.
6. The possession rules: where the submitted engine draws the line and its settings, then the
   rule engine, the fallback, step by step on synthetic scenes, with its rules and settings.
7. Limitations and what next, with the public transitions as a pipeline check.
8. What we tried and did not use: the learned models first, then the rule engine.
9. Reproduction and the team.

The page shows no development history and no internal version names. Its copy follows the
team's guide for public text: the glossary, the rules, the map of the sections that the page
shares with the technical document, and the procedure for an engine change. The guide is in the
private repository. Two rules are special to the page:

- The value chips of the engine section show the value. The constant's name is in the tooltip.
  Only the constants table names the constants. Its developer notes stay hidden until the visitor
  switches them on.
- The organisers' data spells the clubs "Barcelona" and "Atletic Club". The page shows the clubs'
  own names, through `teamDisplayName` in `js/util/format.js`.

The page has no build step. It uses plain HTML, CSS and ES modules, and D3 7.9.0 for the charts.

## Run it on your computer

1. Open a terminal in the repository.
2. Run `cd site && python -m http.server 8000`.
3. Open `http://localhost:8000/` in a browser.

Do not open `index.html` from the file system. The browser blocks the data requests on a
`file://` address.

## The two modes

The page reads one or two data bundles. The schema of both bundles is in `data/SCHEMA.md`.

| mode | bundle | what the engine section shows | top-bar badge |
|---|---|---|---|
| public | `data/public/` only | synthetic scenes built from the real constants | `replay: synthetic illustration` |
| local | `data/public/` and `data/local/` | the real tracking replay and our full-match labels | `replay: real tracking (local)` |

The public bundle is committed. GitHub Pages can serve it after an administrator enables publication.
It holds frame-level detail only inside
the public window, frames `11185` to `18684`, plus per-minute aggregates, the organisers'
baseline labels, the scores, the vetoes, the constants and the summary numbers of the external
matches. It also holds the ledger files `ladder.json` and `cycles.json`. The build of the
technical document reads them. The page does not show them.

The local bundle is never committed. Git ignores `data/local/`. It holds the tracking positions
and our full-match labels. The page looks for it at start. If it is missing, the page runs in
public mode. If its md5 differs from the public bundle, the page refuses it and shows a banner.

## Rebuild the bundles

Run these commands from the repository root after the scorer and the forensics ran on the
version that you want to show. The first command also reads our records and working notes, so it
runs only in our private repository.

```bash
python -m tools.viz.build_site --version <name> --results evaluation/results/<name>.csv
python -m tools.viz.build_replay --version <name> --results evaluation/results/<name>.csv
python -m tools.viz.guard site/data/public
```

The first command writes `data/public/`. It also copies the summary numbers of our file of
external numbers into `data/public/external_summary.json`. The second command writes
`data/local/`. The guard exits 1 if
a public file holds a position, a long per-frame series or a frame outside the public window.

## The no-positions rule

The hosted page must never show a tracking position, a velocity, a distance or our full-match
labels. Two organiser rules decide this: their `RULES.md` and
`evaluation/results/README.md`. The sections that draw real tracking read only `data/local/`.
No module embeds data, and no module reads `data/` or `cache/` of the repository.

## Keys

| key | action |
|---|---|
| `A` | turn arena mode on or off |
| `F` | turn fullscreen on or off |
| `Esc` | leave arena mode |
| Ctrl and scroll, or pinch | zoom the ribbon |
| drag on the ribbon | move the zoomed ribbon |

The engine section adds its own replay keys. Its module documents them.

## Arena mode

Arena mode is for a large screen with nobody at the keyboard. Open the page with `?arena=1`, or
press `A`. The type grows by 1.25 times, the scroll bar disappears, and the page moves to the next
step every 9 s. After the last step it starts again at the top. A wheel or a touch hands the page
back to the viewer.

## D3 and the offline fallback

`index.html` loads D3 7.9.0 from cdnjs with a subresource integrity hash. If that request fails,
the page loads `vendor/d3.min.js`, which is the same file. The arena machine can then run with no
network. To check the fallback, block `cdnjs.cloudflare.com` in the browser developer tools and
reload.

## Motion and screen sizes

If the system asks for reduced motion, every animation jumps to its final state. The layout works
from a phone 375 px wide to a 3840 by 2160 screen. Below 900 px wide the figure sits above the
text. Below 700 px wide each row of a table becomes a card.

## Files

| path | content |
|---|---|
| `index.html` | the page, the text of every step and the static tables: choices, tracking, rules, settings, limitations, ideas not used |
| `css/tokens.css` | colours, type sizes and spacing |
| `css/base.css`, `css/sections.css` | layout and section styles |
| `js/main.js` | loads the bundles and mounts the sections |
| `js/data.js` | bundle loaders and RLE helpers |
| `js/state.js` | shared state: playhead, active step, arena |
| `js/scroller.js` | scroll steps and arena mode |
| `js/ribbon.js` | the full-match ribbon on a canvas |
| `js/util/spells.js` | a copy of the scorer's spell matching, checked against `scores.json` |
| `js/sections/*.js` | one module for each section with data: hero, glance, problem, engine, results, sanity, config, forensics, repro, team |
| `dev/fake_bundle.py` | writes a synthetic public bundle for development, when `data/public/` does not exist |
