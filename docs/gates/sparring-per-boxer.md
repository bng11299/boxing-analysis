# Gate: per-boxer punch counts and attribution on the 10-22 sparring videos -- written 2026-11-02, before any label is scored

Branch `codex/sparring-labels` (PR #43). Committed BEFORE any sparring punch label is compared with any method
output; the commit hash of this file is the evidence of order. gg1022-v1 was labelled (v1, 2.5 h) before this
file was written, but no label has been read by any scoring code. Nothing below changes after a number exists.
A later change is a new, dated section that says why, and the original text stays.

## Question
On real two-person sparring, how far are the per-boxer punch counts from `analyse_clip.py --two-boxer` off,
and how much of that is attribution (the punch was found but given to the wrong boxer)? This is a measurement,
not a pass/fail gate on a method: every identity method already failed its own gate (headgear, glove colour,
linker), so the per-boxer numbers are reported as conditional on identity, and these labels are what lets
that error be measured instead of described.

## Data and handling
- The three face-blurred copies `gg1022-v1..v3` (study FA26-22-NG; ISSUES-LOG O63-O65). Labels, exports,
  scores and per-punch tables stay under `~/comps-participant-data/sparring-labels/`. Only aggregate numbers
  (counts, rates, intervals) enter the repo or `docs/RESULTS.md`; no frame numbers, no per-punch rows, no names.
- Ground truth = Brendan's Label Studio regions (`labelstudio/sparring-punch-labels.xml`), imported by
  `import_labels.py` (the frame checks at burned-in 100 and 2000 must pass, or the video is not scored).
- **Boxer A in the punch labels is the boxer A of `glove-identity/<clip>.labels.csv`** (v1 white gloves,
  v2 blue gloves, v3 white gloves). Brendan confirms this per video in the labelling notes before scoring.
- Tool suggestions (pre-filled regions) may be used to speed labelling. Every suggested region Brendan keeps is
  his label; the scores are reported with how many regions came from suggestions and how many were added,
  moved, retyped or deleted, because a suggested label biases toward the finder.

## Fixed before scoring (no choice below is tuned on these videos)
- Finder rule `config` default (`find_punches.DEFAULT_RULE`), classifier `stgcnpp-boxingvi-seed0`, extent default:
  exactly what `analyse_clip.py --two-boxer` uses on 2026-11-02.
- Stances A and B per video: written by Brendan in the labelling notes before scoring, from memory of the
  session. A mid-round stance switch (v1, ~2:22) is not modelled; type scores near it are reported as is.
- Identity inputs, each scored separately, both named in every row:
  1. **glove-colour** (failed its gate, `docs/gates/glove-identity.md`; verdict `fail`), the primary input.
  2. **tracking only** (track ids / linker chains, ungated), the baseline.
  An identity's internal names mean nothing until anchored. As in the glove gate, each method's A is fixed by
  the FIRST clear moment (`L`/`R`) of `glove-identity/<clip>.labels.csv` at which both its rows are named.
  The punch labels are never used to choose the naming.

## Scoring (per video, per identity input)
- Matching: `punchfind.match_nearest` with tolerance `completeness.TOL_S` = 0.1 s on spans (single-frame
  regions widened by `punchfind.for_matching`, `POINT_TOL_S` = 0.2 s). Label `unclear` counts as a punch for
  counts (a punch whose type is unnameable); `?`-boxer labels count as punches with no owner.
- **Whole-video count error (identity-free reference):** (detections - labels) / labels over all boxers.
- **Per-boxer count error:** for b in {A, B}, (punches the report gives b - labels with boxer b) / labels with
  boxer b. Report both signed values and their mean absolute value. Labels with `?` and punches on unsure
  segments are counted and reported, never assigned.
- **Attribution accuracy:** over matched pairs where the label's boxer is A or B AND the report's boxer is A or
  B: share where they agree, Wilson 95% interval (`headgear.wilson`). Also reported: matched pairs where the
  report said unsure (identity coverage) and where the label said `?`.
- **Detection:** recall and precision of all detections against all labels (identity-free), as context.
- Type accuracy on matched, correctly attributed pairs is reported as secondary, with the stance caveat above.
- Pooled over videos: sum the counts first, then compute; per-video rows always shown beside the pool.

## What it can show, and what it cannot
- Three videos, one session, two-to-four people: a description of these videos, not a generalisation.
- Attribution accuracy near 0.5 means identity is no better than a coin; the count error then cannot be read
  per boxer at all. Whole-video minus per-boxer error separates finder error from identity error only roughly.
- Labels made from suggestions are biased toward agreement with the finder (reported, see above).

## Partial labelling and the deadline
- Labelling ends Thu 2026-11-05, 23:59. A video is scored if it is labelled to the end, OR if Brendan has
  written in the notes, before scoring, the last burned-in frame he labelled fully; then only frames
  [1, that frame] are scored for that video (labels and detections both), it must cover at least 3 minutes,
  and every row carries "partial (x of y min)". A video with neither is recorded as **unscored**.
- If no video qualifies, the sparring per-boxer counts are recorded in `docs/RESULTS.md` as unscored.
