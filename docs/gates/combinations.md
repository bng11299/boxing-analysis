# Gate: punch combinations in the report -- written 2026-10-28, before any combination number exists

Branch `feat/combinations`. Committed BEFORE the combination code is written and before any combination is
built from any label or detection; the commit hash of this file is the evidence of order. Nothing below
changes after a number exists. A later change is a new, dated section that says why, and the original
text stays. Design and alternatives: vault `Decisions/ADR-0019-punch-combinations.md`.

## Question
The report will say things like "favourite combination: jab-cross, 14 times; starts combinations with the
jab 70% of the time". On footage where every punch is labelled with its type, are those statements right
more often than the punch-level accuracy alone would give by chance, and do the headline statements (the
favourite combination, the punch that starts combinations) match the labels?

## The rule (fixed; chosen by Brendan 2026-10-28, not fitted to any clip)
- A **combination** is a maximal run of consecutive punches of ONE person in which each punch's time is at
  most **1.0 s** (`COMBO_GAP_S`) after the previous punch's. A run of one punch is a single, not a
  combination; combinations have length 2 or more.
- A run never crosses a round boundary, and in the per-boxer report never crosses an identity-segment edge
  (track id change, or frames the identity gave to the other boxer, to "unsure" or to nobody). For the two
  own clips below there is one person and one round, so neither break applies.
- A punch's time: a detection's wrist-speed peak frame (the classified CSV's `frame`); a label's instant
  (`punchfind.centers`: the middle of a dragged span, the clicked frame of a point label), both / fps.
- **Naming:** each punch by its type (jab, cross, lead hook, rear hook, lead uppercut, rear uppercut); a
  detection the classifier could not name, or an `unclear` label counted as a punch, is "?". A combination is
  **named** when none of its punches is "?". Classifier confidence is not used.
- Brendan chose 1.0 s over 0.75 s (2 x BoxingVI's median training punch length, 0.375 s over 3,778 clips) on
  2026-10-28: a round number with no measured basis, leaving more room above the finder's 0.4 s minimum gap.

## Known limitation, stated before the number
The finder (`results/punch_finder_v6.json`) reports at most one punch per 0.4 s, whichever hand threw it. PR
#40 measured that 40% (IMG_4406) and 32% (IMG_4395) of labelled punches follow the previous one within 0.4 s.
A labelled combination with such a gap cannot be recovered exactly by construction. The fraction of labelled
combinations with any gap under 0.4 s is computed and reported beside the result (no bar).

## Data (no circularity)
- **Clips:** Brendan's own footage IMG_4406 and IMG_4395, the two labelled clips (`data/own/<clip>.labels.json`),
  orthodox, `--unclear not-a-punch` (an unclear label is framing, not a punch: excluded from the label punches,
  as RESULTS row 6a/6b scores them). Nothing in the rule above was chosen by looking at these labels or at
  any detection on these clips; the session that wrote this did not compute any gap statistic on them.
- **Detections:** regenerated with RESULTS.md command C8 (`find_punches.py` then `classify_punches.py`, rule
  `results/punch_finder_v6.json`, extent `results/punch_extent.json`, classifier
  `weights/classifier/stgcnpp-boxingvi-seed0.json`) into a fresh folder. The result file records the classified
  CSVs' sha256, the classifier sha256 and the finder rule; the report uses the result only for that classifier,
  that rule and that gap.
- **Matching:** detections to labelled punches exactly as RESULTS row 6a/6b: `classify_punches.attribute`
  (nearest one-to-one matching, `find_punches.TOL_S` and `punchfind.POINT_TOL_S`).
- No held-out / tuning distinction is needed: nothing is tuned. IMG_5835 (unlabelled) and every participant
  video play no part.

## Metrics
Per clip, label combinations L (from the label punches) and found combinations F (from all detections).
- **Exact recovery.** A named label combination l_1..l_k is RECOVERED when a found combination f_1..f_m has
  m = k, every f_i is matched to l_i, and every f_i is named with l_i's type. Exact recall = recovered / named
  label combinations. Exact precision = found named combinations that recover one / found named combinations.
- **Structure recovery** (reported, no bar): the same without the type condition, over all label combinations.
- **Favourite:** among named combinations seen at least twice, the set of names with the highest count (ties
  all kept); empty when no named combination occurs twice. This is what the report prints.
- **Starter:** among combinations whose first punch is named, the set of first-punch types with the highest
  count (ties all kept). This is what the report prints as the punch combinations start with.

## Bars (fixed now; PASS = all of them)
1. **Better than chance on names.** Pooled over both clips, exact recall is strictly greater than the 97.5th
   percentile of a floor: the same computation after permuting the types among the NAMED detections within
   each clip ("?" stays "?"; timing, matching and grouping unchanged), 2,000 permutations,
   `numpy.random.default_rng(20261028)`. The floor keeps the classifier's type proportions and the finder's
   timing and asks only whether the types sit on the right punches.
2. **Favourite matches**, in EACH clip: the found favourite set is non-empty and is a subset of the label
   favourite set. A clip whose labels have no named combination seen twice fails this bar (nothing to match).
3. **Starter matches**, in EACH clip: the found starter set is non-empty and is a subset of the label
   starter set.

## What the report does with the verdict
The report reads the gate's result file (`score_combinations.py` writes it; format `boxingvi-combo-gate-1`)
only when its classifier sha256, finder rule and gap equal the report's own. PASS: it states the exact recall,
the floor and that the headline statements matched on both clips. FAIL: it states the same numbers and that
the combinations failed this check, so they are rough. No usable result: it says combinations are unmeasured.
In every case it states the 0.4 s blind spot and that a combination is right only if every punch in it was
found and named right. The per-boxer report adds that this was measured on solo footage, not two boxers.

## Expected
Uncertain, leaning FAIL on bar 1: a 2-punch combination needs four things right (two finds, two names), the
blind spot removes the fast ones, and n is small (94 labelled punches across both clips). A FAIL is reported
as a finding about how far punch-level accuracy carries to sequences, and the report keeps the combination
block with the failure stated in it.
