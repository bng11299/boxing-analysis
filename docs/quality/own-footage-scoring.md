## `feat/own-footage-scoring` -- 2026-10-19

Brendan's labels on IMG_4395 and IMG_4406 are single-frame clicks (94 of 94), not start-to-end
spans, and he used "unclear" for FRAMING (hands out to measure or block), not for a punch nobody
could name. Acceptance: (1) a point label matches within half a punch; (2) "unclear" can be
scored as a non-punch, with what the finder and classifier did with those marks reported; (3) a
per-hand finder option, measured against the pooled rule on own footage AND V6; (4) suites green.

### Iteration 1
- **Build:** `punchfind.POINT_TOL_S = 0.2` and `for_matching` (a single-frame label is widened so
  it matches within 0.2 s; spans keep 0.1 s); `--unclear {punch,not-a-punch}` on find_punches and
  classify_punches (framing marks leave the punch set; detections on them are false positives and
  are reported with the type the classifier gave them); `detect(..., per_hand=True)` and
  `find_punches --per-hand` (each hand's own peaks, the 0.4 s gap within a hand; threshold and gap
  unchanged). classify_punches now accepts a row that is its hand's speed peak above the rule's
  threshold, rather than requiring that hand to be the faster one.
- **The 0.2 s point tolerance was chosen after seeing the scores at 0.1 s and 0.2 s**; both are
  below (iteration 2). A click marks one instant in a ~0.4 s punch, and the finder reports the speed
  peak.
- **Test:** hands 51, classify 211; 8 mutants, one survivor (a click read as a one-frame span)
  given a fixture: a click on a still frame within 0.2 s of a punch's peak.
- **Real-world facts, own footage (orthodox; framing as not-a-punch; point rule):**

  | | IMG_4406 pooled | IMG_4406 per hand | IMG_4395 pooled | IMG_4395 per hand |
  |---|---|---|---|---|
  | punches found (recall) | 35/47 0.745 | 40/47 0.851 | 35/40 0.875 | 40/40 1.000 |
  | detections real (precision) | 35/41 0.854 | 40/59 0.678 | 35/42 0.833 | 40/57 0.702 |
  | type right on matched | 0.886 (35) | 0.825 (40) | 0.600 (35) | 0.700 (40) |
  | framing marks flagged | 1 of 5 (cross) | 1 of 5 | 1 of 2 (jab) | 1 of 2 |

  Why the finder misses: 40% and 32% of these punches follow the previous one within 0.4 s, but
  only 11% and 0% follow one by the same hand. The per-hand rule finds them (recall +0.11, +0.13)
  and pays in precision (-0.18, -0.13): the other wrist's retraction is counted too. V6 says the
  same: per hand recall 0.761 vs 0.741, precision 0.361 vs 0.465, F1 0.489 vs 0.571 (second half).
  **Pooled stays the default.** A cross-hand duplicate rule would be a new parameter; it must be
  tuned on data that is not these clips.
- **Framing:** 2 of 7 framing marks drew a detection, named cross and jab, as Brendan expected; most
  false positives come from elsewhere (6 and 7 on the pooled finder, 1 of each on framing). (This
  count was only a lower bound when written; see iteration 2. Here it is exact.)

### Iteration 2 (review: FIX FIRST, 3 blocking)
- **B1, framing count was a lower bound shown as a count.** Punches are matched first and may take
  the one detection a framing mark also had. Both scripts now report `flagged_as_punches_at_least`
  (marks on detections no punch took) and `_at_most` (marks with any detection in reach), and print
  "a to b" when they differ. On Brendan's clips they agree everywhere (pooled 1/1 and 1/1, per hand
  1/1 and 1/1), so "2 of 7" stands as exact.
- **B2, `labelled_length_s_median` counted clicks as zero-length punches** and was NaN (a crash in
  `json.dumps(allow_nan=False)`) when every label was framing. Now the median of start-to-end
  labels only, None when there are none; `point_labels` counts punch labels only, in both scripts.
- **B3, the 0.1 s scores were claimed and absent.** Re-scored (framing as not-a-punch, pooled):

  | | IMG_4406 +-0.1 s | IMG_4406 +-0.2 s | IMG_4395 +-0.1 s | IMG_4395 +-0.2 s |
  |---|---|---|---|---|
  | punches found (recall) | 28/47 0.596 | 35/47 0.745 | 27/40 0.675 | 35/40 0.875 |
  | detections real (precision) | 28/41 0.683 | 35/41 0.854 | 27/42 0.643 | 35/42 0.833 |
  | type right on matched (floor) | 0.821 of 28 (0.321) | 0.886 of 35 (0.343) | 0.667 of 27 (0.444) | 0.600 of 35 (0.371) |
  | framing marks flagged | 1 of 5 | 1 of 5 | 0 of 2 | 1 of 2 |

  The +-0.2 s column reproduces iteration 1 exactly. The wider tolerance is generous by design: it
  credits a detection anywhere in the punch a click sits in, so it is an upper reading of recall
  for point labels; the 0.1 s column is the strict one. Type accuracy barely moves (0.82 to 0.89,
  0.67 to 0.60): the extra matches are not easier punches.
- **Also from the review:** a negative test for the provenance check's local-maximum clause (a row
  above the threshold on the rising slope is refused); a control that `min_gap_s=0` sees more peaks
  in a fast combination than 0.4 s does.
- **Test:** hands 53, classify 218; 6 new mutants, 6 killed (one survivor on the first run, the
  classifier's upper bound, given a fixture where the bounds differ).
- **Deferred (not blocking):** recording `per_hand` in the punches CSV / classified sidecar (a CSV
  format change that the output branch reads); a hand-specific local peak for the point perfect-
  finder reading.

### Iteration 3 (review of 014762a by the code-reviewer agent and by Codex: 3 blocking) -- 2026-10-21
**Iteration 2 got two things wrong, and this corrects the record:**
- **"They agree everywhere, so '2 of 7' stands as exact" did not follow.** The "lower bound" was the
  framing count under ONE maximum punch matching, the greedy's. Which detections a maximum matching
  takes is a tie-break, so it was not a bound. Counterexample (60 fps, tol 6, point_tol 12):
  detections at 88 and 112, a punch click at 100 (reach 88-112), a framing click at 120 (reach
  108-132). The greedy gives the punch 88 and reported 1 to 1. A punch matching that takes 112 is
  just as large and leaves the mark nothing, so the true minimum is 0.
- **B2 read as fixing the all-framing crash, but find_punches.py still crashed.** (Iteration 2's
  "in both scripts" qualified `point_labels`; `labelled_length_s_median` exists in classify only.)
  find_punches.py crashed when every label was a framing mark: `score()` returns `count_ratio = NaN` with no
  punch spans, and `json.dumps(allow_nan=False)` raised after printing precision and recall 0.000.
  No score.json was written.
- **"6 of 6 mutants killed" missed one.** In every fixture `at_most == marks`, so the mutant
  `at_most = len(framing)` survived.

**The fixes:**
- **`punchfind.framing_bounds`, shared by both scripts.**
  - *Lower bound:* framing marks matched maximally against only the detections in NO punch span's
    reach ([start - tol, end + tol] of the widened span). No punch matching can take those, so the
    bound holds for every matching, maximum or not.
  - *Upper bound:* framing marks against all detections, unchanged.
  - The lower bound's detections are reported as `lower_bound_frames` (find_punches) and
    `lower_bound_named_as` (classify); these keys are new in iteration 4.
- **`match`'s docstring now says the greedy is optimal in COUNT only.** Which detections it leaves
  over is a tie-break, not a fact about the footage.
- **find_punches with no punch spans** writes recall, F1 and count ratio as null, skips the chance
  floor, and prints "no punches labelled". `score()` is unchanged; the V6 evaluation depends on it.
- **The `per_hand` record (iteration 2's deferred item).** find_punches writes `<clip>.punches.json`
  beside the CSV: clip, CSV sha256, rule, per_hand. classify records `finder_per_hand` (provenance)
  and `finder.per_hand` (score) only when the sidecar matches the CSV's bytes and the clip, and
  otherwise null. The CSV header is unchanged; a test asserts it, because feat/output-report reads it.

**Tests:** hands 72, classify 228. New cases:
- the 88/112/100/120 counterexample, at the helper and through both scripts (a jab-cross per-hand
  fixture, with controls that the greedy takes the jab and each detection is within reach of the
  intended click);
- a framing mark with no detection in reach (`at_most < marks`), in both suites;
- reach boundaries: 112 vs 113 for a point label, 106 vs 107 for a drag;
- the framing-only run in find_punches, asserting the null scores, no chance floor on a 20 s clip,
  and the printed line;
- the sidecar's negative cases.

**Mutants (round 1, on 8a5acdf; each applied to a copy of the tree; both suites run):**

| # | mutant | killed |
|---|---|---|
| M1 | lower bound counted against all detections (lower = upper) | yes (hands 6, classify 2 failures) |
| M2 | lower bound = what the greedy left (iteration 2's code) | yes (3, 1) |
| M3 | upper bound = number of marks (the survivor from iteration 2) | yes (3, 1) |
| M4 | upper bound = lower bound | yes (5, 2) |
| M5 | punch reach without match's tol | yes (6, 1) |
| M6 | punch reach on unwidened point labels | yes (5, 1) |
| M7 | lower-bound set taken from the upper-bound matching | yes (3, 2) |
| M8 | find_punches: framing-only score left with NaN | yes (hands crashes: NaN not JSON compliant) |
| M9 | find_punches: chance floor not skipped with no punches | yes (hands 1) |
| M10 | find_punches: F1 left at 0 with no punches | yes (hands 1) |
| M11 | find_punches: framing bounds ignore the punches | yes (hands 2) |
| M12 | classify: framing bounds ignore the punches | yes (classify 2) |
| M13 | classify: sidecar hash not checked | yes (classify 1) |
| M14 | classify: sidecar clip not checked | yes (classify 1) |
| M15 | classify: sidecar per_hand type not checked | yes (classify 1) |
| M16 | find_punches: sidecar always says pooled | yes (hands 1, classify 3) |

**Re-measured on Brendan's clips** (orthodox; framing as not-a-punch), in-process, on iteration 4's
final code. Every finder and type number reproduces iteration 2 exactly:

| | IMG_4406 0.2 s | IMG_4395 0.2 s | IMG_4406 0.1 s | IMG_4395 0.1 s | IMG_4406 per hand 0.2 s | IMG_4395 per hand 0.2 s |
|---|---|---|---|---|---|---|
| punches found | 35/47 | 35/40 | 28/47 | 27/40 | 40/47 | 40/40 |
| detections real | 35/41 | 35/42 | 28/41 | 27/42 | 40/59 | 40/57 |
| type right, matched | 0.886 of 35 | 0.600 of 35 | 0.821 of 28 | 0.667 of 27 | 0.825 of 40 | 0.700 of 40 |
| framing flagged (sound bounds) | 1 to 1 of 5 | 1 to 1 of 2 | 1 to 1 of 5 | 0 to 0 of 2 | 1 to 1 of 5 | 1 to 1 of 2 |
| lower-bound detection named | cross | jab | cross | -- | cross | jab |
| per_hand recorded | false | false | false | false | true | true |

- **The framing counts did not change, and now they are shown, not assumed.** On these clips, the
  old greedy-leftover count also equals the new lower bound in every column (recomputed side by
  side). The two framing detections (frames 1557 and 1436) are out of every punch's reach. So
  "2 of 7 at 0.2 s" (1 of 7 at 0.1 s) is exact. Iteration 2's conclusion was right; its argument
  was not.
- **The upper bound is below the mark count in every column** (1 of 5, 1 of 2): on real data it is
  not the trivial "all marks".

**Review: FIX FIRST** (code-reviewer agent, 1 blocking). See iteration 4.

### Iteration 4 (review of 8a5acdf: FIX FIRST, 1 blocking) -- 2026-10-21
- **What the reviewer checked and found correct:**
  - **Soundness, by brute force.** 6,000 random instances, enumerating every punch matching. The
    instances covered 0-7 detections with duplicates, points and drags, overlapping spans, spans
    clipped at frame 0, tol in {0, 2, 6} and point_tol in {0, 3, 6, 12}. No violation of
    `at_least <= min` or `max <= at_most`.
  - **The bounds hold for non-maximum matchings too.** The docstring now says so.
  - **Loose but sound.** The lower bound is loose in about 10% of random cases and the upper in
    about 14%. Exact min/max over punch matchings is not done (optional).
  - **Both scripts pass the same spans.**
  - **Nothing on any local branch reads the renamed keys.**
- **B1, the sidecar reader crashed on deeply nested JSON.** `"[" * 30000 + "]" * 30000` is 60 KB,
  under the 64,000-byte cap, and makes `json.loads` raise `RecursionError`. That is not a
  ValueError, so classify died on a valid CSV. Now caught, as `lsimport.load_export` already does,
  with a negative test and controls (the input is under the cap; json itself raises).
- **Also fixed:**
  - **The lower-bound set overclaimed.** It is ONE maximum framing matching's detections, not every
    detection that is certainly on a mark (`[118, 122]` around a mark at 120 lists one). Keys
    renamed to `lower_bound_frames` and `lower_bound_named_as`; docstrings and the printed line now
    say "the detections the lower bound counted".
  - **Precision 0/0.** With framing-only labels AND no detections, precision is null, not 0.0.
    Tested on a still clip.
  - **Mixed provenance.** The sidecar's rule must equal this run's `--rule`, or `per_hand` is
    unknown. Tested with a threshold changed by 0.5.
  - **A vacuous test.** The "holds NaN" half of a sidecar test could not fail: the bool, clip and
    rule checks reject that file anyway. Dropped, not reworded.
- **Not done (noted):**
  - feat/analyse-clip's `OUTPUTS` does not publish `punches.json`, so a standalone classify of a
    published out-dir records `per_hand` as null. Decide it at that merge.
  - Exact framing min/max over punch matchings.
- **Tests:** hands 73, classify 232, pipeline 260, labels 50, ingest 109, env 36.
- **Mutants (round 2, on the final code, all 19 re-run; M8 and M10 re-targeted at the edited lines):**
  **19 of 19 killed.** M1-M16 again, with the same suites failing as in round 1, plus:

  | # | mutant | killed |
  |---|---|---|
  | M17 | classify: RecursionError from a nested sidecar not caught | yes (classify crashes: RecursionError) |
  | M18 | classify: sidecar rule not checked | yes (classify 1) |
  | M19 | find_punches: precision 0, not null, with no detections and no punches | yes (hands 1) |
- **Final review (c271d20): SHIP**, no blocking findings. Its re-runs of M5, M17, M18 and M19
  reproduced the results above. Left open (non-blocking):
  - find_punches still reports precision 0.0 via `score()` when punches are labelled but nothing was
    detected (0/0, as in the framing-only case). Override it in the report, not in `score()`.
  - `find_punches.load_rule` accepts NaN/Infinity (no `parse_constant`); this predates the range.
  - an always-true `sx is not None` conjunct in one classify check.
  - this entry's misquote of iteration 2, corrected above.
- **Merge state.** PR #22 is CONFLICTING with its base after #14's fixes (d2e05cb, 5b1488f). A trial
  merge in a throwaway clone conflicts in `classify_punches.py` and this log. Not merged here: no
  gate needed it. At the stack step, keep #22's per-hand peak check, #14's duplicate-row guard
  (threshold and local-peak checks), and `classify.read_clip_window` at both read sites.

### Iteration 5 -- merging #14's fixes in (2026-10-21, base `5b1488f`)
Brendan asked for the conflict to be resolved here, keeping #22's per-hand peak check, #14's
duplicate-row guard, and `classify.read_clip_window` at both read sites. Merged (not rebased, so
no force-push).
- **The peak check: the two sides disagreed in substance, not only in text.**
  - **#14** refuses a row unless its hand is the FASTER one, and judges the threshold and the
    local peak on the faster hand's speed. That is right for the pooled finder.
  - **#22** judges both on the row's OWN hand, because the per-hand finder reports each hand's
    peaks and the other hand may be faster.
  - **Kept #22's**, plus #14's "that hand has speed" clause, which #22's lacked; it matters only
    for a rule whose threshold is 0.
  - **Cost:** #14's accepted rows are a strict subset of the merge's. For a pooled list, three
    kinds of row the pooled finder cannot emit now pass where #14 refused them:
    - a slower hand's own peak;
    - a faster hand's peak that is not a peak of the two hands' max curve (left [0, 5, 3] and
      right [6, 1, 0]: the max curve [6, 5, 3] has no peak at frame 1);
    - the second hand on an exact speed tie.

    #14's comment already called the check "necessary, not sufficient". (Correction, from the
    review: the reason first given here for not using the per_hand sidecar, "recorded, not
    trusted", argues only against LOOSENING by it. Applying #14's pooled check when the sidecar
    says pooled would fail safe. It was not done, because Brendan asked for #22's check; it is an
    open option.) Done in iteration 6, at Brendan's request.
- **The read sites:** `read_clip_window` at the pipeline's read and at the perfect-finder reading,
  as #14 had them. The IoU line keeps #22's exclusion of point labels.
- **#14's duplicate-row guard** (`read_punches`) merged without conflict. Per-hand rows at one
  frame for both hands are different (frame, hand) keys, so they are not duplicates.
- **The log:** #14's iteration 8 now ends its own section, before this one.
- **Mutants on the resolution** (copies of the tree, test_classify):

  | # | mutant | first run | after the new tests |
  |---|---|---|---|
  | X1 | drop the "has speed" clause | survived: `clip_cut`'s `peak_width` refused the row later, so the outcome matched | killed: the test now pins that the PEAK CHECK refuses it (a still frame at threshold 0) |
  | X2 | drop the threshold clause | killed | killed |
  | X3, X4 | allow a faster previous / next frame | killed | killed |
  | X5 | judge on the faster hand's speed (#14's signal) | killed | killed |
  | X6 | restore #14's faster-hand clause | **survived**: no fixture had a per-hand row whose hand is the slower one | killed: both hands punch together, the right at half reach; the right's rows (left faster) must be accepted |
  | X7 | pipeline read back to `cut_span(clip_window)` | **survived** | killed |
  | X8 | perfect-finder read back to `cut_span(clip_window)` | **survived** | killed |
  | X9 | IoU includes point labels | killed | killed |
  | X10 | allow duplicate (frame, hand) rows | killed | killed |
  | X11 | local peak judged on the faster hand's curve, threshold on its own (found by the review) | **survived**: in the slower-hand fixture both hands peak on one frame | killed: the right hand's punches 2 frames later, so at its peak the left is faster AND on its slope; accepted, while a row on the right's own rising slope is still refused |

  - **Why X7 and X8 survived:** #14's frame-rate tests call `read_clip_window` directly, so no
    script-level test pinned either read site. A spy on `classify.read_clip_window` now runs
    classify on a clip with fewer than 5 punches (the training cut, where the two reads differ).
    It asserts that every finder row goes through it, and that each named labelled punch does too
    (3 + 2 calls).
- **Tests:** hands 73, classify 262 (232 + 19 from #14 + 11 new), pipeline 260, labels 50, ingest
  109, env 36.
- **Review of the merge (979ee07, code-reviewer agent): SHIP**, no blocking findings. It diffed the
  merge against both parents: nothing was lost except the deliberate lines. It found the X11
  survivor and the incomplete cost statement above, both fixed in the follow-up commit. Left open:
  - applying #14's pooled check when the sidecar says pooled (above; done in iteration 6);
  - for clips under 5 punches, the printed window length is `cut.length_s` (8/24 s, "0.33 s") where
    the docstring says 0.375 s, and the provenance writes `"per": "clip"` beside
    `source: "training"`. Both predate the merge.
- **Re-measured on the merged code:** every number in iteration 3's table is unchanged, including
  the framing bounds and per_hand. Both clips have far more than 5 punches, so they use the clip's
  own cut, which `read_clip_window` reads exactly as before. #14's change is to the under-5
  fallback only.

### Iteration 6 -- the pooled check, when the sidecar says pooled (2026-10-21)
Brendan's call, on iteration 5's open option. When the verified sidecar says `per_hand: false`,
classify_punches also applies #14's pooled check. Otherwise it keeps the own-hand check.
- **The checks are now two functions,** so each case can be tested on three-frame arrays:
  - `own_hand_peak`: every row, any finder. That hand has speed, reaches the threshold, and no
    neighbouring frame of that hand is faster.
  - `pooled_peak`: only when `finder_per_hand` is False. The hand is the faster one (`np.argmax`,
    so the left on an exact tie, as `punchfind.detect` names it), and the threshold and the local
    peak are judged on the two hands' max.
- **Why it is safe to let the sidecar choose:** the sidecar only TIGHTENS. True and None
  (missing, stale, forged, another clip or rule) both get the own-hand check, which every finder
  passes, so a bad sidecar can never loosen the check. A forged `false` could at worst refuse the
  CSV. The sha256 catches a stale sidecar but authenticates nothing: anyone holding the CSV can
  compute it, so the safety is that `false` only tightens. `finder_per_hand`'s docstring says so;
  it no longer says "never trusted for anything else".
- **Tests (classify 262 to 274):**
  - Unit cases, each passing the own check and refused by the pooled one:
    - the review's left [0, 5, 3] / right [6, 1, 0], a faster hand's peak that is not a peak of
      the max;
    - a slower hand's own peak;
    - an exact tie, refused for the right hand;
    - plus the threshold, the strict neighbourhood, and the first and last rows.
  - Through the script, on IMG_5151 found by the POOLED finder (left only), plus a right-hand row
    at one of the right's own peaks, the sidecar re-hashed to the edited CSV:
    - refused with a verified pooled sidecar, by the pooled check's own message;
    - accepted with a per-hand sidecar;
    - accepted with no sidecar.
  - A control that the pooled finder's own list, with its sidecar, passes.
- **Mutants: 20 of 20 killed.** X1-X11 were retargeted at `own_hand_peak` and re-run:

  | # | mutant | killed by |
  |---|---|---|
  | P1 | pooled check never applied | the pooled-sidecar refusal |
  | P2 | pooled check applied to every list | 4: per-hand rows where the other hand is faster |
  | P3 | an unknown sidecar treated as pooled | the no-sidecar acceptance |
  | P4 | faster-hand clause dropped | the slower-hand unit case (3) |
  | P5 | a tie accepted for either hand | the tie case |
  | P6 | local peak on the max dropped | the [0, 5, 3] / [6, 1, 0] case (3) |
  | P7, P8 | a faster previous / next frame of the max allowed | the [0, 5, 3] case; the neighbourhood case |
  | P9 | threshold on the max dropped | the unit threshold case only. In `main` it is redundant: the own check runs first, and the argmax hand's speed IS the max |

- **Real data:** the stricter check accepted every row of the pooled finder's real output. That is
  41 rows on IMG_4406 and 42 on IMG_4395, at both point tolerances, each with a verified
  `per_hand: false` sidecar, so the check ran. Every number in iteration 3's table is unchanged.
- **Review of f3f5eba (code-reviewer agent): FIX FIRST, 1 blocking, a test gap; the code was
  correct.**
  - **Nothing tested that a row on a plateau is accepted.** `find_peaks` reports the middle of a
    flat top, whose neighbour is EQUAL. So `>` to `>=` in any of the four neighbour clauses would
    refuse genuine finder output, the worst failure this check can have, and all 274 still passed.
  - **The fix:** a 3-long and a 2-long plateau, with the row taken from scipy's own `find_peaks`
    (what `detect` calls), must pass both checks.
  - **Mutants Q1-Q4 (`>` to `>=` in each neighbour clause): 4 of 4 killed**, each by the plateau
    cases.
- **What the reviewer verified by construction:** no false refusal of genuine output.
  - A fuzz of `detect` on 20,000 random tracks with gaps, quantised speeds (so plateaus and ties),
    thresholds 0-3 and gaps 0-0.2 s: of 36,434 pooled and 79,956 per-hand rows, 0 refused by
    either check.
  - 150 synthetic poses through `find_punches.main` and `classify_punches.main`, at 24, 30 and
    60 fps, pooled with verified sidecars and per hand: 0 of 1,796 rows refused.
  - **Why none can be:** `find_peaks` never reports the first or last sample of the segment it is
    given, so the neighbour comparison never crosses a frame-id gap.
- **Also from the review:**
  - The hash wording no longer reads as authentication: it detects staleness only.
  - **The edges test.** Renamed, and the first rename overclaimed. It said "no wrap-around", but no
    case could catch one: the re-review's `i > 0` to `i >= 0` mutants (W1 in `own_hand_peak`, W2 in
    `pooled_peak`) passed all 276.
    - **The fix:** row 0 of `[6, 3, 9]` must pass both checks. Under W1 and W2 it is compared with
      `[-1]`, the 9, and refused. W1 and W2 are killed.
    - **The last row:** it cannot wrap. Without its guard, `i + 1` raises IndexError. Its case
      (`[3, 6]`, row 1) now runs for both checks.
    - **No genuine output is affected:** `find_peaks` never reports row 0.
  - **Noted, not changed:** both checks accept a row on a track's or segment's first or last row,
    which the finder never emits. Tightening that is possible but out of scope.
- **Re-review of 7f2dafc: FIX FIRST**, for the overclaiming test name above; Q1-Q4 confirmed killed.
- **Re-review of f971dfd: SHIP**, no blocking findings.
  - W1 and W2 killed. Removing either last-row guard (`i + 1 < len(...)`) crashes the suite with
    IndexError on the new `[3, 6]` row-1 cases.
  - The test names and this entry now claim only what the tests show.
  - Optional, not done: record a crash as a named FAIL instead of a traceback, and give W1 and W2
    one check each.
- **Tests:** hands 73, classify 277, pipeline 260, labels 50, ingest 109, env 36.

