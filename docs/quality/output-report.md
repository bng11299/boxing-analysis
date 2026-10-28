## `feat/output-report` -- 2026-10-09

The report a boxer reads: `<clip>.report.xlsx` (Punches, Rounds, Reliability, About) and
`<clip>.summary.txt`, from the classified punches. Stacked on `feat/classify-detected`.
Acceptance criteria: (1) the summary is fixed templates and holds exactly the numbers they
inserted; (2) every reliability statement comes from a measured number, linked to the
model that named the punches by hashes, and the summary says when nothing was measured on
your footage; (3) rounding never flips a claim; (4) inputs are validated and the workbook
holds no formulas; (5) all suites green in a clean clone.

Threat sketch. Assets: the honesty of what the user reads. Entry points: classified CSV and
its sidecar, manifest, reference run, finder rule, score JSON, `--rounds`. Trust boundary:
files on disk between tools. Top threats: a number-backed but misleading sentence; a
reliability table borrowed from a different model than the one that named the punches.

Real-world fact checked before building: the reported run's held-out confusion matrix
(results/stgcnpp.json) gives cross +28%, lead uppercut -41%, hand (type-implied) 98.3% --
the figures the project plan states.

### Iteration 1
- **Test:** 65 checks; 8 guards reverted, all caught. The first run caught my own arithmetic
  in a hand-built matrix (a row summing to 50, not 100): the code was right.
- **Review: FIX FIRST.** B1: the finder's 74% recall was stated but not its V6 count ratio
  (648 flagged for 406 labelled, 1.6x), so the totals read as an under-count. B2: "71% from
  your lead hand" came from the finder's wrist while "right 98%" was the classifier's
  type-implied hand -- a figure attached to a reading it never measured. B3: rounding flipped
  claims ("1 punch, 0 a minute"; "0% rear hooks (1)"; "dropped by 8%, from 10 to 10").
  B4: the per-class-F1 match linked two JSON files, never the CSV to the model, and could
  not fail in practice. Also: hardcoded "V2/V5/V9"; "too little of you was in view" for
  every unclassified reason; "between rounds" for punches before the first; floats like
  `1_0` accepted; fractional confusion cells truncated; the provenance check ran on one
  summary only.
- **Security:** bounded JSON reads, NaN refused, formula guard at write time; folded in above.

### Iteration 2
- **Fix:** finder sentence states recall and the count ratio, and that the error on your
  footage is unknown. Lead share by punch TYPE over named punches; type/wrist disagreements
  counted; the 98% tied to the type-implied hand. Rates under 10 keep a decimal; 0%/100%
  only when exact; shares by largest remainder; a rate change only beyond 2 SE of a Poisson
  count, worded from the printed rates. `classify_punches.py` writes
  `<clip>.classified.json` (CSV sha256, classifier sha256, stance, rule, pose frames); the
  report checks CSV -> sidecar -> manifest (classify's own schema) -> per-class F1
  recomputed from the matrix, and an own-footage score must name the same classifier.
  Inputs: decimal regex, integer cells, known notes only, one stance per file.
- **Test:** 110 checks; the provenance check runs on every summary variant and caught a
  real slip on its first run ("benchmark video V6" put an unrecorded digit in the text).
  9 new guards reverted, all caught.

### Iteration 3 -- the extent's columns, caveats, and real reports (2026-10-11 to 10-12)
- **Fix / build:** the report reads `punch_start_frame` / `punch_end_frame` and shows each
  punch's length; a named punch may carry only the two caveats `classify_punches.py` writes
  (wrist dropout beside the peak; longer than the classifier can be shown), an unnamed one only
  a known reason; the summary counts punches whose start and end are unsure. A contract test
  ties the report's caveat list and CSV header to the writer's.
- **Test:** 122 checks. Same numbers in a clean clone.
- **Real reports, after `classify`'s confidence-rule fix:** IMG_4395, 42 punches, 57 a minute:
  31% jabs, 29% crosses, 21% lead hooks, 14% rear uppercuts, 5% lead uppercuts; type and wrist
  disagree on the hand for 12. IMG_4406, 41 punches, 63 a minute: 39% jabs, 20% crosses, 17%
  rear uppercuts, 12% lead hooks, 12% lead uppercuts; disagree for 7. Before the fix the same
  reports led with 63% and 41% lead uppercuts: the report was faithful to a broken input, and
  nothing in it said so beyond the hand-disagreement count (19 and 21 of 41).
- **Residual, open:** the summary does not yet lead with a warning when type-and-wrist hand
  agreement is near chance; it would have flagged the broken input on its own.

### Iteration 4 -- the per-clip cut in the report (2026-10-13)
- Per-punch caveat notes are gone with the per-clip window (a punch's window no longer depends
  on its own peak width): a named punch carries no note, and the "start and end are unsure"
  sentence is removed. The About sheet states how the clip was cut, from the sidecar, validated
  field by field: the clip's median punch length and how many punches set it, or that there
  were too few punches and the training clips' median was used.
- **Test:** 130 checks. Same numbers in a clean clone.
- **Real reports:** IMG_4395 read over 0.41 s (39 punches; 3 left out), 42 punches: 38% jabs,
  24% crosses, 14% lead hooks, 14% rear uppercuts, 10% lead uppercuts. IMG_4406 read over
  0.39 s (41 punches), 41 punches: 41% jabs, 17% crosses, 17% rear uppercuts, 15% lead hooks,
  10% lead uppercuts. Unmeasured against labels.

### Iteration 5 -- the summary leads with a warning (2026-10-19)
Closes iteration 3's open residual. Acceptance: (1) the summary's first sentence is a fixed-template
warning whenever type-hand vs wrist agreement is not above the chance agreement for those proportions
(`classify.hand_agreement`) by a one-sided exact binomial test at 0.05, and that test is the only rule;
(2) every number in it goes through `_Writer`, and the provenance check passes on every variant; (3) the
Reliability sheet carries the agreement, chance and p; (4) the old broken inputs warn, and today's two
clips are reported as they fall; (5) suites green in a clean clone.
- **Build:** `stats.binomial_greater_p` (exact tail, summed in log space so n up to the row cap does not
  overflow); `report.hand_check` in the facts; the warning template; a "this clip's check" block under
  the per-type rows of the Reliability sheet; the rule stated on the About sheet. With no named punch
  there is no punch type to warn about, so no warning.
- **Test:** 186 checks. Hand-worked tails (1/16, 1/32, 56/1024, 11/1024, 0.104, 0.51), an exact
  Fraction oracle sharing no code, 3,000 random cases against scipy (worst 3e-13, scratch); 8 of 10
  at chance 1/2 warns (p 0.055), 9 of 10 does not (0.011); edges: no named punch, all one side agreeing
  (100% vs 100%, p 1) and disagreeing (0 of 9, chance 0), one constant reading, one punch, southpaw.
  13 mutants (a tuned threshold, a tuned margin, chance fixed at 1/2, the warning last, P(X > k),
  two-sided, numbers not recorded...), all caught.
- **Review: FIX FIRST.** B1 the About sheet said the warning fires when agreement is "no more often
  than chance"; IMG_4395 agrees 62% against 49% and warns, so About contradicted the clip in front of
  it. Fixed ("not significantly more often") with a test. Also taken: p under 0.1 printed to three
  decimals (0.055 had read "p = 0.05, not under 0.05"); the sheet keeps three significant figures, so
  6.83e-05 is not shown as 0. 188 checks; 14 mutants, all caught.
- **Security:** no new input; the binomial validates k, n and p (negative tests); its loop is bounded
  by MAX_ROWS; no cell is a formula. No findings.
- **Real-world facts.** Old 0.3 confidence rule, as recorded: IMG_4395 22 of 41 against 0.482, p 0.293;
  IMG_4406 20 of 41 against 0.509, p 0.666. **Both warn.** (0.482 is not reachable with 22 of 41
  agreeing, nearest 0.483; it is iteration 3's figure for 23 of 42, p 0.243, which also warns.) Today
  (`results/punches`, 2026-10-13): IMG_4395 26 of 42 (61.9%) against 49.4%, p **0.071: still warns**;
  IMG_4406 33 of 41 (80.5%) against 50.4%, p 0.00007: no warning. Both reports regenerated; after the
  warning, each summary is identical to the one before.
- **Residual, accepted:** chance is plugged in from the same clip's proportions. Conditioning on both
  margins, the exact test is one-sided Fisher; the binomial is mostly the more cautious of the two
  but can pass a balanced table Fisher would not (n = 42, (16, 9, 6, 11): binomial 0.0500, Fisher
  0.065). Neither real clip changes (Fisher 0.088 and 0.00005). When one reading is constant the check
  cannot tell good from bad and always warns; over hundreds of punches a modest excess over chance
  passes. Both follow from the test being the only rule.

### Iteration 6 -- a floor, both tests, and when the check cannot confirm anything (2026-10-20, vault ADR-0015)
Brendan, on iteration 5's caveats: add an agreement floor ("start 70%, want to work up to 80"); show both tests
("use both tests to output both results and see"); for a clip concentrated on one hand, say so ("there appears
to be an extreme concentration ... if this is not expected, checking the footage is recommended") instead of
"may be wrong"; say the check sees the hand only. Acceptance: (1) agreement under `HAND_FLOOR` (0.70) warns
whatever the tests say; (2) the binomial and Fisher one-sided exact p are both stated, and the clip passes only
if both are under 0.05; (3) when full agreement on the clip's type split could not pass both tests, the summary
leads with a caution (concentrated, or too few punches); (4) the passing summary states both p-values and that
a jab read as a lead hook passes the check; (5) suites green in a clean clone.
- **Build:** `stats.fisher_greater_p` (scipy's `fisher_exact`, one-sided); `hand_check` verdicts in order:
  under the floor -> warn; both tests pass -> pass; full agreement could pass -> warn; otherwise concentrated,
  or too_few (under 7 punches at 0.05). The Reliability row carries both p, the floor and a one-cell result.
- **Test:** 222 checks. Fisher worked by hand (1/10, 26/252, 4/286, 4/56, 3/66) and against an integer
  hypergeometric oracle; the two tests disagreeing in both directions (7 of 8: binomial 9/256 passes, Fisher
  4/56 fails; 11 of 12: binomial 0.054 fails, Fisher 3/66 passes); 68 of 100 failing on the floor alone; IMG_4406
  at an 80% floor passing by one punch. 26 mutants, 2 survived: the float floor (`k < HAND_FLOOR * n`) and a
  binomial-only "could pass". The second got a 5-punch fixture; the first I judged equivalent.
- **Review: FIX FIRST.** B1 scipy returns Fisher's exact 1/20 as 0.04999999999999999: on (12, 2, 0, 2) the
  summary said "p = 0.050 (Fisher); only the Fisher test is under 0.05". B2 "the commonest jabs (6)" picked a
  tie by alphabet. Also: the float floor is not equivalent in general (0.68 at 51 of 75); the "70 of 100"
  test could not see floats; stance went unmentioned where it matters; "That is under..." had no clear subject.
- **Fix:** every comparison with alpha goes through `_under_alpha` (a p within a billionth of alpha is not
  under it); a p under alpha that would print as 0.050 prints "just under 0.05"; every tied type is named
  ("most often jabs and lead hooks (6 each)"); agreement no higher than chance adds that a wrong stance swaps
  lead and rear; "The 62% agreement is under the 70% ..."; floor patches in tests restored in `finally`. 234
  checks; 30 mutants, all caught.
- **Security:** no new input; Fisher validates its cells (negative tests); no cell is a formula. No findings.
- **Real-world facts.** IMG_4395: 26 of 42 (62%) against 49.4%; binomial 0.071, Fisher 0.088; under 70%:
  warns on all three. IMG_4406: 33 of 41 (80.5%) against 50.4%; binomial 0.00007, Fisher 0.00005; passes, and
  still passes at 80% by one punch (32 of 41 would warn). Old 0.3-rule inputs: all under 70%, all warn. Both
  reports regenerated.
- **Residual, accepted:** the floor is chosen, not measured. The check is blind within a hand (jab vs lead
  hook); a within-hand cue needs its own test against labels.

### Iteration 7 -- the four Codex findings (2026-10-21; feat/output-report's iteration 5, merged here)
Codex (10-20) found four ways the report could state a reliability that does not describe
the punches in front of the user; each was confirmed by reading before fixing.
Acceptance criteria: (1) a CSV classified at another fps is refused, at the sidecar AND at
every row's time, by a CLI test that changes only the fps; (2) an own-footage score is
refused unless it names the classified CSV's sha256, which `classify_punches.py --labels`
now writes; (3) the finder's held-out score is refused unless the sidecar's rule (variant,
threshold, min_gap_s) is `--rule`'s; (4) a reference matrix that is not, cell for cell, the
matrix the classifier's manifest pins is refused (transpose included), and
`train_classifier.py` pins it; (5) all suites green in the working tree and a clean clone.
Threat sketch unchanged (files on disk between tools; reliability borrowed from another run).

- **Fix:** (1) `report.check_pose` compares the sidecar's `pose_frames` and `fps` with the
  pose; `read_classified(..., fps=)` (required, keyword-only) refuses a frame past the clip
  and a time more than 2 ms from frame / fps, as `classify_punches.read_punches` does. (2)
  The score records `csv_sha256`; `own_reliability` takes the sidecar's (already verified)
  hash and refuses a mismatch, or a score with no hash. (3) `finder_reliability(rule, used)`
  compares the three settings exactly. (4) The manifest's `held_out.confusion` is the saved
  model's matrix (`run_stgcnpp` saves after scoring); the report requires equality and says
  when the reference is the transpose. Per-class F1 is symmetric under transpose, so the old
  F1 check could not see it.
- **Test:** report 130 -> 185, classify 202 -> 209; other suites unchanged (env 36, hands 38,
  ingest 109, labels 50, pipeline 260). New refusal tests name the phrase the error must
  carry, so none passes on another refusal. A real fact is pinned: the reported held-out
  matrix (V2/V5/V9, 976 punches) gives cross +28% and lead uppercut -41%; its transpose has
  the same F1 and would have said lead uppercuts run HIGH by 69% -- refused. The manifest's
  matrix is checked against one counted by hand from the saved model's own predictions.
  20 guards reverted one at a time in a scratch copy: all 20 caught.
- **Review (self):** one bug in my own fix -- `math.isfinite` on a 400-digit JSON integer
  raises OverflowError, not ValueError (sidecar fps, rule threshold), and the same class in
  `_count` (int64 overflow on a confusion cell, pre-existing). Fixed: exact int/float
  comparison, bounded counts (<= 10^9); tests added. Message for a pre-10-21 score said
  "classified again since" when the truth is "never recorded"; split.
- **Review (Astra / Codex, read-only, effort medium, on `d3975ef`): SHIP.** All four findings
  CLOSED, each cited to file:line; the stacked-branch list confirmed for #17, #18 and #19
  (#21/#24/#25 were outside its brief; `git diff --stat` shows they do not touch these files).
  One nit, confirmed by reading: `read_classified(..., fps=-(10**400))` overflowed, because the
  integer guard bounded only the top. Fixed (`0 < fps <= 10_000` for integers); test and a
  20th reverted guard added.
- **Security (changed files):** no queries, shells or new paths; every new input goes through
  the bounded JSON reader; errors truncate file-supplied numbers to 20 characters. The
  overflow above was the one finding (low: it failed closed, with a traceback).
- **Real data, read-only:** on IMG_4406 and IMG_4395 the fps, row-time (41 and 42 rows) and
  rule checks accept today's outputs. Two refusals, both intended: today's manifest has no
  matrix (re-run `train_classifier.py`), and today's score files have no CSV hash (re-run
  `classify_punches.py --labels`). Until both are re-run the real reports refuse.
- **Stacked branches:** #17 (edits report.py / report_punches.py / test_report.py: conflicts,
  fixtures need the matrix, rule and fps), #18 (`render_punches` calls `read_classified`
  positionally with `data`; needs `fps=`), #19 (`analyse_clip.preflight` calls
  `finder_reliability(a.rule)` and needs the rule; its test manifest lacks the matrix;
  `check_classified` can use `check_pose`). #21 and #24 (and #25) do not touch these files.
  #22 also edits the `--labels` block of `classify_punches.py`: merge by hand.

### Iteration 8 -- feat/output-report's four fixes merged in (2026-10-21)
Acceptance: (1) the merge resolves with both branches' work kept; (2) every suite green, the
report count equal to this branch's 234 plus the 55 checks the fixes added; (3) the hand check's
published real figures unchanged when the real CSVs go through the stricter reader.
- **Merge:** one conflict, this log (both branches appended an iteration 5); kept both, the incoming
  entry renumbered 7. `report.py`, `report_punches.py` and `test_report.py` merged without conflict.
  The hand check adds no input of its own: it reads rows `read_classified` has already validated
  (now fps-checked), so no code change was needed.
- **Test:** report 289 = 234 + 55; classify 209, env 36, hands 38, ingest 109, labels 50, pipeline 260.
- **Review:** the warning's code is untouched by the merge (`git diff origin/feat/output-report`
  shows only the hand check and its templates). No findings.
- **Security:** no new input. No findings.
- **Real-world facts, read-only, through the merged reader:** IMG_4395 26 of 42 (62%) against 49.4%,
  binomial 0.071, Fisher 0.088, warns; IMG_4406 33 of 41 (80.5%) against 50.4%, binomial 0.00007,
  Fisher 0.00005, passes -- iteration 6's numbers exactly. The full reports cannot be regenerated
  until `train_classifier.py` and `classify_punches.py --labels` are re-run (iteration 7).
- **Review (Astra / Codex, read-only, effort medium, one review of the #17/#18/#19 merges): SHIP all.** Merge OK on each branch, no adaptation defects; it checked that the shared checks are identical to feat/output-report's on all three (confirmed here: no changed line). Its sandbox could not run Python, so its verdict is from reading plus the results above; fresh clones of 2e2059f, f2412e8 and a2f90be reproduce every count.

