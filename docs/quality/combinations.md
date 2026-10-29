## `feat/combinations` -- 2026-10-28

Punch combinations in the solo and per-boxer reports (vault ADR-0019): `src/boxingvi/combos.py` (grouping,
the summary sentences, the gate's scoring, the guarded reader of its result), `score_combinations.py` (the gate),
`report.summary(..., combinations=...)`, `report_punches.py` and `report_boxers.py` (wiring and a Combinations
sheet), `tests/test_combos.py`. Gate `docs/gates/combinations.md`, committed `909486e` before any combination
code or number. Design chosen by Brendan before the build: gap 1.0 s, unnamed punches as `?`, a pre-registered
gate on the two labelled own clips, both reports.

**Acceptance criteria.** (1) A combination is a maximal run of one person's punches each at most 1.0 s after
the last, length >= 2, never across a round or an identity-segment edge (hand-worked boundary tests at exactly
1.0 s, one frame over, half-frame label centres, unit breaks). (2) Every number a combination sentence prints
enters through `report._Writer`; both report scripts check the text's numbers against the recorded ones.
(3) The gate's scoring is the pre-registered one (exact recovery, permutation floor with its seed and
percentile, strict inequality, favourite and starter as tie-keeping sets, blind-spot count), checked on
hand-built clips including a floor that cannot be beaten. (4) A gate result is used only for its classifier,
finder rule and gap; a malformed or inconsistent one is refused. (5) Existing suites still pass, with the two
tests that pinned the old summary and sheet list updated to the new contract (not loosened).

**Threat sketch.** Assets: the integrity of the reported reliability (a report must not state a pass or numbers
the gate did not produce) and own-footage-derived files. Entry points: the gate result JSON, classified CSVs and
sidecars, label JSON, CLI paths and clip names. Trust boundary: files on disk into `read_gate`,
`report.read_classified` / `verify_provenance`, `find_punches.load_labels`. Top threats: a forged or stale gate
result making a report claim a pass (or numbers) for another classifier or rule; a crafted file causing a
memory blow-up or a traceback instead of a refusal.

### Iteration 1
- **Build.** As above. `report.summary` gains an optional argument; without it the text is byte-for-byte the old
  one (a test). `boxers.header` and `boxers._check_gate` untouched (the linker-join-check session owns them).
- **Gate test.** `tests/test_combos.py` 137/137 (most checks are refusals: 25 malformed gate files and inputs, 10
  CLI refusals); `test_report.py` 291/291 and `test_boxers.py` 167/167 after updating the solo end-to-end check
  (the summary must now equal the library's WITH combinations, and still start with the old text) and the
  per-boxer sheet list (Identity and Combinations). Full suite: see iteration log below.
- **Gate result (real data, after the code was committed `b68e121`).** C8's commands regenerated the detections
  into `results/combinations/own_tol0.2` with `results/punch_finder_v6.json` and `results/punch_extent.json` (the
  same rule values as `results/repro/`); they reproduce RESULTS 6a/6b exactly (35/47, 35/41; 35/40, 35/42; type
  0.857 and 0.600 of 35, floors 0.314 and 0.371). Score: exact recall 2 of 19 (0.105) vs floor 97.5th percentile
  0.105 (bar 1 FAIL by the strict inequality), favourite fails on IMG_4395 (bar 2 FAIL), starter jab in both (bar 3
  pass): **FAIL**, `results/combinations_gate.json` (first format; re-scored in iteration 2). RESULTS rows 12a-12c, note 12.
- **Rendered and read** (both own clips' solo summaries, and the per-boxer sentences with the real result): the
  recall was printed only as "2 of 19" beside "11% by chance", hiding that they are equal: now "2 of 19 labelled
  combinations exactly (11%)". The per-boxer sentence said each combination "is one person's punches", stronger
  than the evidence (swaps within a track are unmeasured): now it claims only that a wrong join between tracks
  cannot merge two people's punches. Both in `1ac1da2`.
- **Finding for Brendan (not changed: the rule is pre-registered).** At about one punch a second of
  shadowboxing, the 1.0 s gap chains 85-90% of punches into combinations (longest 9 and 14 punches).
- **Gate security (changed files only).** No queries, subprocesses, network, secrets or deserialisation beyond
  JSON. `combos.read_gate`: reads at most 256 KiB through a bounded `read` (not `read_bytes` then slice, which
  would read an arbitrarily large file first: fixed before commit), refuses NaN/Infinity, non-UTF-8, non-objects,
  any key set other than the exact one, non-bool bars, a PASS that is not all bars, counts that are bools,
  negative or over 10^9, recall that is not recovered / labelled, bar 1 inconsistent with recall and floor, and
  huge integers before any float operation; a result for another classifier, rule or gap yields
  `not_applicable` and no number from it (tests for each). `score_combinations.py`: clip names must match
  `[A-Za-z0-9_-]{1,64}` (no path in a name), each clip once, the gate document must exist, outputs written
  through `boxers.replace_bytes` (new file, atomic replace, never through a planted link) and checked by
  `boxers.check_output_location` (participant-derived inputs keep outputs under the participant root); every
  input goes through the existing checks (`verify_provenance`, `check_pose`, `read_classified`, `load_labels`).
  Workbook cells are fixed vocabulary (type names, boxing numbers, `?`) through `report._cell`, which refuses a
  formula lead. No finding open. Residual: if PR #40 merges, `_rule_of` returns a parsed finder block, which
  `score_combinations.py` would then try to write into the result JSON; re-check when #40 lands (the result is
  void after a finder change anyway).

### Iteration 2 (Astra review 1: FIX FIRST, 6 findings; 4 fixed, 2 rejected with reasons)
Review: `codex-companion task`, read-only, effort medium (brief and output in the session scratchpad).
1. **Fixed — a forged PASS was accepted** (per-clip evidence never reconciled). `read_gate` now requires each
   clip block's exact keys, counts that are counts and do not contradict each other, lists of punch-type names,
   `favourite_matches` / `starter_matches` exactly what the lists say, bars 2 / 3 exactly the clips', and the
   pooled counts exactly the clips' sums (zero labels then force recall 0). 11 new refusal tests.
2. **Fixed — an exploratory run passed as the gate.** A result applies only if its clips are the registered
   IMG_4395 + IMG_4406, with 2,000 permutations and seed 20261028; otherwise `not_applicable` (tests for each).
   `score_combinations.py` still runs other clips (for tests and exploration); reports never use those runs.
3. **Fixed — finder mode not bound.** The result records `finder_per_hand: false`; `read_gate` takes the punches'
   mode and applies the result only when both are explicitly false (an old sidecar without the field gets
   `not_applicable`); the blind-spot sentence says "from each hand" for per-hand punches; the scorer refuses a
   sidecar whose `finder_per_hand` is anything but `false` (was: only truthy values). Tests for each.
4. **Rejected — rounds from `time_s`.** The report assigns every punch to a round with `report.round_of(time_s)`
   (per-round counts, rates, outside-rounds count); `summary` checks the combination punches equal the round
   facts' punches. Assigning combination rounds from frame/fps instead would make the two disagree at a boundary,
   which is worse; a combination never crosses a round edge as the report itself defines it. The gate spec's
   frame/fps time is for gaps, which do use frames.
5. **Rejected — the 1e-9 epsilons.** They guard float noise only: gaps are whole frames (>= 1/60 s apart) and
   recalls are k/19 against a percentile of the same quotients, so no real value lies within 1e-9 of a boundary;
   boundary tests at whole frames on both sides exist. Today's tie (2/19 vs 2/19) is unaffected either way.
6. **Fixed — "commonest" also printed lower counts.** Now "commonest" is exactly the gate's favourite (the highest
   count, ties kept), and lower counts follow as "followed by ..." (test).
- **Gate test.** `test_combos.py` 158/158. Result re-scored with the new format: identical numbers, FAIL,
  `results/combinations_gate.json` sha256 `36cd434090e8...`; both own reports re-rendered and read.
