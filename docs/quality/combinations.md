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
  pass): **FAIL**, `results/combinations_gate.json` sha256 `9bc0a69f4543...`. RESULTS rows 12a-12c, note 12.
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
