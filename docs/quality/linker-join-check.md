## feat/linker-join-check -- 2026-10-28: hand-checked linker joins (option A), the tool

Gate: `docs/gates/linker-join-check.md`, committed `ec52160` BEFORE any code, sample or answer (bar agreed by Brendan
2026-10-28: three-way verdict against p\*, n = 100; ADR-0018 in the vault). Stacked on PR #35 (`feat/label-moments`)
to reuse its window.

**Acceptance criteria**
1. The sampler is seeded and reproducible (same frame + seed -> same 100 in the same order, whatever order the frame
   arrives in); strata proportional by the median gap; the shown order shuffled.
2. Resume works: every answer saved at once, a rerun starts at the first unanswered join, a different sample cannot
   be resumed into, a stale window cannot overwrite newer answers.
3. Out-of-range joins are refused: outside the video, gap over the linker's 2 s, fragments out of order, or not
   matching the tracks file.
4. Nothing reveals the linker's decision: no chain / boxer / cost in the sample file or a Join; the window shows
   "join k of n" and the two fragments' boxes only, never a bystander's.
5. The score is computed once, from a complete file, U excluded, three-way verdict against p\*.
6. #35's behaviour is unchanged by the hook refactor (its own tests pass).

**Threat sketch.** Assets: Olympic frames (research-use only) and the integrity of the answers. Entry points: the
sample and answers files, `--data` / `--tracks` / `--dir`, ffmpeg. Trust boundary: files on disk, validated on load.
Top threats: (1) Olympic-derived output leaving `results/`; (2) a tampered or swapped sample / answers file
corrupting the measurement.

**Design.** `src/boxingvi/joincheck.py` holds the frame, draw, files, clip and score; `linker_join_check.py` the
CLI and `JoinApp`, a subclass of #35's `label_moments.App`. #35 gained hooks, not copies: `labelmoments.replace_file`
(atomic save + stale-session refusal), `decode_frames` (the ffmpeg decode), `VideoGuard(path_check=...)`, and App
hook methods (`key`, `decode`, `hold_view`, `play_view`, `status_text`, `answer_blocked`, ...). Frames are decoded in
memory; no clip is written to disk.

### Iteration 1
- GATE test: `tests/test_linker_join_check.py` 130 passed, 0 failed (after 3 first-run failures: Wilson's lower bound
  at x = 0 left 3e-18, now exact at the ends; two test expectations corrected). `tests/test_label_moments.py`
  169 passed, 1 skipped (symlinks, account permission; skipped before this change too). Full suites not run: a seed
  sweep holds the CPU (instruction).
- GATE review, Astra (read-only, code only, medium; the one Astra review): **FIX FIRST**, 3 blocking + 3 should-fix.
  Code-reviewer subagent: **SHIP**, no blocking, 4 should-fix + 6 nits; it confirmed clip boundaries, the gap bound
  (matches `linking.candidate_links`), the draw, the score maths and #35's unchanged behaviour.
  1. two windows could both pass the stale-file check and the second rename drop the first's answer -- Astra
     blocking (inherited from #35) -> `SessionLock` (OS lock, released if the process dies), one window per file;
  2. the stratum was recoverable: `strata` sat in the file the window loads, the gap length was captioned, and
     `sample` printed the median -- Astra blocking, reviewer should-fix -> strata not stored (derived in `score`),
     no gap length shown, no median printed;
  3. `score` trusted the sample's p\* and positions: a p\* of 1 made 100 D answers "acceptable" -- Astra blocking,
     also my security pass -> `parse_sample` enforces the gate's seed / n / target, recomputes p\*, cross-checks
     totals and strata, types every number;
  4. a clip decoded only to fragment 2's first frame was answerable -- Astra -> the whole clip or U only;
  5. any ancestor named `results` was accepted; sampling did not check `--data` containment -- Astra, reviewer nit
     -> `RESULTS_ROOTS` (this checkout's and the main checkout's results/), containment before any read;
  6. boundary frames off the 2-frame grid played at half speed -- Astra -> per-frame durations (the test caught
     one more: the last frame was held two frame-times);
  7. ~400 MB peak per decode -- reviewer -> one buffer viewed, not copied (`decode_raw`);
  8. a bad sample was written before validation and then blocked every rerun; link errors were tracebacks --
     reviewer -> validate bytes first; per-video "did not link";
  9. nits taken: clip plays once before the still (the gate's order), decode failure says "Space retries", item
     noun, exit message. Kept: `lm._under` imported across modules (one canonical path rule).
- GATE security (own pass, changed files): ffmpeg by argument array with `-protocol_whitelist file`; videos resolved
  under `--data` and held open (VideoGuard); sample and result create-only (`xb`, fsync); answers parsed strictly and
  pinned to the sample's sha256; `verify` shows no S / D / U split (no interim peeking); messages name checks, not
  paths. Finding 3 above was mine too. Residual: `check_out_dir` trusts two fixed roots, not git's ignore rules.

### Iteration 2 (`d7d16c6`)
- GATE test: 149 passed (new negative tests: edited p\*, positions, seed, n, totals, strata, median, target, camera;
  second window refused; truncated decode; foreign results/ folder; clip plays before the still; no gap length in
  any caption; real-time durations). #35: 169 passed, 1 skipped.
- GATE review, code-reviewer (re-review; the one Astra review was spent): **SHIP**, every round-1 fix correct,
  `parse_sample` accepts every sample `build_sample` writes, #35 unchanged but for the caption. Should-fix: the lock
  file's `open` outside its try (a traceback with a path); a `Refused` while sampling reported only by type. Nits:
  median bound hard-coded; a float seed accepted; lock not released if `videos.close()` raised.
- GATE security: the two should-fixes are error-path leaks (a path in a traceback; a reason lost); nothing else new.

### Iteration 3
- Fixed both should-fixes and those three nits; new negative tests (unopenable lock file -> refusal; float seed).
- GATE test: 151 passed, 0 failed; #35 169 passed, 1 skipped.
- Not tested: the per-video "did not link" line in `sample` (needs a full Olympic task layout; the gate's "all 23
  or nothing" is tested through `build_sample`).

**Residuals accepted.** A join within about 1 s of a video's end is U-only if its stored frame count overcounts
what decodes (whole-clip rule; constant-rate GoPro footage, unlikely). The checker knows every pair is a linker join
(no foils), which may lean answers towards "same" -- stated in the PR, not designed out (time). No real-footage run
of the window yet: the first `check --dry` on the 23 videos is the real-data smoke test, before Brendan's session.

### Exit
- Gate reproduce: fresh clone of `origin/feat/linker-join-check` at `2f5d0a5`, same interpreter: test_linker_join_check
  151 passed / 0 failed / 0 skipped, test_label_moments 169 / 0 / 1 -- the same numbers as the working tree.
- All gates green: tests (151 + 169), review (Astra FIX FIRST -> fixed; code-reviewer SHIP twice), security (own
  pass; the lock-open and Refused-reason leaks fixed).
## Measured identity error in the per-boxer report (Brendan, 2026-10-28: "wire it in")

**Acceptance criteria.** (1) With the join check's result.json, a linker identity's report quotes the measured
join error, its Wilson interval, the count checked and the footage in its first sentence; (2) every such number is
re-derived from the answers and the sample, never read from a typed field; (3) a sample of videos outside the
gate's 23, edited counts, an inflated p\*, or a gate verdict disagreeing with the measured one is refused; (4)
without the result file, behaviour is unchanged; glove and headgear gates unchanged.

**Threat sketch.** Asset: the report's identity claim. Entry points: the identity sidecar, `--gate-result` and the
files beside it. Top threats: a forged "acceptable" (a PASS with numbers); a number reaching the report that no
answer supports.

**Design.** `GATES["linker"]` registers `docs/gates/linker-join-check.md`; `_check_join_result` loads the
`sample.json` and `answers.csv` beside the result, parses the sample against `joincheck.HELD_OUT_MANIFEST`
(asserted equal to `olympic_headgear.HELD_OUT`), counts S / D / U from the answers and recomputes rate, interval,
p\* and verdict; `Identity.measured` holds only those recomputed numbers. `olympic_boxers.py --join-result`
records the verdict. `report_boxers.py` changes one line (the `header` call); PR #42 edits the lines around it,
so whichever merges second resolves a one-line conflict.

### Iteration 1 (`7d2eb37`)
- GATE test: test_identity_measured 39 passed; test_boxers 168 (one check re-pointed: "linker" was its example of
  a method with NO registered gate; the property is kept with `torso-colour`, plus a new check that the linker's
  gate is its own document); test_linker_join_check 151.
- Real-data fact: Brendan's real result.json reads as `fail`, 64 of 98, 0.6531 [0.5547, 0.7399], and the header
  prints "... failed its pre-registered gate: in a hand check of 98 of its joins on Olympic Boxing video, 65% joined
  two different people (95% interval 55% to 74%), ...".
- GATE review (code-reviewer): **FIX FIRST**. B1 the sample was parsed against its own video list, so made-up
  videos with short chains gave a PASS quoted as Olympic Boxing; B2 the answer total was not tied to the sample.
  Should-fix: counts typed in the result, not counted; "passed" beside a measured "unreliable"; huge integers
  overflowed. Nits: bool / float counts, wilson95 in the nothing-decided branch, a traceback from olympic_boxers.
- GATE security (own): NaN refused at parse (test added); the rest as the review.

### Iteration 2 (`a4be8a7`)
- All of the above fixed: gate manifest, counts from `answers.csv` (pinned to the sample) with n from the sample,
  verdict agreement in the header, strict typing, clean refusal.
- GATE test: test_identity_measured 48 passed (new: foreign videos refused; 10,000 consistent answers refused;
  answers changed after scoring; incomplete answers; huge integer; header disagreement both ways); test_boxers 168;
  test_linker_join_check 152 (manifest copy equals #29's). The real result still reads `fail`, 64 of 98.
- GATE review: re-review pending.
### Iteration 3
- GATE review (code-reviewer re-review): **SHIP**. B1, B2 and every should-fix confirmed fixed, each with a negative test.
  Taken: a "not established" measurement now says "too few clear answers for its gate to judge" (test added).
- GATE test: test_identity_measured 49 passed; test_boxers 168; test_linker_join_check 152.
- **Residuals accepted** (none reaches a number the report quotes): another complete answers.csv pinned to the same
  sample with the same totals (answers moved between rows) would pass, and the result's per-stratum / per-camera /
  minutes fields are not re-derived; `GATES["linker"]["manifest"]` is a test override living in module config
  (None = the real 23; nothing in production sets it); `AnswerFile.load`'s size check is stat-then-read.- Gate reproduce: fresh clone of `origin/feat/linker-join-check` at `b1e4446`: test_identity_measured 49 / 0, test_linker_join_check 152 / 0 / 0, test_boxers 168 / 0, test_label_moments 169 / 0 / 1 skipped -- the same as the working tree.
