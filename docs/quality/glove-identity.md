## `feat/glove-identity` -- 2026-10-23

Glove-colour boxer identity for participant sparring footage (both boxers in black, gloves of different
colours). `src/boxingvi/gloves.py` (the rule), `src/boxingvi/glovescore.py` (the gate's scoring),
`glove_identity.py` (roles / compute / template / freeze / score / gate), `tests/test_gloves.py`. Stacked on
`feat/appearance-identity` (#29). Gate `docs/gates/glove-identity.md`, committed `0e95deb` before any rule
code. **No study frame has been read:** intake stopped at consent scope (vault ISSUES-LOG O65), so every number
below is synthetic and none is a result.

**Acceptance criteria.** (1) On painted fixtures, two boxers in different glove colours are named correctly
through a ByteTrack id swap; same-colour gloves, one-red-one-blue on both, a light drop that changes the colour,
and barely different colours are "not separable" with nobody named; low wrist confidence is "too few rows"; a
clinch makes the touching gloves contested. (2) Scoring is the gate's, checked by hand-counted moments
(anchor, unclear / not tracked / overlapping, swaps, B1 ByteTrack, B2 linker). (3) Every input file is
checked and refused when malformed; every output is under the participant root; TEST / FAIL-CASE are refused
before the freeze, after any rule change, and a second time. (4) Existing suites unchanged. (5) Counts only:
no pixel array is written.

**Threat sketch.** Assets: participant-derived data (pose, glove counts, labels, manifest) and the integrity of
a one-shot held-out result. Entry points: the video decode, tracks / counts .npz, labels .csv, manifest and
score .json, CLI paths. Trust boundary: files on disk into the parsers; the tool's writes. Top threats:
derived participant data written into a synced, git or vault folder; a crafted or mismatched file causing a
memory blow-up or silently wrong numbers (labels of another video, counts of another table); and, for
integrity, scoring TEST with a rule changed after seeing it.

### Iteration 1
- **Build.** The rule: glove disc beyond each confident wrist (wrist >= 0.5, elbow >= 0.3); a read is
  contested when another person's wrist is within 2 radii; 16 pixel classes (12 hues, dark, grey, white,
  skin); per video an unsupervised 2-means in the Hellinger embedding; the video is separable only if the
  centroids are >= 0.30 apart, each cluster holds >= 15% of rows, and co-present boxers fall in different
  clusters in >= 70% of >= 10 frames; a raw decision needs a 0.20 margin; names are smoothed within a track
  id over +-1 s with a 75% majority, and one name per frame. Roles come from the blur reports (DEV = the
  non-fail video with the smaller original sha256), so only blurred copies can be registered.
- **Test:** `tests/test_gloves.py` first run 153 / 153. That is a suite that had never failed, so 22 mutants:
  **17 killed, 4 survived.** Survivors and why: the separation check (the co-present check caught every
  same-colour fixture first; added near-identical colours, split perfectly but under 0.30); the 75% majority
  (no fixture lent an id to the other boxer; added, and it still survived because the one-name-per-frame rule
  masks the error, and the mutant is one-sided; added the lender undetected, in both directions); TEST
  scored twice (refused by "output exists", not by the once-only rule; now the output is moved away first);
  compute on an unregistered video (refused by the fingerprint check; now a matching unregistered clip).
  Final: **163 / 163; 24 / 24 mutants killed** (scratchpad `mutate_gloves.py`).
- **Review (own pass), 3 blocking, fixed:** (B1) the freeze compared only the constants digest: a logic change
  with the same constants would pass; TEST now also needs the rule files byte-identical to the frozen commit
  (`git diff --quiet <commit> -- files`). (B2) a linker refusal in baseline B2 would crash the one-shot TEST
  run; it is recorded as an error, and bar 3 is then not met (never compared against 0). (B3) a failed JSON
  write left a temp file in the participant folder.
- **Security (own pass), changed files only.** Output location: faceblur's deny-by-default check, given a
  `suffixes` argument (default `.mp4`, so the blur's behaviour and its 177 tests are unchanged) -> git /
  vault / Syncthing / cloud / network refused; tested for root, suffix and git checkout. Parsing: counts
  .npz zip directory and every NPY header read before allocation (a 128 GB declaration refused), no pickles,
  exact keys / dtypes / shapes, classes summing to the disc's pixels; CSV and JSON size-capped and strict;
  score files bound to the manifest by sha256. Subprocess: argument arrays; the commit passed to git is
  validated as 40 hex first (no option injection). **Finding (medium), fixed:** a missing input escaped as a
  traceback that prints its path, and an original's path may carry a name -> `main` reports only the OS
  reason; test: a missing `Firstname-Lastname-round1.mp4` never appears in stderr.
- **Regression:** blur 177, headgear 140, appearance 171, linking 266, tracks 79, pipeline 314: all pass,
  same counts as before the branch.
- **Residual risks, accepted for now:** `PoseTable.load` allocates from npy headers until #28 merges (the
  tracks file is the one input not capped here); the counts loader checks the zip and then reopens it with
  `np.load` (a local file swapped in between is out of scope); the rule constants are first-principles
  guesses, untested on any real glove.

### Iteration 2 -- review gate (Astra, read-only, effort medium, on a fresh clone of `6c05b7a`): FIX FIRST, 5 blockers + 4 should-fix, all fixed
- **B1 freeze boundary too narrow:** only the three glove files were pinned, so a change to `headgear.wilson` after
  the freeze passed. Now the freeze and the TEST check cover `src/boxingvi/` whole and the CLI (`git status` /
  `git diff --quiet <commit>`).
- **B2 counts had no rule provenance; table digest left out fps:** counts now carry `provenance` = sha256(rule
  and score constants | every library file and the CLI as on disk); `score` refuses counts of another rule or
  code version. The table digest now includes fps, frame count and size.
- **B3 once-only scoring was not atomic:** a role is CLAIMED by exclusive create (`<manifest>.<role>.claim`)
  before its labels are read. A run that dies after that has used the role up; recovering needs the owner to
  delete the claim deliberately and say why.
- **B4 gate provenance self-asserted:** `gate` now needs a frozen manifest, both scores made under that freeze
  (digest and commit), a claim behind each, the sha256 the run recorded, and a consistent schema
  (correct + wrong = decided <= scored). **Accepted residual:** an owner who deliberately rewrites the manifest
  and the scores together can still forge a pass. Closing that needs a record the owner cannot edit, and
  Brendan is the only user.
- **B5 hard-link escape on `--overwrite`:** every output (npz, csv, json) is written to a fresh temp file and
  swapped in with `os.replace`, so a planted link is replaced, never written through.
- **S1 file names in messages:** output refusals give the reason code only; every path on the command line
  (as typed, resolved, and its file name) is scrubbed from any message.
- **S2 parser and resource failures:** a table claiming over 6 h is refused before the label grid is built (an
  fps of 1e-300 looped effectively forever); `csv.Error` (a 140,000-character field) and `RecursionError`
  (JSON nested 100,000 deep) become refusals.
- **S3 float equality at a bar:** `1 - 35/50` is `0.30000000000000004` and failed bar 2. All share bars now
  compare whole counts (10 x unsure <= 3 x scored, etc.); tested at exactly 0.90, 0.30 and 10%.
- **S4 smoothing overrode direct evidence:** a named row is now unsure if ANY confident raw decision of its
  track within one sampling step disagrees. A first version used the nearest decision only, and a new fixture
  showed it missed the last frame of a lend. Then a second hole, found while chasing a surviving mutant: an id
  lent while the borrower's gloves are hidden had no evidence at all, and the 1 s window named the borrower
  as the lender. **New rule constant `SUPPORT_S = 0.3`:** a row is named only with an agreeing raw decision of its
  track within 0.3 s. Tested: wrong names occur only within 0.3 s (+1 sample) of the last glove seen.
  Both are rule changes made before any study frame, on synthetic fixtures only.
- **Test:** 189 / 189. **Mutants: 36 of 37 killed.** The survivor is the 75% majority (made 50%): the support and
  contradiction rules now cover every case it was added for. It stays as defence in depth and is recorded
  here rather than given a contrived fixture. Four survivors on the way were each one guard masked by another
  (digest vs provenance, scored-record vs claim, edited score vs schema, name in a message vs scrub); each
  test now defeats the other guard first.

### Iteration 3 -- review gate (Astra, second round, on a fresh clone of `8efc7bd`): FIX FIRST; 5 of 9 fixed, 4 partly, 3 new; all fixed
- **B1/B2 partly (blocker): untracked code.** After a freeze, an untracked `src/boxingvi/appearance/__init__.py`
  shadows `appearance.py`, and `git diff` never sees it; the code digest hashed top-level files only. Now:
  `git_same` also needs `git status --porcelain --untracked-files=all` empty for `src/boxingvi` and the CLI
  (the freeze's own check too); `code_digest` hashes every file under `src/boxingvi`, subfolders included
  (`__pycache__` skipped); and `check_imports` refuses to run unless every loaded `boxingvi.*` module is a
  top-level file of this library under its own name. Found while building it: `boxingvi` is a NAMESPACE package
  (no `__init__.py`), which merges every `boxingvi` folder on `sys.path`, so it must have exactly one portion.
  Tested on a throwaway git repo (untracked shadow, tracked change, unknown commit) and with fake modules.
- **S1 partly: the CSV parser echoed bad values** (a `t_s` of `Firstname-Lastname.mp4` reached stderr; it is not
  on the command line, so `scrub` cannot remove it). Messages now give the row number only. The first test was
  case-sensitive and missed an echo of the upper-cased side; a mutant showed it.
- **NEW-1 (blocker): the evidence filters ran before one-name-per-frame,** so a filter could remove the right
  boxer and leave the wrong one named. `resolve_labels` judges duplicates on the smoothed names before any
  filter, then applies the filters; unit-tested on a hand-built two-frame table.
- **NEW-2: my hidden-borrower fixture never hid anything.** It lowered the PREVIOUS row's confidence, and the
  current row was then created with visible wrists. Fixed, and the test now asserts its premise (no read
  during the lend, no row of the lender) and that nobody is named in the middle of the lend. So the previous
  "wrong only near the edges" pass was on an easier case than the one claimed.
- **NEW-3: manifest read-modify-write was unlocked** (TEST and FAIL-CASE finishing together could overwrite each
  other's record). `manifest_lock` (exclusive-create lock file, bounded wait) around the freeze and both
  record steps.
- **Test:** 210 / 210. **Mutants: 43 of 44 killed;** the survivor is still the redundant 75% majority.

### Iteration 3 review (Astra, third round, on a fresh clone of `27717ec`): **SHIP**
All six open items FIXED (B1, B2, S1, NEW-1, NEW-2, NEW-3); no new finding. Astra could not run Python in its
sandbox in any round ("Access is denied"), so its three reviews are static; the runtime evidence is the
suites and mutants above, all run here and reproduced in a fresh clone.

### Exit
- **Gates:** test 210 / 210 (fresh clone of `27717ec`: identical; blur 177, headgear 140, appearance 171, linking
  266, tracks 79, pipeline 314, all unchanged from before the branch); mutants 43 / 44 (survivor: the redundant
  majority guard); review: Astra FIX FIRST -> FIX FIRST -> SHIP; security: every finding above closed.
- **Residual risks, accepted:**
  - An owner who deliberately rewrites the manifest and the scores together can forge a pass. Closing that
    needs a record the owner cannot edit.
  - `PoseTable.load` is capped only once #28 merges.
  - The counts loader checks the zip, then reopens it with `np.load` (a file swapped in between is out of
    scope).
  - Counts must be computed and scored in the SAME checkout: the code digest hashes the bytes on disk, and line
    endings differ between checkouts.
  - An id lent to the other boxer while the borrower's gloves are hidden and the lender is undetected can be
    named as the lender for up to 0.3 s.
  - Every constant is a first-principles guess, untested on any real glove. **No study frame has been read**,
    and the gate is not evaluated until Hector rules (vault ISSUES-LOG O65).

### Iteration 4 -- the very-large-square rule (Brendan, 2026-10-25), after the face blur of the study videos
- **Why.** The face blur (pure deface) left large false face squares on the study videos (video 1: 5.4% of frames,
  mid-frame, over bodies and gloves; measured as face-box sizes only, vault ISSUES-LOG O65). Brendan accepted
  the blur residual, and decided that glove reading treats those squares as unreadable.
- **Rule.** A glove read whose disc centre lies under the mosaic of a face box taller than `LARGE_FACE_PX` = 160
  px is contested. The boxes come from the blur's own detector on the ORIGINAL video (`FaceMasks`, every sampled
  frame), and the mosaic rectangle is the blur's own `faceblur.det_region`. Face-sized mosaics are counted
  (`under_any_face_mosaic`) but NOT excluded, per the decision. **Disclosed:** 160 px was read off the face-box
  size distribution of all three study videos, TEST and FAIL-CASE included (real faces 40-100 px, the false
  large squares above 160). That is a property of the face detector, read before any glove pixel and with no
  label or glove result, so it is not tuning on TEST; it is still recorded here.
- **Binding.** The face-box file is an input: size-capped, zip and NPY headers checked before allocation, exact
  keys and dtypes, boxes inside the frame, frames on the glove sampler's grid, scores in [0, 1], and bound to
  its ORIGINAL by sha256. `compute` requires it and checks that sha against the manifest role's original. The
  three study files were bound after the fact by hashing the originals in the same sorted order, each
  matching its blur report's `source_sha256`. Counts record the mask file's sha256 (counts format
  `boxingvi-gloves-2`), and `score` refuses counts made without a mask.
- **Test:** 236 / 236. A very tall box over one glove makes exactly that glove's reads contested; the other boxer's
  glove 30 px away does not change; a face-sized mosaic over a glove is counted, not excluded; outside the
  masked frames the counts are unchanged; plus loader negatives and CLI refusals (no `--faceboxes`, another
  original, unmasked counts). **Mutants: 49 of 50 killed** (the 6 new ones all; the survivor is still the redundant majority). The full re-run found one more guard masked by a new one: the unregistered-video refusal was being made by the face-box frame-count check, so that test now passes matching face boxes and only registration can refuse it.

### Iteration 5 -- review gate (Astra, on a fresh clone of `01ee446`): FIX FIRST, 1 blocker + 1 should-fix, both fixed
- **Blocker: the face boxes were bound to the original, not to the detector run that drew the mosaics.** A file
  made from the right original with another threshold or model passed every check and could omit large
  squares the blur actually drew. **Fix:**
  - `compute` now requires the blur report and checks that it is the report of THIS blurred video
    (`output_sha256`);
  - it checks the same original as the face boxes and the manifest role, the deface policy, and the padding
    `det_region` assumes (`face_det_pad`);
  - and, the content binding, on every sampled frame exactly as many boxes as the blur says it found
    (`gloves.check_blur_report`).
  - The producer moved from a scratch script into `glove_identity.py faceboxes`: it re-runs the blur's
    detector on the original at the report's own threshold, and refuses to write unless its counts match the
    report.
  - **Measured on the three study files: all 9,589 sampled frames agree with the blur reports, zero
    differences.**
  - Residual: someone who deliberately swaps one large box for a fake small one in the same frame keeps the
    counts; that is forgery, outside this threat model.
- **Should-fix: a 0-d `frame` array crashed `FaceMasks.load` with a TypeError** (`len()` before the shape
  check). Now refused first; tested.
- **Environment:** the desktop's pose venv lacked `lap`, which ultralytics' ByteTrack imports. It installs `lap`
  on first use only when online, never with `YOLO_OFFLINE=1`, so the first pose run on the study videos
  stopped at frame 0, with nothing written. Brendan approved the install (`lap` 0.5.13, a pre-built wheel from
  PyPI), now pinned in `requirements-pose.txt`. `check_env.py --pose` will now flag any other `lap` version,
  e.g. one ultralytics put on the GPU laptop.
- **Found while running pose:** `run_ingest.py` does not set `YOLO_OFFLINE` (the blur does), and its default
  `--out-dir` is `results/ingest`, a Syncthing folder. For participant footage both were set by hand. A guard
  in `run_ingest.py` is a follow-up, not in this PR.

### Iteration 6 -- no FAIL-CASE video exists (gate addendum `60a3e29`, 2026-10-26)
- Brendan watched all three blurred copies: every pair of boxers wears different-colour gloves. The addendum
  (committed before any glove count, label or score) makes DEV the smallest original sha256 of all three, both
  others TEST, bars 1-3 on EACH (never pooled), and bar 4 NOT TESTABLE (`None`, never `True`).
- **Code:** `roles --fail-case none` writes a `glove-identity-manifest-2` (roles dev / test / test2). The
  manifest loader accepts each format only with its own role set. `gate` takes `--test2` for format 2 and
  refuses `--fail-case`, and the reverse for format 1. `glovescore._bars_123` is shared by both verdicts.
  The format-1 path is unchanged.
- **Tests:** 272 / 272 (17 new: verdict unit cases incl. never-pooled and a baseline that failed on one video;
  roles none / 0 / NONE; gate flag-mode refusals both ways; three cross-format manifest refusals).
  **Mutants: 3 of 3 killed** (bar 4 reported True; pooled `or` instead of `and`; roles not hash-ordered).
- **Full mutant re-run over iterations 1-5 (57 mutants, after `02f9af9`): 56 of 57 killed**; the survivor is still
  the redundant 75% majority. The run found two more guards masked by newer ones, both now isolated in tests only
  (no rule file changed, so the freeze is unaffected):
  - the unregistered-video refusal was being made by the blur-report output check (the default report describes
    another video), so that test now passes the video's own report and only registration can refuse it;
  - the manifest-role check on the face boxes' original was being made by `check_blur_report` (boxes and report
    named different originals), so a new test gives boxes and report that agree with each other and with the
    video but name another original than the manifest's role.
  Tests 273 / 273.

### Gate result (2026-10-27): FAIL
Frozen at `cc48822` (rule digest `65909195f855`), no change after DEV, each TEST scored once. TEST: separable,
28 of 44 decided, 28 right, unsure 0.36 (bars 1 and 2 fail, bar 3 passes). TEST2: not separable, 0 of 50.
DEV: not separable, 0 of 59. Bar 4 not testable. About 51% of the pixels read were skin. Full tables: the gate
result comment on PR #31 (2026-10-28) and `docs/RESULTS.md` "Pre-registered negatives" N2. Future Work.
