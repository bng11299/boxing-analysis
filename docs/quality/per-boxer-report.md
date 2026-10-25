## per-boxer-report -- 2026-10-25: one report per boxer from an explicit identity input

Branch `feat/per-boxer-report` from `origin/main` (`bddef45`). Design record: vault ADR-0016 (proposed).

### Setup
**Task.** The plan's Phase 2 deliverable: from a two-boxer video, one report per boxer (what they throw, how
often, when), built against an identity INPUT so it does not depend on which identity method wins.

**Interfaces reused, not reinvented.** Neither identity session writes an identity file. Glove colour (#31,
`feat/glove-identity`) names each pose-table row 0 / 1 / unsure in memory and its gate scores methods through
per-row name functions (`glovescore.glove_names`, `linker_names`, `bytetrack_names`); its gate writes a JSON with
`"PASS": true | false`. Headgear / kit colour (#29) decide per fragment; the linker (#11) per track id per frame.
Chosen: per-SEGMENT records, which all three reduce to without loss (`segments_from_row_names` takes glove's
per-row names; `row_names_from_linking` the linker's), and a recorded PASS is checked against the glove gate's own
JSON. The finder's segment scan is `completeness._scan_segments` itself (new `keep=` argument), not a copy.

**Acceptance criteria.**
1. `tests/test_boxers.py` passes, with a named negative test for every refusal the brief lists: forged sidecar,
   wrong hashes, frame-count and fps mismatch, an identity for another video, an empty boxer, all segments unsure.
2. On a linker identity the per-boxer finder equals `completeness.find_bursts` (synthetic, and all 29 Olympic
   videos); on one unbroken person boxer A's rows equal the solo `classified.csv` byte for byte; no punch at a
   handover where the same rows stitched into one track do fire.
3. Fail / ungated: the first sentence of each boxer's summary is the unverified-split warning; every number in
   the text is traced to a template.
4. `analyse_clip.py --two-boxer` publishes its seven outputs; a refusal leaves `--out-dir` unchanged; the whole
   existing suite passes unchanged (the solo path's outputs are untouched).

**Threat sketch.** Assets: the correctness of per-boxer counts; any participant-derived file. Entry points: the
identity CSV and sidecar, the gate result JSON, the tracks npz, the boxers / unsure CSVs and sidecar, CLI
arguments. Trust boundary: every file read from disk (other sessions and tools write them). Top threats: (1) a
mismatched or forged identity mis-splits punches while claiming a gate PASS; (2) crafted input crashes a parser,
or participant-derived output is written outside the encrypted participant folder.

### Iteration 1 -- build, test gate, own review and security gate
- Built: `src/boxingvi/boxers.py`, `boxer_punches.py`, `report_boxers.py`, `analyse_clip.py --two-boxer`,
  `olympic_boxers.py`; behaviour-preserving hooks in `classify.py` (`cut_from_widths`), `completeness.py`
  (`keep=`) and `report.py` (`extra=` sheets).
- Test gate: `tests/test_boxers.py` 135 then 144 passed; the whole existing suite green (18 files + machine
  scripts, counts unchanged: analyse_clip 113, report 289, classify 297, completeness_credited 95, linking 266 ...).
- Real-world facts the gate asserts (not shape): on the 29 Olympic videos the per-boxer finder reproduces three
  recorded numbers exactly -- 7,984 punches on the two linked boxers (RESULTS row 7c), 10,077 on credited
  fragments (7d), 1,585 of 2,423 credited punches owned (65.4%, 7b) -- and equals `find_bursts` on 29 of 29.
- Own review, three blocking findings, fixed: (a) `csv.Error` (a field over the csv module's limit) escaped as a
  traceback from three CSV readers -> one `_csv_rows` reader that refuses with ValueError, negative tests added
  (NUL bytes are accepted by Python 3.13's csv and refused by the field checks; tested); (b) nothing kept
  participant-derived outputs inside the participant folder -> `check_output_location`, deny by default, paths
  resolved first, in both stages and the two-boxer preflight, four tests; (c) the two-boxer staging folder was not
  removed on an interrupt -> `finally`, as the solo path.
- Security gate (changed files): every disk input size-capped before parsing; JSON refuses NaN / Infinity and
  deep nesting; exact key sets; no pickles (tracks via `PoseTable.load`); gate document path regex-limited to
  `docs/gates/<name>.md` and resolved inside it; no secrets, no participant names in messages (file names only).
  Residual, accepted: the tracks file is hashed by path after it is loaded (a local check-then-use window on a
  single-user machine; the report stage re-hashes and re-runs the finder).

### Iteration 1 -- review gate (Astra, read-only, effort medium, fresh clone at `4ff4b60`): FIX FIRST, all fixed
Astra confirmed no solo-path regression and no number reaching a summary outside `report._Writer`; the warning
leads every fail / ungated summary. Four blocking findings, each checked against the code before fixing:
| # | Finding | Fix | Test |
|---|---|---|---|
| R1 high | A boxer's rows were merged before the gap split, so frames the identity left unsure (or gave B) inside one track, if no longer than the bridge (3 frames at 60 fps), were interpolated back into A | `scan_segments_of`: every identity segment is its own scan unit; its edges are always cuts | three unsure frames inside A's track: segments `(0,99) (103,749) (750,1199)`, no A row in 100-102, and the punch stage's output is accepted by the report stage; Olympic re-run: still 29/29 identical to `find_bursts` (the linker never leaves such a hole) |
| R2 high | A recorded PASS needed only a JSON saying `PASS: true`, against any gate document | `GATES` registry, deny by default: a gated verdict only for a method with a pre-registered gate, and only that method's document; a PASS needs the result in the gate's own format (glove: four bars, each true / false, PASS = all four); headgear colour (#29) may record its FAIL, never a PASS (no result format registered) | bare `{"PASS": true}`, PASS against its own bars, a non-boolean bar, the linker claiming a gate, another method's document, a headgear result, a headgear pass: all refused |
| R3 medium | Well-formed JSON escaped as TypeError (a list stance) and OverflowError (a 400-digit extent) | type before membership; range before float; the sidecar's extent is now exactly `frac, lead_widths, tail_widths, variant` (the classifier's) | a sweep over EVERY leaf of both sidecars x 8 bad values (list, dict, null, string, 10^400, -1, true, 1.5): 79 + 440 mutations, none accepted, none another exception. It found one more: `True == 1` let a scan count of `true` through; recorded blocks are now compared as canonical JSON (`_same`) |
| R4 high | A link planted at an output path in a participant folder made the standalone stages write participant-derived text outside it | every output written to a fresh file created exclusively in the folder, then `os.replace` (the link is replaced, never written through); a failed write leaves the old file and no temporary | a hard link at `boxers.csv` to a file outside: unchanged; symbolic links cannot be made without privilege on this machine (Developer Mode off), so that case is written but skipped here and covered by the same write; an interrupted write leaves the old output and no `.tmp` |

Not changed, by decision: an ungated method's name is its writer's own label (the sweep accepts `method: "x"` and
says why). Residual, stated: a gate result is a JSON file; its hash and format show it is the recorded one and
internally consistent, not that the gate was actually run (it is not signed).

### Iteration 2 -- test gate
`tests/test_boxers.py` 163 passed, 0 failed. Olympic re-run on the fixed code: identical aggregates.
