## `feat/own-footage-labels` -- 2026-10-06

Label Studio config (`labelstudio/punch-labels.xml`) and importer (`src/boxingvi/lsimport.py`,
`import_labels.py`) for the ground truth every punch-finding number on own footage is scored
against. Label the OVERLAY mp4: H.264 CFR (browsers rarely play the iPhone's HEVC) with the
true frame number burned in.

- **Acceptance:** (1) the config passes Label Studio 1.23.2's own validator; (2) a correct export
  imports with every punch on the right video frame; (3) a wrong frame rate, misplaced frame
  checks, a duplicate, an unknown label, an out-of-range frame, a second annotation or hostile
  JSON is refused with the reason named; (4) existing labels are never silently replaced.
- **Threat sketch:** asset: the ground-truth labels. Entry point: the export JSON (untrusted).
  Top threats: a silent frame shift (LS defaults to 24 fps; numbering base undocumented) that
  corrupts every score with nothing visible on screen; a duplicated punch quietly lowering recall.

### Iteration 1
- **Real-world facts:** config VALID under LS 1.23.2's `validate_label_config` (from its own venv),
  parsed as TimelineLabels with 8 labels; controls: a broken `toName` is rejected, a misspelt tag
  is NOT (so the parsed type, not the validator, is the evidence the tag is right).
- **Test:** 34; each of 10 guards reverted fails >= 1 test.
- **Review: SHIP**, with findings from the installed LS source: frames are 1-based
  (`TimelineRegionEditor` `min={1}`, `VideoCanvas` `ceil(t*fps)`), so the offset is exactly +1 and
  "within +-1" would let checks marked one frame early shift every punch; regions are not checked
  for `from_name`/`to_name`; drafts give a misleading "found 0"; `lead_time` is cumulative
  wall-clock including idle; an `unclear` duplicate escapes the per-hand overlap rule; `Infinity`,
  `NaN` and deep nesting fail without a clean error; VFR video and overwrites not refused.
- **Security:** export size capped before parsing; nothing in the export names a file to open;
  output path from the local video name only; `allow_nan=False` on write.

### Iteration 2
- **Fix:** all of the above. Offset must be exactly +1; tag names checked; drafts named; lead time
  reported as an upper bound; unclear-on-unclear overlap and unclear IoU >= 0.5 with any punch
  refused; `parse_constant` rejects NaN/Infinity; RecursionError becomes a clean refusal; VFR
  video refused; `--force` required to replace labels; the export's sha256 recorded with them.
- **Test:** 50; each of 8 new guards reverted fails >= 1 test. All suites: labels 50, ingest 109,
  env 36, pipeline 260, scripts 38.
- **Not yet covered:** every fixture is hand-built. The first real export is the real-world test,
  and the importer enforces it: it refuses unless both frame checks give exactly +1.

