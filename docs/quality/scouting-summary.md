## `feat/scouting-summary` -- 2026-11-03

The summary restructured into a scouting layout, and an optional LLM narrator held to the report's facts.
`src/boxingvi/report.py` (`_Writer` records each sentence's section, trust tag and numbers; `summary_writer`;
`_headline`), `src/boxingvi/combos.py` (combination statements tagged by the gate; a failed one marked inline),
`src/boxingvi/narrate.py` (prompt builder, verifier, assembly), `narration_check.py`, `report_punches.py
--prompt-out`, `report_boxers.py` (header as its own paragraph), `tests/test_narrate.py`. Brendan's request
(2026-11-03): "restructure, but give good prompt building". Nothing calls a model.

**Acceptance criteria.** (1) Warnings still lead; then "In short:", "What the tool saw:", "How much to trust it:";
every sentence keeps its wording; every number still enters through the writer and belongs to exactly one sentence
record (test). (2) The headline follows a fixed rule (About sheet): pace always; the commonest type only if the
hand check did not warn and its benchmark recall >= LOW_RECALL; how combinations start only if the gate measured
it and it held. (3) The prompt gives the model only headline/facts sentences with ids and trust tags, inside
delimiters declared as data; warnings and trust sentences are placed verbatim by the program. (4) The verifier
refuses any draft whose sentence states a number not in the facts it cites (multiset), spells a number, uses an
absolute, puts a failed fact in the headline or uses one unhedged, cites an unknown or fixed id, breaks the shape
or size; refused drafts leave the template text. (5) A prompt from participant inputs is marked
`may_leave_machine: false`.

**Threat sketch.** Assets: the report's factual integrity (no invented or softened claim reaching a boxer) and
participant-derived data (must not go to a cloud model). Entry points: the model's reply (untrusted), the prompt
file on disk, CLI paths. Trust boundary: the reply into `narrate.check`. Top threats: a fluent draft that changes
a number, computes a new one or drops the failed-gate caveat; participant numbers pasted into a cloud chat.

### Iteration 1
- **Build.** As above. Prompt designed with the context-engineering skill: coach persona for the reader, task and
  output contract as a JSON schema, positive rules each matching one verifier check, an escape value (null
  headline), one shape example with other numbers, the key rule restated last; the model never handles warnings
  or caveats (structure, not request).
- **Gate test.** `test_narrate.py` 69/69 (29 refused drafts, 5 CLI cases, record invariants on the real summary);
  `test_report.py` 293 (the IMG_4395 exact-text check now pins the new layout: warning, then "In short:", then the
  sections, and that a warned clip's headline names no type; plus `--prompt-out` end to end), `test_combos` 158,
  `test_boxers` 168, `test_analyse_clip` 113.
- **Rendered and read.** IMG_4406 summary in the new layout; its real prompt; a realistic hand-written draft (as a
  model would answer) accepted and assembled; the closing line said "from the facts above", which the narrated
  version does not show: reworded. The dry-run output was deleted.
