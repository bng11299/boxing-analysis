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

### Iteration 2 (Astra review 1: FIX FIRST, 6 findings; all addressed, one as a documented residual)
Review: `codex-companion task`, read-only, effort medium, on `27f0b2c` (its runtime probe was denied by the
sandbox, so its bypasses were traced statically; each one is now a test that runs).
1. **Fixed — numeric bypasses.** Fullwidth / other Unicode digits, Roman numeral characters, vulgar fractions:
   any numeric character outside 0-9 is refused. Ordinals (third .. twelfth) and quantity words (couple,
   several, handful) join the spelled-number list. A number keeps its unit: tokens are taken WITH a trailing `%`
   from the cited facts' text, so "8%" cannot reuse the count 8. Digits glued to letters (`8e9`, `1st`) and
   ASCII Roman numerals (`IV`) are refused. Tests for each.
2. **Fixed in part, residual documented — inverted claims and fake hedges.** Negation (not, no, never, -n't,
   without, rarely, ...) is refused in the headline and points, except inside the required hedge phrases, so
   "You do not lead combinations with the jab" and "this is not unreliable" are refused (tests); the prompt says
   to state facts positively. Residual: the verifier checks numbers, citations, words and hedges, not
   entailment; a positively worded claim that misreads a cited fact (or a focus line, where negation is allowed)
   can pass. Mitigation by construction: warnings and every trust sentence are the tool's, verbatim, and the
   closing line says which parts a model wrote.
3. **Fixed — a failed starter was not marked inline.** It now carries the same "(this failed its check on
   labelled footage; see below)" as the other failed statements (test).
4. **Fixed — delimiter separation.** Fact and fixed lines are JSON-quoted with `<` and `>` escaped, so no data
   line can close or open a tag (test with `</facts>` and `</printed_by_the_program>` in the data); the
   data-only rule is in the system message as well.
5. **Fixed — privacy inputs.** `may_leave_machine` now covers every input that contributes a number: pose,
   classified CSV, `--score` and `--combo-gate` (resolved: relative paths and links followed). Residual: a copy
   of participant data moved outside the participant folder is not recognised (no provenance to check).
6. **Fixed — warning order with a passing identity.** When the identity passed (its header carries no warning)
   and the summary opens with a "Warning:" paragraph, that warning now leads the per-boxer text and the header
   follows (`report_boxers.order_paragraphs`); a Caution does not move; a failed or ungated identity's warning
   stays first. Two `test_boxers.py` checks that encoded "header first" for a fixture whose hand check warns now
   assert the stronger order (warning, then header, no identity warning).
- **Gate test.** `test_narrate.py` 88/88, `test_boxers.py` 168, `test_report.py` 293, `test_combos.py` 158. The
  realistic IMG_4406 draft is still accepted (output deleted again).
- **Gate security (changed files).** No network, subprocess or model call. The model's reply is untrusted: read
  through a 16 KiB cap, parsed strictly (no NaN, exact keys, one line per sentence), every rule checked, refused
  drafts write nothing. The prompt file is read through a 1 MiB cap and must be a `<clip>.narration-prompt.json`
  object; outputs go through `boxers.replace_bytes`. Participant inputs mark the prompt `may_leave_machine:
  false` (findings 4-5 above). No finding open beyond the two documented residuals.
- **Gate reproduce.** Fresh `git clone`, branch tip `0d8afa8`, no `data/` or `results/`: 28 Python suites and the
  machine-script suite green, 4,146 checks (1 existing skip in label_moments): narrate 88, report 293, boxers
  168, combos 158, analyse_clip 113, and every other suite as on main.
- **Exit: all gates green.** Residuals: entailment is not verified (positively worded misreadings and the focus
  line can pass; warnings and trust notes stay verbatim); participant data copied out of its folder is not
  recognised as such.
