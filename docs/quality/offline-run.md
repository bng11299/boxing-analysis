# Quality loop: `feat/offline-run` (2026-11-05)

Post-freeze; touches no RESULTS.md number. Built on `feat/scouting-summary` (#45).

## Acceptance criteria
1. A fresh `git clone` runs solo and `--two-boxer` analysis, video in -> every output out, with every socket
   call refused: `python tests/test_offline.py` exits 0.
2. Every step that needed a person at a Python prompt has a CLI: identity (`make_identity.py`), weights
   (`scripts/fetch_weights.py`); data download (Drive sign-in) and stance stay explicit, documented inputs.
3. `--narrate gemini` is optional and fails closed: no key, malformed key, bad key, timeout, network error,
   junk reply, invented number, injected text, participant footage -> template kept (`tests/test_gemini.py`,
   participant case in `scripts/offline_smoke.py`).
4. The default (`--narrate template`) makes no network call; ultralytics runs with `YOLO_OFFLINE=1`.
5. README "Run it yourself": install to outputs, from the repo and the pinned weights.

## Threat sketch
Assets: the API key; participant-derived numbers. Entry points: `GEMINI_API_KEY`, the Gemini reply, fact text
in the prompt. Trust boundaries: this machine -> Google; model reply -> report. Top threats: (1) participant
numbers sent to a cloud model; (2) a model reply (or text planted in a label) putting a false number in a report.

## Iteration 1
- Gates: full suite (30 files, all green, incl. new test_gemini 46 and test_offline 12); review (code-reviewer
  subagent); security pass on the changed files.
- Found while building: a digit planted in a fact's text (a label) was lendable to the model's sentence, so an
  obeyed injection passed `narrate.check`. Fixed: `_fact_tokens` lends only numbers the templates produced.
  Also: `run_ingest.py` never set `YOLO_OFFLINE` (ultralytics DNS and usage events); fixed in `pose_yolo`.
- Review verdict FIX FIRST: solo `analyse_clip --narrate gemini` on a participant video sent the prompt,
  because `report_punches` only sees staged files under `--out-dir`. Fixed: `analyse_clip` checks the source
  video against the participant roots and keeps the template; negative test added and mutation-checked (fails
  with the guard removed).
- Security: key from the environment only, validated `[A-Za-z0-9_-]{20,200}` (no header injection), sent in
  `x-goog-api-key`, never in the URL; reasons carry an HTTP status at most, never server text or the key;
  reply capped at 256 KiB and parsed strictly; model name allowlisted by regex; stdlib HTTPS with certificate
  verification and a 30 s timeout, so no SDK to pin. Weights downloads are size-capped and kept only on a
  sha256 match.

## Iteration 2 (after the fix)
- Gates: test_offline 12, test_gemini 46, test_narrate 88, test_analyse_clip 113, test_ingest 185,
  test_boxers 168: all green.
- Reproduce: test_offline clones HEAD `32edb4b` (no results/, weights/, data/) and runs offline_smoke there:
  green, same checks as the working tree.

## Residual risks (accepted)
- Two-boxer reports stay template-only (no narrator path there yet).
- `make_identity.py` records `by-hand, ungated`: a person's reading of the overlay, not a gated method.
- The four BoxingVI-derived model files cannot ship (licence); a fresh clone rebuilds them (~70 min CPU).
- `fetch_weights.py`'s download path was checked against tampered and existing files, not a live download.
- No live Gemini call was made: the HTTP layer is tested through an injected `post` only.
