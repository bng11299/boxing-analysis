# Gate: appearance keeps two boxers apart (Olympic Boxing) -- written 2026-10-22, before any held-out number

Branch `feat/appearance-identity`. Committed BEFORE the v3 rule is designed and before any held-out video is
read by it; the commit hash of this file is the evidence of order. Nothing below changes after a held-out
number exists. If a later change is needed, it is a new, dated section that says why, and the original stays.

## Question
Can appearance (pixels, read for identity only, never for punch classification) keep the two boxers'
identities apart across a bout, where pose alone failed (PR #11 linker: 3 identity errors in about a minute
on GH078416; pose continuity rules cut credited punches on A or B from 65.4% to 38.4%)?

## Why this cue (evidence gathered 2026-10-22, before this gate)
- Brendan's eye check of a torso-region overlay on GH159681 (a DEV video): at 1:37-1:45 BOTH boxers wear red
  vests; one has blue gloves and blue headgear, the other red gloves and red headgear. The corner colour is on
  the headgear and gloves, not (always) the vest. That explains why the v2 torso rule (ADR-0013) scored chance.
- The same check: ByteTrack id 1102 jumps to the referee (frame 4880) and to the other boxer (frame 5039).
  Fragments are NOT always one person. This caps any per-fragment score and is reported, not excused.
- Brendan also answered for GH089681 and GH118416 (TEST videos under ADR-0013): one boxer red, one blue.
  Those two videos are therefore removed from the held-out set below.
- So the route is a COLOUR cue v3 on headgear and gloves (no weights, CPU). A learned re-ID embedding stays the
  fallback for footage without corner kit; it is not built here.

## Split (fixed before the design; never below the video)
- DEV (design only, numbers are not evidence): ADR-0013's frozen set, GH078416, GH159681, GH128416, GH069681.
- HELD-OUT (gate data): the other 25 minus GH089681 and GH118416 = **23 videos**.
- Robustness row (reported, not gated): also drop the held-out videos whose GoPro chapter number equals a dev
  video's on the other camera (GH079681, GH158416, GH129681), in case kam2 and kam4 film the same minutes
  (a label-timing cross-correlation on 10-22 found no clear twin: best r 0.16-0.31, never clearly above the
  runner-up; weak evidence, hence the row) = 20 videos.
- Disclosed: the 25 ADR-0013 test videos were read once before, by the v2 TORSO rule (52.2%, chance). v3 reads
  no statistic of any held-out video during its design. Boxers may recur across chapters of one bout, so a
  held-out video can show a boxer also seen on dev; a fixed red/blue rule is not fitted to individuals, so this
  is a stated residual, not a leak of labels.

## Freeze
The v3 rule (regions, colour bands, decision constants) is designed on DEV only and committed; the PR records
that commit hash. Only after it is the held-out run made, once. `olympic_colour.py` keeps `--split dev` as the
default so a careless run cannot read a held-out number.

## G1 -- does the cue tell the two boxers apart? (non-circular, ADR-0013's design)
Pairs: two track ids that BOTH carry a labelled punch credited to them and are on screen together for at least
0.5 s are two different people, so a working corner-colour rule calls them OPPOSITE colours. The labels never
say red or blue, so this cannot pass by construction. Chance is about 50% of decided pairs.

PASS needs all three, pooled over the 23 held-out videos:
1. opposite share >= **0.90** of decided pairs (ADR-0013's bar, unchanged);
2. >= **100** decided pairs;
3. the 95% video-cluster bootstrap lower bound (resample videos with replacement, 10,000 draws, seed
   20261022, percentile) >= **0.80**.

Reported with it (not gated): the Wilson 95% interval, per camera (kam2 / kam4; a camera with < 20 decided pairs
is "not established"), the robustness row, the share of co-present credited pairs that are decided at all (how
often the cue speaks), and the credited fragments' red / blue / none / unsure counts.

## G2 -- does it keep identities apart over a bout? (run only if G1 passes)
The linker (PR #11, as merged) is the BASELINE; the NEW linker is the same with the colour constraint (a red
fragment may join only the red identity, blue only blue; none / unsure as today). Both on the same 23 videos in
the same run, without and with `--reacquire`.

1. **Coverage kept:** credited punches on A or B. PASS: NEW >= BASELINE - 5 percentage points (pose continuity
   fell 27 points; that is what this guards against). Both-identified share reported.
2. **Swap rate, baseline:** links inside a baseline chain that join two fragments of opposite decided colour, per
   bout-minute of chain. Valid only because G1 validated the colour independently. NEW has none by construction,
   so it is not reported as a win.
3. **Swap rate by eye (the real-world fact):** two held-out videos drawn with `random.Random(20261022)` from the 23,
   one 2-minute stretch each (start drawn with the same generator). Both overlays, BASELINE and NEW, are named X
   and Y in an order drawn by the same generator, so Brendan does not know which is which. He counts identity
   errors: a swap between the boxers, or an identity jumping to a non-boxer. PASS: NEW total errors < BASELINE
   total errors, and NEW has 0 jumps to a non-boxer.

## If it fails
G1 or G2 failing is a clean Future Work finding (`PAPER-future-work.md` item 1): written up once, with the
numbers, and NOT iterated on held-out data. No v4 on these 23 videos.

## Data rules
Olympic Boxing is research-use only. Nothing derived (clips, crops, embeddings, colour files, overlays) is
committed or shown publicly; results stay in the gitignored `results/`. Pixels are reduced to counts in memory;
the reviewer never views a frame. No re-ID weights are used by this gate.

## Dated note, 2026-10-22 evening: settled AFTER this gate was committed, BEFORE any held-out read
Nothing above is changed; these make it precise. All come from dev numbers or from review, none from held-out data.
- **The reading unit.** For G1 each fragment of a co-present pair is decided from its rows on the frames the two
  SHARE, not over its whole length. Why (dev only): ids hop between people (the eye check above), and the same crown
  rule scored 89 / 124 opposite over whole fragments against 98 / 108 over the shared frames. Fragment-level decisions
  are still reported (`credited_by_decision`).
- **The rule** is `src/boxingvi/headgear.py` (headgear crown above the face, ADR-0013 v2's colour decisions); its
  dev numbers are in `docs/quality-loop.md` (`feat/appearance-identity`). kam2 is expected to be "not established":
  its blue kit is washed out (dev: 2 decided pairs).
- **The held-out manifest is enforced** (Astra review 1): the gate is evaluated only on exactly the 23 named videos,
  each read once; a missing or failed one means "not evaluated", an unknown id is never read.
- **Bootstrap draws with no decided pair** are dropped and counted; more than 1% dropped fails G1.
- The rule is FROZEN at the commit that adds this note; the PR records its hash.
