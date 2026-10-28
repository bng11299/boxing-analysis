# Gate: glove colour tells two sparring boxers apart -- written 2026-10-23, before any rule or number exists

Branch `feat/glove-identity` (on top of `feat/appearance-identity`, PR #29). Committed BEFORE the glove rule is
written and before any frame of the study footage is read by anything; the commit hash of this file is the
evidence of order. Nothing below changes after a test number exists. A later change is a new, dated section
that says why, and the original text stays.

## Question
In sparring where BOTH boxers wear black and their GLOVES differ in colour, can glove colour alone say which
boxer is which, moment by moment, better than tracking alone? Pixels are read for identity only (never for
punch classification), reduced to counts in memory, and never written as images.

## Status of the data (read first)
The three videos are participant footage (study FA26-22-NG, Garage Gang amendment). On 2026-10-23 their
intake STOPPED at consent scope, with a procedure deviation (vault ISSUES-LOG O65). **Nothing in this gate runs
on them until Hector rules and the faces are blurred.** Until then only synthetic fixtures exist, and they
never produce a reported number. If no ruling allows it by Fri Oct 30, the gate is recorded as NOT EVALUATED
and glove identity goes to Future Work.

## Prior evidence (stated so a pass or fail is read against it)
- On Olympic Boxing dev videos (#29), a fixed red/blue rule read on the gloves decided 82-86 of 141-145 pairs
  "opposite" (about 58-60%), worse than the headgear crown (90.7%), which itself failed its held-out bar (83.2%).
  Gloves are small, move fast (motion blur), and sit against other people's kit in a clinch.
- Differences here: two people only, side camera, 1080p, 30 fps; glove colours are NOT fixed in advance (they
  are learned per video, unsupervised), and the clothes are black, so the glove is the main colour on the arm.
  Expectation: uncertain. A fail is a clean Future Work finding.

## Roles (fixed before any frame is read; never below the video)
- **FAIL-CASE:** the video in which both boxers' gloves are the same colour (Brendan names it from memory of the
  recording, before any processing). It is the expected failure: reported as one, never tuned on.
- **DEV:** of the other two, the one whose ORIGINAL file has the lexicographically smaller SHA-256 (hashes taken
  at intake). Not hand-picked.
- **TEST:** the remaining one.
- The manifest (roles, hashes, frozen rule digest) lives in `~/comps-participant-data/glove-identity/`, never in
  the repo. With one TEST video, nothing here can show that the rule generalises (see "What it can show").

## Ground truth (Brendan, on the BLURRED copies only)
- Moments every **5 s** from 0 s. At each moment Brendan writes `L`, `R` or `?`: boxer A on the left, on the
  right, or not clear (overlapping, out of frame, gloves hidden, unsure). He never sees any method's output
  before labelling.
- A per video is written in the label file's header in words: "A = the boxer in <colour> gloves". In FAIL-CASE
  the gloves cannot define A, so A = whoever is on the left at the first clear moment, followed by eye.
- Time-boxed at about **30 min in total** for all three videos; the minutes per video are recorded in the label
  file and reported. Labels are written for all three at once; DEV labels may be read during design, TEST and
  FAIL-CASE labels are read only by the single frozen run (the tool refuses them before the freeze).

## Scoring
- A label at time t is scored on the pose-table frame round(t x fps). The two boxers are the two rows with the
  largest boxes. Fewer than two rows = **not tracked**; box centres closer than 0.25 x the mean box width =
  **overlapping**. A `?` label, not tracked and overlapping moments are counted and reported, never scored.
  Left and right come from the box centres.
- Each method gives each of the two rows **A**, **B** or **unsure**. If it gives both the same name, both count as
  unsure. A moment is **decided** when both rows are named, **correct** when the names match the label, and
  **wrong** otherwise.
- **Naming anchor.** A method's internal names (glove cluster 0/1, linker chain 0/1, track id) mean nothing on
  their own. Each method's FIRST decided scored moment fixes which internal name is A, and that moment is then
  left out of its scores. Every method, the new one and both baselines, gets exactly this one label for free per
  video.
- Per video and method, over the scored moments after the anchor:
  - **accuracy** = correct / decided, with a Wilson 95% interval;
  - **unsure share** = 1 - decided / scored;
  - **correct share** = correct / scored (how often the method gets it right, counting unsure as not right);
  - **swaps**: how many times, among decided moments in time order, the outcome flips between correct and wrong.
    Reported per 3-minute round, i.e. swaps / (span of scored moments in seconds / 180).

## Baselines (same moments, same anchor rule)
- **B1, ByteTrack alone:** a row is A or B if its track id is the id A or B had at the anchor moment, and unsure
  otherwise. A new id cannot be named.
- **B2, the pose-only linker** (`linking.link`, PR #11, unchanged). Sparring has no punch labels, so every
  fragment is credited one punch and the region is the whole frame. That adaptation is part of the baseline and
  is stated with its numbers. A row is A or B if its track id is chain 0 or 1 at that frame.

## PASS (TEST video unless stated; all four needed)
1. **Accuracy:** >= **0.90**, Wilson 95% lower bound >= **0.80**, over >= **30** decided moments.
2. **Speaks often enough:** unsure share <= **0.30**.
3. **Better than tracking:** swaps per round <= **1.0**, AND correct share > max(B1, B2) correct share.
4. **Fails safe on FAIL-CASE:** wrong <= **0.10** of FAIL-CASE's scored moments. Accuracy there is not demanded:
   the right behaviour is to say unsure, or to call the video not separable, rather than guess.

DEV numbers are reported and marked "design data, not evidence".

## Freeze
The rule's constants are written from first principles and synthetic fixtures. After that they are adjusted on
DEV only, if at all, and every adjustment is recorded in `docs/quality-loop.md` with its DEV number. Then
`glove_identity.py freeze` writes a digest of the rule's constants into the manifest. `score` refuses TEST and
FAIL-CASE unless the code's digest equals the frozen one. Each is scored once, and the output records the
digest and the commit.

## What it can show (said plainly)
One held-out video of one pair of boxers, one gym, one camera, one lighting setup, one glove-colour pair.
Moments within a video are strongly correlated (same people, same light), so the Wilson interval is optimistic,
and no video-level interval is possible with one TEST video. A PASS means "it worked once, under conditions that
make it easy". It does not mean the method works. A FAIL is reported as Future Work. Neither is iterated on
TEST or FAIL-CASE.

## Data rules
Participant footage. Everything derived from it (pose tables, glove counts, labels, manifest, score JSON)
stays under `~/comps-participant-data/`, and the tools refuse any other output location (deny by default). None
of it goes to the repo, its `results/`, the vault, git, slides, the poster, Astra, ChatGPT or any cloud service.
Tests use synthetic fixtures only. Reported numbers are counts and shares; no file name, frame or person
appears anywhere.

## Addendum 2026-10-26 -- no FAIL-CASE exists (written before any glove count, label or score on the study videos)
**Why.** Brendan watched the three blurred copies on 2026-10-26 and reports that in all three the two boxers
wear different-colour gloves. The FAIL-CASE role above assumed one same-colour video; there is none. Nothing
has yet been computed from the videos except pose tracks (no glove counts, labels, freeze or score), so no
number can have influenced this change.

**What changes (Brendan's decision, 2026-10-26).**
- **Roles:** DEV is still chosen by rule, now the video with the lexicographically smallest ORIGINAL SHA-256
  of all three. The other two are both TEST (`test` = the smaller remaining hash, `test2` = the larger).
  Each is scored once, after the freeze.
- **PASS:** bars 1-3 must hold on EACH TEST video, separately (not pooled). Bar 4 (fails safe on FAIL-CASE)
  is reported as **NOT TESTABLE**, never as met. A pass is reported as "bars 1-3 passed on two held-out videos;
  bar 4 not testable", never as a full four-bar pass.
- **What it can show:** two held-out videos of the same pair of boxers, in the same gym and light. That is
  still not evidence that the method generalises. Safety on same-colour gloves is untested and goes to Future Work.
- Everything else above is unchanged: ground truth, scoring, baselines, freeze, data rules.
