# Gate: how often does the linker join two different people? (Olympic Boxing, hand-checked joins) -- written 2026-10-27, before any check

Branch `feat/linker-join-check`. This is option A of `docs/quality/olympic-swap-rate.md` (PR #32). It is committed
BEFORE the sampler, the checker or any sampled join exists, and before Brendan answers a single join. The commit
hash of this file is the evidence of order. Nothing below changes after an answer exists; a later change is a
new, dated section that says why, and the original stays.

## Question
The linker (`src/boxingvi/linking.py`, PR #11) chains ByteTrack fragments into boxers A and B. A chain can only
change person at a JOIN (two consecutive fragments of one chain). How often does a join connect two fragments
that are NOT the same person? This is the identity input of the per-boxer report (ADR-0016, PR #33), which
today says "unverified". Headgear colour (#29) and glove colour (#31) both failed their gates, so this is the
remaining measured identity number.

Not in scope (as the option A record says): ByteTrack's own switches INSIDE one track id (e.g. id 1102 on
GH159681). Both the linker and raw ByteTrack inherit those, and the clip shows only the join.

## Where this follows the option A record rather than the session brief
- **Videos:** the 23 held-out videos of #29's manifest (`olympic_headgear.HELD_OUT`), not all 29. The 4 dev
  videos shaped the linker (GH078416 is the video Brendan eye-counted, and the re-acquisition rule came from
  them); GH089681 and GH118416 are out because Brendan saw them by eye on 10-22.
- **Rates:** the Wilson interval on the wrong-join share (the brief) PLUS a video-cluster bootstrap on wrong
  joins per minute (the record).
- **the reviewer never views a clip** (the record), which is stricter than "never judges a join" (the brief).

## Sampling frame (fixed now)
- The linker exactly as the per-boxer report runs it: `linking.link_video(table, attrs)` with default
  `LinkParams()` (no `--reacquire`), on the tracks and annotations that `olympic_link.load_video` accepts, as in
  `olympic_boxers.py`. Every join therefore has a gap of 1 to 100 frames (at most 2 s at 50 fps).
- A **join** is every consecutive pair (fragment 1, fragment 2) of every non-empty chain, in every one of the 23
  videos. Its **gap** is fragment 2's first frame minus fragment 1's last frame.
- The manifest is enforced: all 23 must link, or nothing is sampled. An id outside the manifest is never read.
- What the frame stores per join: video id, the two track ids, fragment 1's last frame, fragment 2's first frame,
  the gap. It does NOT store which chain (A or B) the join came from, the link cost, or the jump.

## Sample (fixed now)
- **Strata by gap:** short = gap <= the median gap of the frame; long = gap > the median. Proportional
  allocation, n_h = round(100 x N_h / N) (adjusted by one so the total is 100), so the sample is self-weighting
  and the pooled share needs no reweighting. Both strata are about 50.
- **Draw:** `random.Random(20261027)`; the frame is sorted by (video, fragment 1 id, fragment 2 id); each stratum
  is sampled without replacement in that order; then the 100 are shuffled with the same generator. That shuffled
  order is the order shown, so consecutive clips rarely come from one chain.
- **n = 100** (Brendan, 2026-10-28).

## What Brendan sees, and what he answers
For each join, one window (the keypress pattern of `label_moments.py`, PR #35) loops a short clip:
fragment 1's last 1.0 s, the gap, then fragment 2's first 1.0 s, back to back at real speed, then holds a
side-by-side pair of stills (fragment 1's last frame | fragment 2's first frame). Fragment 1's box is outlined in
**cyan**, fragment 2's in **yellow**, the same two colours on every join. No other box is drawn.

Never shown: which boxer (A or B) the linker made the chain, the link cost or jump, the stratum, any other
fragment of the chain, or anything about other joins. The window shows "join k of 100" only.

Answer per join, for the person in the cyan box at its end and the person in the yellow box at its start:
- **S** same person;
- **D** different person (the other boxer, the referee, anyone else);
- **U** can't tell (occluded, the box holds two people, too small, the clip does not decode).

Every answer is saved at once; the session times itself (idle gaps capped as in PR #35) and resumes at the first
unanswered join; the sample file's sha256 is pinned in the answers file so a different sample cannot be resumed
into. The tool stops for a break at 50 minutes. the reviewer never views a clip, never judges a join and never
suggests an answer. Samples, clips and answers stay under the gitignored `results/`; only aggregates enter docs.

## Metric (computed once, after the last answer)
- **Join error rate** = D / (S + D), with the **Wilson 95% interval**. U is excluded from the rate and reported
  as its own count and share.
- **Implied misattribution (an estimate, stated as one).** A wrong join switches the chain to another person
  from that join on, until another wrong join. If each join is wrong independently with probability p, a
  fragment k joins into its chain is someone other than the chain's first person with probability
  (1 - (1 - 2p)^k) / 2 (an odd number of switches; it treats every wrong join as a switch between two people).
  M(p) is that probability averaged over the frame's credited punches by the position of the fragment each is
  credited to. Reported as M(p_hat) with [M(lower), M(upper)]. Its assumptions (independent errors; jumps to a
  non-boxer treated like a switch; measured against the chain's first person, not its majority person) are
  reported with it.
- **Wrong joins per chain-minute** = rate x (joins in the frame) / (minutes of chain time in the frame), with a
  95% video-cluster bootstrap (resample the 23 videos with replacement, 10,000 draws, seed 20261027,
  percentile; a draw with no decided sampled join is dropped and counted, and more than 1% dropped means no
  interval is reported). Descriptive, not gated.
- Also descriptive: the rate per stratum (short / long gap) and per camera (kam2 / kam4; under 20 decided is
  "not established"), and the minutes Brendan spent.

## Bar -- AGREED by Brendan 2026-10-28, before any check (option 1 of three; ADR-0018 in the vault)
The identity gates so far asked for 0.90 correct (ADR-0013, #29). The same bar, stated for punches: the per-boxer
report may call identity **acceptable** only if at most 10% of credited punches are on the wrong person. Let
**p\*** be the join error at which M(p\*) = 0.10 on this frame (solved by bisection; computed from the frame's
structure when the sample is drawn, before any answer, and recorded in the sample file).

| verdict | condition (all need: U <= 30% of answers and >= 70 decided) |
|---|---|
| **measured, acceptable** | Wilson upper bound <= p\* |
| **measured, unreliable** | Wilson lower bound > p\* |
| **measured, inconclusive** | the interval contains p\* |
| **not established** | U > 30% of answers, or fewer than 70 decided, or not all 23 videos linked |

Said plainly before any check: with about 23 joins per chain (the dev videos' count, the only link count read so
far), p\* is about 1%. At n = 100, even 0 wrong joins gives a Wilson upper bound of 3.7%, so **"acceptable" cannot
be reached at n = 100** (it would need about 400 checks with no error). The check can establish "unreliable"
(from about 4 wrong joins in 100), or end "inconclusive". Either is a measured number the per-boxer report and the
paper can state, which "unverified" is not.

Rejected (ADR-0018): "acceptable" when p_hat <= p\* (it could pass on 0 or 1 errors while the interval still allows
about 25% misattribution), and a raw join error <= 5% (it ignores compounding: 5% per join is about 30% of punches).

## If it fails
"Unreliable" or "inconclusive" is reported as measured, once, with the numbers: the per-boxer counts stay
conditional on identity. Nothing is re-sampled, re-asked or re-drawn on these 23 videos, and the linker is not
tuned on the answers.

## Data rules
Olympic Boxing is research-use only and never redistributed. Frames are decoded in memory for the window; the
sample, the answers and any rendered clip stay under `results/` (gitignored). No Olympic frame on a slide or poster.

## Result, 2026-10-28: MEASURED, UNRELIABLE (sample `775d7fa8292a`, scored once)
Sample: 1,030 joins in the 23 held-out videos (`sample` linked all 23, nothing missing); 100 drawn; p\* = 0.0088
from this frame's chains. Brendan answered all 100 in 11.97 minutes; the reviewer viewed no clip and suggested no answer.

| | S | D | U | join error (D / decided) | Wilson 95% |
|---|---|---|---|---|---|
| **all 100 (gate)** | 34 | 64 | 2 | **0.653** | **[0.555, 0.740]** |
| short gaps (<= median) | 24 | 25 | 1 | 0.510 | [0.375, 0.644] |
| long gaps (> median) | 10 | 39 | 1 | 0.796 | [0.664, 0.885] |
| kam2 | 16 | 25 | 0 | 0.610 | [0.457, 0.743] |
| kam4 | 18 | 39 | 2 | 0.684 | [0.555, 0.790] |

U = 2% (limit 30%); 98 decided (>= 70). Lower bound 0.555 > p\* 0.0088: **measured, unreliable**. Implied
misattribution 0.457 [0.454, 0.459] (the two-person model saturates at 0.5: a chain's later fragments are a coin
flip between people). Wrong joins per chain-minute 5.78, video-cluster bootstrap [4.82, 6.77] (0 of 10,000 draws
dropped).

**Checked after scoring, not changing it:** the window's seek lands on the frame the tracks call frame n. On 8
boundary frames of the 4 earliest sampled joins (3 videos), the seek-decoded frame equals the sequentially decoded
frame n exactly (mean |difference| 0.00; frames n +- 1 differ by 0.5 to 9). Numbers only; no frame was saved or viewed.
So the boxes were drawn on the frames they belong to; the rate is not a decoding offset.