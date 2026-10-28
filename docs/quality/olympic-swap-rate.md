## olympic-swap-rate -- 2026-10-24: not measured (no label pins identity over time)

**Question.** How often does the two-boxer linker (`src/boxingvi/linking.py`, PR #11) swap the two boxers on
held-out Olympic Boxing videos, in swaps per co-present boxer-minute with an interval, against raw ByteTrack
fragments?

**Finding: no existing label can answer it.** The brief assumed each credited punch carries the labelled corner
or side. It does not.
- The 29 raw CVAT files (`annotations.json`) hold 4,609 punch tracks. Every track's and every shape's `attributes` list
  is empty, and `group` is 0 on all but one track. There are exactly 8 labels: {head, body, block, miss} x {the
  thrower's own left hand, own right hand}. The remaining keys are geometry and CVAT bookkeeping (`frame`,
  `points`, `outside`, `occluded`, `z_order`, `source`). No field names a corner, a colour or a boxer.
- #29's held-out pairs ("two co-present credited people are different boxers") are cannot-link constraints.
  They stay true when A and B are swapped, so they cannot detect a swap. A chain's fragments are time-disjoint
  by construction, so the constraint never applies inside one identity.
- Colour cannot stand in for ground truth: torso colour scored at chance (ADR-0013, 52.2%), and headgear colour
  failed its pre-registered held-out gate (83.2% vs 0.90, `docs/gates/appearance-identity.md`).
- The only identity ground truth anywhere is Brendan's eye count on DEV video GH078416: 3 identity errors in
  about a minute (1 boxer swap, 2 jumps to a spectator). That is one video and not held-out, so it is not a rate.

**Decision (Brendan, 2026-10-24): option C.** No held-out swap rate is reported. Identity on Olympic Boxing rests
on the two pre-registered failures above. No gate was committed and nothing was built. No held-out linking
number was read: only the 4 dev videos' link counts were read, to size option A (188 links over 12.7 min of
video, about 15 per minute; from `results/olympic_linking.json`, 2026-10-07, without `--reacquire`).

**Still open (option A).** Brendan checks a seeded sample of about 100 linker joins (a linker swap can only
happen at a join), under a gate committed before any held-out number:
- the 23 held-out videos of #29's manifest;
- Wilson interval on the wrong-join share, and a video-cluster bootstrap on the per-minute rate;
- the joins rendered as short before/after clips under `results/`, which the reviewer never views.

The cost is about 25 minutes of his viewing plus about half a day of tooling. It does not count ByteTrack's own
within-id switches (e.g. id 1102 on GH159681), which both methods inherit. This is worth doing only if glove
colour (#31) also fails its gate on Oct 28 and there is time before the Nov 6 freeze.

**Rejected (option B).** An automatic cut-and-relink proxy is optimistic: its cuts fall where ByteTrack was coping.
It also assumes an uncut fragment is one person, which id 1102 disproves. Usable as context only, never as the
paper's number.
