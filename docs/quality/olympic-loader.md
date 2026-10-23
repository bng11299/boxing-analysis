## `feat/olympic-loader` -- 2026-10-06

Strict loader for the Olympic Boxing release (`src/boxingvi/olympic.py`) and an audit
(`olympic_audit.py`, writes `results/olympic_audit.json`).

- **Acceptance:** every one of the 29 annotation files parses under strict rules; the release
  reproduces the paper's countable facts or the mismatch is reported; CVAT track semantics
  (visible until the next `outside` keyframe) are honoured.
- **Threat sketch:** asset: training labels whose "no punch" frames must really be no punch.
  Entry point: third-party CVAT JSON. Top threats: a mis-read track end shortening every
  punch; an under-labelled video teaching "no punch" on real punches.

### Iteration 1
- **Real-world facts:** 29 videos, **312,774 frames: exactly the paper's total**; 4,605 events
  + 4 empty tracks = 4,609 tracks. The paper: a licensed referee "looked at 312,774 frames" and
  judged each punch or not, misses and blocks included.
- **Test:** 23 incl. real-data; each of 6 guards reverted fails a test.
- **Review: SHIP**, findings: an event must end at the frame before its `outside` keyframe
  (965 of 4,605 ran 2-12 frames past the last visible box); one track reappears after a 1-frame
  hide (kam4_gh069681) and a longer reappearance would have merged silently; box bounds too
  loose (none of 17,649 real boxes crosses the edge); the audit called the release "matching
  the paper" while the paper's 11,345 punch frames matches no derivable count; minutes
  hard-coded at 50 fps; kam2/kam4 may film the same punches, so splits must hold out bouts.

### Iteration 2
- **Fix:** all of the above. Reappearance merged only after <= 2 hidden frames, else refused.
- **Real-world facts:** punch frames to the outside keyframe **21,564 (paper: 11,345) -- reported
  as a mismatch, not fitted**; 1 reappearing track, 0 open-ended; strict bounds accept every box.
  Flags (each needs a viewed spot-check before any exclusion): kam2_gh208416 and kam2_gh218416
  at 21 and 18 punches/min against a median of 42.7; kam4_gh099681 95% single-keyframe labels.
  Cameras cover 47.1 vs 57.2 min, so their density gap (38.5 vs 48.9/min) is not like-for-like.
- **Test:** 33; all suites green.
- **Open:** completeness can only be tested with pose -- fast wrist bursts far from any label --
  which needs both boxers tracked (YOLO + ByteTrack), i.e. Phase 2's tracking brought forward.

