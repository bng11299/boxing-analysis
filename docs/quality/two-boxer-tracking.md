## `feat/two-boxer-tracking` -- 2026-10-06

Every tracked person per frame (`pose_yolo.PoseTable`, `extract_tracks`, `yolo_tracker` with
ultralytics' ByteTrack, `config/bytetrack_50fps.yaml`), `run_ingest.py --track`, a tracks
overlay with ids, and crediting each labelled Olympic punch to a person (`olympic.attribute`,
`olympic_attribute.py`).

- **Acceptance:** (1) nobody is chosen at extraction (the ring has a referee); (2) a punch is
  credited without using the label's hand, so the left/right convention is TESTED; (3) blocks
  and clinches come out ambiguous, not settled arbitrarily; (4) ids on the overlay are legible.
- **Threat sketch:** asset: per-boxer punch sets that later train and score the finder. Entry
  points: tracker output, CVAT boxes. Top threats: a statistic that cannot fail certifying a
  convention; an attribution rule that quietly drops one kind of punch from the training set.

### Iteration 1
- **Test:** 28 (fake tracker: two boxers + referee, ids -1 and duplicates); 8 guards bite.
- **Review: FIX FIRST.** Blocking: the thrower was chosen BY the labelled hand's wrist, so under
  a mirrored convention events dropped out and "labelled hand nearer" stayed high whatever was
  true -- a gate that could not fail. Also: first-box-only distances; 10% boxer rule shape-only;
  ByteTrack buffer 0.6 s at 50 fps and conf 0.25 starving its second pass; overlay ids clipped
  at the edge, border only when nobody tracked (never, with a referee present).

### Iteration 2
- **Fix:** nearest wrist of ANY person, either hand, against the per-frame interpolated glove box;
  credit only with a margin (else ambiguous); convention = share of credited punches whose
  nearest wrist is the labelled hand. Tracker yaml (buffer 100 frames), `--track-conf 0.1`.
  Overlay: border under 2 people; ids clamped whole and below the frame counter.
- **Test:** a mirrored pose model gives 0% agreement vs 100% unmirrored; restoring the old
  choice fails it. The edge-id test was wrong twice (the id still fit; then a pixel COUNT that
  could not separate the cases) and is now a position check.
- **Review: SHIP**, with two biases: a one-frame defender touch vetoed credit (minimum over
  frames), which would under-represent blocks and head shots in any credited subset; and a
  left-sorts-first tie in the thrower's own wrists inflated agreement for left-labelled punches.

### Iteration 3
- **Fix:** rank and margin on the MEDIAN distance over the event's frames (minimum kept only for
  "reached the glove"); punches whose thrower's two wrists are equally near give no hand evidence
  and are counted apart; ambiguous share reported by target.
- **Test:** 34; each of 5 attribution rules reverted fails a test. All suites green.
- **Not yet run on real footage:** the laptop smoke run (one Olympic video, `--track
  --max-frames 3000`) is the first real-world fact: ids, boxer co-presence, credited share,
  the convention share, and ambiguity by target.

### 2026-10-07 -- first real-world run (29 Olympic videos, RTX 3060 laptop, YOLO11m-pose + ByteTrack)
- **Coverage:** 28 of 29 tables complete; GH088416 holds only its first 3,000 of 12,524 frames
  (the smoke run's file: the full run's resume check tested that a file existed, not that it was
  complete). 303,250 of 312,774 frames tracked.
- **Crowd:** a median of 8 people per frame and 541 track ids per video (116-916): spectators,
  judges and coaches are in view, and boxers fragment across many ids. The two most-credited ids
  take 7-33% of a video's credited punches and are almost never on screen together, so boxer
  identity CANNOT be read from ids yet: tracklet linking is Phase 2 work, now measured.
- **Attribution (4,490 punches in tracked frames):** credited 2,363 (53%), too close to call ~46%,
  no confident wrist near the glove < 1% -- the pose stage reaches almost every labelled punch.
- **Left/right convention, tested:** the nearest wrist is on the labelled hand in **78% of 1,756**
  credited punches with clear hand evidence (607 ties left out). A mirrored convention would sit
  near 0%; chance is 50%. COCO's left is the boxer's left. The 22% disagreement is the size of
  wrist confusion at contact, worth knowing before per-hand training.
- **Ambiguity by target:** head 41%, miss 47%, body 61%, block 64%. The credited subset therefore
  under-represents body shots and blocks; any training set drawn from it must be reweighted or
  say so. kam4_gh099681 (95% single-keyframe labels) is 82% ambiguous: its labels give the
  median rule a single frame, at contact.

