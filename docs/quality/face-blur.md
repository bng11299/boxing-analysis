## `feat/face-blur` -- 2026-10-19

A privacy control: protocol FA26-22-NG as amended (vault ISSUES-LOG O57, O58, O63)
says the second annotator sees face-blurred clips only and is never told names.
`blur_faces.py` + `src/boxingvi/faceblur.py`.

**Acceptance criteria.** (1) Every planted face in synthetic frames is covered by the
blur AND its texture is destroyed in the pixels, in every mode, including after H.264
encoding. (2) Every failure path (predictor raised, malformed output, nobody found, a
person missing against neighbouring frames, overlapping people with a missing or
ambiguous face, a face outside its box) blurs the whole frame. Under random confidence
loss on every joint, random body sizes and cropped boxes, no planted face survives.
(3) The output has one H.264 video stream, no audio, and none of the input's planted
metadata, or the participant name in its file name, anywhere in its bytes; the report
names no file. (4) Output is refused, with a reason code and before a frame is
decoded: outside `~/comps-participant-data/` (unless overridden, recorded in the
report), and inside a git checkout, the vault, a Syncthing folder (config or marker),
a cloud-synced folder or a network path, override or not. (5) No unblurred frame is
ever handed to the encoder (a spy on the writer); a failure leaves no file behind.
(6) On own footage only, no face is recognisable in sampled frames that include the
fastest motion; the frames are saved for Brendan to confirm.

**Threat sketch.** Assets: participants' identities (faces; names in file names and
metadata; GPS and creation time in phone video). Entry points: the input video
(untrusted container), the predictor's output (a model can be wrong in any way), the
output path, the YOLO checkpoint (a pickle; pinned sha256, as in ingest). Trust
boundaries: predictor output into `plan_frame` (validated, and every doubt goes to
more blur); the output path into `check_output_location`. Top threats: (1) a face
that the detector misses or misplaces reaches the annotator unblurred, most likely in
fast motion or a clinch; (2) blurred footage, or the report, lands somewhere it syncs
or is committed, or carries a name or a location.

### Iteration 1

- **Build:** mosaic (block means; cannot be undone the way a Gaussian can be partly
  deconvolved). 3x3 cells over a face patch, 16 cells along the long side of a frame
  blurred whole. Two passes: predictions only are kept from the first. Frames reach
  the encoder through a pipe, and the writer's `private=True` strips the muxer's own
  tags. `YOLO_OFFLINE=1` is set before ultralytics is imported: it sends usage events
  to Google Analytics when online (`ultralytics/utils/events.py`).
- **Test:** the random-degradation test found its first hole at once. When the only
  confident face point is one ear, the patch centre sits on the side of the head, and
  PAD 0.6 left a 2-px sliver uncovered. PAD is now 0.8, with a named regression test.
  73 checks. All other suites green.
- **Review (code-reviewer subagent): FIX FIRST.** B1: a head-and-shoulders box with
  one face point got a patch covering 31% of the head (the box-height floor assumed a
  whole body). B2: in a clinch both skeletons can claim one head, leaving the other
  bare. B3: counting people lets a false detection stand in for a missed boxer. B4:
  the output check was a denylist. B5: the network-path test could not fail (an
  earlier rule fired first), and no refusal test checked which rule fired.
- **Security (own pass):** the report carried the input's file name, and participant
  clips may be named after the participant. It now carries the input's sha256 instead.

### Iteration 2

- **Fixes:** B1, box size when the torso is not measurable. B2, overlapping people
  whose faces are close together, or inside each other's box, blur the whole frame.
  B3, one-to-one matching of boxes by position against every frame within 0.25 s.
  B4, deny by default under the participant root. B5, reason codes asserted in every
  refusal test. Also: input hashed before and after (the plans must fit the pixels);
  a re-check of the output location before the final rename; overflow-safe sizes;
  `missing_conf=0.0` (a model with no keypoint confidences blurs more, not less);
  `\\?\` and network-drive handling; cloud names matched exactly where they are
  generic ("boxing" is not "Box").
- **Test:** the random test, extended to cropped boxes and body sizes, found 20
  scenes where the box-only patch (top third) missed the chin of a cut-off box. A box
  that is not body-shaped (height < 2 x width) is now blurred whole. This departs
  from the brief's "top third" toward more blur, and is flagged in the PR. 103
  checks. 15,000 extra random scenes on three other seeds: 0 faces left. All other
  suites green.

### Iteration 3

- **Review of iteration 2: FIX FIRST.** B1-B5 and every suggestion taken were verified
  closed. Two new holes appeared in the new rules. N1: the distinct-head test ran only
  for boxes that intersect, but a face may sit up to 25% outside its own box, so a
  skeleton reached a neighbour's head across a gap, leaving the neighbour's real head
  0% covered. N2: hips the model imagines just under the shoulders in a crop "measured"
  a torso and switched off the box-size floor (31% of the head, B1 again).
- **Fixes:** the distinct-head test runs on every pair with faces. The box floor stays
  unless the skeleton reaches the legs (`_whole_body`). A legs-only box no longer
  vouches for a person in neighbour matching (top edges must agree). An
  `InvalidPrediction` sentinel. `report_extra` cannot overwrite core fields. A failed
  report move removes the video and any stale report.
- **Test:** 108 checks. The random test now imagines hips and legs inside crops.
  15,000 extra scenes: 0 faces left.
- **Real data (iteration-2 code, own footage IMG_4395, 2659 frames at 60 fps):** 25.9%
  of frames blurred whole: 577 a person missing against the neighbouring frames, 52 nobody
  found, 61 overlaps. YOLO found 2-4 "people" in 96 frames of a solo clip. Over-blur,
  not leakage; it is the cost of DET_CONF 0.10 and the neighbour rule.

### Iteration 4 -- past the cap, NOT independently re-reviewed

- **Review of iteration 3: FIX FIRST.** N1 and N2 closed. Two siblings: P1, one person
  with a face and a neighbour without, the face sitting on the neighbour's head across a
  gap (owner's head 0% covered); P2, a knee imagined BELOW a mid-frame crop satisfied
  `_whole_body` (31%).
- **Fixes (each only adds blur):** a lone face within the slack of another person's box
  blurs the whole frame. `_whole_body` requires the leg joint inside the box. Both are
  named tests; the random test also imagines legs below crops. 110 checks, 15,000 extra
  scenes with 0 faces left, every other suite green.
- **Not taken:** P3, a tall crop with three layers of wrong confident joints reaches 80%
  head coverage. Closing it (a patch floor of half the box width) enlarges every patch
  on a wide guard.

### Blocker report (iteration cap reached)

**Gate that keeps failing:** review. Each round closed what it found, and the next round
found a sibling of the same shape: a rule that trusts keypoint geometry, defeated by a
second piece of confidently wrong geometry (N1 -> P1, N2 -> P2).

**Root cause hypothesis:** the patch is placed and sized from the keypoints, and each
new guard enumerates one more way they can be wrong. The space of wrong-but-confident
skeletons is open-ended, so a guard list never closes it. Tests and random scenes show
no remaining hole, but they only sample the failure shapes we already thought of.

**Options:**
1. **Box-based blur is the floor; keypoints only ADD blur.** Every person always gets
   the box-only patch (top third, or the whole box if not body-shaped), and the
   keypoint patch is unioned on top. Closes the whole class, P3 included, because no
   keypoint can shrink what the box alone would blur. Cost: the top third of every
   boxer (head, shoulders, guard) is always mosaicked, which is where feints show.
   That is a real cost to the feint labels this clip exists for.
2. **Keep the guard list; make the human check part of the protocol.** Ship as is,
   with a per-clip spot check of sampled frames (fastest motion, every `box` and
   `ambiguous_faces` frame) by Brendan before a clip goes to the annotator. Cheapest.
   It relies on a person catching what the code missed.
3. **Another review round with a raised cap** (`--max-iterations 5`), specifically on
   P3 and on sampled frames from the first real sparring clip. That is real data with
   two boxers, which no test here has.

### Real check -- own footage only, final code (2026-10-19)

Run with `--allow-outside-participant-root` (own footage, scratchpad output), YOLO11m-pose
on CPU, weights pinned by sha256, `YOLO_OFFLINE=1`.

| Clip | Frames | Face patch | Box patch | Whole frame | People per frame |
|---|---|---|---|---|---|
| IMG_4395 | 2659 | 1828 | 117 | 714 (26.9%): 603 missing vs neighbours, 52 nobody, 33 overlap, 26 ambiguous | 0:52, 1:2511, 2:85, 3:10, 4:1 |
| IMG_4406 | 2353 | 1877 | 90 | 386 (16.4%): 376 missing vs neighbours, 10 overlap | 1:2265, 2:88 |

Twelve frames were extracted from the blurred output into `results/face-blur-check/`
(gitignored): per clip, the three fastest-motion frames (measured on the original in
memory), one box-patch frame and one whole-frame blur, plus the two fastest
face-patch frames of IMG_4395. Looked at one by one: **no face is recognisable in any
of them.** Patch frames show a 3x3 mosaic covering the head with a wide margin
(shoulders and guard included). Whole-frame blurs leave a close-up face as at most a
few flat 120-px cells. Pending Brendan's own look.

Observed costs, not leaks: the solo clips have 2-4 "people" in 96 and 88 frames (false
detections at DET_CONF 0.10, e.g. furniture at the frame edge). Each one blurs about
+/-15 frames whole through the neighbour rule. Patches are large (about 3 head widths),
so the guard is mosaicked whenever the face is.

### Iteration 5 -- option 1, the box is the floor (Brendan's choice, 2026-10-20)

- **Change:** every person's box patch is always blurred; the face patch is added on
  top, so keypoints only add blur. All whole-frame rules unchanged.
- **Test:** the floor at a third missed P3 (96%). The floor became max(1/3 of the box,
  1 box width), then 1.5 box widths after review H1 (a box-wide head at exactly 2:1,
  no face points: 96%). New tests: every person's box patch is blurred in every
  non-full plan (500 noisy-skeleton scenes); face points confidently on the chest
  still leave the head covered; the random test also scatters face points confidently
  anywhere in the box. 114 checks; 15,000 extra scenes with 0 faces left; every other
  suite green; a clean clone of `08ec430` gives the same counts.
- **Review: SHIP.** All eight earlier leak probes (B1-B3, N1, N2, P1-P3) closed. The
  reviewer found no guard rule dominated, so all are kept. Residuals, now in the
  operator notes: a person never detected, and a box that consistently leaves out the
  head with no confident face points.
- **Security:** the change is blur geometry only, with no new input, path or I/O
  handling. The iteration-1-4 findings stand.
- **Real check (own footage, final code):** modes and whole-frame counts are as before
  (26.9% and 16.4%); the floor changes regions, not frames. The 12 frames in
  `results/face-blur-check/` were re-extracted and looked at again: **no face
  recognisable.** **Cost seen:** a boxer's box in guard is ~0.4x as wide as tall, so
  1.5 box widths is ~60% of the box. The floor mosaics head to hips (IMG_4406 frame
  209), or to the knees (frame 2204): the guard, the torso and most of the arms. That
  is far more than the "top third" option 1 described, driven by P3 and H1, which are
  synthetic close-up crops. Open decision for Brendan.

### Iteration 6 -- option (c): the 1.5-width floor only at the frame edge (Brendan, 2026-10-20)

- **Change:** a body-shaped box gets the top third (+pad); it gets 1.5 box widths only
  when it touches the frame edge, where real close-up crops come from. A box that is not
  body-shaped still gets all of it.
- **Review: FIX FIRST, then SHIP.** The first version used a fixed 2 px edge band, which
  is finer than YOLO places an edge (one model pixel is about 3 frame pixels at 1080p).
  A crop 3 px short of the edge lost the extension, leaving 70% of the head covered. The
  band is now 1% of the frame's long side (6.4 model pixels at any resolution). New
  tests: crops 3 px short at 1080p and 6 px short at 4K, in all four orientations
  (69.6% each with the old band, 100% now), plus a mid-frame whole boxer whose floor
  must stop at the top third. The reviewer would move to 2% unless a histogram of
  real box-to-edge gaps argues for 1%; left as a tuning decision. Black bars baked
  into a clip are an operator rule ("feed the camera's own file"), not a code check.
- **Accepted residual (Brendan's decision):** a mid-frame body-shaped box whose head
  reaches below the top third + pad, with no face patch.
- **Test:** 121 checks; 15,000 extra scenes, 0 faces left; every other suite green; a
  clean clone of `c5728ed` gives the same counts.
- **Real check (own footage, final code, `edge_frac` 0.01 in both reports):** mode and
  whole-frame counts are unchanged (IMG_4395: 26.9%; IMG_4406: 16.4%). All 12 frames
  were re-extracted and looked at: **no face recognisable.** The floor on a mid-frame
  boxer now stops at about the chest: in IMG_4406 frame 2204 the torso, arms and a
  hand are visible again. Close-up and edge-touching boxes (IMG_4395) are still
  mosaicked wholly or to 1.5 widths.

### Iteration 7 -- tighter blur for feint labels (Brendan, 2026-10-20)

- **Why:** Brendan found no face recognisable, but he "genuinely can't see" the boxer. Measured over
  every frame (16x16 blocks that were textured in the original and flat in the output):
  44-53% of the textured picture was mosaicked, and even typical frames hid the head,
  guard, shoulders and chest.
- **Change:**
  - Face patch: with two or more confident face points, the head is measured from the
    points (`head_width`: max over visible pairs of distance x frontal ratio), and the
    patch is ~1.8 head widths (was 3-4). One point, or clumped points, keep the
    body-sized patch.
  - Floor: the top fifth of a whole body (was a third), or down to the LOWER confident
    shoulder; a third + 10% with no confident shoulder, and always for boxes 4:1 or
    taller.
- **Review:** FIX FIRST twice, then SHIP.
  - R1: with arms raised the box top is the gloves, so the head is below the top fifth.
    With face points on a glove the head was 0% covered; the shoulder anchor fixes it.
  - Then the skeleton fitted up a raised arm (face on the glove, shoulders on the
    elbows) is ONE failure (0%). It is closed by the box shape: 4:1 or taller always
    gets the deep floor. Guard (2.6-3.4:1) and punch boxes keep the shallow floor.
  - A mean of the shoulders gave 47% with one shoulder high; the lower shoulder is now
    used.
  - The factors were checked against a 3-D head model (frontal 1.09-1.41x the true
    width), not against real heads.
- **Test:** 134 checks; 15,000 extra scenes with 0 faces left; every other suite green; a
  clean clone of `eae5b3d` gives the same counts.
- **Real check, own footage, det-conf 0.10 vs 0.25:**

  | Clip | Whole-frame blur | Nobody found | Textured area mosaicked (mean / median) |
  |---|---|---|---|
  | IMG_4395 @0.10 | 26.9% | 52 | 53% / 40% |
  | IMG_4395 @0.25 | 15.3% | 119 | 46% / 35% |
  | IMG_4406 @0.10 | 16.4% | 0 | 44% / 33% |
  | IMG_4406 @0.25 | 3.8% | 0 | 38% / 28% |

  0.25 removes most of the false "people" in these solo clips (2-4 people in 96 and 88
  frames, down to 35 and 2), and with them most whole-frame blurs. Its cost is misses
  (IMG_4395: nobody found in 119 frames instead of 52); those frames are blurred whole,
  so they are safe here. On two-boxer footage a sustained miss is not caught by the
  neighbour rule; pass --expect-persons 2. Recall on sparring is unmeasured.
- **What the tighter blur bought:** in face-patch frames the median mosaicked area moved
  little (27-31%), because the box floor now dominates. It must be box-wide (the head
  can be anywhere across the box when no keypoint is trusted) and reach the shoulders,
  so gloves held in guard at face height stay mosaicked. Below the shoulders the arms,
  hands and torso are visible again (IMG_4406 frame 970: arm, hand and watch). Typical
  frames for both settings are in `results/face-blur-typical/`.

### Iteration 8 -- hybrid with deface's face detector (Brendan, 2026-10-21/22)

- **Why:** Brendan found the blur "terrible" for labelling and asked for a free tool.
  deface (CenterFace, MIT, local) on own footage obscured ~8% of the textured picture
  (ours 39-47%) and left the guard visible, but alone it found no face in 3.8% (IMG_4406)
  and 28.5% (IMG_4395) of frames where YOLO saw a person. A contact sheet of those frames
  showed real faces unblurred: mid-punch, in profile, behind a glove.
- **Change:** a person drops the box floor only when the CenterFace box and YOLO's
  keypoints agree. That requires a frontal face (nose and both eyes confident, all inside
  the box), a box at least half the head wide (head measured from eye-to-eye distance), a
  box centre on that person, and no confident wrist inside the box. The agreed person gets
  the box grown 1.3x plus a 1.2-head-width square on the nose and eyes; a box 4:1 or
  taller keeps its deep floor anyway. Everyone else keeps the floor + face patch.
  Unclaimed detections are blurred; malformed output or a failing detector means no
  detections. The model is pinned by sha256; deface 1.5.0 is installed `--no-deps`.
- **Test:** 155 checks. A 1000-scene random hybrid test found 79 leaks before the keypoint
  square was unioned in (one-sided face points "agreeing" with a shifted box); now 0, and
  15,000 more scenes on other seeds give 0.
- **Review:** FIX FIRST, then SHIP with one residual for Brendan.
  - D1: arms up, face points and box both on the glove: 0%.
  - D2: guard, glove decoy: 30%.
  - Partial three-quarter boxes.
  - Closed by the arms-up floor under agreement, a wrist veto and frontal-only agreement.
  - **Residual:** a glove at the cheek fools both models while YOLO misses that wrist.
- **Real data:** only 6 of 200 sampled persons agreed. The cause was the max-over-pairs
  head estimate running ~2x the true head on real keypoints, so every CenterFace box
  failed the size check. With the head measured from the eyes: 97 of 200.

  | Clip @det-conf | Face blur only (agreed) | Fallback | Whole frame | Textured area obscured (median) |
  |---|---|---|---|---|
  | IMG_4406 @0.25 | 48% | 48% | 3.8% | 17% (deface 7%, previous blur 28%) |
  | IMG_4395 @0.25 | 28% | 66% | 15.3% | 30% (deface 8%, previous 36%) |
  | IMG_4406 @0.05 | 27% | 28% | 44.5% | -- |
  | IMG_4395 @0.05 | 23% | 45% | 32.3% | -- |

  0.05 (Brendan's ask) multiplies false "people" (up to 6 in a frame) and with them
  neighbour-rule whole-frame blurs; 0.25 is the better setting. At every frame where deface
  leaked a face (18 sampled per clip), the hybrid covers it. Agreed frames look like
  deface: face mosaicked, arms, hands and guard visible. Three-quarter and profile faces
  (the bladed stance) keep the fallback.

### Iteration 9 -- pure deface as the default, leak recorded as an accepted residual (Brendan, 2026-10-22)

- **Decision (Brendan, 2026-10-22):** "keep deface. deface only doesn't work amazingly when
  the face is occluded, so it's fine." Asked again after side-by-sides of the middle ground
  (deface plus the body-based blur only where deface finds no face) against pure deface:
  "Use pure deface with the leak rate recorded as an accepted residual in the PR and
  quality-loop log."
- **The measured residual, stated plainly:**
  - **Detector misses:** on own shadowboxing footage, deface (CenterFace, threshold 0.2)
    found no face in 89 of 2353 frames of IMG_4406 (3.8%) and 843 of 2659 of IMG_4395
    (31.7%; 724 of the 2540 frames with a person, 28.5%).
  - **Visible faces among the misses:** in contact sheets of 18 sampled no-face frames per
    clip, about 6 per clip showed the face substantially visible and unblurred (looking
    down mid-combination, profile while punching, half behind a glove); the rest were
    occluded, turned away or out of frame.
  - **Estimated leak rate (eyeballed sample, not a measurement of every frame):** roughly
    1% (IMG_4406) and 9% (IMG_4395) of frames with a person show an unblurred, visible
    face.
  - **Unmeasured:** sparring footage, with more profile, motion blur, two boxers and
    clinches.
  - **Disagreement on record:** the reviewer's assessment differs from the stated reason. Not
    all misses are occluded faces; some are visible faces in profile or mid-punch.
- **Protocol note (stated once, per the provenance/disclosure rule's spirit):** amendment
  O57 promises the second annotator face-blurred clips only. A visible face reaching the
  annotator is a deviation from the approved procedure; whether that needs reporting to
  the HSRRC is Brendan's call with Hector.
- **Change:**
  - `--face-policy {deface (default), agree, trust}`. deface means `plan_faces_only`: every
    CenterFace box grown 1.3x and mosaicked, and a frame with no detection is left
    untouched.
  - A detector error or malformed output still blurs the whole frame. No YOLO is needed.
  - The report counts `frames_without_face` and lists `no_face_ranges`; the CLI prints them
    on every run and warns above 30%.
  - **Review F1:** a detector that (nearly) never fires is broken, not the accepted miss
    rate. Over 50% faceless frames is refused before anything is written, unless
    `--allow-no-faces` (recorded). `--face-thresh` above 0.5 is refused under deface, and
    the threshold is recorded in the report.
  - `MODE_CODES` replaces first letters, because "faces" and "full" collided.
  - agree/trust (the earlier fail-closed paths, and the middle ground) remain as options.
- **Test:** 177 checks. New for the deface policy:
  - faces mosaicked, nothing else touched;
  - a no-face frame passes through unchanged, counted (the residual, tested so it stays
    visible);
  - errors and malformed output blur the whole frame;
  - a never-firing, a one-hit and a raise-once detector are all refused; 25% faceless
    passes;
  - the threshold cap; no YOLO needed.
- **Review:** FIX FIRST twice (F1, then its one-hit near-miss), then the remaining change
  was the fraction-based refusal, applied. agree/trust are unchanged from the approved
  round.
- **Real check (own footage, default policy):** IMG_4406 3.8% no-face, IMG_4395 31.7% (under
  the 50% refusal, over the 30% warning). At deface's miss frames the output is
  pixel-for-pixel the same residual as the deface CLI (sheet compared). These runs started
  on the code before the F1 refusal, which changes only refusals and report fields, not
  pixels.

