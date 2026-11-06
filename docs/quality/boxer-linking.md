## `feat/boxer-linking` -- 2026-10-07

Link each Olympic Boxing video's ByteTrack ids into two boxer identities (`src/boxingvi/linking.py`,
`olympic_link.py`, a linked overlay), then the completeness test per boxer (`completeness.py`,
`olympic_completeness.py`). ADR-0012: link first, test completeness second.

- **Acceptance:** (1) from planted scenes (two boxers each in 4 fragments with gaps, a moving referee,
  static spectators outside the ring) exactly the two boxers' fragment sets come back; (2) a chain's
  fragments are pairwise time-disjoint and the two chains share no fragment; (3) every parameter is
  fixed, with its reason beside it, before any real-data number is looked at; (4) the real-data gate
  asserts facts that can fail: how many credited punches sit inside the two identities (also with half
  the punches held out of the linking) and whether the identities are on screen together; (5) the
  completeness test judges nothing: it counts bursts away from labels and hands a seeded sample to a human.
- **Threat sketch:** asset: per-boxer minutes and burst counts that decide whether the referee's labels
  are complete. Entry points: tracks `.npz` (loaded without pickle), CVAT JSON (already strict), a GoPro id
  from the command line that becomes part of an output file name. Top threats: (a) an identity swap or a
  referee absorbed into a boxer's track silently moves the unexplained-burst rate (a referee waving is a
  burst); (b) a circular gate: punches drive the linking, so "most punches are inside the identities"
  certifies itself. Mitigations: punch evidence is required for any identity; held-out share and
  co-presence reported beside the in-sample share; the overlay exists to see swaps by eye.

### Iteration 1 -- build, test, real data (as specified)
- **Parameters** (value and reason beside each constant in `linking.py`): max gap 2 s (ByteTrack's own
  buffer); speed limit 5 torso lengths/s + 1 of slack; fragments under 0.2 s ignored; ring = 1st-99th
  percentile of credited glove-box centres widened by 2 torsos; a punch is worth 10 points, a second in
  the ring 0.5, a link 1 (+ up to 3 jump, 1 gap, 2 scale).
- **Found by the first synthetic tests, fixed before any real run:** with a free second chain, the
  cheapest use of it was to cut the ONLY boxer in two to save a link (`CHAIN_POINTS` = 8: more than the
  dearest plausible link, less than one punch); a chain of ring-time alone (a lone referee) could win the
  second slot (`MIN_CHAIN_PUNCHES` = 1); `--overlay ""` was falsy and silently skipped the id check.
- **Real-world facts, 29 videos, as specified (`results/olympic_linking.json`):** both boxers identified in
  50.5% of frames (frames-weighted; median video 57%), boxer A 76.6%, B 65.4%, either 91.5%.
  **65.4% of the 2,423 credited punches are on A or B (901 / 684); 50.7% with half the punches held out of the
  linking.** Co-presence (both identified, over the shorter identity's time) median 78%; one video
  (GH208416) 0%: its identity B is two fragments and A starts at frame 4557. Fragments per identity: median
  27 / 19 of a median 541 ids. Longest stretch with a boxer unidentified: median 4.2 s (A) and 13.9 s (B),
  worst 91 s and 156 s. Diagnosis (`credited punches by fate`): of 2,423, 1,585 inside, 803 in
  eligible fragments that overlap a chain in time (the chain was cut elsewhere), 12 unreachable, 23 outside
  the ring or too short. A chain is one path, so a stretch longer than 2 s with the boxer untracked ends it.

### Iteration 2 -- one change made AFTER seeing the real-data numbers (declared)
- **Change:** re-acquisition links (`reacquire_s`, off by default; `--reacquire` = 60 s): across gaps
  longer than 2 s, no speed limit (after 2 s a boxer can be anywhere in the ring), a flat price (4 points
  + 0.05/s). Reason: the diagnosis above, not a number to hit. It was NOT tuned: one value (a round's rest)
  was chosen and run once; no other parameter changed after seeing results.
- **Result (`results/olympic_linking.reacquire.json`):** credited punches on A or B 81.7% (59.1% held
  out), but both-identified falls to 44.5% (median 43%) and co-presence to 69%: the chain hops to
  punch-bearing fragments and skips punch-less time. It buys punches, not minutes, and the A/B labels are then only
  reliable between hops (nothing recognises a person). Hence off by default; both are reported.
- **Held out is the honest figure.** The in-sample share is high partly because punches choose the chains.
  Half held out, 51% (59% with re-acquisition) of the other half's credited punches are on the two identities
  -- a lower bound, because a fragment whose only punch is held out has no evidence left.

### Iteration 3 -- completeness (`results/olympic_completeness*.json`, sample `*.csv`)
- Rule: `results/punch_finder_v6.json` "chosen" (relative speed > 3.62 body lengths/s, 0.4 s apart) on each
  boxer's own rows, dropouts <= 0.05 s bridged, tolerance 0.1 s to ANY labelled punch of the video.
- **As specified:** 8,135 bursts on the two identified boxers (148 identified boxer-minutes: A 79.8, B 68.2);
  **5,506 are more than 0.1 s from any of the 4,605 labelled punches = 37.2 per identified boxer-minute**
  (A 38.1, B 36.1), against 31.1 labelled punches per boxer-minute; 2,705 of the 4,605 labelled punches have a
  burst within 0.1 s. With re-acquisition: 8,694 / 5,647 / 39.8 per boxer-minute, 3,035 labelled punches seen.
- **Superseded by Iteration 4** (the matching, the rates and the "upper bound" wording below were wrong; the numbers
  there are the pre-fix ones).
- **Not a finding about the labels.** On V6's tuning half the same rule had precision 0.46, i.e. over half its
  detections were not labelled punches; footwork, guard changes and slips move a wrist fast. "Unexplained" is a candidate for a
  missed label (NOT an upper bound: see Iteration 4) until a human classifies the sample (40 bursts, seed 20261007, stratified over videos):
  `results/olympic_completeness_sample.csv`, column `user_label` empty. Nothing was viewed or judged here.

### Iteration 4 -- Astra 6 review fixes (2026-10-07; verdict was FIX FIRST: 5 blocking, 4 not)

> **Partly superseded by Iteration 5** (the second review). Three things below were wrong in one respect each and are
> marked where they occur: the denominator of the completeness rates (F4, F5: identified minutes, not the frames the
> rule ran on), the greedy matching (F2) and the "like for like" comparison in the numbers. The rest stands.

Order of work: each finding got a failing test first (or a mutation that shows the old behaviour
fails it), then the fix. Nothing was re-tracked and no existing `.npz` was rewritten.

| # | Finding | Fix | Evidence |
|---|---|---|---|
| F1 blocking | A punch-free chain (a referee) took a slot, was dropped AFTER solving, and left boxer B empty (A 60 s / 10 punches, B 10 s / 1, referee 60 s / 0) | `linking.solve_chains`: an exact integer program (`scipy.optimize.milp`, zero gap, 300 s limit, anything but "optimal" is a `RuntimeError`) over a two-state layered path model (state 0 = no punch yet, state 1 = holds one; only state 1 may end). networkx (never pinned) is gone. Costs are rounded to 1/1000 point so the solver's absolute gap cannot hide a difference. | Astra's three tracks return A + B = 129 (was A + empty); 800 random instances (7 parameter sets) match an exhaustive enumeration of every pair of vertex-disjoint chains exactly, and the punch rule changes the optimum in at least 20 of them (asserted); negative: a time-limited or failed solve raises |
| F2 blocking | A burst was "explained" by ANY label within 0.1 s, including one thrown by the other boxer, and one burst near two labels counted both | `completeness.find_bursts`: one-to-one, nearest first (**greedy; replaced by an optimal assignment in Iteration 5, R2**), a boxer's OWN labels first (a label is owned by the boxer whose identity holds its credited track id in the event's frames), then the remaining bursts of both boxers against the UNATTRIBUTED labels, each label once. `match` is own / unattributed / none; the matched label's hand is recorded (`hand_agrees`, information only, 62% of matched bursts) and not required. The count is now called "unexplained = candidates for missed labels", not an upper bound; `events_with_burst` is gone. | both boxers burst together, only A labelled: B's burst is unexplained; one burst between two labels matches one; one unattributed label near two bursts matches one; greedy nearest-first unit tests |
| F3 blocking | Joined fragments read a pose jump at a handover as wrist speed (~10 body lengths/s at 50 fps, threshold 3.62) | Rows are split into segments at an id change or a missing frame; `bridge`, `wrist_signal`, `detect` run per segment; segments under the smoothing window (5 frames) are skipped and counted (`short_segments`), segments with no confident body are `bodyless_segments`. | two stationary fragments a torso apart: the joined track reads a burst (asserted in the test as the fault), per segment there is none; a real punch inside one fragment is still found |
| F4 blocking | All labels of the video were divided by identified boxer-minutes (**the matched fix still divided by identified minutes: superseded by Iteration 5, R1**) | `own_labels_per_identified_boxer_min` (labels owned by an identified boxer / his identified minutes) and `labels_per_video_boxer_min` (all labels / 2 x video minutes); unattributed labels reported as a count and share; durations and counts aggregated unrounded, rounded only when the json is written; the CLI line compares unexplained with OWN labels per identified boxer-minute | Astra's example (60 labels, 0.1 min each): 6 / 0.2 = 30, not 300; two videos of 0.0004 min each still divide by 0.0016 |
| F5 | "Signal minutes" averaged the two hands | (`signal_min` is now `exposure_min`, an explicit mask: Iteration 5) `signal_min` = elapsed minutes with EITHER hand valid (new `punchfind.signal_frames`, the one definition shared with `signal_coverage`), plus per-hand minutes and per-hand burst / unexplained counts | left wrist invalid, right valid throughout: signal minutes = elapsed, not half |
| F6 | Stratified quota: ties took the first 40 video names whatever the seed, small pools were over-sampled | `completeness.random_sample`: seeded simple random sample without replacement over ALL unexplained bursts, inclusion probability n/N for every burst; the csv has `inclusion_prob` (`user_label` still empty), the json the population N | 2,000 seeds over 12 bursts: every inclusion rate within 3 standard errors of 1/3; 60 videos x 3 bursts, sample 40: every video appears across seeds |
| F7 | The held-out test would pass with leakage (every fragment had many labels) | `olympic_link.analyse` returns `ring_heldout`; new scene where B's last fragment is supported only by held-out (odd) labels | held-out share 14/18 = 77.8%, and it WOULD read 100% if all labels leaked in (asserted); changing the held-out events' glove boxes moves the full ring and not `ring_heldout`; mutation: pass all attrs, 4 checks fail |
| F8 blocking | Size, fps and length do not identify a video; a renamed table was accepted by `load_video` and the overlays | `pose_yolo.verify_binding`: the table's `source` must equal the video's file name; `PoseTable` / `PoseSequence` gain `video_fp` = `video.fingerprint(path)` (sha256 of size as 8 bytes big-endian + first 4 MiB + last 4 MiB), written by `run_ingest`, round-tripped by save/load (old files load as ""), verified when present. Empty = accepted on the name alone and reported `video_bound: false`. Used by `olympic_link.load_video`, `olympic_attribute.py`, `render_tracks_overlay`, `render_linked_overlay`, and (content only) `render_overlay`. | same size, rate and length, other name: refused; same name, other recording, with a fingerprint: refused; legacy table: accepted, reported unbound; the fingerprint does not see an edit in the middle of a file (asserted, documented) |
| F9 | `PoseTable` allowed id -1, and with empty identities (`membership == -1`) both boxers selected every -1 row | `boxer_rows`: `(want >= 0) & (want == track_id)`; `PoseTable` refuses negative ids on construction and on load | empty linking against a table holding id -1: no rows, no bursts, no minutes |

**Existing assertions replaced (not weakened), and fixtures changed**
- `ring time valued 10x: a chain with no credited punch is emptied` asserted `identities[1] == ()` and the
  first identity equal to boxer A. That encoded the post-filter. With the rule inside the optimisation and
  ring time worth 5 points a second, the exact optimum may use a referee fragment as the tail of a punching
  chain (documented residual risk 2), so the assertion is now "no identity is punch-free and no
  referee/spectator-only chain is reported".
- The completeness tests that called `_near_label`, `events_with_burst`, `stratified_sample` and the
  csv header were rewritten for `span_gap`, `labels`, `random_sample` and the new header; the 5-frame
  tolerance boundary (5 frames inside, 6 not) is kept as `span_gap`.
- Fixtures: the planted scene's table source is now `GH000001.mp4` (the video's real name; it was
  `.MP4`), and the tables built to be "another video" (frame rate, length) now carry the right name, so those
  tests still fail for their own reason and not for the new name check.

**Decisions that differ from or sharpen the brief**
- Segments break at every id change. As first built (the brief's wording) they also broke on ANY missing
  frame inside one id, which dropped the 0.05 s pose bridge the rule was tuned with and skipped 549 of 3,542
  segments as too short. **Changed in review (a reviewer, after a second reviewer's report):** inside ONE id a hole of up to
  `BRIDGE_S` (2 frames at 50 fps) is the same person and is bridged by `punchfind.bridge` as before; only a
  longer hole ends the segment. Now 262 of 2,945 segments are under 5 frames (default linking). Tests: a 2-frame
  hole is bridged (its frames count as signal), a 3-frame hole splits, an id change splits even with no hole;
  mutations "strict split" (1 fail) and "no id split" (6 fail) are caught.
- A segment is "too short" under 5 frames (0.1 s at 50 fps = the smoothing window); this value was set
  before the real run.
- Greedy nearest-first matching, ties by frame, boxer, label index: deterministic, not a maximum matching.
  **Superseded by Iteration 5, R2** (Astra's counterexample: bursts at 100 and 120, owned labels [99,120] and [95,95]).
- `Linking` gained `solve_s` (not part of equality); `olympic_link.py` reports `link_solve_s`,
  `video_bound`, and pooled `videos_content_bound`.
- The held-out linking is solved too; its time is not recorded, only the full-label solve.

**Mutation check (each guard reverted in a scratch copy; the named suites must fail)**
| Guard reverted | Result | First failing checks |
|---|---|---|
| F1 punch rule moved back outside the optimisation (all fragments treated as punching, dead chains dropped after) | caught, 5 fail | Astra's three tracks (114 instead of 129); brute force 746/800 exact; lone punch-free fragment |
| F2a ownership ignored (any label explains any boxer) | caught, 4 | burst at 300 explained by the unattributed label; label counts; coincident bursts |
| F2b not one-to-one | caught, 6 | label used up; label already taken; unattributed label matched once |
| F3 segments ignore id changes (joined) | caught, 4 | per-segment: no burst at a handover; real punch still found; short-segment count |
| F4 rate uses all labels | caught, 2 | 6 / 0.2 = 30 not 300 |
| F5 signal minutes averaged over hands | caught, 1 | left invalid, right valid: elapsed minutes |
| F6 sample = first n bursts | caught, 4 | different seed differs; inclusion rate within 3 SE; every video can appear |
| F7 held-out linking sees all labels | caught, 4 | held-out share below 100%; would-be 100%; ring_heldout |
| F8a source not compared with the video name | caught, 4 (+1 test_tracks) | renamed table refused (load_video, CLI, tracks overlay) |
| F8b fingerprint not verified | caught, 3 (+1 test_tracks, +1 test_ingest) | same name other content refused |
| F8c overlays skip `verify_binding` | caught, 4 | renamed / wrong-fingerprint table refused by both overlays |
| F9a `(want >= 0)` removed | caught, 3 | empty linking selects no rows; no bursts, no minutes |
| F9b negative ids accepted | caught, 2 (+2 test_tracks) | construction and load refused |

**Numbers, before -> after (29 Olympic videos, default parameters, `results/*.json`, gitignored)**
| Quantity | Before | After |
|---|---|---|
| credited punches on A or B, all / half held out | 65.4% / 50.7% | 65.4% / 50.7% (901 + 684 of 2,423) |
| both identified: median video / frame-weighted | 57.2% / 50.5% | 57.2% / 50.5% |
| co-presence (median) | 78% | 77.9% |
| with `--reacquire`: credited on A or B, all / held out | 81.7% / 59.1% | 81.7% / 59.1% |
| `--reacquire`: both identified, median / frame-weighted | 43.5% / 44.5% | 43.5% / 44.5% |
| videos whose fragments in A / B changed | | 0 of 29 (and 0 of 29 with `--reacquire`) |
| MILP solve time per video (full-label linking) | | default: max 0.09 s, total 1.3 s; `--reacquire`: max 2.6 s, total 30 s; none near 60 s |
| bursts / unexplained | 8,135 / 5,506 | 7,984 / 5,772 (own 973, unattributed 1,239, none 5,772) |
| unexplained per identified boxer-minute (**denominator superseded: Iteration 5 divides by exposure**) | 37.2 | 39.0 (148.1 identified boxer-minutes) |
| own labels per identified boxer-minute | (31.1 = ALL labels over the same minutes, wrong) | 10.7 |
| labels per video boxer-minute (all labels / 2 x 104.3 min) | | 22.1 |
| unattributed labels | | 3,020 of 4,605 (65.6%); 57.0% with `--reacquire` |
| labels matched to a burst | 2,705 "within 0.1 s" (not one-to-one) | 2,212 one-to-one (973 own, 1,239 unattributed) |
| `--reacquire`: unexplained / bursts, per identified boxer-minute, own labels | 5,647 / 8,694, 39.8 | 6,185 / 8,657, 43.6, 14.0 |

- The linking is unchanged on this data: the punch-free-chain displacement of F1 is real (the unit test shows
  it) but did not occur in any of the 29 videos. The completeness numbers move because of F2-F4: the
  unexplained count rises (a label that sat next to two bursts, or thrown by the other boxer, no longer
  excuses them) and the like-for-like labelled rate falls to 10.7 per identified boxer-minute, because two
  thirds of the labels cannot be credited to an identified boxer (the identities cover 65.4% of credited
  punches and only 53% of labels are credited at all).
- **Still not a finding about the labels.** Unexplained bursts are candidates; nothing about missed labels follows until
  the sample is classified. (This bullet used to compare 53.9 bursts per identified boxer-minute with 22.1 labels per
  video boxer-minute as "like for like" and read the 2.4 times ratio against the rule's V6 precision. Those are two
  different observation populations and the inference needs recall and transfer assumptions nobody has tested: withdrawn
  in Iteration 5, R1.) The sample (seed 20261007, 40 of 5,772, inclusion probability 0.0069 each) is for you to
  classify; nothing was viewed or judged here.
  The GH078416 linked overlay was re-rendered (`results/ingest/olympic/GH078416.linked.overlay.mp4`) and NOT opened.
- All 29 tables are unbound by content (`video_bound: false`, `videos_content_bound: 0`): they predate the
  fingerprint and were accepted on the file name, which matched for all 29. Re-running `run_ingest.py --track`
  would bind them; that was not done.

- **Tests:** `test_linking.py` 224 (was 130), `test_tracks.py` 55 (34), `test_ingest.py` 114 (109);
  unchanged and green: olympic 33, hands 38, labels 50, pipeline 260, env 36.
- **Security pass:** a stored fingerprint is validated as 64 lowercase hex on construction and load (a loaded
  file is untrusted); the name and fingerprint checks run before the video is decoded or any output is
  written; the fingerprint reads two 4 MiB windows with `open(..., "rb")` and no subprocess; negative track
  ids and malformed fingerprints fail closed; the solver cannot return a non-optimal answer silently (status,
  fractionality, chain structure and the objective are all re-checked from the extracted chains); no new
  dependency (networkx removed from the code, scipy was already pinned).
- **Residual risks:** (1) the fingerprint identifies a file, not every byte (an edit in the middle of a
  recording is invisible to it; asserted and documented); (2) the 29 existing tables are bound by name only;
  (3) nearest-first matching is greedy, not a maximum matching (**fixed in Iteration 5**); (4) a referee fragment can still be absorbed
  as a tail or connector of a punching chain; (5) the identity swap after a crossing is unchanged;
  (6) 262 short segments and 6 bodyless ones are skipped, so some real bursts are unseen.

### Iteration 5 -- Astra 6 second review (2026-10-07; verdict FIX FIRST: 1 blocking, 5 not)

Reviewed `a9204be..2d5f244`. 2d5f244 (a reviewer, after the round-2 fixes) had already changed `boxer_segments` to bridge holes of
up to `BRIDGE_S` inside ONE id and to split only at an id change or a longer hole; that stays.

| # | Finding | Fix | Evidence |
|---|---|---|---|
| R1 blocking | `identified_min` counted original rows, but detection runs on BRIDGED segments (a peak can land on an interpolated frame) and skips short / bodyless segments that still counted as identified time; the report called bursts per identified boxer-minute against labels per video boxer-minute "like for like" | `completeness.find_bursts` builds a per-boxer **exposure** mask over the video frames: every frame of every processed segment (not short, not bodyless) where the bridged track has a valid signal for at least one hand (`signal_frames(track).any(axis=1)`), bridged frames INCLUDED (policy stated in the docstring). `exposure_min` = mask.sum() / fps / 60. Every burst frame is asserted to lie in its boxer's exposure (`RuntimeError` otherwise). Own labels are "in exposure" when their span touches the owner's exposure (`own_labels_in_exposure` / `_outside_exposure`); unattributed labels when it touches either boxer's. Rates per boxer and pooled (raw counts and raw frame counts summed, divided last): `unexplained_per_exposure_min`, `bursts_per_exposure_min`, `own_labels_in_exposure_per_exposure_min`. `identified_min` stays as a description and nothing divides by it; `labels_per_video_boxer_min` stays and is labelled a different population. The like-for-like sentence and the precision-based reading are removed from the CLI print and from Iteration 4. | integrated through `find_bursts`: (a) the peak at frame 102, which has no row in the table, is in the numerator and the mask (398 identified rows, 400 exposure frames); (b) a 3-row skipped segment adds 3 identified frames and 0 exposure frames; (c) an owned label wholly inside the skipped segment is `own_labels_outside_exposure`; a patched `signal_frames` that empties the mask makes the run raise; no key anywhere is named `*per_identified*` |
| R2 | Greedy nearest-first matching strands a burst (tolerance 5, bursts at 100 and 120, owned labels [99,120] and [95,95]: the broad label goes to 100, 120 is left unexplained) and favours A on ties | `completeness._match` is an optimal assignment (`scipy.optimize.linear_sum_assignment`), **optimal within each phase, not necessarily the global maximum (Iteration 6, D1)**: cost = span gap for feasible pairs (gap <= tolerance), a larger-than-any-feasible-assignment M otherwise, infeasible pairs dropped afterwards, so it maximises the number of matches and then minimises the total gap. Two phases kept (own labels per boxer, then both boxers' leftovers against unattributed labels; a label once). Exact ties: bursts sorted by (frame, key) and labels by (first, last, index), and two integer tie-break terms scaled below one frame of gap, so the burst that sorts first by (frame, boxer) takes a contested label. | the counterexample, a unit test and the same case through `find_bursts` (303 and 337, a broad and a narrow owned label); 400 random small instances equal a brute-force enumeration of (most matches, least total gap); simultaneous bursts of both boxers for one label: A wins the exact tie whatever the list order (exact ties go to the burst that sorts first by (frame, boxer)) |
| R3 | `PoseTable.load` converted (`astype`, `int`, `float`) before checking: a stored id -0.5 became id 0, a frame 1.9 became frame 1 | `pose_yolo._whole_array` / `_scalar` / `_float_array`: dtype kind, finiteness, whole values, range, 1-D shape, scalar shape, `n_frames`/`width`/`height` whole and >= 1, `fps` finite and > 0, a missing field is a `ValueError` naming it. `PoseSequence.load` has the same checks. | fractional negative id, fractional frame, NaN id, whole-but-negative frame, string ids, 2-D frame array, NaN / zero fps, fractional n_frames and width, zero height, vector fps, missing field; whole-valued FLOAT arrays still load unchanged |
| R4 | Labels enough for a ring but no eligible identity: the rate was `None` and `:.1f` raised after the files were written | `olympic_completeness.main` computes the summary and both files' contents first, then writes, then prints; absent rates print `n/a` and are `null` in the json | every person re-identified every 5 frames (under the 0.2 s fragment minimum): 33 labels, no identity, exit 0, rates null, `n/a` printed, sample file header only |
| R5 | Equal MILP optima have no canonical selection rule | **Claim narrowed, no secondary optimisation** (as instructed): `linking.py` now says the answer is repeatable for the same input and solver version and that among exactly equal optima the choice is the solver's; the A / B sort names the chains after the selection | docstring only |
| R6 | Held-out pooling rebuilt counts from rounded shares | `analyse` keeps `credited_heldout_inside` (integer); `olympic_link.pooled_heldout_share` pools integer counts and divides once | 1 of 5 and 4 of 6: rebuilt from rounded shares 0.4546, the integer ratio 5 / 11 = 0.4545 |

**Existing assertions replaced or extended (nothing weakened)**
- `signal_min` -> `exposure_min` in four assertions: the same quantity (the frames with a valid signal in processed segments) under its new
  name, now an explicit mask. The values asserted are unchanged (200 / 400 frames).
- The label-count dict equality gains `unattributed_in_exposure: 1`.
- The Iteration 4 rate tests (`own_labels_per_identified_boxer_min`, the 60-labels / 0.2 min = 300 example, the 0.0004-minute rounding
  case, and the end-to-end checks on `ca["...per_identified_boxer_min"]`) encoded the identified-minute denominator that R1 shows to
  be the wrong observation. They are replaced by exposure-based equivalents with the same intent (matched numerator and denominator;
  aggregation on unrounded numbers; a real fact from the planted run); the 6 / 0.2 = 30 and 4 / 0.0016 = 2500 arithmetic is kept.
- The test titled "greedy nearest-first" is renamed; its assertion (the burst on the label takes it, the other takes the farther one)
  is unchanged and still holds under the optimal assignment.

**Mutation check (each change reverted or broken in the working tree, `git checkout` after; the named suite must fail)**
| Mutation | Suite | Result | First failing checks |
|---|---|---|---|
| R1 rates divide by `identified_min` | test_linking | caught, 4 fail | 6 / 0.2 = 30; 2 / 0.2 = 10; per boxer 1 / 0.1; own-labels rate 1 over 200 frames not 2 over 203 |
| R1 burst-in-exposure assertion removed | test_linking | caught, 1 | "a burst outside the exposure ... is a RuntimeError" |
| R1 bridged frames excluded from exposure | test_linking | caught (the run raises the new RuntimeError at frame 102, the suite aborts) + 1 fail | the 2-frame-hole segment test |
| R1 skipped segment counted in exposure | test_linking | caught, 4 | (b), (c), the video-row rate, the short-segment test |
| R2 greedy matching restored | test_linking | caught, 4 | counterexample; labels used up; 400-instance brute force; in the scene |
| R3 `PoseTable.load` coerces before validating | test_tracks | caught, 13 | -0.5 id, 1.9 frame, NaN id, negative frame, string ids, 2-D frame, fractional / zero scalars |
| R3 `PoseSequence.load` coerces | test_ingest | caught, 3 | NaN fps, fractional width, zero height |
| R4 `n/a` formatting removed (`None:.1f`) | test_linking | caught (the run crashes on the new test) | zero-identity video |
| R6 pooled from rounded shares | test_linking | caught, 2 | 5 / 11 = 0.4545 |

**Numbers, before (2d5f244) -> after (29 Olympic videos, default parameters; `results/*.json`, gitignored)**
| Quantity | Before | After |
|---|---|---|
| bursts / unexplained | 7,984 / 5,772 | 7,984 / 5,766 (the optimal assignment explains 6 more) |
| exposure boxer-minutes vs identified boxer-minutes | (identified 148.06; signal 146.40) | exposure 146.40, identified 148.06 |
| unexplained per exposure-minute | 39.0 per identified | 39.4 |
| bursts per exposure-minute | 53.9 per identified | 54.5 |
| own labels in exposure, per exposure-minute | 10.7 (1,585 own labels / identified) | 10.8 (1,580 in exposure) |
| own labels outside exposure | | 5 of 1,585 |
| unattributed labels in exposure | | 2,749 of 3,020 |
| labels matched (own / unattributed) | 973 / 1,239 | 973 / 1,245 |
| sample N (population), inclusion probability | 5,772 | 5,766, 0.00694 |
| `--reacquire`: bursts / unexplained | 8,657 / 6,185 | 8,657 / 6,178 |
| `--reacquire`: exposure vs identified boxer-minutes | | 140.78 vs 141.79 |
| `--reacquire`: unexplained / bursts / own labels in exposure per exposure-minute | 43.6 / 61.1 / 14.0 per identified | 43.9 / 61.5 / 14.0 |
| `--reacquire`: own labels outside exposure; unattributed in exposure | | 7 of 1,980; 2,440 of 2,625 |
| `--reacquire`: labels matched (own / unattributed) | 1,212 / 1,260 | 1,212 / 1,267; sample N 6,178 |
| held-out credited share, pooled | 50.7% | 601 of 1,185 = 50.72% (credited on A or B, all: 1,585 of 2,423 = 65.4%) |
| MILP solve time | max 0.09 s, total 1.3 s | max 0.09 s, total 1.22 s |

- **The correction is one of definition, not of size.** On this data the exposure is 98.9% of the identified time (146.40 of 148.06
  minutes; 99.2% with `--reacquire`), no burst fell outside it in any of the 29 videos (the new assertion held in both runs), and
  the rates move by about 1%. What changed is that the numerator and the denominator are now guaranteed to describe the same frames,
  and that the unsupported comparison is gone. Nothing new is learned about the labels: unexplained bursts are candidates for missed
  labels, and the sample (40 of 5,766, for YOU to classify; nothing was viewed or judged) is what will say how many are punches.
- 2,749 of the 3,020 unattributed labels lie in time when a linked boxer was in the exposure: they have no owner among the selected boxer
  identities (ambiguous, unreached, or credited to a track outside A and B, a referee for instance), not that nobody was being observed.
  That is a statement about attribution, not about completeness.
- The linking is unchanged on this data: 0 of 29 videos changed which fragments are in A / B (R5 and R6 do not touch the optimisation);
  all 29 tables are still unbound by content (`videos_content_bound: 0`). The GH078416 linked overlay was not re-rendered or opened this round.

- **Tests:** `test_linking.py` 256 (was 224), `test_tracks.py` 71 (55), `test_ingest.py` 117 (114); unchanged and green: olympic 33,
  hands 38, labels 50, pipeline 260, env 36.
- **Security pass:** a stored pose file is validated before conversion (fractions, NaN, negative, wrong dtype, wrong shape, missing field
  all fail closed); the burst-in-exposure assertion turns a silent numerator / denominator mismatch into an error; the CLI computes
  everything before it writes, so a late formatting failure cannot leave half a result; no new dependency (`scipy.optimize` was already
  used by the MILP).
- **Residual risks:** (1) exposure is a property of the rule's signal, not of the footage: a frame can be in the exposure while the
  boxer is hidden behind the referee but the pose is confident; (2) the matching is optimal within each phase only (Iteration 6) and has ties beyond its tie-break (two different
  sets of bursts with the same position sum); there the solver decides, deterministically for the same input; (3) among exactly equal
  MILP optima the identity pick is the solver's (R5, claim narrowed and not engineered away); (4) the rule is V6's (AlphaPose, 24 fps)
  and untuned for 50 fps sparring, so "unexplained" says nothing about missed labels until the sample is classified; (5) the 29 tables
  are bound by file name only.

### Iteration 6 -- Astra 6 third review (2026-10-07; verdict FIX FIRST: 2 blocking, 2 documentation)

| # | Finding | Fix | Evidence |
|---|---|---|---|
| B1 blocking | `PoseSequence` accepted a `(F, 17)` bool array as the per-frame `detected` mask (`len()` gave F, the dtype passed, `xy[detected]` is valid indexing); both loaders `str()`'d `source` / `model` / `video_fp`, so a number or a vector became plausible text; "every stored field is validated before conversion" was untrue | `detected` must be 1-D (constructor, and on load also bool dtype via `_bool_vector`); `_text` requires a 0-d unicode array before `str()`; every stored field is read through the one `_field` missing-field helper in both loaders; the load docstrings now list exactly what is checked and what is left to the constructor | `(F, 17)` mask refused on load and, with finite coordinates, at construction; integer `detected` refused; numeric `source`, numeric `model`, vector `model`, vector `source`, numeric `video_fp` refused; missing `source` / `model` / `detected` are `ValueError`s naming the field; proper strings load unchanged |
| B2 blocking | No held-out observations reported as 0%: every video unable to estimate a held-out ring gave `0 / max(0, 1)`, printed "0.0% held out" | `pooled_heldout_share` returns `None` on a zero denominator (json `null`); the per-video share is `None` when there is no held-out punch to evaluate; the pooled print uses the absent-safe `_pct` | 16-label video (ring for the full linking, 8 per half: none for the held-out one): exit 0, pooled and per-video share null, counts exactly 0, CLI prints `n/a held out` and `n/a with half held out`, never `0.0%`; empty and all-zero pools are `None`; 0 of 4 is `0.0` (a real zero); a held-out half with a ring but no credited punch is `None` |
| D1 | "Optimal assignment" omitted that the two phases are optimised separately | Wording in `completeness.py` (module and `_match` docstrings), the json `matching` field, the project rules and the Iteration 5 row: optimal within each phase (own labels first, then unattributed), not necessarily the global maximum of explained bursts; exact ties go to the burst that sorts first by (frame, boxer), so to A on a simultaneous tie | the two-phase counterexample is a test of the CURRENT behaviour: owned label [302,337] takes burst 303, unattributed label [298,298] cannot reach 337, which stays `none`, although narrow -> 303 and broad -> 337 would explain both |
| D2 | "Every rate divides by exposure" excluded the whole-video density; "the attribution could not tell whose punch it was" did not describe labels credited to a track outside A and B | the project rules, `find_bursts` and the Iteration 5 text now say the burst rates and the exposure-qualified own-label rate divide by exposure and labels per video boxer-minute is a whole-video density; unattributed means no owner among the selected boxer identities (ambiguous, unreached, or credited to a track outside A and B, a referee for instance) | wording only |

**Mutation check (change broken in the working tree, restored with `git checkout`; the named suite must fail)**
| Mutation | Suite | Result | First failing checks |
|---|---|---|---|
| B1 `detected` shape check removed (load and constructor) | test_ingest | caught, 2 fail | `(F, 17)` mask on load; the same at construction |
| B1 string-field check removed | test_tracks | caught, 5 | numeric source, numeric model, vector model, vector source, numeric video_fp |
| B1 string-field check removed | test_ingest | caught, 2 | numeric source, vector model |
| B2 zero denominator -> 0 | test_linking | caught, 4 | null pooled share; per-video null and `n/a` print; empty / all-zero pool; held-out half with no credited punch |

The first version of the construction-time `(F, 17)` test passed under the mutation (the mask selected NaN coordinates and another
ValueError refused it for the wrong reason); it now uses finite coordinates and asserts the shape rule's message.

**Real data:** `olympic_link.py` on the 29 videos is unchanged: 601 of 1,185 held-out credited punches inside A or B = 0.5072
(50.7%), 1,585 of 2,423 = 65.4% overall; solver max 0.09 s, total 1.24 s. No pooled or per-video share was null there.
**Tests:** `test_linking.py` 263 (was 256), `test_tracks.py` 79 (71), `test_ingest.py` 124 (117); olympic 33, hands 38, labels 50,
pipeline 260, env 36 unchanged.
**Residual risks:** (1) the two phases are a policy (ownership first), so the explained count can be lower than a global assignment
would give; (2) ~~`credited_in_either_share` divides by `max(credited_punches, 1)`~~ **closed in review (a reviewer):** with no
credited punch the share is None (json null, printed n/a) per video and pooled; a measured 0.0 (credited punches, none on A or B) is
still 0.0; 3 tests, mutation `max(nc, 1)` restored -> 1 fail; (3) the two output files of `olympic_completeness.py` are
written one after the other, not as one atomic transaction.

### Mutation check, iterations 1-3 (each guard reverted in a copy; `tests/test_linking.py` must fail)
| Guard reverted | Result | First failing test |
|---|---|---|
| gap limit (reach unbounded) | caught, 4 fail | gap limit inclusive; 3 s hole never bridged |
| speed limit (always plausible) | caught, 6 | 139 px plausible / 141 px not |
| forward-only: both `lo` and `gap < 1` removed | caught, 2 | overlap by 5 frames; same-frame start |
| forward-only: either one alone | SURVIVED (redundant by design: each guards the other) | -- |
| node capacity 2 (overlapping chains) | caught, 7 | one boxer: second identity must be empty |
| chain start fee 0 | caught, 3 | same test (the only boxer gets cut in two) |
| punches worth nothing | caught (crash + fails) | exact boxer sets recovered |
| a chain needs no punch | caught, 1 | ring time valued 10x: no identity without a punch |
| ring prior: eligibility | caught, 3 | punches wrongly credited to a spectator |
| ring prior: share as reward weight | caught, 2 | reward = 10/punch + 0.5/s x ring share |
| credited ids must be in the table | caught, 1 | attributions from another video refused |
| re-acquisition: ordinary gap un-gated | caught, 5 | speed-limit tests |
| overlay: table vs video size/rate | caught, 2 | refuses another frame rate / size |
| overlay: linking vs table | caught, 2 | other source / other length |
| overlay: table longer than video | caught, 1 | longer recording refused |
| completeness: boxer rows by membership | caught (crash) | frame ids not increasing |
| completeness: tolerance inclusive | caught, 1 | 5 frames explains, 6 do not |
| completeness: explained flag | caught, 4 | planted bursts |
| CLI: `--overlay` id check | caught, 5 | `../GH000001` etc. refused |
| CLI: refuse incomplete tracks | caught, 1 | a `--max-frames` table refused |

- **Test:** `tests/test_linking.py` 130 (new); all other suites unchanged and green (tracks 34, olympic 33,
  hands 38, labels 50, ingest 109, env 36, pipeline 260, machine scripts 38).
- **Security pass:** tracks loaded by `PoseTable.load` (`allow_pickle=False`); the only subprocesses are in
  `video.py` (argument lists); the GoPro id is `[A-Za-z0-9]{1,32}` before it becomes a file name, the overlay
  is written only inside `--tracks`, refuses to overwrite the input, and refuses a table or linking from
  another video; stale attributions (ids not in the table), negative counts, NaN parameters and an incomplete
  tracks file are refused; JSON written with `allow_nan=False`; a video that fails is reported and the exit code is 1.
- **Residual risks:** (1) no appearance model: after two people cross with ids lost, the nearer continuation
  wins and the boxers swap (asserted in the tests as the actual behaviour); (2) a referee fragment beside a
  boxer's hole can be absorbed (punch evidence is the only separator), and a punch credited to the referee
  makes his fragment worth 10 points; (3) coverage is deliberately conservative (ring time is worth 0.5/s), so
  holes of seconds remain; (4) a burst on a bridged or fragment-handover frame is not flagged; (5) the rule is
  V6's (AlphaPose, 24 fps, hooks/uppercuts) and untuned for 50 fps sparring; (6) the overlay for GH078416
  (`results/ingest/olympic/GH078416.linked.overlay.mp4`) has NOT been viewed by anyone yet: identity swaps in
  the real data are unmeasured.

